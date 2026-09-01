"""Serving model predictions to the trading loop.

The loop reads :attr:`TradingLoop.model_probabilities`, a mapping of ticker to
``(probability, model_id, model_version)``. Nothing filled it in until this
module existed, so every strategy that consults a model saw ``None`` and fell
back to its own arithmetic. This is the worker that fills it.

What a prediction is allowed to be
----------------------------------
A calibrated probability that the YES outcome resolves true, or nothing at all.
There is deliberately no third option. Every path that cannot produce an honest
number -- no deployed model, missing features, a feature vector that fails its
point-in-time check, an estimator that raises -- returns ``None`` and the
strategy proceeds without a model view. A default of 0.5 would be indefensible:
it is a real prediction that the market is a coin flip, and a strategy sizing on
it would take positions against a number nobody computed.

The model has no authority
--------------------------
A prediction is one input among several. It reaches the strategy through
:class:`~kat.strategies.base.StrategyContext` and the risk engine's decision
record, and it has no route to an order of its own -- the same rule that governs
the LLM advisor. A miscalibrated model can make a strategy wrong; it cannot make
it act unsupervised.

Staleness
---------
Predictions carry the time they were computed and the feature vector they were
computed from. A prediction older than :data:`DEFAULT_MAX_AGE_MS` is dropped
rather than served, because a model output is exactly as perishable as the
features behind it, and a stale probability is more dangerous than none: it
looks current.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np

from kat.common.enums import MarketType
from kat.features.spec import FeatureVector
from kat.models.registry import ModelRegistry, RegisteredModel
from kat.observability.logging import get_logger
from kat.observability.metrics import metrics

log = get_logger(__name__)

__all__ = ["InferenceEngine", "Prediction", "predict_one"]

#: Beyond this a prediction is dropped rather than served. Matched to the
#: feature layer's own staleness tolerance: a probability cannot be fresher than
#: the vector it came from.
DEFAULT_MAX_AGE_MS = 30_000

#: Probabilities are clipped away from the boundaries. A model asserting 0.0 or
#: 1.0 is asserting certainty, and downstream arithmetic that divides by
#: ``1 - p`` would produce an infinite position from a rounding artefact.
MIN_PROBABILITY = 1e-4
MAX_PROBABILITY = 1.0 - MIN_PROBABILITY


@dataclass(slots=True)
class Prediction:
    """One model output, with everything needed to audit it later."""

    ticker: str
    probability: float
    model_id: str
    model_version: str
    computed_ms: int
    #: Information time of the oldest feature behind this number. The prediction
    #: is no fresher than this, whatever ``computed_ms`` says.
    feature_ts_ms: int
    feature_count: int = 0

    def age_ms(self, now_ms: int) -> int:
        return max(0, now_ms - self.computed_ms)

    def is_fresh(self, now_ms: int, *, max_age_ms: int = DEFAULT_MAX_AGE_MS) -> bool:
        return self.age_ms(now_ms) <= max_age_ms

    def as_loop_entry(self) -> tuple[float, str, str]:
        """The shape :attr:`TradingLoop.model_probabilities` stores."""
        return (self.probability, self.model_id, self.model_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "probability": round(self.probability, 6),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "computed_ms": self.computed_ms,
            "feature_ts_ms": self.feature_ts_ms,
            "feature_count": self.feature_count,
        }


def predict_one(
    model: RegisteredModel,
    vector: FeatureVector,
    *,
    now_ms: int,
) -> Prediction | None:
    """Score one feature vector, or return ``None`` with a logged reason.

    The feature vector is reordered into the exact column order the model was
    trained on. That reordering is not a nicety: a model handed its features in
    a different order produces confident nonsense, and nothing downstream would
    detect it.
    """
    names = model.feature_names
    if not names:
        log.warning("inference.model_has_no_feature_names", model=model.key)
        return None

    values: list[float] = []
    oldest_ts = now_ms
    for name in names:
        value = vector.values.get(name)
        if value is None or value.value is None:
            # A missing feature is not a zero. Imputing one would silently move
            # the model along whichever axis that feature controls.
            log.debug(
                "inference.feature_missing",
                model=model.key, ticker=vector.ticker, feature=name,
            )
            return None
        values.append(float(value.value))
        oldest_ts = min(oldest_ts, value.info_ts_ms)

    matrix = np.asarray([values], dtype=float)
    if not np.isfinite(matrix).all():
        log.warning(
            "inference.non_finite_features", model=model.key, ticker=vector.ticker
        )
        return None

    try:
        raw = model.estimator.predict_proba(matrix)[:, 1]
        if model.calibrator is not None:
            # The calibrator travels with the estimator for a reason: an
            # uncalibrated probability was not what the model was selected on.
            raw = model.calibrator.transform(raw)
        probability = float(np.clip(raw, MIN_PROBABILITY, MAX_PROBABILITY)[0])
    except Exception as exc:
        log.warning(
            "inference.failed", model=model.key, ticker=vector.ticker, error=str(exc)
        )
        return None

    if not np.isfinite(probability):
        log.warning("inference.non_finite_probability", model=model.key)
        return None

    return Prediction(
        ticker=vector.ticker,
        probability=probability,
        model_id=model.model_id,
        model_version=model.version,
        computed_ms=now_ms,
        feature_ts_ms=oldest_ts,
        feature_count=len(names),
    )


class InferenceEngine:
    """Keeps a trading loop's ``model_probabilities`` current.

    One instance per market type, because the two domains are independent and a
    perpetual model must never score an event contract. The engine holds the
    deployed model for its domain and refreshes it when the registry deploys a
    new one.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        market_type: MarketType,
        *,
        # None means "serve whatever is deployed for this domain". Naming a
        # target here is how this engine spent its life looking for
        # "settlement" while the trainer only ever deploys "forward_move_30m".
        target: str | None = None,
        max_age_ms: int = DEFAULT_MAX_AGE_MS,
        refresh_interval_s: float = 300.0,
    ) -> None:
        self.registry = registry
        self.market_type = market_type
        self.target = target
        self._reported_absent = False
        self.max_age_ms = max_age_ms
        self.refresh_interval_s = refresh_interval_s

        self.model: RegisteredModel | None = None
        self.predictions: dict[str, Prediction] = {}
        self.scored = 0
        self.skipped = 0
        self.dropped_stale = 0
        # None, not 0. With `0` the "is a refresh due" arithmetic only comes
        # out true once the clock passes the interval, which is always the case
        # for an epoch clock and never the case for a simulated one starting
        # near zero -- so the engine would silently never load a model in
        # simulation while working in production.
        self._last_refresh_ms: int | None = None

    # -- model lifecycle -----------------------------------------------------

    async def refresh_model(self) -> RegisteredModel | None:
        """Pick up whatever the registry currently has deployed.

        A deployment swap invalidates every cached prediction: they were
        computed by a different model, and serving them under the new model's id
        would make the audit trail a lie.
        """
        try:
            deployed = await self.registry.deployed(self.market_type, self.target)
        except Exception as exc:
            log.error(
                "inference.registry_unavailable",
                market_type=self.market_type.value, error=str(exc),
            )
            return self.model

        if deployed is None:
            if self.model is not None:
                log.warning(
                    "inference.model_undeployed",
                    market_type=self.market_type.value, previous=self.model.key,
                )
                self.model = None
                self.predictions.clear()
            elif not self._reported_absent:
                # Said once, because it is otherwise indistinguishable from a
                # worker that is running fine. Serving nothing was the single
                # largest reason the platform declined to trade, and it looked
                # exactly like everything working.
                self._reported_absent = True
                log.warning(
                    "inference.no_model_deployed",
                    market_type=self.market_type.value,
                    target=self.target or "(any)",
                    note="strategies needing a model probability will stand aside",
                )
            return None
        self._reported_absent = False

        if self.model is None or deployed.key != self.model.key:
            log.info(
                "inference.model_deployed",
                market_type=self.market_type.value,
                model=deployed.key,
                features=len(deployed.feature_names),
                previous=self.model.key if self.model else None,
            )
            self.model = deployed
            self.predictions.clear()
        return self.model

    @property
    def ready(self) -> bool:
        return self.model is not None and bool(self.model.feature_names)

    # -- scoring -------------------------------------------------------------

    def score(self, vectors: dict[str, FeatureVector], now_ms: int) -> dict[str, Prediction]:
        """Score every market we have a usable feature vector for.

        Markets that cannot be scored are *removed* from the prediction map
        rather than left holding their previous value. Leaving a stale entry
        behind is how a strategy ends up trading a probability computed from
        features that have since gone missing.
        """
        if not self.ready:
            return {}

        model = self.model
        assert model is not None  # narrowed by self.ready

        fresh: dict[str, Prediction] = {}
        for ticker, vector in vectors.items():
            prediction = predict_one(model, vector, now_ms=now_ms)
            if prediction is None:
                self.skipped += 1
                continue
            fresh[ticker] = prediction
            self.scored += 1

        self.predictions = fresh
        metrics.gauge("inference.markets_scored", len(fresh))
        return fresh

    def loop_entries(self, now_ms: int) -> dict[str, tuple[float, str, str]]:
        """The mapping the trading loop assigns to ``model_probabilities``.

        Stale predictions are filtered here rather than at write time, so a loop
        that stalls for a minute sees an empty map instead of a minute-old view
        of the market.
        """
        entries: dict[str, tuple[float, str, str]] = {}
        for ticker, prediction in self.predictions.items():
            if not prediction.is_fresh(now_ms, max_age_ms=self.max_age_ms):
                self.dropped_stale += 1
                continue
            entries[ticker] = prediction.as_loop_entry()
        return entries

    async def update(self, loop: Any, vectors: dict[str, FeatureVector], now_ms: int) -> int:
        """Refresh the model if due, score, and publish into the loop.

        This is the single call the engine makes. Returning the count lets the
        supervisor distinguish "the model is serving nothing" from "the worker
        is not running", which look identical from the loop's side.
        """
        due = (
            self._last_refresh_ms is None
            or now_ms - self._last_refresh_ms >= self.refresh_interval_s * 1000
        )
        if due:
            self._last_refresh_ms = now_ms
            await self.refresh_model()

        if not self.ready:
            # Publish an empty map rather than leaving the previous one in
            # place: no model means no view, and the loop must see that.
            loop.model_probabilities = {}
            loop.model_target = None
            return 0

        self.score(vectors, now_ms)
        entries = self.loop_entries(now_ms)
        loop.model_probabilities = entries
        # Published beside the numbers, because a probability whose subject is
        # unknown cannot be used safely by anything downstream.
        loop.model_target = self.model.target if self.model else None
        return len(entries)

    async def run(self, loop: Any, *, interval_s: float = 5.0) -> None:
        """Poll the loop's features and keep its predictions current.

        Registered as a supervised component. A failure inside one cycle is
        logged and the loop continues without predictions, which is a
        degradation rather than an outage: strategies that need a model stand
        aside, and the ones that do not carry on.
        """
        log.info(
            "inference.started",
            market_type=self.market_type.value,
            target=self.target,
            interval_s=interval_s,
        )
        while True:
            try:
                now_ms = loop.clock.now_ms()
                await self.update(loop, dict(getattr(loop, "feature_vectors", {})), now_ms)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error(
                    "inference.cycle_failed",
                    market_type=self.market_type.value, error=str(exc), exc_info=True,
                )
                loop.model_probabilities = {}
            await asyncio.sleep(interval_s)

    def stats(self) -> dict[str, Any]:
        return {
            "market_type": self.market_type.value,
            "target": self.target,
            "model": self.model.key if self.model else None,
            "features": len(self.model.feature_names) if self.model else 0,
            "live_predictions": len(self.predictions),
            "scored": self.scored,
            "skipped": self.skipped,
            "dropped_stale": self.dropped_stale,
        }
