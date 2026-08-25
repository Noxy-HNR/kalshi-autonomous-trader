"""The autonomous research loop.

Three jobs, on three different cadences, because they answer questions at
different timescales:

**Regime detection** (fast). Classify each market's current conditions and
publish them into the trading loops, so a strategy can decline a regime it was
never validated in. Seconds.

**Validation** (slow). Re-run the full protocol in :mod:`kat.simulation.
validation` for every strategy below LIVE and record what the gates say.
Minutes to hours.

**Promotion** (slowest, and conditional). Act on the gate verdicts -- or, in the
recommended default mode, decline to act and stage the recommendation for a
human instead.

Autonomy is a property of this loop, not of the gates
-----------------------------------------------------
The gates are identical in every mode: :func:`~kat.discovery.lifecycle.
evaluate_promotion` does not know or care what mode the platform is in, and
there is no mode in which a gate is skipped. What the mode changes is who acts
on the verdict:

``FULL_AUTONOMOUS``
    Promotions are applied, including into live capital.
``AUTONOMOUS_DISCOVERY``
    Promotions are applied up to PAPER. Anything that would put real money
    behind a strategy waits for a human.
``GATED_DISCOVERY`` (default)
    Nothing is applied. Every promotion that passed its gates is recorded as a
    recommendation for approval from the dashboard.
``HUMAN_STRATEGIES``
    No promotions at all; the loop measures and reports.

A strategy failing its gates is never promoted in any mode, so the modes differ
only in what happens to a *pass*. That asymmetry is deliberate: autonomy should
widen what the platform may do on good evidence, never narrow what counts as
good evidence.

Demotion is not gated by mode
-----------------------------
A live strategy whose measured performance has collapsed is demoted immediately
in every mode, including the ones where promotion needs a human. Waiting for
approval to *stop* doing something is the wrong default, and a mode setting is
not a reason to keep losing money.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from kat.common.enums import MarketType, OperatingMode, StrategyState
from kat.common.timeutil import Clock
from kat.config.settings import ResearchSettings
from kat.discovery.lifecycle import evaluate_promotion
from kat.marketdata.state import MarketState
from kat.observability.logging import get_logger
from kat.observability.metrics import metrics
from kat.research.regime import RegimeDetector, RegimeReading
from kat.strategies.registry import StrategyRegistry

log = get_logger(__name__)

__all__ = ["PromotionRecommendation", "ResearchLoop"]

#: Observations kept per market for regime classification. Two hours at a
#: five-second cadence, which is enough for the sixty-minute features and short
#: enough that a regime change is visible rather than averaged away.
REGIME_WINDOW = 1_440

#: States from which a promotion is even worth evaluating. LIVE has no next
#: rung, and RETIRED is not coming back without a human.
_PROMOTABLE = (
    StrategyState.DISCOVERY,
    StrategyState.DEVELOPMENT,
    StrategyState.BACKTEST,
    StrategyState.OUT_OF_SAMPLE,
    StrategyState.WALK_FORWARD,
    StrategyState.ROBUSTNESS,
    StrategyState.PAPER,
    StrategyState.PROBATION,
)

#: Above this rung a promotion puts real money behind a strategy.
_REAL_MONEY_STATES = (StrategyState.PROBATION, StrategyState.LIVE)


@dataclass(slots=True)
class PromotionRecommendation:
    """A promotion whose gates passed but which was not applied.

    Held for the dashboard to approve or reject. Carries every gate result, so
    an operator approving one is looking at the same evidence the loop was.
    """

    strategy_id: str
    current_state: StrategyState
    target_state: StrategyState
    gates: list[dict[str, Any]]
    created_ms: int
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "current_state": self.current_state.value,
            "target_state": self.target_state.value,
            "gates": self.gates,
            "created_ms": self.created_ms,
            "reason": self.reason,
        }


@dataclass(slots=True)
class _Window:
    """Rolling observations for one market's regime classification."""

    prices_cc: list[int] = field(default_factory=list)
    baseline: float | None = None

    def observe(self, price_cc: int, limit: int = REGIME_WINDOW) -> None:
        self.prices_cc.append(price_cc)
        if len(self.prices_cc) > limit:
            del self.prices_cc[: len(self.prices_cc) - limit]


class ResearchLoop:
    """Regime classification, validation and mode-dependent promotion."""

    def __init__(
        self,
        *,
        settings: ResearchSettings,
        registry: StrategyRegistry,
        market_states: dict[MarketType, MarketState],
        loops: dict[MarketType, Any],
        clock: Clock,
        detector: RegimeDetector | None = None,
        mode: OperatingMode = OperatingMode.GATED_DISCOVERY,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.market_states = market_states
        self.loops = loops
        self.clock = clock
        self.detector = detector or RegimeDetector()
        self.mode = mode

        self._windows: dict[str, _Window] = {}
        self.readings: dict[str, RegimeReading] = {}

        #: Metric blocks per strategy, produced by whatever ran the validation
        #: protocol. Keyed strategy_id -> the kwargs evaluate_promotion takes.
        self.evidence: dict[str, dict[str, Any]] = {}
        self.recommendations: dict[str, PromotionRecommendation] = {}

        self.regime_cycles = 0
        self.promotion_cycles = 0
        self.promoted = 0
        self.recommended = 0
        self.refused = 0

    # -- regime --------------------------------------------------------------

    def observe_markets(self) -> None:
        """Fold current prices into each market's rolling window."""
        for state in self.market_states.values():
            for ticker in state.tickers():
                snapshot = state.get(ticker)
                if snapshot is None:
                    continue
                price = snapshot.mid_cc or snapshot.reference_price_cc()
                if price is None:
                    continue
                self._windows.setdefault(ticker, _Window()).observe(price)

            # Markets that have left the watchlist take their history with them.
            live = set(state.tickers())
            for ticker in [t for t in self._windows if t not in live]:
                # Only drop a ticker no domain still holds; the same window map
                # serves both market states.
                if all(s.get(ticker) is None for s in self.market_states.values()):
                    del self._windows[ticker]

    def classify(self) -> dict[str, RegimeReading]:
        """Classify every market with enough history, and publish the results."""
        self.regime_cycles += 1
        readings: dict[str, RegimeReading] = {}

        for market_type, state in self.market_states.items():
            per_domain: dict[str, str] = {}
            for ticker in state.tickers():
                window = self._windows.get(ticker)
                snapshot = state.get(ticker)
                if window is None or snapshot is None:
                    continue

                book = snapshot.book
                reading = self.detector.detect(
                    prices_cc=window.prices_cc,
                    baseline_volatility=window.baseline,
                    spread_cc=snapshot.spread_cc,
                    # Mean of the two sides: a book that is deep on one side
                    # only is not a liquid book, and taking the max would call
                    # it one.
                    depth_qc=(
                        (book.depth_qc(is_bid=True) + book.depth_qc(is_bid=False)) // 2
                        if book is not None
                        else None
                    ),
                )
                readings[ticker] = reading
                per_domain[ticker] = reading.primary

            loop = self.loops.get(market_type)
            if loop is not None:
                loop.regimes = per_domain

        self.readings = readings
        metrics.gauge("research.markets_classified", len(readings))
        return readings

    # -- promotion -----------------------------------------------------------

    def submit_evidence(self, strategy_id: str, gate_kwargs: dict[str, Any]) -> None:
        """Record the metric blocks a validation run produced.

        Accepts :meth:`~kat.simulation.validation.ValidationReport.gate_kwargs`
        verbatim. Kept as a separate entry point so validation can run wherever
        it is cheapest -- in-process, or offline and submitted later -- without
        this loop needing to know which.
        """
        self.evidence[strategy_id] = dict(gate_kwargs)
        log.info(
            "research.evidence_submitted",
            strategy_id=strategy_id,
            blocks=sorted(k for k, v in gate_kwargs.items() if v),
        )

    def _may_apply(self, target: StrategyState) -> bool:
        """Whether this mode lets the loop act on a pass, rather than stage it."""
        if self.mode is OperatingMode.FULL_AUTONOMOUS:
            return True
        if self.mode is OperatingMode.AUTONOMOUS_DISCOVERY:
            return target not in _REAL_MONEY_STATES
        # GATED_DISCOVERY and HUMAN_STRATEGIES never apply a promotion.
        return False

    def evaluate_all(self) -> list[PromotionRecommendation]:
        """Run the gates for every promotable strategy that has evidence."""
        self.promotion_cycles += 1
        staged: list[PromotionRecommendation] = []
        now_ms = self.clock.now_ms()

        for strategy in self.registry.all():
            if strategy.state not in _PROMOTABLE:
                continue
            evidence = self.evidence.get(strategy.strategy_id)
            if not evidence:
                continue

            decision = evaluate_promotion(
                strategy_id=strategy.strategy_id,
                current_state=strategy.state,
                settings=self.settings,
                **evidence,
            )
            gates = [g.to_dict() for g in decision.gates]

            if not decision.promoted:
                self.refused += 1
                failed = [g for g in decision.gates if not g.passed]
                log.info(
                    "research.promotion_refused",
                    strategy_id=strategy.strategy_id,
                    state=strategy.state.value,
                    failed=[g.name for g in failed],
                )
                continue

            target = decision.target_state
            if target is None:  # pragma: no cover - promoted implies a target
                continue

            if self._may_apply(target):
                applied = self.registry.promote(
                    strategy.strategy_id, actor="research", gate_results={"gates": gates}
                )
                if applied is not None:
                    self.promoted += 1
                    log.info(
                        "research.promoted",
                        strategy_id=strategy.strategy_id,
                        target=applied.value,
                        mode=self.mode.value,
                    )
                continue

            recommendation = PromotionRecommendation(
                strategy_id=strategy.strategy_id,
                current_state=strategy.state,
                target_state=target,
                gates=gates,
                created_ms=now_ms,
                reason=(
                    f"every gate passed; {self.mode.value} requires human approval "
                    f"before {target.value}"
                ),
            )
            self.recommendations[strategy.strategy_id] = recommendation
            staged.append(recommendation)
            self.recommended += 1
            log.info(
                "research.promotion_recommended",
                strategy_id=strategy.strategy_id,
                target=target.value,
                mode=self.mode.value,
            )

        metrics.gauge("research.pending_recommendations", len(self.recommendations))
        return staged

    def approve(self, strategy_id: str, *, actor: str) -> StrategyState | None:
        """Apply a staged recommendation. Called from the dashboard.

        The gates are re-run rather than trusted: the recommendation may be
        hours old, and a strategy whose evidence has since deteriorated must not
        be promoted on a stale verdict just because someone clicked approve.
        """
        recommendation = self.recommendations.get(strategy_id)
        if recommendation is None:
            return None

        strategy = self.registry.get(strategy_id)
        evidence = self.evidence.get(strategy_id)
        if strategy is None or not evidence:
            self.recommendations.pop(strategy_id, None)
            return None

        decision = evaluate_promotion(
            strategy_id=strategy_id,
            current_state=strategy.state,
            settings=self.settings,
            **evidence,
        )
        if not decision.promoted:
            log.warning(
                "research.approval_refused",
                strategy_id=strategy_id,
                actor=actor,
                reason="gates no longer pass on re-evaluation",
                failed=[g.name for g in decision.gates if not g.passed],
            )
            self.recommendations.pop(strategy_id, None)
            return None

        applied = self.registry.promote(
            strategy_id,
            actor=actor,
            gate_results={
                # The gates that actually authorised this, not the ones shown at
                # staging time. registry.promote writes these verbatim into the
                # audit row, so recording the stale snapshot would make the
                # permanent answer to "what evidence put this strategy on real
                # money?" a number that was already known to be wrong when the
                # promotion was granted.
                "gates": [g.to_dict() for g in decision.gates],
                "staged_gates": recommendation.gates,
                "staged_at_ms": recommendation.created_ms,
                "approved_by": actor,
            },
        )
        self.recommendations.pop(strategy_id, None)
        if applied is not None:
            self.promoted += 1
            log.info(
                "research.promotion_approved",
                strategy_id=strategy_id, actor=actor, target=applied.value,
            )
        return applied

    def reject(self, strategy_id: str, *, actor: str, reason: str = "") -> bool:
        """Discard a staged recommendation."""
        if self.recommendations.pop(strategy_id, None) is None:
            return False
        log.info(
            "research.promotion_rejected",
            strategy_id=strategy_id, actor=actor, reason=reason,
        )
        return True

    # -- component wiring ----------------------------------------------------

    async def run_regimes(self, *, interval_s: float = 5.0) -> None:
        log.info("research.regime_loop_started", interval_s=interval_s)
        while True:
            try:
                self.observe_markets()
                self.classify()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("research.regime_cycle_failed", error=str(exc), exc_info=True)
            await asyncio.sleep(interval_s)

    async def run(self) -> None:
        """The slow promotion cycle."""
        log.info(
            "research.loop_started",
            interval_s=self.settings.loop_interval_s,
            mode=self.mode.value,
        )
        while True:
            try:
                self.evaluate_all()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("research.cycle_failed", error=str(exc), exc_info=True)
            await asyncio.sleep(self.settings.loop_interval_s)

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "regime_cycles": self.regime_cycles,
            "promotion_cycles": self.promotion_cycles,
            "markets_tracked": len(self._windows),
            "markets_classified": len(self.readings),
            "strategies_with_evidence": len(self.evidence),
            "promoted": self.promoted,
            "recommended": self.recommended,
            "refused": self.refused,
            "pending_recommendations": [r.to_dict() for r in self.recommendations.values()],
        }
