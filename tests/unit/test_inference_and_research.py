"""Tests for model inference, the live feature worker, and the research loop.

The theme running through these: a system that cannot compute an honest answer
must say so rather than produce a plausible one. A missing feature is not zero,
an absent model is not 0.5, and an unmeasured robustness test is not a pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from kat.common.enums import MarketType, OperatingMode, StrategyState
from kat.common.timeutil import SimulatedClock
from kat.config.settings import ResearchSettings
from kat.engine.features import FeatureWorker
from kat.features.spec import FeatureValue, FeatureVector
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketState, funding_rate_bps_per_hour, funding_rate_per_hour
from kat.models.inference import (
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    InferenceEngine,
    predict_one,
)
from kat.models.registry import DeploymentState, RegisteredModel
from kat.research.loop import ResearchLoop
from kat.strategies.registry import StrategyRegistry

HOUR_MS = 3_600_000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Estimator:
    """A stand-in that returns whatever probability it is told to."""

    def __init__(self, probability: float = 0.7, raises: bool = False) -> None:
        self.probability = probability
        self.raises = raises
        self.seen: list[np.ndarray] = []

    def predict_proba(self, X):
        if self.raises:
            raise RuntimeError("estimator exploded")
        self.seen.append(np.asarray(X))
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.probability), np.full(n, self.probability)])


class _OrderSensitiveEstimator:
    """Returns the first column value, so column order is observable."""

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=float)
        first = np.clip(arr[:, 0], 0.0, 1.0)
        return np.column_stack([1 - first, first])


def model(estimator=None, names=("a", "b", "c"), calibrator=None) -> RegisteredModel:
    from pathlib import Path

    return RegisteredModel(
        model_id="event-settlement-logistic",
        version="v1",
        name="logistic",
        model_type="logistic",
        market_type=MarketType.EVENT,
        target="settlement",
        artifact_path=Path("unused.pkl"),
        artifact_sha256="0" * 64,
        feature_names=list(names),
        deployment_state=DeploymentState.DEPLOYED,
        estimator=estimator or _Estimator(),
        calibrator=calibrator,
    )


def vector(ticker="M-1", values=None, ts_ms=1_000) -> FeatureVector:
    values = values if values is not None else {"a": 0.1, "b": 0.2, "c": 0.3}
    return FeatureVector(
        ticker=ticker,
        market_type=MarketType.EVENT,
        feature_set_id="test.v1",
        as_of_ms=ts_ms,
        values={
            name: FeatureValue(name=name, value=v, info_ts_ms=ts_ms, availability_lag_ms=0)
            for name, v in values.items()
        },
    )


class _Loop:
    """The two attributes the workers publish into."""

    def __init__(self) -> None:
        self.model_probabilities: dict = {}
        self.features: dict = {}
        self.feature_vectors: dict = {}
        self.research: dict = {}
        self.regimes: dict = {}


class _Registry:
    def __init__(self, model_=None) -> None:
        self._model = model_
        self.calls = 0

    async def deployed(self, market_type, target):
        self.calls += 1
        return self._model


# ---------------------------------------------------------------------------
# predict_one
# ---------------------------------------------------------------------------


class TestPredictOne:
    def test_a_calibrated_probability_comes_back(self):
        prediction = predict_one(model(_Estimator(0.73)), vector(), now_ms=2_000)
        assert prediction is not None
        assert prediction.probability == pytest.approx(0.73)
        assert prediction.model_id == "event-settlement-logistic"
        assert prediction.feature_count == 3

    def test_features_are_reordered_into_the_models_own_column_order(self):
        """A model handed its features in the wrong order is confidently wrong.

        Nothing downstream could detect it, so the ordering has to be explicit
        here rather than left to whatever order the vector happens to iterate in.
        """
        estimator = _OrderSensitiveEstimator()
        # The vector is built in a different order from the model's.
        v = vector(values={"c": 0.9, "b": 0.5, "a": 0.25})
        prediction = predict_one(model(estimator, names=("a", "b", "c")), v, now_ms=2_000)
        assert prediction is not None
        # The first column must be "a" = 0.25, not "c" = 0.9.
        assert prediction.probability == pytest.approx(0.25)

    def test_a_missing_feature_yields_no_prediction(self):
        """A missing feature is not a zero.

        Imputing one silently moves the model along whichever axis that feature
        controls, and reports the result with full confidence.
        """
        assert predict_one(model(), vector(values={"a": 0.1, "b": 0.2}), now_ms=2_000) is None

    def test_a_non_finite_feature_yields_no_prediction(self):
        v = vector(values={"a": float("nan"), "b": 0.2, "c": 0.3})
        assert predict_one(model(), v, now_ms=2_000) is None

    def test_an_estimator_that_raises_yields_no_prediction(self):
        assert predict_one(model(_Estimator(raises=True)), vector(), now_ms=2_000) is None

    def test_a_model_with_no_feature_names_is_refused(self):
        assert predict_one(model(names=()), vector(), now_ms=2_000) is None

    def test_probabilities_are_clipped_away_from_certainty(self):
        """A model asserting 0 or 1 asserts certainty.

        Downstream arithmetic dividing by ``1 - p`` would turn a rounding
        artefact into an infinite position.
        """
        high = predict_one(model(_Estimator(1.0)), vector(), now_ms=2_000)
        low = predict_one(model(_Estimator(0.0)), vector(), now_ms=2_000)
        assert high is not None and low is not None
        assert high.probability == pytest.approx(MAX_PROBABILITY)
        assert low.probability == pytest.approx(MIN_PROBABILITY)

    def test_the_prediction_is_no_fresher_than_its_oldest_feature(self):
        v = FeatureVector(
            ticker="M-1", market_type=MarketType.EVENT,
            feature_set_id="test.v1", as_of_ms=10_000,
            values={
                "a": FeatureValue(name="a", value=0.1, info_ts_ms=9_000, availability_lag_ms=0),
                "b": FeatureValue(name="b", value=0.2, info_ts_ms=4_000, availability_lag_ms=0),
                "c": FeatureValue(name="c", value=0.3, info_ts_ms=8_000, availability_lag_ms=0),
            },
        )
        prediction = predict_one(model(), v, now_ms=10_000)
        assert prediction is not None
        assert prediction.feature_ts_ms == 4_000


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------


class TestInferenceEngine:
    @pytest.mark.asyncio
    async def test_no_deployed_model_means_no_predictions_not_a_default(self):
        """0.5 would be a real prediction that nobody computed."""
        engine = InferenceEngine(_Registry(None), MarketType.EVENT)
        loop = _Loop()
        assert await engine.update(loop, {"M-1": vector()}, 1_000) == 0
        assert loop.model_probabilities == {}

    @pytest.mark.asyncio
    async def test_predictions_reach_the_loop_in_the_shape_it_reads(self):
        engine = InferenceEngine(_Registry(model(_Estimator(0.62))), MarketType.EVENT)
        loop = _Loop()
        assert await engine.update(loop, {"M-1": vector()}, 1_000) == 1
        probability, model_id, version = loop.model_probabilities["M-1"]
        assert probability == pytest.approx(0.62)
        assert model_id == "event-settlement-logistic"
        assert version == "v1"

    @pytest.mark.asyncio
    async def test_a_stale_prediction_is_dropped_rather_than_served(self):
        """A stale probability is more dangerous than none: it looks current."""
        engine = InferenceEngine(_Registry(model()), MarketType.EVENT, max_age_ms=5_000)
        await engine.refresh_model()
        engine.score({"M-1": vector()}, 1_000)
        assert engine.loop_entries(3_000) != {}
        assert engine.loop_entries(30_000) == {}
        assert engine.dropped_stale == 1

    @pytest.mark.asyncio
    async def test_a_market_that_stops_scoring_disappears(self):
        """Leaving the previous value behind is how a strategy trades a ghost."""
        engine = InferenceEngine(_Registry(model()), MarketType.EVENT)
        await engine.refresh_model()
        engine.score({"M-1": vector()}, 1_000)
        assert "M-1" in engine.predictions
        engine.score({"M-1": vector(values={"a": 0.1})}, 2_000)
        assert engine.predictions == {}

    @pytest.mark.asyncio
    async def test_a_deployment_swap_invalidates_the_cached_predictions(self):
        """They were computed by a different model.

        Serving them under the new model's id would make the audit trail a lie.
        """
        registry = _Registry(model(_Estimator(0.6)))
        engine = InferenceEngine(registry, MarketType.EVENT)
        await engine.refresh_model()
        engine.score({"M-1": vector()}, 1_000)
        assert engine.predictions

        replacement = model(_Estimator(0.4))
        replacement.version = "v2"
        registry._model = replacement
        await engine.refresh_model()
        assert engine.predictions == {}

    @pytest.mark.asyncio
    async def test_a_registry_failure_keeps_the_previous_model(self):
        class _Broken(_Registry):
            async def deployed(self, market_type, target):
                raise RuntimeError("database gone")

        engine = InferenceEngine(_Registry(model()), MarketType.EVENT)
        await engine.refresh_model()
        engine.registry = _Broken()
        assert await engine.refresh_model() is not None


# ---------------------------------------------------------------------------
# funding units
# ---------------------------------------------------------------------------


class TestFundingUnits:
    def test_the_interval_is_applied_not_ignored(self):
        """The defect: a per-interval rate was assigned to a per-hour field.

        On Kalshi's eight-hour schedule that overstated three perp features by
        a factor of eight.
        """
        assert funding_rate_per_hour(0.0008, 8.0) == pytest.approx(0.0001)
        assert funding_rate_bps_per_hour(0.0008, 8.0) == pytest.approx(1.0)

    def test_the_two_helpers_cannot_drift_apart(self):
        for rate in (0.0001, 0.0025, -0.0008):
            for hours in (1.0, 4.0, 8.0):
                assert funding_rate_bps_per_hour(rate, hours) == pytest.approx(
                    funding_rate_per_hour(rate, hours) * 10_000.0
                )

    def test_unknown_stays_unknown(self):
        """An absent rate is not a rate of zero."""
        assert funding_rate_per_hour(None, 8.0) is None
        assert funding_rate_per_hour(0.001, 0.0) is None


# ---------------------------------------------------------------------------
# the live feature worker
# ---------------------------------------------------------------------------


def _seed_market(state: MarketState, ticker: str, mid_cc: int, now_ms: int) -> None:
    book = OrderBook.from_two_sided_levels(
        ticker,
        bid_levels=[(mid_cc - 50, 5_000)],
        ask_levels=[(mid_cc + 50, 5_000)],
        ts_ms=now_ms,
    )
    snapshot = state.ensure(ticker)
    snapshot.book = book
    snapshot.book_updated_ms = now_ms
    snapshot.updated_ms = now_ms


class TestFeatureWorker:
    def test_bars_close_on_the_period_boundary(self):
        state = MarketState(MarketType.EVENT)
        clock = SimulatedClock(start_ms=0)
        worker = FeatureWorker(MarketType.EVENT, state, clock)

        for step in range(150):
            now = step * 5_000
            _seed_market(state, "M-1", 5_000 + step, now)
            worker.observe(now)

        series = worker._series["M-1"]
        assert len(series.bars) >= 10
        for bar in series.bars:
            assert (bar.ts_ms + 1) % 60_000 == 0, "bars are stamped with their close"
            assert bar.low_cc <= bar.close_cc <= bar.high_cc

    def test_an_out_of_order_update_is_dropped_not_fatal(self):
        """Market data can deliver these on a reconnect."""
        state = MarketState(MarketType.EVENT)
        worker = FeatureWorker(MarketType.EVENT, state, SimulatedClock(start_ms=0))
        _seed_market(state, "M-1", 5_000, 10_000)
        worker.observe(10_000)
        _seed_market(state, "M-1", 5_100, 5_000)
        worker.observe(5_000)  # must not raise
        assert len(worker._series["M-1"].mids) == 1

    def test_no_features_during_warmup_is_healthy(self):
        """There genuinely are no features yet; that is not a fault."""
        state = MarketState(MarketType.EVENT)
        worker = FeatureWorker(MarketType.EVENT, state, SimulatedClock(start_ms=0))
        loop = _Loop()
        _seed_market(state, "M-1", 5_000, 0)
        worker.publish(loop, 0)
        assert worker.healthy() is True
        assert loop.features == {}

    def test_published_maps_are_replaced_not_merged(self):
        """A market that stops producing features must vanish, not go stale."""
        state = MarketState(MarketType.EVENT)
        worker = FeatureWorker(MarketType.EVENT, state, SimulatedClock(start_ms=0))
        loop = _Loop()
        loop.features = {"GHOST": {"x": 1.0}}
        _seed_market(state, "M-1", 5_000, 0)
        worker.publish(loop, 0)
        assert "GHOST" not in loop.features


# ---------------------------------------------------------------------------
# the research loop
# ---------------------------------------------------------------------------


class _Strategy:
    def __init__(self, strategy_id: str, state: StrategyState) -> None:
        self.strategy_id = strategy_id
        self.state = state
        self.market_type = MarketType.EVENT
        self.version = "1.0.0"
        self.enabled = True


def _passing_evidence() -> dict:
    """Metric blocks strong enough to clear every gate."""
    block = {
        "trades": 500,
        "sharpe": 2.0,
        "max_drawdown_pct": 0.05,
        "expectancy_micros": 25_000,
    }
    return {
        "backtest_metrics": dict(block),
        "oos_metrics": dict(block),
        "walk_forward_metrics": {"windows": 6, "positive_windows": 5, "sharpe": 1.8},
        "robustness_metrics": {
            "monte_carlo_p5_sharpe": 0.9,
            "perturbation_sharpe_ratio": 0.85,
        },
        "paper_metrics": dict(block),
        "live_metrics": dict(block),
    }


def _loop(mode: OperatingMode, state=StrategyState.BACKTEST) -> tuple[ResearchLoop, StrategyRegistry]:
    registry = StrategyRegistry()
    registry._strategies["s1"] = _Strategy("s1", state)
    loop = ResearchLoop(
        settings=ResearchSettings(),
        registry=registry,
        market_states={MarketType.EVENT: MarketState(MarketType.EVENT)},
        loops={},
        clock=SimulatedClock(start_ms=0),
        mode=mode,
    )
    loop.submit_evidence("s1", _passing_evidence())
    return loop, registry


class TestResearchLoop:
    def test_gated_discovery_stages_rather_than_applies(self):
        """The recommended default: passing gates is necessary, not sufficient."""
        loop, registry = _loop(OperatingMode.GATED_DISCOVERY)
        staged = loop.evaluate_all()
        assert len(staged) == 1
        assert registry.get("s1").state is StrategyState.BACKTEST
        assert loop.promoted == 0
        assert loop.recommended == 1

    def test_full_autonomy_applies_the_promotion(self):
        loop, registry = _loop(OperatingMode.FULL_AUTONOMOUS)
        loop.evaluate_all()
        assert registry.get("s1").state is StrategyState.OUT_OF_SAMPLE
        assert loop.promoted == 1

    def test_autonomous_discovery_stops_short_of_real_money(self):
        """Mode C may reach PAPER on its own; PROBATION spends real capital."""
        loop, registry = _loop(OperatingMode.AUTONOMOUS_DISCOVERY, StrategyState.PAPER)
        loop.evaluate_all()
        assert registry.get("s1").state is StrategyState.PAPER
        assert loop.recommended == 1

        below, registry_below = _loop(OperatingMode.AUTONOMOUS_DISCOVERY, StrategyState.BACKTEST)
        below.evaluate_all()
        assert registry_below.get("s1").state is StrategyState.OUT_OF_SAMPLE

    def test_a_failing_strategy_is_refused_in_every_mode(self):
        """Autonomy widens what may be done on good evidence.

        It never changes what counts as good evidence.
        """
        for mode in OperatingMode:
            loop, registry = _loop(mode)
            loop.evidence["s1"]["backtest_metrics"]["trades"] = 3
            loop.evaluate_all()
            assert registry.get("s1").state is StrategyState.BACKTEST, mode
            assert loop.promoted == 0, mode
            assert loop.recommended == 0, mode

    def test_a_strategy_without_evidence_is_never_promoted(self):
        loop, registry = _loop(OperatingMode.FULL_AUTONOMOUS)
        loop.evidence.clear()
        loop.evaluate_all()
        assert registry.get("s1").state is StrategyState.BACKTEST

    def test_approval_re_runs_the_gates_rather_than_trusting_them(self):
        """A recommendation may be hours old.

        A strategy whose evidence has since deteriorated must not be promoted
        because someone clicked approve on a stale verdict.
        """
        loop, registry = _loop(OperatingMode.GATED_DISCOVERY)
        loop.evaluate_all()
        assert "s1" in loop.recommendations

        loop.evidence["s1"]["backtest_metrics"]["expectancy_micros"] = -5_000
        assert loop.approve("s1", actor="operator") is None
        assert registry.get("s1").state is StrategyState.BACKTEST
        assert "s1" not in loop.recommendations

    def test_approval_promotes_when_the_evidence_still_holds(self):
        loop, registry = _loop(OperatingMode.GATED_DISCOVERY)
        loop.evaluate_all()
        assert loop.approve("s1", actor="operator") is StrategyState.OUT_OF_SAMPLE
        assert registry.get("s1").state is StrategyState.OUT_OF_SAMPLE

    def test_rejecting_discards_the_recommendation(self):
        loop, registry = _loop(OperatingMode.GATED_DISCOVERY)
        loop.evaluate_all()
        assert loop.reject("s1", actor="operator", reason="not convinced") is True
        assert loop.recommendations == {}
        assert registry.get("s1").state is StrategyState.BACKTEST
        assert loop.reject("s1", actor="operator") is False


class TestApprovalAuditTrail:
    """What gets written down as the reason a strategy was promoted.

    `approve` re-runs the gates rather than trusting the staged verdict, then
    used to persist the *staged* gate block as the promotion's justification.
    The decision was always correct; the permanent record of why was not.
    """

    def test_the_persisted_gates_are_the_ones_that_authorised_it(self):
        loop, registry = _loop(OperatingMode.GATED_DISCOVERY)
        loop.evaluate_all()

        staged = loop.recommendations["s1"].gates
        staged_sharpe = next(g for g in staged if g.get("name") == "backtest_present")

        # Evidence changes between staging and approval, but still passes.
        loop.evidence["s1"]["backtest_metrics"]["sharpe"] = 0.9
        loop.evidence["s1"]["backtest_metrics"]["expectancy_micros"] = 11_111

        captured: dict = {}
        original = registry.promote

        def spy(strategy_id, *, actor, gate_results=None):
            captured.update(gate_results or {})
            return original(strategy_id, actor=actor, gate_results=gate_results)

        registry.promote = spy
        assert loop.approve("s1", actor="operator") is StrategyState.OUT_OF_SAMPLE

        assert captured["approved_by"] == "operator"
        # Both are kept, and they are distinguishable.
        assert "gates" in captured and "staged_gates" in captured
        assert captured["staged_gates"] == staged
        assert captured["staged_gates"] is not captured["gates"]
        assert staged_sharpe in captured["staged_gates"]
        # The authorising block reflects the evidence at approval time.
        observed = {g["name"]: g["observed"] for g in captured["gates"]}
        assert "11111" in observed["backtest_expectancy"]

    def test_a_refused_approval_writes_nothing(self):
        loop, registry = _loop(OperatingMode.GATED_DISCOVERY)
        loop.evaluate_all()
        loop.evidence["s1"]["backtest_metrics"]["expectancy_micros"] = -1

        calls: list = []
        registry.promote = lambda *a, **kw: calls.append(kw) or None

        assert loop.approve("s1", actor="operator") is None
        assert calls == []
        assert registry.get("s1").state is StrategyState.BACKTEST
