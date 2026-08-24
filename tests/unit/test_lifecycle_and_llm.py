"""Lifecycle gates and the LLM boundary.

Two things are established here:

1. A strategy cannot reach live capital without evidence, and missing evidence
   counts against it rather than for it.
2. The LLM is structurally incapable of placing an order and its influence on
   sizing is bounded by arithmetic, not by policy.
"""

from __future__ import annotations

import inspect
from decimal import Decimal

import pytest

from kat.common.enums import StrategyState
from kat.config.settings import LLMSettings, ResearchSettings
from kat.discovery.lifecycle import evaluate_promotion, may_demote
from kat.llm import advisor as advisor_module
from kat.llm.advisor import Advice, LLMAdvisor
from kat.llm.client import extract_json
from kat.llm.schemas import LLMRejection, MarketThesis

SETTINGS = ResearchSettings()


def good(**overrides) -> dict:
    """Metrics that clear every threshold, before overrides."""
    base = {
        "trades": 500,
        "sharpe": 1.5,
        "max_drawdown_pct": 0.10,
        "expectancy_micros": 25_000,
        "windows": 8,
        "positive_windows": 7,
        "monte_carlo_p5_sharpe": 0.4,
        "perturbation_sharpe_ratio": 0.8,
    }
    base.update(overrides)
    return base


class TestPromotionLadder:
    def test_promotion_moves_exactly_one_rung(self):
        decision = evaluate_promotion(
            strategy_id="s",
            current_state=StrategyState.OUT_OF_SAMPLE,
            settings=SETTINGS,
            backtest_metrics=good(),
            oos_metrics=good(),
        )
        assert decision.target_state is StrategyState.WALK_FORWARD

    def test_live_is_the_top_of_the_ladder(self):
        decision = evaluate_promotion(
            strategy_id="s", current_state=StrategyState.LIVE, settings=SETTINGS
        )
        assert decision.target_state is None
        assert decision.promoted is False

    @pytest.mark.parametrize(
        "state", [StrategyState.PAUSED, StrategyState.RETIRED, StrategyState.REVIEW]
    )
    def test_a_paused_or_retired_strategy_is_not_promoted_automatically(self, state):
        decision = evaluate_promotion(
            strategy_id="s", current_state=state, settings=SETTINGS,
            backtest_metrics=good(), oos_metrics=good(), paper_metrics=good(),
        )
        assert decision.target_state is None
        assert decision.promoted is False

    def test_demotion_is_always_allowed(self):
        """Stopping something misbehaving must never require evidence."""
        assert may_demote(StrategyState.LIVE, StrategyState.PAUSED) is True
        assert may_demote(StrategyState.LIVE, StrategyState.RETIRED) is True
        assert may_demote(StrategyState.PAPER, StrategyState.PAUSED) is True


class TestMissingEvidenceFails:
    """Missing evidence is not favourable evidence."""

    @pytest.mark.parametrize(
        ("state", "kwargs"),
        [
            (StrategyState.BACKTEST, {}),
            (StrategyState.OUT_OF_SAMPLE, {}),
            (StrategyState.WALK_FORWARD, {}),
            (StrategyState.ROBUSTNESS, {}),
            (StrategyState.PAPER, {}),
            (StrategyState.PROBATION, {}),
        ],
    )
    def test_no_metrics_at_all_fails(self, state, kwargs):
        decision = evaluate_promotion(
            strategy_id="s", current_state=state, settings=SETTINGS, **kwargs
        )
        assert decision.promoted is False
        assert decision.failures

    def test_an_unmeasurable_gate_fails_rather_than_passing(self):
        """A metric present but non-numeric is treated as absent."""
        decision = evaluate_promotion(
            strategy_id="s",
            current_state=StrategyState.OUT_OF_SAMPLE,
            settings=SETTINGS,
            backtest_metrics=good(),
            oos_metrics=good(sharpe=float("nan")),
        )
        assert decision.promoted is False
        assert any(g.name == "oos_sharpe" and not g.passed for g in decision.gates)


class TestSampleSize:
    def test_too_few_trades_blocks_promotion_regardless_of_sharpe(self):
        """A Sharpe of 5 over 12 trades is not a Sharpe of 5."""
        decision = evaluate_promotion(
            strategy_id="s",
            current_state=StrategyState.OUT_OF_SAMPLE,
            settings=SETTINGS,
            backtest_metrics=good(),
            oos_metrics=good(trades=12, sharpe=5.0),
        )
        assert decision.promoted is False
        assert any(g.name == "oos_sample_size" and not g.passed for g in decision.gates)

    def test_enough_trades_with_a_good_sharpe_promotes(self):
        decision = evaluate_promotion(
            strategy_id="s",
            current_state=StrategyState.OUT_OF_SAMPLE,
            settings=SETTINGS,
            backtest_metrics=good(),
            oos_metrics=good(),
        )
        assert decision.promoted is True


class TestDegradation:
    """The gate that catches an edge fitted to its own sample."""

    def test_a_collapsed_out_of_sample_sharpe_is_rejected(self):
        # Each number individually looks acceptable; the ratio does not.
        decision = evaluate_promotion(
            strategy_id="s",
            current_state=StrategyState.OUT_OF_SAMPLE,
            settings=SETTINGS,
            backtest_metrics=good(sharpe=4.0),
            oos_metrics=good(sharpe=0.6),
        )
        assert decision.promoted is False
        failure = next(g for g in decision.gates if g.name == "degradation")
        assert not failure.passed
        assert "did not survive" in failure.detail

    def test_a_modest_decay_is_accepted(self):
        decision = evaluate_promotion(
            strategy_id="s",
            current_state=StrategyState.OUT_OF_SAMPLE,
            settings=SETTINGS,
            backtest_metrics=good(sharpe=2.0),
            oos_metrics=good(sharpe=1.5),
        )
        assert next(g for g in decision.gates if g.name == "degradation").passed

    def test_out_of_sample_beating_in_sample_is_not_penalised(self):
        decision = evaluate_promotion(
            strategy_id="s",
            current_state=StrategyState.OUT_OF_SAMPLE,
            settings=SETTINGS,
            backtest_metrics=good(sharpe=1.0),
            oos_metrics=good(sharpe=1.8),
        )
        assert next(g for g in decision.gates if g.name == "degradation").passed


class TestWalkForward:
    def test_too_few_windows_blocks_promotion(self):
        decision = evaluate_promotion(
            strategy_id="s",
            current_state=StrategyState.WALK_FORWARD,
            settings=SETTINGS,
            walk_forward_metrics=good(windows=2, positive_windows=2),
        )
        assert decision.promoted is False

    def test_an_average_carried_by_a_minority_of_windows_is_rejected(self):
        """One extraordinary window is not a repeatable edge."""
        decision = evaluate_promotion(
            strategy_id="s",
            current_state=StrategyState.WALK_FORWARD,
            settings=SETTINGS,
            walk_forward_metrics=good(windows=10, positive_windows=3, sharpe=2.0),
        )
        assert decision.promoted is False
        failure = next(g for g in decision.gates if g.name == "walk_forward_consistency")
        assert not failure.passed


class TestRobustness:
    def test_a_strategy_that_loses_under_resampling_is_rejected(self):
        decision = evaluate_promotion(
            strategy_id="s",
            current_state=StrategyState.ROBUSTNESS,
            settings=SETTINGS,
            robustness_metrics=good(monte_carlo_p5_sharpe=-0.3),
        )
        assert decision.promoted is False
        assert any(g.name == "monte_carlo" and not g.passed for g in decision.gates)

    def test_a_strategy_that_collapses_under_a_parameter_shift_is_rejected(self):
        decision = evaluate_promotion(
            strategy_id="s",
            current_state=StrategyState.ROBUSTNESS,
            settings=SETTINGS,
            robustness_metrics=good(perturbation_sharpe_ratio=0.1),
        )
        assert decision.promoted is False
        failure = next(g for g in decision.gates if g.name == "parameter_perturbation")
        assert "fitted to the sample" in failure.detail


class TestPathToLive:
    def test_a_good_backtest_alone_does_not_reach_live(self):
        """The whole point of the ladder."""
        state = StrategyState.BACKTEST
        decision = evaluate_promotion(
            strategy_id="s", current_state=state, settings=SETTINGS,
            backtest_metrics=good(),
        )
        assert decision.target_state is StrategyState.OUT_OF_SAMPLE
        assert decision.target_state is not StrategyState.LIVE

    def test_reaching_live_requires_a_probation_record_not_a_paper_one(self):
        """Probation already trades real money; that is the evidence for full size."""
        decision = evaluate_promotion(
            strategy_id="s",
            current_state=StrategyState.PROBATION,
            settings=SETTINGS,
            paper_metrics=good(),   # excellent paper record
            live_metrics=None,      # but nothing from probation
        )
        assert decision.promoted is False

    def test_a_full_ladder_walk_reaches_live(self):
        """Every rung, in order, with evidence at each."""
        metrics = {
            "backtest_metrics": good(),
            "oos_metrics": good(),
            "walk_forward_metrics": good(),
            "robustness_metrics": good(),
            "paper_metrics": good(),
            "live_metrics": good(),
        }
        state = StrategyState.BACKTEST
        visited = [state]
        for _ in range(10):
            decision = evaluate_promotion(
                strategy_id="s", current_state=state, settings=SETTINGS, **metrics
            )
            if not decision.promoted or decision.target_state is None:
                break
            state = decision.target_state
            visited.append(state)
        assert state is StrategyState.LIVE, f"stopped at {state.value}: {visited}"
        assert StrategyState.PAPER in visited
        assert StrategyState.PROBATION in visited


# ---------------------------------------------------------------------------
# The LLM boundary
# ---------------------------------------------------------------------------


class TestLLMHasNoAuthority:
    @pytest.mark.parametrize(
        "forbidden",
        ["ExecutionEngine", "EventContractsREST", "PerpsREST", "KalshiHTTPClient", ".submit("],
    )
    def test_the_advisor_module_cannot_reach_the_exchange(self, forbidden: str):
        """Structural, not policy: there is nothing here it could call."""
        source = inspect.getsource(advisor_module)
        assert forbidden not in source

    def test_advice_carries_no_order(self):
        """No side, no size, no instruction -- only a probability."""
        fields = set(Advice.__slots__)
        for tradeable in ("side", "qty", "quantity", "size", "order", "ticker", "intent"):
            assert tradeable not in fields


class TestLLMSchemaRejectsBadOutput:
    def valid(self, **overrides) -> dict:
        base = {
            "probability": 0.62,
            "confidence": 0.75,
            "thesis": "Polling has been stable for three weeks and turnout models agree.",
            "key_facts": ["poll average 62%", "turnout stable"],
            "invalidation": ["a poll below 55%"],
            "information_sufficiency": "sufficient",
        }
        base.update(overrides)
        return base

    def test_a_valid_thesis_is_accepted(self):
        thesis = MarketThesis.model_validate(self.valid())
        assert thesis.is_actionable is True

    @pytest.mark.parametrize(
        ("label", "overrides"),
        [
            ("probability above one", {"probability": 1.4}),
            ("probability below zero", {"probability": -0.1}),
            ("confidence above one", {"confidence": 1.2}),
            ("no key facts", {"key_facts": []}),
            ("no invalidation", {"invalidation": []}),
            ("thesis too short", {"thesis": "yes"}),
        ],
    )
    def test_malformed_output_is_rejected(self, label: str, overrides: dict):
        with pytest.raises(Exception):  # noqa: B017 - pydantic raises its own type
            MarketThesis.model_validate(self.valid(**overrides))

    def test_a_model_that_contradicts_itself_is_rejected(self):
        """High confidence alongside admitted insufficiency cannot both be true."""
        with pytest.raises(Exception):  # noqa: B017
            MarketThesis.model_validate(
                self.valid(information_sufficiency="insufficient", confidence=0.95)
            )

    def test_an_injected_extra_field_is_rejected(self):
        """A model cannot smuggle an instruction into the schema."""
        with pytest.raises(Exception):  # noqa: B017
            MarketThesis.model_validate(self.valid(place_order=True))

    def test_admitting_insufficient_information_is_a_valid_answer(self):
        thesis = MarketThesis.model_validate(
            self.valid(information_sufficiency="insufficient", confidence=0.1)
        )
        assert thesis.is_actionable is False


class TestJSONExtraction:
    @pytest.mark.parametrize(
        "text",
        [
            '{"a": 1}',
            '```json\n{"a": 1}\n```',
            '```\n{"a": 1}\n```',
            'Here is my answer:\n{"a": 1}\nHope that helps.',
        ],
    )
    def test_messy_packaging_is_tolerated(self, text: str):
        assert extract_json(text) == {"a": 1}

    @pytest.mark.parametrize("text", ["no json here", "", "42", "[1, 2, 3]"])
    def test_content_that_is_not_an_object_is_rejected(self, text: str):
        with pytest.raises(LLMRejection):
            extract_json(text)


class TestInfluenceIsBounded:
    def _advisor(self, cap: str = "0.25") -> LLMAdvisor:
        return LLMAdvisor(LLMSettings(enabled=True, max_influence_pct=Decimal(cap)))

    def _advice(self, probability: float, confidence: float) -> Advice:
        thesis = MarketThesis.model_validate(
            {
                "probability": probability,
                "confidence": confidence,
                "thesis": "A sufficiently long rationale for the schema to accept.",
                "key_facts": ["a fact"],
                "invalidation": ["a disproof"],
                "information_sufficiency": "sufficient",
            }
        )
        return Advice(
            signal_id="s", probability=probability, confidence=confidence,
            accepted=True, reason="ok", thesis=thesis,
        )

    def test_maximum_influence_is_the_configured_cap(self):
        """LLM at 1.00 with full confidence moves 0.50 by at most 25% of the gap."""
        advisor = self._advisor("0.25")
        assert advisor.blend(0.5, self._advice(1.0, 1.0)) == pytest.approx(0.625)

    def test_lower_confidence_scales_the_influence_down(self):
        advisor = self._advisor("0.25")
        blended = advisor.blend(0.5, self._advice(1.0, 0.5))
        assert blended == pytest.approx(0.5625)

    def test_unusable_advice_changes_nothing(self):
        advisor = self._advisor()
        unusable = Advice(
            signal_id="s", probability=1.0, confidence=1.0, accepted=False, reason="rejected"
        )
        assert advisor.blend(0.42, unusable) == 0.42

    def test_a_zero_cap_removes_all_influence(self):
        advisor = self._advisor("0.0")
        assert advisor.blend(0.5, self._advice(1.0, 1.0)) == pytest.approx(0.5)

    def test_the_cap_is_clamped_by_settings(self):
        """Configuration cannot hand the model unlimited influence."""
        settings = LLMSettings(enabled=True, max_influence_pct=Decimal("5.0"))
        assert settings.max_influence_pct <= Decimal("0.50")

    def test_the_blend_stays_a_probability(self):
        advisor = self._advisor("0.5")
        for quantitative in (0.0, 0.01, 0.5, 0.99, 1.0):
            for llm_p in (0.0, 0.5, 1.0):
                result = advisor.blend(quantitative, self._advice(llm_p, 1.0))
                assert 0.0 <= result <= 1.0


class TestDegradesWhenUnavailable:
    async def test_a_disabled_llm_returns_unusable_advice_without_raising(self):
        advisor = LLMAdvisor(LLMSettings(enabled=False))
        advice = await advisor.advise(ticker="T", title="Will it rain?")
        assert advice.accepted is False
        assert advice.usable is False
        assert "disabled" in advice.reason

    async def test_an_unreachable_server_returns_unusable_advice_without_raising(self):
        advisor = LLMAdvisor(
            LLMSettings(enabled=True, base_url="http://127.0.0.1:59999/v1", timeout_s=5.0)
        )
        try:
            advice = await advisor.advise(ticker="T", title="Will it rain?")
            assert advice.accepted is False
            assert advice.usable is False
            assert "unavailable" in advice.reason
        finally:
            await advisor.client.aclose()

    async def test_availability_is_checkable_without_raising(self):
        advisor = LLMAdvisor(
            LLMSettings(enabled=True, base_url="http://127.0.0.1:59999/v1", timeout_s=5.0)
        )
        try:
            assert await advisor.client.available() is False
        finally:
            await advisor.client.aclose()
