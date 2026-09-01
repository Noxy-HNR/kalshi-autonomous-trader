"""The properties that must hold no matter what else changes.

Most tests here assert something that is currently true and easy to keep true.
These assert things that would be catastrophic to break silently, and that a
plausible future change could break without any other test noticing: an order
escaping paper mode, a risk slider reaching the leverage ceiling, a money value
turning into a float.

Written after a full-codebase sweep, so each one records a property that was
verified by hand and is now checked automatically.
"""

from __future__ import annotations

import inspect

import pytest

from kat.common.enums import TradingEnvironment
from kat.config.settings import AppSettings
from kat.execution.base import ExecutionEngine
from kat.risk.invariants import assert_invariant_intact, evaluate_perp_leverage
from kat.risk.tunables import TUNABLES, TunableError, apply_override


class TestRealOrdersNeedTwoDeliberateChoices:
    """Six combinations of environment and armed flag; exactly one may trade."""

    @pytest.mark.parametrize(
        ("environment", "armed", "expected"),
        [
            (TradingEnvironment.PAPER, False, False),
            (TradingEnvironment.DEMO, False, False),
            (TradingEnvironment.LIVE, False, False),
            (TradingEnvironment.LIVE, True, True),
        ],
    )
    def test_the_matrix(self, environment, armed, expected):
        settings = AppSettings(environment=environment, live_trading_armed=armed)
        assert settings.can_submit_real_orders is expected

    @pytest.mark.parametrize(
        "environment", [TradingEnvironment.PAPER, TradingEnvironment.DEMO]
    )
    def test_arming_without_going_live_will_not_even_load(self, environment):
        """Stronger than refusing to trade: the configuration is unloadable.

        A half-set pair is far more likely to be a mistake than an intention, so
        it fails at start-up rather than running in a state nobody chose.
        """
        with pytest.raises(Exception, match="requires environment=live"):
            AppSettings(environment=environment, live_trading_armed=True)


class TestPaperModeCannotReachTheExchange:
    def test_dispatch_returns_before_any_exchange_path(self):
        """The paper branch is the first statement, not a later condition.

        Checked structurally rather than by mocking, because the property that
        matters is that no code was added *above* the check.
        """
        source = inspect.getsource(ExecutionEngine._dispatch)
        body = [
            line.strip()
            for line in source.split("\n")
            if line.strip() and not line.strip().startswith(("#", '"""', "async def"))
        ]
        assert body[0].startswith("if self.environment is TradingEnvironment.PAPER")
        assert "return" in body[:3]

    def test_submitting_requires_a_risk_verdict(self):
        """The type system carries the rule: only a risk engine makes a verdict."""
        assert "verdict" in inspect.signature(ExecutionEngine.submit).parameters


class TestNoTunableCanReachTheLeverageCeiling:
    def test_every_tunable_at_its_most_aggressive_still_caps_at_one(self):
        settings = AppSettings()
        for tunable in TUNABLES:
            target = tunable.maximum if tunable.higher_is_riskier else tunable.minimum
            apply_override(settings, tunable.key, target)

        equity = 200_000_000
        at_the_line = evaluate_perp_leverage(
            equity_micros=equity, current_exposure_micros=0,
            pending_exposure_micros=0, requested_delta_micros=equity,
        )
        one_micro_over = evaluate_perp_leverage(
            equity_micros=equity, current_exposure_micros=0,
            pending_exposure_micros=0, requested_delta_micros=equity + 1,
        )
        assert at_the_line.permitted is True
        assert one_micro_over.permitted is False

    def test_leverage_is_not_a_tunable_key(self):
        with pytest.raises(TunableError, match="not an adjustable risk parameter"):
            apply_override(AppSettings(), "perp.max_leverage", 2.0)

    def test_the_invariant_self_check_passes(self):
        assert assert_invariant_intact() is None


class TestDeRiskingIsAlwaysAllowed:
    """Being over-exposed must never trap a position.

    Equity can fall after a position is opened, putting the account above the
    ceiling through no new action. Refusing to let it close then would be the
    worst possible moment to enforce a limit.
    """

    def test_reducing_is_permitted_even_when_far_over_the_ceiling(self):
        assessment = evaluate_perp_leverage(
            equity_micros=100,
            current_exposure_micros=10**9,
            pending_exposure_micros=0,
            requested_delta_micros=-10**6,
        )
        assert assessment.permitted is True

    def test_increasing_is_refused_when_over_the_ceiling(self):
        assessment = evaluate_perp_leverage(
            equity_micros=100,
            current_exposure_micros=10**9,
            pending_exposure_micros=0,
            requested_delta_micros=1,
        )
        assert assessment.permitted is False

    @pytest.mark.parametrize("equity", [0, -1, -(10**12)])
    def test_an_unknown_or_negative_equity_permits_no_increase(self, equity):
        """Zero equity is not a licence to trade freely."""
        assessment = evaluate_perp_leverage(
            equity_micros=equity, current_exposure_micros=0,
            pending_exposure_micros=0, requested_delta_micros=1,
        )
        assert assessment.permitted is False


class TestMoneyStaysExact:
    def test_no_float_appears_in_a_fixed_point_conversion(self):
        """A float in a money path is a bug, not a rounding preference."""
        from kat.common.fixedpoint import notional_micros, parse_price_cc

        value = parse_price_cc("0.4237")
        assert isinstance(value, int)
        assert isinstance(notional_micros(value, 100), int)

    @pytest.mark.parametrize(
        "bad", ["", "abc", "nan", "inf", "-inf", None, True, [], {}]
    )
    def test_unparseable_money_is_refused_rather_than_guessed(self, bad):
        from kat.common.fixedpoint import parse_price_cc

        with pytest.raises((ValueError, TypeError, ArithmeticError)):
            parse_price_cc(bad)


class TestEdgeIsSignedTheSameWayEverywhere:
    """The gate and the audit record must not compute edge separately.

    They did, and only one signed the NO side. A NO contract is worth
    ``1 - fair_value``, so the plain subtraction misprices every NO trade by the
    whole dollar. The gate approved a Chelsea NO at +492, which was right; the
    audit trail recorded the identical order at -7,492, which is the YES formula
    applied to a NO. Both described the same trade, and the one a person reads
    said the platform was buying things worth ten cents for eighty-five.
    """

    @staticmethod
    def _intent(side, limit_price_cc, fair_value_cc):
        from kat.common.enums import MarketType
        from kat.risk.types import OrderIntent

        return OrderIntent(
            market_type=MarketType.EVENT,
            ticker="KXPREMIERLEAGUE-27-CFC",
            qty_qc=100,
            limit_price_cc=limit_price_cc,
            outcome_side=side,
            fair_value_cc=fair_value_cc,
        )

    def test_a_no_buy_below_its_worth_has_positive_edge(self):
        """The real trade: fair YES $0.1008, NO bought at $0.85, worth $0.8992."""
        from kat.common.enums import OutcomeSide

        intent = self._intent(OutcomeSide.NO, 8_500, 1_008)
        assert intent.edge_cc() == 492

    def test_a_yes_buy_above_its_worth_has_negative_edge(self):
        from kat.common.enums import OutcomeSide

        assert self._intent(OutcomeSide.YES, 1_550, 1_008).edge_cc() == -542

    def test_a_yes_buy_below_its_worth_has_positive_edge(self):
        from kat.common.enums import OutcomeSide

        assert self._intent(OutcomeSide.YES, 1_000, 1_500).edge_cc() == 500

    def test_the_risk_gate_and_the_audit_record_agree(self):
        """The property that was violated, stated directly."""
        from kat.common.enums import OutcomeSide
        from kat.engine.decision import _edge
        from kat.risk.event_risk import EventRiskEngine

        for side in (OutcomeSide.YES, OutcomeSide.NO):
            for price, fair in ((8_500, 1_008), (1_550, 1_008), (5_000, 5_000)):
                intent = self._intent(side, price, fair)
                assert EventRiskEngine._edge_cc(intent) == _edge(intent, None), (
                    f"gate and record disagree for {side} at {price} vs {fair}"
                )

    def test_no_fair_value_is_no_edge_rather_than_zero_edge(self):
        from kat.common.enums import OutcomeSide

        assert self._intent(OutcomeSide.NO, 8_500, None).edge_cc() is None
