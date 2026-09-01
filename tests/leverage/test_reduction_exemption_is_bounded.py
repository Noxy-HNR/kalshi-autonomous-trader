"""An exit is exempt from the entry limits. A reversal is not.

Being classified as reducing exempts an event-contract order from every check in
the risk engine except freshness: the daily-loss and drawdown halts, the
order-rate limit, available capital, max loss per trade, the per-market and
portfolio exposure caps, the correlated-event cap, the per-order notional cap,
the open-position count, minimum edge and minimum expected value.

That exemption has to exist. Closing a position must never be blocked by the
limits that govern opening one, or a halted platform could not get out of the
trade that halted it.

It was granted on direction alone. Any opposite-side buy qualified, at any size,
so a single residual contract bought an unbounded position on the other side
with nothing checked. It is reachable from the shipped strategies rather than
hypothetical: ``headroom_qc(..., increases=False)`` returns ``cap + position_qc``
precisely so a reversing signal can close and re-open in one intent.

Event contracts have no leverage invariant behind them -- that is a perpetuals
protection -- so this engine is the only thing standing between a strategy and
the account.
"""

from __future__ import annotations

import pytest

from kat.common.enums import MarketType, OutcomeSide, RiskDecision
from kat.config.settings import EventRiskSettings
from kat.risk.event_risk import EventAccountState, EventPositionState, EventRiskEngine
from kat.risk.types import OrderIntent

NOW_MS = 1_000


def _engine() -> EventRiskEngine:
    return EventRiskEngine(EventRiskSettings())


def _halted_account(*, position_qc: int) -> EventAccountState:
    """An account past every halt the engine has, holding one contract."""
    state = EventAccountState(
        balance_micros=0,
        portfolio_value_micros=5_000,             # half a cent of equity
        realized_pnl_today_micros=-100_000_000,   # $100 down, limit is 5%
        peak_equity_micros=200_000_000,           # ~100% drawdown, limit is 20%
        orders_this_minute=10_000,                # limit is 30
        updated_at_ms=NOW_MS,
    )
    if position_qc:
        state.positions["X"] = EventPositionState(
            ticker="X", position_qc=position_qc, exposure_micros=5_000
        )
    return state


def _healthy_account(*, position_qc: int) -> EventAccountState:
    state = EventAccountState(
        balance_micros=100_000_000,        # $100
        portfolio_value_micros=10_000_000,  # $10
        peak_equity_micros=110_000_000,
        updated_at_ms=NOW_MS,
    )
    if position_qc:
        state.positions["X"] = EventPositionState(
            ticker="X",
            position_qc=position_qc,
            exposure_micros=abs(position_qc) * 5_000,
        )
    return state


def _intent(*, side: OutcomeSide, qty_qc: int, ev: int = 1_000_000) -> OrderIntent:
    return OrderIntent(
        market_type=MarketType.EVENT,
        ticker="X",
        qty_qc=qty_qc,
        limit_price_cc=5_000,
        outcome_side=side,
        expected_value_micros=ev,
    )


def _evaluate(engine, intent, state):
    return engine.evaluate(intent, state, book=None, now_ms=NOW_MS, estimated_fees_micros=0)


class TestAnExitIsStillExempt:
    """The behaviour the exemption exists for, which must not regress."""

    def test_closing_the_whole_position_is_allowed_on_a_halted_account(self):
        verdict = _evaluate(
            _engine(),
            _intent(side=OutcomeSide.NO, qty_qc=100),
            _halted_account(position_qc=100),
        )
        assert verdict.approved
        assert verdict.approved_qty_qc == 100

    def test_a_partial_exit_is_allowed_on_a_halted_account(self):
        verdict = _evaluate(
            _engine(),
            _intent(side=OutcomeSide.NO, qty_qc=40),
            _halted_account(position_qc=100),
        )
        assert verdict.approved

    def test_a_short_position_exits_by_buying_yes(self):
        """position_qc < 0 is long NO; the exit is a YES buy."""
        verdict = _evaluate(
            _engine(),
            _intent(side=OutcomeSide.YES, qty_qc=100),
            _halted_account(position_qc=-100),
        )
        assert verdict.approved

    def test_an_exit_with_negative_expected_value_is_still_allowed(self):
        """Getting out is not required to be profitable."""
        verdict = _evaluate(
            _engine(),
            _intent(side=OutcomeSide.NO, qty_qc=100, ev=-999_999),
            _halted_account(position_qc=100),
        )
        assert verdict.approved


class TestAReversalIsNotAnExit:
    def test_one_contract_does_not_exempt_five_thousand(self):
        """The measured case: $2,500 approved against a $25 cap on $0.005."""
        verdict = _evaluate(
            _engine(),
            _intent(side=OutcomeSide.NO, qty_qc=500_000, ev=-999_999),
            _halted_account(position_qc=100),
        )

        assert verdict.decision is RiskDecision.REJECTED
        assert verdict.approved_qty_qc == 0
        # It must be stopped by a real limit, not merely resized.
        assert [c.name for c in verdict.checks] != ["account_state_freshness"]

    def test_one_contract_over_the_position_already_faces_the_limits(self):
        """No tolerance band: the boundary is the size actually held."""
        engine, state = _engine(), _halted_account(position_qc=100)

        assert _evaluate(engine, _intent(side=OutcomeSide.NO, qty_qc=100), state).approved
        assert not _evaluate(
            engine, _intent(side=OutcomeSide.NO, qty_qc=101), state
        ).approved

    def test_a_reversal_on_a_healthy_account_is_sized_rather_than_waved_through(self):
        """It is not forbidden -- it is measured, like any other entry."""
        verdict = _evaluate(
            _engine(),
            _intent(side=OutcomeSide.NO, qty_qc=500_000),
            _healthy_account(position_qc=100),
        )

        if verdict.approved:
            assert verdict.approved_qty_qc < 500_000
            notional = verdict.approved_qty_qc * 5_000
            assert notional <= int(_engine().settings.max_order_notional_usd * 1_000_000)
        assert len(verdict.checks) > 1

    def test_adding_to_a_position_is_never_reducing(self):
        verdict = _evaluate(
            _engine(),
            _intent(side=OutcomeSide.YES, qty_qc=100),
            _halted_account(position_qc=100),
        )
        assert not verdict.approved


class TestTheStrategyCannotAuthoriseItself:
    """reduce_only is set by the strategy, and this engine exists so a strategy
    cannot authorise its own risk."""

    def test_reduce_only_does_not_exempt_an_order_with_no_position(self):
        intent = _intent(side=OutcomeSide.NO, qty_qc=500_000, ev=-999_999)
        intent.reduce_only = True

        verdict = _evaluate(_engine(), intent, _halted_account(position_qc=0))

        assert not verdict.approved
        assert [c.name for c in verdict.checks] != ["account_state_freshness"]

    def test_reduce_only_does_not_exempt_an_oversized_reversal(self):
        intent = _intent(side=OutcomeSide.NO, qty_qc=500_000, ev=-999_999)
        intent.reduce_only = True

        verdict = _evaluate(_engine(), intent, _halted_account(position_qc=100))

        assert not verdict.approved

    def test_a_genuine_exit_marked_reduce_only_still_works(self):
        intent = _intent(side=OutcomeSide.NO, qty_qc=100)
        intent.reduce_only = True

        verdict = _evaluate(_engine(), intent, _halted_account(position_qc=100))

        assert verdict.approved


class TestTheExemptionIsDecidedBySizeAndSide:
    @pytest.mark.parametrize(
        ("position_qc", "side", "qty_qc", "reducing"),
        [
            (0, OutcomeSide.NO, 100, False),      # nothing to reduce
            (100, OutcomeSide.NO, 1, True),
            (100, OutcomeSide.NO, 100, True),
            (100, OutcomeSide.NO, 101, False),    # a reversal
            (100, OutcomeSide.YES, 1, False),     # adds
            (-100, OutcomeSide.YES, 100, True),
            (-100, OutcomeSide.YES, 101, False),
            (-100, OutcomeSide.NO, 1, False),     # adds to the short
        ],
    )
    def test_classification(self, position_qc, side, qty_qc, reducing):
        position = EventPositionState(ticker="X", position_qc=position_qc)
        intent = _intent(side=side, qty_qc=qty_qc)

        assert EventRiskEngine._is_reducing(intent, position) is reducing
