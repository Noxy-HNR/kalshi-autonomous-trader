"""A paper account must record what its own simulated fills did to it.

In live and demo the reconciler rebuilds positions from the exchange. Paper has
no exchange to ask, and nothing was doing that job locally: 108 fills were
recorded against an account that still read exactly its seeded $200, with no
positions, no exposure and no profit or loss. There was no way to tell a
strategy that was working from one that was not.

The arithmetic mirrors the backtester's EventLedger, which is the reference the
platform already trusts. ``position_qc`` follows Kalshi: positive is YES,
negative is NO, and selling YES you do not hold is buying NO at what is left of
the dollar.
"""

from __future__ import annotations

from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.risk.event_risk import EventAccountState

SEED = 200_000_000  # $200


def _account() -> EventAccountState:
    state = EventAccountState(balance_micros=SEED, updated_at_ms=1_000)
    state.peak_equity_micros = state.equity_micros
    return state


def test_buying_yes_costs_the_yes_price() -> None:
    state = _account()
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    assert state.balance_micros == SEED - 4_000 * 1_000
    position = state.positions["M"]
    assert position.position_qc == 1_000
    assert position.avg_entry_cc == 4_000


def test_selling_to_open_is_buying_no_at_the_rest_of_the_dollar() -> None:
    """The half of Kalshi's model that is easy to get backwards."""
    state = _account()
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=False)
    assert state.positions["M"].position_qc == -1_000
    assert state.positions["M"].avg_entry_cc == CC_PER_DOLLAR - 4_000
    assert state.balance_micros == SEED - 6_000 * 1_000


def test_closing_a_long_at_a_higher_price_is_a_profit() -> None:
    state = _account()
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    realised = state.apply_fill(ticker="M", qty_qc=1_000, price_cc=5_000, is_buy=False)
    assert realised == 1_000 * 1_000  # a cent per contract
    assert state.realized_pnl_today_micros == realised
    assert state.balance_micros == SEED + realised
    assert "M" not in state.positions, "a flat position must not linger"


def test_closing_a_long_at_a_lower_price_is_a_loss() -> None:
    state = _account()
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    realised = state.apply_fill(ticker="M", qty_qc=1_000, price_cc=3_500, is_buy=False)
    assert realised == -500 * 1_000
    assert state.balance_micros == SEED + realised


def test_closing_a_short_at_a_lower_price_is_a_profit() -> None:
    """Short YES is long NO, so the profit runs the other way."""
    state = _account()
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=False)
    realised = state.apply_fill(ticker="M", qty_qc=1_000, price_cc=3_000, is_buy=True)
    assert realised == 1_000 * 1_000
    assert state.balance_micros == SEED + realised


def test_a_partial_close_releases_a_proportional_basis() -> None:
    state = _account()
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    realised = state.apply_fill(ticker="M", qty_qc=400, price_cc=5_000, is_buy=False)
    assert realised == 400 * 1_000
    position = state.positions["M"]
    assert position.position_qc == 600
    assert position.avg_entry_cc == 4_000, "the remainder keeps its cost basis"


def test_adding_to_a_position_averages_the_entry() -> None:
    state = _account()
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=6_000, is_buy=True)
    position = state.positions["M"]
    assert position.position_qc == 2_000
    assert position.avg_entry_cc == 5_000


def test_a_fill_that_crosses_through_flat_closes_then_opens() -> None:
    """Selling 1500 against a 1000 long leaves a 500 short, not a -500 long."""
    state = _account()
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    realised = state.apply_fill(ticker="M", qty_qc=1_500, price_cc=5_000, is_buy=False)
    assert realised == 1_000 * 1_000, "only the 1000 held are closed"
    position = state.positions["M"]
    assert position.position_qc == -500
    assert position.avg_entry_cc == CC_PER_DOLLAR - 5_000


def test_fees_are_charged_on_both_sides_of_a_round_trip() -> None:
    state = _account()
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True, fee_micros=7_000)
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=False, fee_micros=7_000)
    # Flat at the same price: the only change is the two fees.
    assert state.balance_micros == SEED - 14_000


def test_exposure_reflects_what_the_position_can_lose() -> None:
    """For an event contract the whole premium is at risk, so exposure is cost."""
    state = _account()
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    assert state.positions["M"].exposure_micros == 4_000 * 1_000
    assert state.total_exposure_micros() == 4_000 * 1_000


def test_equity_is_unchanged_by_opening_a_position_at_its_own_price() -> None:
    """Cash falls by exactly what the position is worth, so equity holds.

    This is the property that makes the number on the dashboard mean something:
    opening a trade is not a loss.
    """
    state = _account()
    before = state.equity_micros
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    state.portfolio_value_micros = state.total_exposure_micros()
    assert state.equity_micros == before


def test_a_round_trip_moves_equity_by_the_realised_amount() -> None:
    state = _account()
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    state.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_500, is_buy=False)
    assert state.equity_micros == SEED + 500 * 1_000


def test_the_timestamp_is_refreshed_when_supplied() -> None:
    """A fill is first-hand evidence the account state is current."""
    state = _account()
    state.apply_fill(ticker="M", qty_qc=100, price_cc=4_000, is_buy=True, now_ms=9_999)
    assert state.updated_at_ms == 9_999
