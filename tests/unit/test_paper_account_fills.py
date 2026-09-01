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


# -- mark to market ----------------------------------------------------------
#
# Without marking, a position is only ever worth what it cost: equity moves when
# a trade closes and at no other time, and every open position reports zero
# unrealised P&L for its whole life. "Is this working?" then has no answer until
# something is sold.

from kat.common.enums import MarketType  # noqa: E402
from kat.common.timeutil import SimulatedClock  # noqa: E402
from kat.engine.trading import TradingLoop  # noqa: E402
from kat.marketdata.book import OrderBook  # noqa: E402
from kat.marketdata.state import MarketState  # noqa: E402

MARK_T0 = 1_700_000_000_000


class _Stub:
    market_type = MarketType.EVENT

    def working_orders(self) -> list[object]:
        return []


def _loop() -> tuple[TradingLoop, MarketState, EventAccountState]:
    state = MarketState(MarketType.EVENT)
    loop = TradingLoop(
        market_type=MarketType.EVENT,
        market_state=state,
        risk_engine=None,
        execution=_Stub(),  # type: ignore[arg-type]
        recorder=None,  # type: ignore[arg-type]
        clock=SimulatedClock(MARK_T0),
    )
    loop.paper_fills = True
    loop.account_state = _account()
    return loop, state, loop.account_state


def _quote(state: MarketState, ticker: str, mid_cc: int, now_ms: int) -> None:
    book = OrderBook.from_event_contract_levels(
        ticker,
        yes_levels=[(mid_cc - 50, 100_000)],
        no_levels=[(CC_PER_DOLLAR - (mid_cc + 50), 100_000)],
    )
    state.apply_snapshot(ticker, book, now_ms=now_ms)


def test_a_long_gains_when_the_market_rises() -> None:
    loop, state, account = _loop()
    account.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    _quote(state, "M", 5_000, MARK_T0)
    loop._mark_positions()
    assert account.positions["M"].unrealized_pnl_micros == 1_000 * 1_000
    assert account.portfolio_value_micros == 5_000 * 1_000


def test_a_short_gains_when_the_market_falls() -> None:
    """Short YES is long NO, so the mark runs the other way."""
    loop, state, account = _loop()
    account.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=False)
    _quote(state, "M", 3_000, MARK_T0)
    loop._mark_positions()
    # Paid 6000cc for NO; NO is now worth 7000cc.
    assert account.positions["M"].unrealized_pnl_micros == 1_000 * 1_000


def test_opening_a_position_does_not_move_equity() -> None:
    """The property that makes the dashboard number honest."""
    loop, state, account = _loop()
    _quote(state, "M", 4_000, MARK_T0)
    before = account.equity_micros
    account.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    loop._mark_positions()
    assert account.equity_micros == before


def test_equity_tracks_the_market_while_a_position_is_open() -> None:
    loop, state, account = _loop()
    _quote(state, "M", 4_000, MARK_T0)
    account.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    loop._mark_positions()
    opened = account.equity_micros

    _quote(state, "M", 4_300, MARK_T0 + 1_000)
    loop.clock = SimulatedClock(MARK_T0 + 1_000)
    loop._mark_positions()
    assert account.equity_micros == opened + 300 * 1_000


def test_a_stale_market_keeps_its_last_mark_rather_than_going_to_zero() -> None:
    """An absent price is not a price of nothing.

    Marking to zero would report a total loss every time a feed hiccuped, which
    is exactly the moment an operator is most likely to be looking.
    """
    loop, state, account = _loop()
    _quote(state, "M", 4_000, MARK_T0)
    account.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    loop._mark_positions()
    marked = account.portfolio_value_micros

    # Far past any freshness budget, with no new quote.
    loop.clock = SimulatedClock(MARK_T0 + 3_600_000)
    loop._mark_positions()
    assert account.portfolio_value_micros == marked


def test_marking_is_inert_outside_paper() -> None:
    """Live and demo take positions from the reconciler; a second source for the
    same number is how two truths appear."""
    loop, state, account = _loop()
    loop.paper_fills = False
    account.apply_fill(ticker="M", qty_qc=1_000, price_cc=4_000, is_buy=True)
    account.portfolio_value_micros = 12_345
    _quote(state, "M", 9_000, MARK_T0)
    loop._mark_positions()
    assert account.portfolio_value_micros == 12_345


def test_no_positions_marks_to_nothing() -> None:
    loop, _state, account = _loop()
    loop._mark_positions()
    assert account.portfolio_value_micros == 0
    assert account.equity_micros == SEED


class TestMarkingIsGatedOnTheFeedNotTheMarket:
    """A quiet market is not an unpriced one.

    On a delta feed, no message means the book has not changed, so the last mid
    is still the current mid. Marking used to require ``is_fresh`` -- the test
    for whether a market may be *traded*, which is a stricter question. The two
    came apart on the real board: the median watched market updates every 489
    seconds against a 300-second quiet bound, so most positions failed the test,
    kept a frozen mark, and equity sat at exactly $190.346066 across eighteen
    open positions indefinitely. Paper equity that cannot move cannot answer the
    one question paper trading exists to answer.
    """

    def test_a_quiet_market_is_still_marked(self):
        loop, state, account = _loop()
        _quote(state, "Q", 5_000, MARK_T0)
        account.apply_fill(
            ticker="Q", qty_qc=1_000, price_cc=5_000, is_buy=True, now_ms=MARK_T0
        )

        # Well past the quiet bound, but the feed is alive: another market just
        # spoke, so silence on this one means the book has not changed.
        later = MARK_T0 + 600_000
        # In time order: Q moves, goes quiet, and something else keeps the feed
        # demonstrably alive. Quoting out of order would wind the feed clock back.
        _quote(state, "Q", 6_000, later - 400_000)
        _quote(state, "OTHER", 5_000, later)
        loop.clock = SimulatedClock(later)
        loop._mark_positions()

        assert not state.is_fresh("Q", now_ms=later), "fixture must be past the bound"
        assert account.positions["Q"].unrealized_pnl_micros > 0, (
            "a market that moved in our favour but has been quiet since must "
            "still be marked at the price it moved to"
        )

    def test_a_dead_feed_keeps_the_last_mark(self):
        """The failure that marking on the feed must still refuse."""
        loop, state, account = _loop()
        _quote(state, "Q", 5_000, MARK_T0)
        account.apply_fill(
            ticker="Q", qty_qc=1_000, price_cc=5_000, is_buy=True, now_ms=MARK_T0
        )
        _quote(state, "Q", 6_000, MARK_T0 + 1_000)
        loop.clock = SimulatedClock(MARK_T0 + 2_000)
        loop._mark_positions()
        marked = account.positions["Q"].unrealized_pnl_micros

        # Nothing anywhere for far longer than the feed budget.
        silent = MARK_T0 + 2_000 + state.feed_max_age_ms + 60_000
        loop.clock = SimulatedClock(silent)
        loop._mark_positions()

        assert account.positions["Q"].unrealized_pnl_micros == marked
