"""A market is only worth watching if a move there pays for itself.

Round-trip cost is the spread plus two fees, and the fee is quadratic in price:
175 centicents a side at $0.50, 33 at $0.95, 14 at $0.98. So a market's price
level decides much of its own tradeability before any strategy has an opinion.

Measured over a week of banked quotes, every market the platform was watching
cleared its own round-trip cost in 0.0% of thirty-minute windows -- moves of 0 to
12 centicents against costs of 136 to 466. The ranking asked for volume and a
spread cap, and those are not the same question: volume says a market traded, not
that it moved, and nothing at all about the toll.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kat.engine.ingestion import (
    _MIN_MOVE_VS_COST,
    _MIN_WATCH_VOLUME_QC,
    _move_vs_cost,
    _round_trip_cost_cc,
)


@dataclass
class _Market:
    """Only the fields the scoring reads."""

    yes_bid_cc: int | None = 4_900
    yes_ask_cc: int | None = 5_100
    last_price_cc: int | None = 5_000
    previous_price_cc: int | None = 5_000
    volume_24h_qc: int | None = 100_000


class TestRoundTripCost:
    def test_the_fee_is_charged_twice_and_the_spread_once(self):
        # At $0.50 the fee is 0.07 * 0.5 * 0.5 = $0.0175 = 175cc a side.
        cost = _round_trip_cost_cc(_Market(yes_bid_cc=4_900, yes_ask_cc=5_100))
        assert cost == 200 + 2 * 175

    def test_cost_collapses_at_the_extremes(self):
        """The quadratic fee is why the edges are where trading is affordable."""
        middle = _round_trip_cost_cc(_Market(yes_bid_cc=4_950, yes_ask_cc=5_050))
        edge = _round_trip_cost_cc(_Market(yes_bid_cc=9_750, yes_ask_cc=9_850))

        assert middle is not None and edge is not None
        assert edge < middle / 3

    @pytest.mark.parametrize("bid,ask", [(None, 5_100), (4_900, None)])
    def test_an_unpriceable_market_has_no_cost(self, bid, ask):
        assert _round_trip_cost_cc(_Market(yes_bid_cc=bid, yes_ask_cc=ask)) is None

    def test_a_market_pinned_at_a_dollar_has_no_cost(self):
        """A mid at the boundary is a settled market, not a cheap one."""
        assert _round_trip_cost_cc(_Market(yes_bid_cc=10_000, yes_ask_cc=10_000)) is None


class TestMoveVersusCost:
    def test_a_move_larger_than_the_cost_scores_above_one(self):
        market = _Market(last_price_cc=5_000, previous_price_cc=4_000)  # 1000cc move
        score = _move_vs_cost(market)
        assert score is not None and score > 1.0

    def test_a_move_smaller_than_the_cost_scores_below_one(self):
        """The state of every market the platform was actually watching."""
        market = _Market(last_price_cc=5_000, previous_price_cc=4_990)  # 10cc move
        score = _move_vs_cost(market)
        assert score is not None and score < 1.0

    def test_direction_does_not_matter(self):
        up = _move_vs_cost(_Market(last_price_cc=5_500, previous_price_cc=5_000))
        down = _move_vs_cost(_Market(last_price_cc=4_500, previous_price_cc=5_000))
        assert up == down


class TestAMissingPreviousPriceIsNotAFreeDollar:
    """The failure this scoring nearly shipped with.

    9,752 of 26,479 live markets report a previous price of exactly 0. Read
    literally, a market last trading at $0.99 then looks like a 99-cent move, and
    because the score is a ratio those markets sort straight to the top -- the
    absent value becoming the most attractive one. A live contract cannot trade
    at $0.0000.
    """

    def test_a_zero_previous_price_is_unknown_not_a_move(self):
        assert _move_vs_cost(_Market(last_price_cc=9_900, previous_price_cc=0)) is None

    def test_a_none_previous_price_is_unknown(self):
        assert _move_vs_cost(_Market(previous_price_cc=None)) is None

    def test_a_zero_last_price_is_unknown_too(self):
        assert _move_vs_cost(_Market(last_price_cc=0, previous_price_cc=5_000)) is None

    def test_an_unknown_market_never_outranks_a_real_mover(self):
        """The property that matters: absent data must not win the ranking."""
        unknown = _move_vs_cost(_Market(last_price_cc=9_900, previous_price_cc=0))
        real = _move_vs_cost(_Market(last_price_cc=5_500, previous_price_cc=5_000))

        assert real is not None
        assert (unknown or 0.0) < real


class TestTheThresholdsAreSane:
    def test_the_movement_bar_is_a_screen_not_a_promise(self):
        """Below 1.0 by design: it excludes the unwinnable without claiming the
        next move will pay."""
        assert 0.0 < _MIN_MOVE_VS_COST < 1.0

    def test_the_volume_floor_admits_a_small_account(self):
        """Fifty contracts a day, not a deep book."""
        assert 0 < _MIN_WATCH_VOLUME_QC <= 50_000


class TestVolatilityFallsBackToTheWindowThisVenueSupports:
    """``realized_vol_5m_cc`` needs four bar-to-bar changes inside fifteen
    minutes, which assumes a market that reprices continuously.

    Kalshi event contracts reprice a few times an hour. Of sixty watched
    markets, eight cleared that bar, and ``realized_vol_unavailable`` was the
    single largest reason the platform declined to trade -- 76,644 refusals
    across a thousand cycles. The hour-long window is the same statistic over
    more data, so it is a substitute rather than a fudge.
    """

    @staticmethod
    def _ctx(features):
        from kat.common.enums import MarketType
        from kat.marketdata.state import MarketSnapshot
        from kat.strategies.base import StrategyContext

        return StrategyContext(
            now_ms=0,
            market=MarketSnapshot(ticker="X", market_type=MarketType.EVENT),
            features=features,
        )

    def test_the_short_window_is_preferred(self):
        from kat.strategies.common import per_bar_volatility_cc

        ctx = self._ctx({"realized_vol_5m_cc": 12.0, "realized_vol_60m_cc": 30.0})
        assert per_bar_volatility_cc(ctx) == 12.0

    def test_the_hour_window_is_used_when_the_short_one_is_absent(self):
        from kat.strategies.common import per_bar_volatility_cc

        assert per_bar_volatility_cc(self._ctx({"realized_vol_60m_cc": 30.0})) == 30.0

    def test_a_zero_short_window_falls_through_rather_than_dividing_by_it(self):
        """Zero is a missing estimate, not a calm market."""
        from kat.strategies.common import per_bar_volatility_cc

        ctx = self._ctx({"realized_vol_5m_cc": 0.0, "realized_vol_60m_cc": 30.0})
        assert per_bar_volatility_cc(ctx) == 30.0

    def test_neither_window_available_is_none_not_zero(self):
        from kat.strategies.common import per_bar_volatility_cc

        assert per_bar_volatility_cc(self._ctx({})) is None
        assert per_bar_volatility_cc(self._ctx({"realized_vol_5m_cc": 0.0})) is None


class TestSomeSlotsGoToMarketsThatWillResolve:
    """Trading and learning want different markets.

    The economic filter picks markets where a move can pay for itself, and on
    the live board 717 of the 734 that qualify close more than a month out --
    only one inside a week. Those never resolve while we watch them, so they can
    never carry a settlement label, and a settlement label is the only target
    MispricingStrategy can read.

    Every settlement the platform has recorded came from the old volume-ranked
    watchlist, which happened to include same-day baseball, and all of them
    closed near the end of the history. That is why the first settlement dataset
    could not be split honestly: purging emptied validation even at a zero
    embargo.
    """

    from datetime import UTC, datetime, timedelta

    @staticmethod
    def _market(ticker, days_to_close, volume=1_000_000):
        from dataclasses import dataclass
        from datetime import UTC, datetime, timedelta

        @dataclass
        class _M:
            ticker: str
            close_time: object
            volume_24h_qc: int
            yes_bid_cc: int = 4_900
            yes_ask_cc: int = 5_100
            last_price_cc: int = 5_000
            previous_price_cc: int = 4_000

        return _M(
            ticker=ticker,
            close_time=datetime.now(UTC) + timedelta(days=days_to_close),
            volume_24h_qc=volume,
        )

    def test_soon_closing_markets_take_reserved_slots(self):
        from kat.engine.ingestion import _LEARNING_SLOTS, _with_learning_slots

        far = [self._market(f"FAR{i}", 200) for i in range(60)]
        soon = [self._market(f"SOON{i}", 2) for i in range(20)]

        picked = _with_learning_slots(far, far + soon, limit=60)

        chosen = {m.ticker for m in picked}
        assert len(picked) == 60
        assert sum(1 for t in chosen if t.startswith("SOON")) == _LEARNING_SLOTS

    def test_the_trading_slice_keeps_the_rest(self):
        from kat.engine.ingestion import _LEARNING_SLOTS, _with_learning_slots

        far = [self._market(f"FAR{i}", 200) for i in range(60)]
        soon = [self._market(f"SOON{i}", 2) for i in range(20)]

        picked = _with_learning_slots(far, far + soon, limit=60)

        far_kept = sum(1 for m in picked if m.ticker.startswith("FAR"))
        assert far_kept == 60 - _LEARNING_SLOTS

    def test_no_soon_closing_market_leaves_the_ranking_alone(self):
        """A board with nothing resolving soon is not a reason to watch less."""
        from kat.engine.ingestion import _with_learning_slots

        far = [self._market(f"FAR{i}", 200) for i in range(60)]

        picked = _with_learning_slots(far, far, limit=60)

        assert len(picked) == 60
        assert all(m.ticker.startswith("FAR") for m in picked)

    def test_an_illiquid_soon_closing_market_is_not_reserved_a_slot(self):
        """Resolving is necessary; being tradeable at all is still required."""
        from kat.engine.ingestion import _with_learning_slots

        far = [self._market(f"FAR{i}", 200) for i in range(60)]
        thin = [self._market("THIN", 2, volume=10)]

        picked = _with_learning_slots(far, far + thin, limit=60)

        assert "THIN" not in {m.ticker for m in picked}

    def test_a_watchlist_smaller_than_the_reservation_is_untouched(self):
        from kat.engine.ingestion import _with_learning_slots

        far = [self._market(f"FAR{i}", 200) for i in range(5)]
        soon = [self._market("SOON", 2)]

        picked = _with_learning_slots(far, far + soon, limit=5)

        assert [m.ticker for m in picked] == [f"FAR{i}" for i in range(5)]


class TestSomeSlotsGoToMarketsThatTrade:
    """A taker needs the price to move. A maker needs somebody to trade.

    The move-versus-cost filter is the right question for one and silently
    excludes the best markets for the other. It excluded them badly: the board
    carries 879 markets trading over a thousand contracts a day and 159 over ten
    thousand, while the markets the platform watched averaged fourteen trades a
    day -- and a passive quote managed one round trip in eight days for want of
    anybody to trade with.
    """

    @staticmethod
    def _market(ticker, *, volume, bid=9_400, ask=9_500, last=9_450, prev=9_440):
        from dataclasses import dataclass
        from datetime import UTC, datetime, timedelta

        @dataclass
        class _M:
            ticker: str
            volume_24h_qc: int
            yes_bid_cc: int
            yes_ask_cc: int
            last_price_cc: int
            previous_price_cc: int
            close_time: object

        return _M(
            ticker=ticker, volume_24h_qc=volume, yes_bid_cc=bid, yes_ask_cc=ask,
            last_price_cc=last, previous_price_cc=prev,
            close_time=datetime.now(UTC) + timedelta(days=200),
        )

    def test_a_maker_edge_is_the_spread_less_both_fees(self):
        from kat.engine.ingestion import _maker_edge_cc

        # At 94-95 cents the fee is small, so a one-cent spread pays.
        assert (_maker_edge_cc(self._market("X", volume=1)) or 0) > 0

    def test_a_spread_that_cannot_cover_two_fees_is_not_a_maker_market(self):
        from kat.engine.ingestion import _maker_edge_cc

        # Mid-board the fee is 175cc a side, so a one-cent spread loses.
        tight = self._market("Y", volume=1, bid=4_950, ask=5_050)
        assert (_maker_edge_cc(tight) or 0) < 0

    def test_liquid_markets_take_the_reserved_slots(self):
        from kat.engine.ingestion import _MAKER_SLOTS, _with_maker_slots

        quiet = [self._market(f"QUIET{i}", volume=6_000) for i in range(60)]
        busy = [self._market(f"BUSY{i}", volume=5_000_000) for i in range(20)]

        picked = _with_maker_slots(quiet, quiet + busy, limit=60)

        chosen = {m.ticker for m in picked}
        assert len(picked) == 60
        assert sum(1 for t in chosen if t.startswith("BUSY")) == _MAKER_SLOTS

    def test_a_liquid_market_with_no_spread_edge_is_not_reserved(self):
        """Flow is necessary and not sufficient: the spread still has to pay."""
        from kat.engine.ingestion import _with_maker_slots

        quiet = [self._market(f"QUIET{i}", volume=6_000) for i in range(60)]
        # Enormous volume, but a spread that cannot cover two mid-board fees.
        churn = [self._market("CHURN", volume=9_000_000, bid=4_950, ask=5_050)]

        picked = _with_maker_slots(quiet, quiet + churn, limit=60)

        assert "CHURN" not in {m.ticker for m in picked}

    def test_a_board_with_nothing_liquid_leaves_the_ranking_alone(self):
        from kat.engine.ingestion import _with_maker_slots

        quiet = [self._market(f"QUIET{i}", volume=6_000) for i in range(60)]
        picked = _with_maker_slots(quiet, quiet, limit=60)

        assert len(picked) == 60
        assert all(m.ticker.startswith("QUIET") for m in picked)
