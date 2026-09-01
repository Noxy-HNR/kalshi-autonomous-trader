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
