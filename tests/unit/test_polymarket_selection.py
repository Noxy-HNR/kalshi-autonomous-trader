"""Which sixty markets out of four thousand get watched closely.

The collector can follow about sixty at a resolution fine enough to measure a
fill. That choice decides what every later question can be asked of, so it is
worth more than the eight lines it takes to write.

The ranking encodes the one thing the Kalshi work established and then failed to
act on: it measured spread and queue depth separately, found 879 markets trading
over a thousand contracts a day, and still produced nine round trips in
seventy-two hours -- because it never divided one by the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from kat.marketdata.book import OrderBook
from kat.polymarket.fees import PolymarketFeeSchedule
from kat.polymarket.selection import maker_edge_cc, rank_by_reachability

EXACT = PolymarketFeeSchedule(
    taker_theta=Decimal("0.06"), maker_theta=Decimal("-0.0125"), exact=True
)


@dataclass
class _Market:
    slug: str
    fees: PolymarketFeeSchedule = EXACT


def _book(bid_cc: int, ask_cc: int, size_qc: int = 1_000) -> OrderBook:
    return OrderBook.from_two_sided_levels(
        "m", bid_levels=[(bid_cc, size_qc)], ask_levels=[(ask_cc, size_qc)]
    )


class TestTheEdgeIncludesTheRebate:
    def test_a_round_trip_earns_more_than_the_spread(self):
        """Because the maker is paid on both legs. On Kalshi the same book
        earns the spread *minus* two fees, which at these prices is negative."""
        edge = maker_edge_cc(_Market("m"), _book(1_540, 1_600))
        assert edge is not None
        assert edge > 60  # the spread alone

    def test_a_one_tick_spread_still_pays(self):
        """A tenth of a cent is the venue's minimum increment. On Kalshi the
        minimum tick is a whole cent and the fee at this price is 92 centicents
        a side, so the same trade is a loss there and a gain here."""
        edge = maker_edge_cc(_Market("m"), _book(1_540, 1_550))
        assert edge is not None
        assert edge > 0

    def test_an_unpriceable_book_is_none_not_zero(self):
        """Zero is an edge. Unknown is not, and ranking on it would follow a
        market chosen by a missing value."""
        assert maker_edge_cc(_Market("m"), _book(1_600, 1_540)) is None  # crossed
        assert maker_edge_cc(_Market("m"), _book(1_540, 1_540)) is None  # locked

    def test_a_settled_market_earns_nothing(self):
        """At the bounds the quadratic vanishes; quoting a resolved market must
        not look like free income."""
        edge = maker_edge_cc(_Market("m"), _book(9_990, 9_999))
        assert edge is None or edge <= 9


class TestRankingIsOnReachabilityNotEdge:
    def test_a_short_queue_beats_a_bigger_edge_behind_a_long_one(self):
        """The whole point. A wide spread behind ten thousand contracts pays
        nothing, because everything ahead trades first."""
        rich_but_buried = (_Market("buried"), 500, 1_000_000)
        thin_but_reachable = (_Market("reachable"), 50, 100)
        assert rank_by_reachability([rich_but_buried, thin_but_reachable], 1) == [
            "reachable"
        ]

    def test_markets_that_do_not_pay_are_dropped_entirely(self):
        """Following one to see whether it starts paying later spends a slot a
        paying market could have used."""
        assert rank_by_reachability(
            [(_Market("loses"), -10, 100), (_Market("pays"), 10, 100)], 5
        ) == ["pays"]

    def test_an_empty_touch_does_not_divide_by_zero(self):
        """Being first in line is the best case, not an infinite one."""
        assert rank_by_reachability([(_Market("empty"), 50, 0)], 1) == ["empty"]

    def test_the_limit_is_respected(self):
        candidates = [(_Market(f"m{i}"), 100, 100) for i in range(200)]
        assert len(rank_by_reachability(candidates, 60)) == 60

    def test_the_order_is_stable_across_sweeps(self):
        """An unstable follow set silently restarts each market's history every
        six hours, which is the one thing a long collection cannot afford."""
        tied = [(_Market("b"), 100, 100), (_Market("a"), 100, 100)]
        assert rank_by_reachability(tied, 2) == rank_by_reachability(
            list(reversed(tied)), 2
        )

    def test_nothing_in_nothing_out(self):
        assert rank_by_reachability([], 60) == []


class TestOnlyMarketsThatTradeAreFollowed:
    """Ranking on reachability alone was wrong in a way that hid its own error.

    It favours short queues; a short queue usually means few people quoting;
    few quoters correlates with few takers. The first follow set built that way
    reported zero trades in 297 intervals, while a broad unranked sample of the
    same venue found 15% of counter-reporting markets trading and 1,473
    contracts an hour. The ranking had selected the markets most likely to
    report nothing, and then been believed.
    """

    def test_a_market_that_has_never_traded_is_not_followed(self):
        from kat.polymarket.selection import rank_tradeable

        assert rank_tradeable([(_Market("never"), 500, 100, None)], 10) == []
        assert rank_tradeable([(_Market("never"), 500, 100, 0)], 10) == []

    def test_among_traders_the_short_queue_still_wins(self):
        """The original insight was sound and is kept: measured against markets
        that did trade, six contracts of queue against 239 traded an hour is
        twenty fills an hour, and 102,299 of queue against 63 is none."""
        from kat.polymarket.selection import rank_tradeable

        deep = (_Market("deep"), 44, 10_229_900, 6_300)
        thin = (_Market("thin"), 44, 600, 23_900)
        assert rank_tradeable([deep, thin], 1) == ["thin"]

    def test_a_dead_market_loses_to_a_worse_but_live_one(self):
        from kat.polymarket.selection import rank_tradeable

        dead_but_perfect = (_Market("dead"), 900, 100, None)
        live_but_thin = (_Market("live"), 10, 100_000, 500)
        assert rank_tradeable([dead_but_perfect, live_but_thin], 2) == ["live"]

    def test_it_still_drops_markets_whose_round_trip_does_not_pay(self):
        from kat.polymarket.selection import rank_tradeable

        assert rank_tradeable([(_Market("loses"), -5, 100, 9_999)], 5) == []
