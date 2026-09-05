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
