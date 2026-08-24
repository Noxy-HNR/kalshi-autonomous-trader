"""Order book construction and cost estimation.

The YES/NO reflection tested here is the single most common Kalshi integration
bug: Kalshi publishes two arrays of *bids*, and a client that treats the NO array
as an ask array produces a book where every price is inverted while still looking
entirely plausible.
"""

from __future__ import annotations

import pytest

from kat.marketdata.book import OrderBook


@pytest.fixture
def event_book() -> OrderBook:
    """YES bids at $0.42/$0.41; NO bids at $0.55/$0.54.

    The NO side reflects to YES asks at $0.45 and $0.46.
    """
    return OrderBook.from_event_contract_levels(
        "MKT",
        yes_levels=[(4200, 1300), (4100, 5000)],
        no_levels=[(5500, 2000), (5400, 3000)],
    )


class TestEventContractReflection:
    def test_no_bids_become_yes_asks_at_one_minus_price(self, event_book: OrderBook):
        assert event_book.best_bid_cc == 4200
        assert event_book.best_ask_cc == 4500  # 1.0000 - 0.5500
        assert event_book.asks == {4500: 2000, 4600: 3000}

    def test_spread_and_mid_are_computed_on_the_reflected_book(self, event_book: OrderBook):
        assert event_book.spread_cc == 300
        assert event_book.mid_cc == 4350

    def test_a_no_side_delta_moves_the_ask_side(self, event_book: OrderBook):
        event_book.apply_event_contract_delta(5500, -2000, is_yes=False)
        assert event_book.best_ask_cc == 4600

    def test_a_yes_side_delta_moves_the_bid_side(self, event_book: OrderBook):
        event_book.apply_event_contract_delta(4200, -1300, is_yes=True)
        assert event_book.best_bid_cc == 4100

    def test_zero_size_levels_are_dropped_not_stored(self):
        book = OrderBook.from_event_contract_levels(
            "MKT", yes_levels=[(4200, 0)], no_levels=[(5500, 0)]
        )
        assert book.best_bid_cc is None
        assert book.best_ask_cc is None


class TestEmptyAndDegenerateBooks:
    def test_an_empty_book_reports_none_not_zero(self):
        """A missing bid and a bid of zero are different facts."""
        book = OrderBook("MKT")
        assert book.best_bid_cc is None
        assert book.best_ask_cc is None
        assert book.spread_cc is None
        assert book.mid_cc is None

    def test_a_one_sided_book_has_no_midpoint(self):
        """Inventing a mid from one side is how you price against a market
        that is not really there."""
        book = OrderBook.from_two_sided_levels("MKT", [(4000, 1000)], [])
        assert book.best_bid_cc == 4000
        assert book.mid_cc is None
        assert book.spread_cc is None

    def test_a_crossed_book_is_detected(self):
        book = OrderBook.from_two_sided_levels("MKT", [(4500, 1000)], [(4400, 1000)])
        assert book.is_crossed is True

    def test_an_uncrossed_book_is_not_flagged(self, event_book: OrderBook):
        assert event_book.is_crossed is False


class TestWalkingTheBook:
    def test_walk_consumes_levels_in_price_order(self, event_book: OrderBook):
        # Buying 40 contracts takes 20 at $0.45 and 20 at $0.46.
        filled, notional, worst = event_book.walk(4000, is_buy=True)
        assert filled == 4000
        assert notional == 18_200_000  # $9.00 + $9.20
        assert worst == 4600

    def test_vwap_rounds_against_the_buyer(self, event_book: OrderBook):
        assert event_book.vwap_cc(4000, is_buy=True) == 4550

    def test_slippage_is_measured_from_the_touch(self, event_book: OrderBook):
        assert event_book.slippage_cc(4000, is_buy=True) == 50

    def test_vwap_is_none_when_depth_is_insufficient(self, event_book: OrderBook):
        """A partial answer would understate the true cost of the order."""
        assert event_book.vwap_cc(10_000, is_buy=True) is None

    def test_walk_on_an_empty_side_returns_none(self):
        book = OrderBook.from_two_sided_levels("MKT", [(4000, 1000)], [])
        assert book.walk(100, is_buy=True) is None

    def test_walk_rejects_non_positive_size(self, event_book: OrderBook):
        assert event_book.walk(0, is_buy=True) is None
        assert event_book.walk(-100, is_buy=True) is None

    def test_selling_walks_the_bid_side(self, event_book: OrderBook):
        filled, notional, worst = event_book.walk(6300, is_buy=False)
        assert filled == 6300
        # 13 contracts at $0.42 plus 50 at $0.41.
        assert notional == 4200 * 1300 + 4100 * 5000
        assert worst == 4100


class TestDeltas:
    def test_a_level_reaching_zero_is_removed(self):
        book = OrderBook.from_two_sided_levels("MKT", [(4000, 1000)], [(4100, 1000)])
        book.apply_delta(4000, -1000, is_bid=True)
        assert 4000 not in book.bids
        assert book.best_bid_cc is None

    def test_a_negative_resting_size_is_treated_as_removal(self):
        """Going below zero means we missed a message; do not store nonsense."""
        book = OrderBook.from_two_sided_levels("MKT", [(4000, 1000)], [(4100, 1000)])
        book.apply_delta(4000, -5000, is_bid=True)
        assert 4000 not in book.bids

    def test_adding_size_at_a_new_level_reorders_the_book(self):
        book = OrderBook.from_two_sided_levels("MKT", [(4000, 1000)], [(4100, 1000)])
        book.apply_delta(4050, 500, is_bid=True)
        assert book.best_bid_cc == 4050

    def test_depth_counts_only_the_requested_levels(self):
        book = OrderBook.from_two_sided_levels(
            "MKT", [(4000, 1000), (3900, 2000), (3800, 4000)], [(4100, 1000)]
        )
        assert book.depth_qc(is_bid=True, levels=2) == 3000
        assert book.total_depth_qc(is_bid=True) == 7000


class TestSnapshot:
    def test_snapshot_is_ordered_best_first(self, event_book: OrderBook):
        snap = event_book.snapshot(levels=2)
        assert snap["bids"] == [[4200, 1300], [4100, 5000]]
        assert snap["asks"] == [[4500, 2000], [4600, 3000]]
