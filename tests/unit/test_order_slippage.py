"""Slippage is the price paid against the price asked, in the order's own terms.

This platform only ever buys an event outcome: it exits YES by buying NO. Slippage
read a NO buy as a sell and so reversed the sign of every NO order -- and with
paper recording NO fills on the YES scale besides, a NO buy that filled exactly at
its limit reported -4,100 centicents into the decision record, the orders table
and the dashboard.
"""

from __future__ import annotations

from kat.common.enums import BookSide, MarketType, OutcomeSide
from kat.execution.order import ManagedOrder


def _event(side: OutcomeSide, *, limit: int, filled_at: int) -> ManagedOrder:
    order = ManagedOrder(market_type=MarketType.EVENT, ticker="X", requested_qty_qc=100,
                         limit_price_cc=limit, outcome_side=side)
    order.avg_fill_price_cc = filled_at
    return order


class TestEventOrdersAreBuys:
    def test_a_no_buy_at_its_limit_has_no_slippage(self):
        assert _event(OutcomeSide.NO, limit=2_950, filled_at=2_950).slippage_cc() == 0

    def test_a_no_buy_paying_more_than_its_limit_is_positive(self):
        assert _event(OutcomeSide.NO, limit=2_950, filled_at=2_960).slippage_cc() == 10

    def test_a_no_buy_paying_less_is_negative(self):
        assert _event(OutcomeSide.NO, limit=2_950, filled_at=2_940).slippage_cc() == -10

    def test_a_yes_buy_is_unchanged(self):
        assert _event(OutcomeSide.YES, limit=4_050, filled_at=4_060).slippage_cc() == 10


class TestPerpetualsStillHaveTwoSides:
    def test_an_ask_receiving_less_than_its_limit_is_positive(self):
        order = ManagedOrder(market_type=MarketType.PERP, ticker="P", requested_qty_qc=100,
                             limit_price_cc=10_000, book_side=BookSide.ASK)
        order.avg_fill_price_cc = 9_990
        assert order.slippage_cc() == 10
