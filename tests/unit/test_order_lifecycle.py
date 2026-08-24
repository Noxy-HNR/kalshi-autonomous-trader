"""Order state machine.

The premise under test: an HTTP 200 is not a fill and a timeout is not a
rejection. Our order state is ours, the exchange's is the exchange's, and the
transitions between them are the ones enumerated here.
"""

from __future__ import annotations

import pytest

from kat.common.enums import BookSide, MarketType, OrderState, OutcomeSide
from kat.execution.order import ManagedOrder, OrderTransitionError, is_legal_transition


@pytest.fixture
def order() -> ManagedOrder:
    return ManagedOrder(
        market_type=MarketType.EVENT,
        ticker="MKT",
        requested_qty_qc=1000,
        limit_price_cc=4000,
        outcome_side=OutcomeSide.YES,
    )


class TestTransitions:
    def test_happy_path(self, order: ManagedOrder):
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.ACKNOWLEDGED)
        order.apply_fill(fill_id="f1", qty_qc=1000, price_cc=4000)
        assert order.state is OrderState.FILLED

    def test_terminal_states_accept_nothing_further(self, order: ManagedOrder):
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.REJECTED)
        with pytest.raises(OrderTransitionError):
            order.transition(OrderState.ACKNOWLEDGED)

    def test_a_cancel_can_lose_the_race_to_a_fill(self):
        """The matching engine can fill an order we asked to cancel.

        A state machine that forbade this would reject the truth and leave us
        believing we were flat when we were not.
        """
        assert is_legal_transition(OrderState.CANCEL_REQUESTED, OrderState.FILLED) is True

    def test_unknown_can_resolve_to_anything(self):
        """Only reconciliation moves an order out of UNKNOWN, and it may land
        anywhere the exchange says."""
        for target in (
            OrderState.ACKNOWLEDGED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        ):
            assert is_legal_transition(OrderState.UNKNOWN, target) is True

    def test_cannot_skip_straight_from_created_to_filled(self, order: ManagedOrder):
        with pytest.raises(OrderTransitionError):
            order.transition(OrderState.FILLED)

    def test_non_strict_mode_logs_instead_of_raising(self, order: ManagedOrder):
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.REJECTED)
        order.transition(OrderState.ACKNOWLEDGED, strict=False)
        assert order.state is OrderState.REJECTED  # unchanged

    def test_history_records_every_transition(self, order: ManagedOrder):
        order.transition(OrderState.SUBMITTED, reason="sent")
        order.transition(OrderState.ACKNOWLEDGED, reason="acked")
        assert [h["to"] for h in order.state_history] == ["submitted", "acknowledged"]
        assert order.state_history[0]["reason"] == "sent"


class TestExposureAccounting:
    def test_unknown_orders_still_consume_exposure(self):
        """An order that might be resting is an order that might fill."""
        assert OrderState.UNKNOWN.is_working is True

    def test_submitted_orders_consume_exposure(self):
        assert OrderState.SUBMITTED.is_working is True

    @pytest.mark.parametrize(
        "state",
        [
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.RISK_REJECTED,
            OrderState.EXPIRED,
        ],
    )
    def test_terminal_orders_consume_nothing(self, state: OrderState):
        assert state.is_working is False
        assert state.is_terminal is True

    def test_remaining_quantity_drives_exposure(self, order: ManagedOrder):
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.ACKNOWLEDGED)
        order.apply_fill(fill_id="f1", qty_qc=400, price_cc=4000)
        assert order.remaining_qty_qc == 600
        assert order.exposure_qty_qc == 600

    def test_a_terminal_order_exposes_nothing_even_if_unfilled(self, order: ManagedOrder):
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.CANCELLED)
        assert order.exposure_qty_qc == 0

    def test_signed_remaining_follows_direction(self):
        sell = ManagedOrder(
            market_type=MarketType.PERP,
            ticker="P",
            requested_qty_qc=1000,
            limit_price_cc=5000,
            book_side=BookSide.ASK,
        )
        sell.transition(OrderState.SUBMITTED)
        assert sell.signed_remaining_qc == -1000


class TestFills:
    def test_partial_then_complete(self, order: ManagedOrder):
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.ACKNOWLEDGED)
        order.apply_fill(fill_id="f1", qty_qc=400, price_cc=4000)
        assert order.state is OrderState.PARTIALLY_FILLED
        order.apply_fill(fill_id="f2", qty_qc=600, price_cc=4100)
        assert order.state is OrderState.FILLED
        # Weighted average: (400*4000 + 600*4100) / 1000
        assert order.avg_fill_price_cc == 4060

    def test_duplicate_fills_are_ignored(self, order: ManagedOrder):
        """A websocket replay after reconnect resends fills; double-counting one
        would corrupt both the position and the average price."""
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.ACKNOWLEDGED)
        assert order.apply_fill(fill_id="f1", qty_qc=400, price_cc=4000) is True
        assert order.apply_fill(fill_id="f1", qty_qc=400, price_cc=4000) is False
        assert order.filled_qty_qc == 400

    def test_fees_accumulate_across_fills(self, order: ManagedOrder):
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.ACKNOWLEDGED)
        order.apply_fill(fill_id="f1", qty_qc=500, price_cc=4000, fee_micros=1000)
        order.apply_fill(fill_id="f2", qty_qc=500, price_cc=4000, fee_micros=1500)
        assert order.fees_micros == 2500

    def test_overfill_still_lands_in_filled(self, order: ManagedOrder):
        """Defensive: a fill larger than requested should not leave the order
        stuck in a partial state forever."""
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.ACKNOWLEDGED)
        order.apply_fill(fill_id="f1", qty_qc=1500, price_cc=4000)
        assert order.state is OrderState.FILLED
        assert order.remaining_qty_qc == 0


class TestSlippage:
    def test_paying_above_a_buy_limit_is_positive_slippage(self, order: ManagedOrder):
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.ACKNOWLEDGED)
        order.apply_fill(fill_id="f1", qty_qc=1000, price_cc=4060)
        assert order.slippage_cc() == 60

    def test_filling_below_a_buy_limit_is_negative_slippage(self, order: ManagedOrder):
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.ACKNOWLEDGED)
        order.apply_fill(fill_id="f1", qty_qc=1000, price_cc=3900)
        assert order.slippage_cc() == -100

    def test_sell_slippage_is_measured_in_the_other_direction(self):
        sell = ManagedOrder(
            market_type=MarketType.PERP,
            ticker="P",
            requested_qty_qc=1000,
            limit_price_cc=5000,
            book_side=BookSide.ASK,
        )
        sell.transition(OrderState.SUBMITTED)
        sell.transition(OrderState.ACKNOWLEDGED)
        sell.apply_fill(fill_id="f1", qty_qc=1000, price_cc=4950)
        assert sell.slippage_cc() == 50

    def test_no_slippage_before_any_fill(self, order: ManagedOrder):
        assert order.slippage_cc() is None


class TestIdempotencyKey:
    def test_each_order_gets_a_distinct_client_order_id(self):
        ids = {
            ManagedOrder(
                market_type=MarketType.EVENT,
                ticker="M",
                requested_qty_qc=100,
                limit_price_cc=4000,
            ).client_order_id
            for _ in range(500)
        }
        assert len(ids) == 500

    def test_the_client_order_id_is_stable_across_the_order_lifetime(
        self, order: ManagedOrder
    ):
        """A retry or a reconciliation query reuses it; that is what makes an
        ambiguous submission recoverable rather than a coin flip."""
        original = order.client_order_id
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.UNKNOWN)
        order.transition(OrderState.ACKNOWLEDGED)
        assert order.client_order_id == original


class TestPersistenceShape:
    def test_to_row_covers_the_orders_table(self, order: ManagedOrder):
        from kat.db.models import Order as OrderRow

        order.transition(OrderState.SUBMITTED)
        row = order.to_row()
        columns = {c.name for c in OrderRow.__table__.columns}
        # Every key we emit must be a real column, or the write fails at runtime.
        assert set(row) <= columns
        # And the row must be constructible.
        assert OrderRow(**row) is not None
