"""Event-contract execution engine."""

from __future__ import annotations

from kat.common.enums import MarketType, OrderAction, OrderState, OutcomeSide
from kat.common.timeutil import Clock
from kat.execution.base import ExecutionEngine
from kat.execution.order import ManagedOrder
from kat.kalshi.event_contracts.rest import EventContractsREST
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketState
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["EventExecutionEngine"]


class EventExecutionEngine(ExecutionEngine):
    """Sends event-contract orders and tracks their lifecycle."""

    market_type = MarketType.EVENT

    def __init__(
        self,
        *,
        rest: EventContractsREST | None,
        market_state: MarketState,
        clock: Clock,
        **kwargs: object,
    ) -> None:
        super().__init__(clock=clock, **kwargs)  # type: ignore[arg-type]
        self.rest = rest
        self.market_state = market_state

    # -- exchange interaction ------------------------------------------------

    async def _submit_to_exchange(self, order: ManagedOrder) -> None:
        if self.rest is None:
            raise RuntimeError("event-contract REST client is not configured")

        outcome = order.outcome_side or OutcomeSide.YES
        # Kalshi expresses direction as (outcome_side, action). This platform
        # only ever *buys* an outcome: to exit a YES position it buys NO rather
        # than selling YES. Both are economically identical on Kalshi, and
        # buying is the form that never depends on holding inventory the
        # exchange thinks we do not have.
        action = OrderAction.BUY

        response = await self.rest.create_order(
            ticker=order.ticker,
            client_order_id=order.client_order_id,
            outcome_side=outcome,
            action=action,
            count_qc=order.requested_qty_qc,
            price_cc=order.limit_price_cc,
            time_in_force=order.time_in_force,
            post_only=order.post_only,
            reduce_only=order.reduce_only,
        )

        order.exchange_order_id = response.order_id
        order.raw_response = response.raw
        self._apply_exchange_state(order, response)

    def _apply_exchange_state(self, order: ManagedOrder, response: object) -> None:
        """Move our order to whatever the exchange says it is now."""
        state = response.to_internal_state()  # type: ignore[attr-defined]
        filled = getattr(response, "filled_qc", 0)

        if filled and filled > order.filled_qty_qc:
            price = (
                getattr(response, "yes_price_cc", None)
                if order.outcome_side is not OutcomeSide.NO
                else getattr(response, "no_price_cc", None)
            )
            order.apply_fill(
                fill_id=f"{order.exchange_order_id}:ack",
                qty_qc=filled - order.filled_qty_qc,
                price_cc=price or order.limit_price_cc or 0,
            )
            return

        if state in (OrderState.ACKNOWLEDGED, OrderState.FILLED, OrderState.CANCELLED,
                     OrderState.REJECTED, OrderState.EXPIRED):
            order.transition(state, reason="exchange response", strict=False)
        else:
            order.transition(OrderState.ACKNOWLEDGED, reason="exchange accepted", strict=False)

    async def _cancel_at_exchange(self, order: ManagedOrder) -> None:
        if self.rest is None:
            raise RuntimeError("event-contract REST client is not configured")
        if not order.exchange_order_id:
            # Never acknowledged, so there is nothing at the exchange to cancel
            # under an id we hold. Resolve what actually happened instead of
            # assuming it is gone.
            await self.resolve_unknown(order)
            return
        await self.rest.cancel_order(order.exchange_order_id)

    def _current_book(self, ticker: str) -> OrderBook | None:
        return self.market_state.book(ticker)

    async def resolve_unknown(self, order: ManagedOrder) -> OrderState:
        """Find out what really happened to an order we lost track of.

        Queries by ``client_order_id`` -- the idempotency key assigned before the
        submission -- and never resubmits. If the order exists at the exchange we
        adopt its state; if it demonstrably does not, only then is it rejected.
        """
        if self.rest is None:
            return order.state

        try:
            orders = await self.rest.get_orders(ticker=order.ticker)
        except Exception as exc:
            log.error(
                "execution.resolve_unknown_failed",
                order_id=order.order_id, client_order_id=order.client_order_id, error=str(exc),
            )
            return order.state  # stays UNKNOWN and keeps consuming exposure

        match = next((o for o in orders if o.client_order_id == order.client_order_id), None)
        if match is None:
            # Absent from an authoritative listing means it never rested.
            order.transition(
                OrderState.REJECTED,
                reason="not present at exchange on reconciliation; submission did not take effect",
                strict=False,
            )
            log.warning(
                "execution.unknown_resolved_absent",
                order_id=order.order_id, client_order_id=order.client_order_id,
            )
            # Persisted, because nothing else on this path did: the order's row
            # kept whatever state was last written -- UNKNOWN, typically -- after
            # the exchange had told us its real fate, so the audit trail and memory
            # disagreed from then on.
            self._persist(order)
            return order.state

        order.exchange_order_id = match.order_id
        if match.filled_qc > order.filled_qty_qc:
            order.apply_fill(
                fill_id=f"{match.order_id}:reconciled",
                qty_qc=match.filled_qc - order.filled_qty_qc,
                price_cc=(
                    match.no_price_cc if order.outcome_side is OutcomeSide.NO else match.yes_price_cc
                ) or order.limit_price_cc or 0,
            )
        else:
            order.transition(match.to_internal_state(), reason="reconciled", strict=False)

        log.info(
            "execution.unknown_resolved",
            order_id=order.order_id, exchange_order_id=match.order_id, state=order.state.value,
        )
        self._persist(order)
        return order.state
