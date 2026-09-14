"""Perpetual execution engine."""

from __future__ import annotations

from kat.common.enums import BookSide, MarketType, OrderState
from kat.common.timeutil import Clock
from kat.execution.base import ExecutionEngine
from kat.execution.order import ManagedOrder
from kat.kalshi.perps.rest import PerpsREST
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketState
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["PerpExecutionEngine"]


class PerpExecutionEngine(ExecutionEngine):
    """Sends perpetual orders and tracks their lifecycle.

    Separate from the event engine because the wire format, the order semantics
    and the failure modes all differ: margin orders are two-sided (bid/ask
    rather than yes/no), priced in decimal dollars, and carry margin-specific
    flags such as ``cancel_order_on_pause`` and a subaccount.
    """

    market_type = MarketType.PERP

    def __init__(
        self,
        *,
        rest: PerpsREST | None,
        market_state: MarketState,
        clock: Clock,
        subaccount: int = 0,
        **kwargs: object,
    ) -> None:
        super().__init__(clock=clock, **kwargs)  # type: ignore[arg-type]
        self.rest = rest
        self.market_state = market_state
        self.subaccount = subaccount

    async def _submit_to_exchange(self, order: ManagedOrder) -> None:
        if self.rest is None:
            raise RuntimeError("perpetual REST client is not configured")
        if order.limit_price_cc is None:
            raise ValueError("perpetual orders require a limit price")

        response = await self.rest.create_order(
            ticker=order.ticker,
            client_order_id=order.client_order_id,
            side=order.book_side or BookSide.BID,
            count_qc=order.requested_qty_qc,
            price_cc=order.limit_price_cc,
            time_in_force=order.time_in_force,
            post_only=order.post_only,
            reduce_only=order.reduce_only,
            subaccount=self.subaccount,
            # A resting perpetual order that survives an exchange pause would be
            # measured against a stale mark when trading resumes, so let the
            # exchange drop it.
            cancel_order_on_pause=True,
        )

        order.exchange_order_id = response.order_id
        order.raw_response = response.raw

        if response.fill_qc > order.filled_qty_qc:
            order.apply_fill(
                fill_id=f"{response.order_id}:ack",
                qty_qc=response.fill_qc - order.filled_qty_qc,
                price_cc=response.price_cc or order.limit_price_cc,
            )
        else:
            order.transition(
                response.to_internal_state(), reason="exchange response", strict=False
            )

    async def _cancel_at_exchange(self, order: ManagedOrder) -> None:
        if self.rest is None:
            raise RuntimeError("perpetual REST client is not configured")
        if not order.exchange_order_id:
            await self.resolve_unknown(order)
            return
        await self.rest.cancel_order(order.exchange_order_id)

    def _current_book(self, ticker: str) -> OrderBook | None:
        return self.market_state.book(ticker)

    async def resolve_unknown(self, order: ManagedOrder) -> OrderState:
        """Resolve an order whose submission outcome was never observed.

        Same contract as the event engine: query by ``client_order_id``, never
        resubmit. On a perpetual this matters more, because a phantom position
        would also distort the leverage denominator until reconciliation caught
        it -- which is why an unresolved order stays ``UNKNOWN`` and keeps
        consuming exposure rather than being optimistically discarded.
        """
        if self.rest is None:
            return order.state

        try:
            orders = await self.rest.get_orders(ticker=order.ticker)
        except Exception as exc:
            log.error(
                "execution.resolve_unknown_failed",
                domain="perp", order_id=order.order_id,
                client_order_id=order.client_order_id, error=str(exc),
            )
            return order.state

        match = next((o for o in orders if o.client_order_id == order.client_order_id), None)
        if match is None:
            order.transition(
                OrderState.REJECTED,
                reason="not present at exchange on reconciliation; submission did not take effect",
                strict=False,
            )
            log.warning(
                "execution.unknown_resolved_absent",
                domain="perp", order_id=order.order_id,
                client_order_id=order.client_order_id,
            )
            # Persisted, because nothing else on this path did: the order's row
            # kept whatever state was last written -- UNKNOWN, typically -- after
            # the exchange had told us its real fate, so the audit trail and memory
            # disagreed from then on.
            self._persist(order)
            return order.state

        order.exchange_order_id = match.order_id
        if match.fill_qc > order.filled_qty_qc:
            order.apply_fill(
                fill_id=f"{match.order_id}:reconciled",
                qty_qc=match.fill_qc - order.filled_qty_qc,
                price_cc=match.price_cc or order.limit_price_cc or 0,
            )
        else:
            order.transition(match.to_internal_state(), reason="reconciled", strict=False)

        log.info(
            "execution.unknown_resolved",
            domain="perp", order_id=order.order_id,
            exchange_order_id=match.order_id, state=order.state.value,
        )
        self._persist(order)
        return order.state
