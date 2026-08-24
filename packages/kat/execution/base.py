"""Execution engine base.

The signature that matters is :meth:`ExecutionEngine.submit`, which takes a
:class:`~kat.risk.types.RiskVerdict` rather than an order or an intent. There is
no overload that accepts anything else. A strategy cannot construct a
``RiskVerdict`` meaningfully -- only a risk engine returns one -- so the
architecture

    Strategy -> Portfolio -> Risk Engine -> Execution Engine -> Kalshi

is enforced by the type system rather than by convention. There is no "submit
directly" path for emergencies, for the dashboard, for models, or for an LLM;
every one of those routes through a risk engine like everything else.

Environment gating
------------------
``PAPER`` never touches an order endpoint at all: fills are simulated against
live market data. ``DEMO`` and ``LIVE`` send real requests to the respective
exchange. ``LIVE`` additionally requires both ``environment=live`` and
``live_trading_armed=true``, which are separate settings on purpose.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from kat.common.enums import MarketType, OrderState, TradingEnvironment
from kat.common.errors import RiskRejection
from kat.common.ids import use_correlation_id
from kat.common.timeutil import Clock
from kat.db.models import Fill as FillRow
from kat.db.models import Order as OrderRow
from kat.db.writer import AsyncWriter
from kat.execution.order import ManagedOrder
from kat.kalshi.http import AmbiguousSubmission
from kat.marketdata.book import OrderBook
from kat.observability.logging import get_logger
from kat.observability.metrics import (
    M_EXEC_LATENCY,
    M_ORDERS_FILLED,
    M_ORDERS_REJECTED,
    M_ORDERS_SUBMITTED,
    metrics,
)
from kat.risk.killswitch import KillSwitchManager
from kat.risk.types import RiskVerdict

log = get_logger(__name__)

__all__ = ["ExecutionEngine"]


class ExecutionEngine(ABC):
    """Common order lifecycle for one domain."""

    market_type: MarketType

    def __init__(
        self,
        *,
        clock: Clock,
        writer: AsyncWriter | None = None,
        kill_switches: KillSwitchManager | None = None,
        environment: TradingEnvironment = TradingEnvironment.PAPER,
        live_armed: bool = False,
    ) -> None:
        self.clock = clock
        self.writer = writer
        self.kill_switches = kill_switches
        self.environment = environment
        self.live_armed = live_armed
        # Working orders by our internal id. This is the in-memory book of
        # record on the trading path; the database is a durable copy of it.
        self.orders: dict[str, ManagedOrder] = {}
        self._by_client_id: dict[str, ManagedOrder] = {}
        self._lock = asyncio.Lock()

    # -- public API ----------------------------------------------------------

    async def submit(self, verdict: RiskVerdict) -> ManagedOrder:
        """Submit an order that a risk engine has authorised.

        Raises :class:`RiskRejection` if handed an unapproved verdict, which
        would indicate a caller bypassing the pipeline.
        """
        if not verdict.approved:
            raise RiskRejection(
                verdict.reason_code,
                f"execution engine refused an unapproved verdict: {verdict.reason}",
            )
        if verdict.approved_qty_qc <= 0:
            raise RiskRejection("ZERO_QUANTITY", "approved quantity is zero")

        intent = verdict.intent
        if intent.market_type is not self.market_type:
            raise RiskRejection(
                "WRONG_DOMAIN",
                f"{self.market_type.value} execution engine received a "
                f"{intent.market_type.value} order",
            )

        # Kill switches are checked here as well as upstream. The duplication is
        # deliberate: this is the last point before an order leaves the process,
        # and a switch engaged between risk approval and submission must still
        # stop it.
        if self.kill_switches is not None:
            self.kill_switches.assert_clear(self.market_type, strategy_id=intent.strategy_id)

        order = ManagedOrder(
            market_type=intent.market_type,
            ticker=intent.ticker,
            requested_qty_qc=verdict.approved_qty_qc,
            limit_price_cc=intent.limit_price_cc,
            outcome_side=intent.outcome_side,
            book_side=intent.book_side,
            post_only=intent.post_only,
            reduce_only=intent.reduce_only,
            strategy_id=intent.strategy_id,
            decision_id=intent.decision_id,
            correlation_id=intent.correlation_id,
            environment=self.environment.value,
        )
        async with self._lock:
            self.orders[order.order_id] = order
            self._by_client_id[order.client_order_id] = order

        with use_correlation_id(order.correlation_id or None):
            await self._dispatch(order, verdict)
        self._persist(order)
        return order

    async def cancel(self, order_id: str) -> bool:
        """Request cancellation. Safe to call more than once."""
        order = self.orders.get(order_id)
        if order is None:
            log.warning("execution.cancel_unknown_order", order_id=order_id)
            return False
        if order.state.is_terminal:
            return True

        order.transition(OrderState.CANCEL_REQUESTED, reason="cancel requested", strict=False)
        try:
            if self.environment is TradingEnvironment.PAPER:
                order.transition(OrderState.CANCELLED, reason="paper cancel", strict=False)
            else:
                await self._cancel_at_exchange(order)
                order.transition(OrderState.CANCELLED, reason="exchange cancel", strict=False)
            self._persist(order)
            return True
        except Exception as exc:
            # A failed cancel leaves the order working, which is the truthful
            # state. It is NOT marked cancelled on our side, because believing
            # we are flat when we are not is the dangerous error.
            log.error(
                "execution.cancel_failed",
                order_id=order_id, ticker=order.ticker, error=str(exc),
            )
            return False

    async def cancel_all(self, *, ticker: str | None = None) -> int:
        """Cancel every working order, optionally limited to one market."""
        targets = [
            o for o in list(self.orders.values())
            if o.is_working and (ticker is None or o.ticker == ticker)
        ]
        results = await asyncio.gather(
            *(self.cancel(o.order_id) for o in targets), return_exceptions=True
        )
        return sum(1 for r in results if r is True)

    def working_orders(self, *, ticker: str | None = None) -> list[ManagedOrder]:
        return [
            o for o in self.orders.values()
            if o.is_working and (ticker is None or o.ticker == ticker)
        ]

    def find_by_client_order_id(self, client_order_id: str) -> ManagedOrder | None:
        return self._by_client_id.get(client_order_id)

    # -- dispatch ------------------------------------------------------------

    async def _dispatch(self, order: ManagedOrder, verdict: RiskVerdict) -> None:
        if self.environment is TradingEnvironment.PAPER:
            await self._simulate(order, verdict)
            return

        if self.environment is TradingEnvironment.LIVE and not self.live_armed:
            # Belt and braces: settings validation should already have caught
            # this, but an order must never reach a live exchange because one
            # flag was missed.
            order.transition(
                OrderState.RISK_REJECTED,
                reason="live environment is not armed; refusing to submit",
                strict=False,
            )
            metrics.increment(M_ORDERS_REJECTED, domain=self.market_type.value, reason="not_armed")
            return

        start = self.clock.monotonic_ns()
        order.transition(OrderState.SUBMITTED, reason="sent to exchange")
        metrics.increment(M_ORDERS_SUBMITTED, domain=self.market_type.value)
        try:
            await self._submit_to_exchange(order)
            order.submit_latency_ms = (self.clock.monotonic_ns() - start) / 1_000_000
            metrics.observe(M_EXEC_LATENCY, order.submit_latency_ms, domain=self.market_type.value)
        except AmbiguousSubmission as exc:
            # The order may or may not exist at the exchange. Do NOT retry and
            # do NOT assume failure: mark it unknown so it keeps consuming
            # exposure, and let reconciliation determine the truth.
            order.transition(
                OrderState.UNKNOWN,
                reason=f"submission outcome unknown: {exc}",
                strict=False,
            )
            log.error(
                "execution.ambiguous_submission",
                order_id=order.order_id, client_order_id=order.client_order_id,
                ticker=order.ticker, error=str(exc),
            )
            metrics.increment(M_ORDERS_REJECTED, domain=self.market_type.value, reason="ambiguous")
        except Exception as exc:
            order.transition(
                OrderState.REJECTED, reason=f"{type(exc).__name__}: {exc}", strict=False
            )
            metrics.increment(M_ORDERS_REJECTED, domain=self.market_type.value, reason="error")
            log.error(
                "execution.submit_failed",
                order_id=order.order_id, ticker=order.ticker, error=str(exc),
            )

    async def _simulate(self, order: ManagedOrder, verdict: RiskVerdict) -> None:
        """Paper fill against the live book.

        Charges the real cost of crossing the spread rather than assuming a fill
        at the touch, so paper results are comparable to live ones. A limit that
        is not marketable rests rather than filling, exactly as it would live.
        """
        order.transition(OrderState.SUBMITTED, reason="paper submission")
        book = self._current_book(order.ticker)
        if book is None:
            order.transition(
                OrderState.ACKNOWLEDGED, reason="paper: no book available; resting", strict=False
            )
            return

        is_buy = order.book_side is not None and order.book_side.value == "bid"
        if order.book_side is None:
            is_buy = order.outcome_side is not None and order.outcome_side.value == "yes"

        touch = book.best_ask_cc if is_buy else book.best_bid_cc
        marketable = (
            touch is not None
            and order.limit_price_cc is not None
            and (order.limit_price_cc >= touch if is_buy else order.limit_price_cc <= touch)
        )
        if not marketable or order.post_only:
            order.transition(
                OrderState.ACKNOWLEDGED, reason="paper: resting (not marketable)", strict=False
            )
            return

        walk = book.walk(order.requested_qty_qc, is_buy=is_buy)
        if walk is None:
            order.transition(OrderState.ACKNOWLEDGED, reason="paper: empty book", strict=False)
            return
        filled_qc, notional, _worst = walk
        if filled_qc <= 0:
            order.transition(OrderState.ACKNOWLEDGED, reason="paper: no depth", strict=False)
            return
        avg_price = (
            -(-notional // filled_qc) if is_buy else notional // filled_qc
        )
        order.apply_fill(
            fill_id=f"paper-{order.order_id}",
            qty_qc=filled_qc,
            price_cc=avg_price,
            fee_micros=verdict.estimated_fees_micros,
        )
        metrics.increment(M_ORDERS_FILLED, domain=self.market_type.value, mode="paper")
        self._persist_fill(
            order,
            fill_id=f"paper-{order.order_id}",
            qty_qc=filled_qc,
            price_cc=avg_price,
            fee_micros=verdict.estimated_fees_micros,
        )

    # -- persistence ---------------------------------------------------------

    def _persist(self, order: ManagedOrder) -> None:
        if self.writer is not None:
            self.writer.enqueue(OrderRow(**order.to_row()), critical=True)

    def _persist_fill(
        self, order: ManagedOrder, *, fill_id: str, qty_qc: int, price_cc: int, fee_micros: int
    ) -> None:
        if self.writer is None:
            return
        self.writer.enqueue(
            FillRow(
                fill_id=fill_id,
                exchange_order_id=order.exchange_order_id,
                order_id=order.order_id,
                market_type=order.market_type.value,
                ticker=order.ticker,
                strategy_id=order.strategy_id,
                ts=self.clock.now(),
                book_side=order.book_side.value if order.book_side else None,
                outcome_side=order.outcome_side.value if order.outcome_side else None,
                price_cc=price_cc,
                count_qc=qty_qc,
                fee_micros=fee_micros,
                environment=self.environment.value,
            ),
            critical=True,
        )

    # -- subclass responsibilities -------------------------------------------

    @abstractmethod
    async def _submit_to_exchange(self, order: ManagedOrder) -> None:
        """Send the order and record the exchange's response on ``order``."""

    @abstractmethod
    async def _cancel_at_exchange(self, order: ManagedOrder) -> None:
        """Cancel at the exchange, raising on failure."""

    @abstractmethod
    def _current_book(self, ticker: str) -> OrderBook | None:
        """Latest known book for paper simulation, or ``None``."""

    @abstractmethod
    async def resolve_unknown(self, order: ManagedOrder) -> OrderState:
        """Determine the true state of an order whose outcome was never seen.

        Implementations query the exchange by ``client_order_id`` -- never by
        resubmitting.
        """

    def stats(self) -> dict[str, Any]:
        working = self.working_orders()
        return {
            "domain": self.market_type.value,
            "environment": self.environment.value,
            "live_armed": self.live_armed,
            "tracked_orders": len(self.orders),
            "working_orders": len(working),
            "unknown_orders": sum(
                1 for o in self.orders.values() if o.state is OrderState.UNKNOWN
            ),
        }
