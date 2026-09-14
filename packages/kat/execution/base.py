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

from kat.common.enums import MarketType, OrderState, OutcomeSide, TradingEnvironment
from kat.common.errors import RiskRejection
from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.common.ids import use_correlation_id
from kat.common.timeutil import Clock
from kat.db.models import Fill as FillRow
from kat.db.models import Order as OrderRow
from kat.db.writer import AsyncWriter
from kat.execution.order import ManagedOrder
from kat.kalshi.fees import FeeSchedule, realised_fee_micros
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

#: The window ``orders_in_last_minute`` counts over. Named because pruning must
#: never shorten it: an order forgotten inside this window is an order the
#: runaway-order limit can no longer see, which quietly loosens a risk check.
_RATE_WINDOW_MS = 60_000

#: How long a finished order stays in memory after it finishes.
#:
#: Every order used to stay forever, and the scans over this dict run every
#: trading cycle. On 2026-09-13 it held 103,455 orders, 103,423 of them
#: cancelled, and the cycle had grown from 23ms to 3.6 seconds over twenty
#: hours -- starving the database writer and flapping trading in and out of
#: DEGRADED 189 times. The same growth was behind the 1.7-second cycles of
#: 2026-09-11. A finished order is already in the database; memory only needs
#: it for the rate window, and for an exchange message that arrives after we
#: believed the order was done. Ten minutes covers both with room to spare.
_TERMINAL_RETENTION_MS = 10 * 60_000


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
        fee_schedules: dict[str, FeeSchedule] | None = None,
    ) -> None:
        self.clock = clock
        self.writer = writer
        self.kill_switches = kill_switches
        self.environment = environment
        self.live_armed = live_armed
        #: Per-market fee terms, keyed by ticker. Shared by reference with the
        #: trading loop that sizes orders against this engine, so the fee used
        #: to judge a trade and the fee charged for it cannot come from
        #: different schedules.
        self.fee_schedules: dict[str, FeeSchedule] = (
            fee_schedules if fee_schedules is not None else {}
        )
        # Working orders by our internal id. This is the in-memory book of
        # record on the trading path; the database is a durable copy of it.
        self.orders: dict[str, ManagedOrder] = {}
        self._by_client_id: dict[str, ManagedOrder] = {}
        self._lock = asyncio.Lock()
        #: Finished orders forgotten so far, so the stats say memory is bounded
        #: rather than leaving it to be inferred from a small number.
        self._pruned = 0

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

    async def expire_stale_orders(self, *, now_ms: int, ttl_ms: int) -> int:
        """Cancel working orders that have rested longer than ``ttl_ms``.

        These are good-till-cancelled limit orders priced off a book that has
        since moved, and nothing was ever cancelling them. Two things follow from
        leaving them, and both are bad.

        The first is that a resting order holds its market's position headroom
        for as long as it rests, because pending exposure counts against the
        concentration cap exactly as a holding does. Nothing released it, so
        every market the platform touched became permanently unavailable to it --
        observed as ``POSITION_LIMIT`` on markets holding no position at all,
        while the account still had five sixths of its cash.

        The second is worse than idleness. A stale limit does not simply fail to
        fill: it fills precisely when the market crosses it, which is when the
        price has moved against the opinion that placed it. Leaving it resting
        turns a cancelled idea into a standing offer to be picked off.

        A strategy re-prices every cycle, so an order that has not filled within
        the TTL has been superseded by its own author.
        """
        cutoff = now_ms - ttl_ms
        stale = [
            order
            for order in list(self.orders.values())
            if order.is_working
            and order.submitted_ts is not None
            and order.submitted_ts.timestamp() * 1000.0 < cutoff
        ]
        if not stale:
            return 0
        results = await asyncio.gather(
            *(self.cancel(o.order_id) for o in stale), return_exceptions=True
        )
        cancelled = sum(1 for r in results if r is True)
        if cancelled:
            log.info(
                "execution.stale_orders_expired",
                domain=self.market_type.value,
                cancelled=cancelled,
                ttl_ms=ttl_ms,
            )
        return cancelled

    def working_orders(self, *, ticker: str | None = None) -> list[ManagedOrder]:
        return [
            o for o in self.orders.values()
            if o.is_working and (ticker is None or o.ticker == ticker)
        ]

    def prune_terminal_orders(self, *, now_ms: int) -> int:
        """Forget orders that finished longer ago than the retention period.

        Only terminal orders are candidates. Anything that can still consume
        exposure stays however old it is -- including ``UNKNOWN``, which
        reconciliation has yet to resolve, and ``CANCEL_REQUESTED``, which the
        exchange may still fill.

        Two guards keep this from loosening anything. Retention is never shorter
        than the rate window, and an order submitted inside the window is kept
        even if its timestamps say otherwise, so every order
        ``orders_in_last_minute`` could count is still here to be counted. And an
        order marked terminal without a ``terminal_ts`` is kept: when it finished
        is unknown, and an unknown age is not an old one.
        """
        horizon = now_ms - max(_TERMINAL_RETENTION_MS, _RATE_WINDOW_MS)
        rate_floor = now_ms - _RATE_WINDOW_MS
        finished: list[str] = []
        for order_id, order in self.orders.items():
            if not order.state.is_terminal or order.terminal_ts is None:
                continue
            if order.terminal_ts.timestamp() * 1000.0 >= horizon:
                continue
            submitted = order.submitted_ts
            if submitted is not None and submitted.timestamp() * 1000.0 >= rate_floor:
                continue
            finished.append(order_id)
        for order_id in finished:
            order = self.orders.pop(order_id)
            if self._by_client_id.get(order.client_order_id) is order:
                del self._by_client_id[order.client_order_id]
        self._pruned += len(finished)
        return len(finished)

    def find_by_client_order_id(self, client_order_id: str) -> ManagedOrder | None:
        return self._by_client_id.get(client_order_id)

    def orders_in_last_minute(self, *, now_ms: int) -> int:
        """How many orders this engine has submitted in the last sixty seconds.

        The event risk engine compares this against ``max_orders_per_minute`` and
        calls the rejection "runaway-order protection engaged". Nothing outside
        the backtester ever supplied the number, so the field sat at its default
        of zero and the comparison was ``0 < 30`` on every evaluation: the limit
        could not fire, in paper or in live.

        It was not idle protection. A strategy re-proposing the same trade every
        cycle submitted 55 good-till-cancelled orders in 55 seconds against one
        market -- roughly double the configured limit -- and nothing objected.
        """
        cutoff = now_ms - _RATE_WINDOW_MS
        count = 0
        for order in self.orders.values():
            submitted = order.submitted_ts
            if submitted is None:
                continue
            if submitted.timestamp() * 1000.0 >= cutoff:
                count += 1
        return count

    def working_exposure_micros(self) -> dict[str, int]:
        """Capital committed to still-working orders, keyed by ticker.

        ``qty_qc * price_cc`` is already micros -- 1e-2 contracts times 1e-4 USD
        -- so no conversion belongs here. The platform only ever buys an outcome,
        so the limit price is the cost per contract and also the most that
        contract can lose, which is what makes this a sound exposure figure.
        """
        committed: dict[str, int] = {}
        for order in self.orders.values():
            remaining = order.exposure_qty_qc
            price = order.limit_price_cc
            if remaining <= 0 or not price:
                continue
            committed[order.ticker] = committed.get(order.ticker, 0) + remaining * price
        return committed

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

        # The book is YES-denominated, and so is everything downstream of this
        # walk: the fill price it reports and the account's apply_fill both speak
        # YES. The limit has to as well. A NO buy at p is a YES sell at 1 - p, the
        # conversion the backtester's yes_denominated_order has always made.
        #
        # Comparing the NO limit itself against the YES bid got both halves of the
        # board wrong. Below fifty cents it refused marketable orders -- a NO buy
        # at the NO ask with three thousand contracts of depth rested unfilled,
        # which is why model_drift's exits never filled in paper and stacked. Above
        # fifty cents it filled orders that were not marketable at all, at a price
        # past their own limit.
        limit_on_book = order.limit_price_cc
        if (
            order.book_side is None
            and order.outcome_side is OutcomeSide.NO
            and limit_on_book is not None
        ):
            limit_on_book = CC_PER_DOLLAR - limit_on_book

        touch = book.best_ask_cc if is_buy else book.best_bid_cc
        marketable = (
            touch is not None
            and limit_on_book is not None
            and (limit_on_book >= touch if is_buy else limit_on_book <= touch)
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
        # Recorded in the order's own outcome's terms, as the live path records it
        # from the exchange's no_price. The walk ran on the YES book, so a NO buy's
        # cost is what is left of the dollar -- and with YES rounded down above,
        # that rounds the NO cost up, against us, as a YES buy's rounding is.
        # Recording the YES side instead put a NO order's fill and its limit on
        # different scales, and its slippage read -4,100cc at a fill exactly on
        # the limit.
        fill_price_cc = (
            CC_PER_DOLLAR - avg_price
            if order.book_side is None and order.outcome_side is OutcomeSide.NO
            else avg_price
        )
        # Charged on what actually filled, not on ``verdict.estimated_fees_micros``.
        # That figure is an *estimate for a decision*, priced for the quantity the
        # strategy asked for; the risk engine then resizes, routinely by an order of
        # magnitude. Billing the pre-resize estimate against a post-resize fill
        # charged one 4.32-contract fill the fee for 29 contracts, and over one
        # session paper paid $101.17 in fees on $2,092.01 of notional -- 4.84%,
        # against the 1.74% the same book costs once the fee follows the fill. On
        # edges measured in single cents that difference is most of the edge. A
        # realised fee is computed from the fill, which is also what the live path
        # does: there it comes from the exchange's fill payload, never from this
        # model.
        # realised_ rather than estimate_: an estimate for an unknown schedule is
        # padded by 50% so that an unknown fee cannot make a marginal trade look
        # profitable. That padding belongs to the decision. Charging it would
        # claim the exchange took more than its published schedule says, and
        # would make paper results worse than the reality they exist to predict.
        fee_micros = realised_fee_micros(
            price_cc=avg_price,
            qty_qc=filled_qc,
            schedule=self.fee_schedules.get(order.ticker),
            is_taker=True,  # a resting order returned above; this one crossed
        )
        order.apply_fill(
            fill_id=f"paper-{order.order_id}",
            qty_qc=filled_qc,
            price_cc=fill_price_cc,
            fee_micros=fee_micros,
        )
        metrics.increment(M_ORDERS_FILLED, domain=self.market_type.value, mode="paper")
        self._persist_fill(
            order,
            fill_id=f"paper-{order.order_id}",
            qty_qc=filled_qc,
            price_cc=fill_price_cc,
            fee_micros=fee_micros,
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
            "pruned_orders": self._pruned,
            "working_orders": len(working),
            "unknown_orders": sum(
                1 for o in self.orders.values() if o.state is OrderState.UNKNOWN
            ),
        }
