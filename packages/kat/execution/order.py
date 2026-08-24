"""Internal order state machine.

The premise of this module is that **an HTTP 200 is not a fill and a timeout is
not a rejection**. Our order state is ours; the exchange's is the exchange's;
reconciliation is what reconciles them. Conflating the two is how a trading
system ends up with a position it does not know about.

Legal transitions::

    CREATED ──────► RISK_REJECTED          (our own risk engine refused it)
    CREATED ──────► SUBMITTED              (request left the process)
    SUBMITTED ────► ACKNOWLEDGED           (exchange confirmed it rests)
    SUBMITTED ────► REJECTED               (exchange refused it)
    SUBMITTED ────► FILLED / PARTIALLY_FILLED   (immediate execution)
    SUBMITTED ────► UNKNOWN                (ambiguous: outcome not observed)
    ACKNOWLEDGED ─► PARTIALLY_FILLED / FILLED / CANCEL_REQUESTED / EXPIRED
    PARTIALLY_FILLED ► FILLED / CANCEL_REQUESTED / CANCELLED / EXPIRED
    CANCEL_REQUESTED ► CANCELLED / FILLED  (a cancel can lose the race)
    UNKNOWN ──────► any state              (only reconciliation resolves it)

``UNKNOWN`` is the important one. It is entered when a submission's outcome was
never observed, and it counts fully against exposure until reconciliation proves
otherwise -- because an order that might be resting is an order that might fill.

``CANCEL_REQUESTED ──► FILLED`` is deliberately legal: a cancel races the
matching engine and can lose. A state machine that forbade it would reject the
truth and leave us believing we were flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kat.common.enums import (
    BookSide,
    MarketType,
    OrderState,
    OrderType,
    OutcomeSide,
    TimeInForce,
)
from kat.common.ids import new_client_order_id, new_id
from kat.common.timeutil import utcnow_aware
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["ManagedOrder", "OrderTransitionError", "is_legal_transition"]


class OrderTransitionError(RuntimeError):
    """An illegal state transition was attempted. Always a bug, never data."""


# Terminal states accept no further transitions.
_LEGAL: dict[OrderState, frozenset[OrderState]] = {
    OrderState.CREATED: frozenset(
        {OrderState.RISK_REJECTED, OrderState.SUBMITTED, OrderState.REJECTED}
    ),
    OrderState.SUBMITTED: frozenset(
        {
            OrderState.ACKNOWLEDGED,
            OrderState.REJECTED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.ACKNOWLEDGED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_REQUESTED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_REQUESTED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.CANCEL_REQUESTED: frozenset(
        {
            OrderState.CANCELLED,
            # A cancel races the matching engine and can lose.
            OrderState.FILLED,
            OrderState.PARTIALLY_FILLED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
    # Reconciliation is the only way out of UNKNOWN, and it may land anywhere.
    OrderState.UNKNOWN: frozenset(
        {
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.FILLED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.RISK_REJECTED: frozenset(),
    OrderState.EXPIRED: frozenset(),
}


def is_legal_transition(current: OrderState, target: OrderState) -> bool:
    return target in _LEGAL.get(current, frozenset())


@dataclass(slots=True)
class ManagedOrder:
    """One order and everything we know about it.

    ``client_order_id`` is generated once at construction and is the idempotency
    key. It is never regenerated -- a retry or a reconciliation query reuses it,
    which is precisely what makes an ambiguous submission recoverable.
    """

    market_type: MarketType
    ticker: str
    requested_qty_qc: int
    limit_price_cc: int | None

    order_id: str = field(default_factory=lambda: new_id("ord"))
    client_order_id: str = field(default_factory=new_client_order_id)
    exchange_order_id: str | None = None

    # Direction: event contracts use outcome_side, perpetuals use book_side.
    outcome_side: OutcomeSide | None = None
    book_side: BookSide | None = None

    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.GOOD_TILL_CANCELED
    post_only: bool = False
    reduce_only: bool = False

    state: OrderState = OrderState.CREATED
    filled_qty_qc: int = 0
    avg_fill_price_cc: int | None = None
    fees_micros: int = 0

    strategy_id: str | None = None
    decision_id: str | None = None
    correlation_id: str | None = None

    created_ts: datetime = field(default_factory=utcnow_aware)
    submitted_ts: datetime | None = None
    acknowledged_ts: datetime | None = None
    terminal_ts: datetime | None = None
    submit_latency_ms: float | None = None

    reject_reason: str | None = None
    environment: str = "paper"
    state_history: list[dict[str, Any]] = field(default_factory=list)
    fill_ids: set[str] = field(default_factory=set)
    raw_response: dict[str, Any] | None = None

    # -- state -----------------------------------------------------------

    def transition(
        self, target: OrderState, *, reason: str | None = None, strict: bool = True
    ) -> None:
        """Move to ``target``, recording the transition.

        A no-op transition to the same state is allowed only where the state
        machine lists it (repeated partial fills), so a duplicate websocket
        message does not become an error.
        """
        if target is self.state and target is not OrderState.PARTIALLY_FILLED:
            return
        if not is_legal_transition(self.state, target):
            message = (
                f"illegal order transition {self.state.value} -> {target.value} "
                f"for {self.order_id} ({self.ticker})"
            )
            if strict:
                raise OrderTransitionError(message)
            log.error("order.illegal_transition", order_id=self.order_id, message=message)
            return

        previous = self.state
        self.state = target
        now = utcnow_aware()
        self.state_history.append(
            {
                "from": previous.value,
                "to": target.value,
                "ts": now.isoformat(),
                "reason": reason,
                "filled_qty_qc": self.filled_qty_qc,
            }
        )
        if target is OrderState.SUBMITTED:
            self.submitted_ts = now
        elif target is OrderState.ACKNOWLEDGED and self.acknowledged_ts is None:
            self.acknowledged_ts = now
        if target.is_terminal:
            self.terminal_ts = now
        if reason and target in (OrderState.REJECTED, OrderState.RISK_REJECTED):
            self.reject_reason = reason

        log.info(
            "order.transition",
            order_id=self.order_id, ticker=self.ticker, market_type=self.market_type.value,
            **{"from": previous.value}, to=target.value, reason=reason,
            strategy_id=self.strategy_id, correlation_id=self.correlation_id,
        )

    def apply_fill(self, *, fill_id: str, qty_qc: int, price_cc: int, fee_micros: int = 0) -> bool:
        """Record a fill. Idempotent on ``fill_id``.

        Returns ``False`` when this fill was already applied. Duplicate fill
        messages are normal -- a websocket replay after reconnect will resend
        them -- and double-counting one would corrupt both the position and the
        average price.
        """
        if fill_id in self.fill_ids:
            log.debug("order.duplicate_fill_ignored", order_id=self.order_id, fill_id=fill_id)
            return False
        self.fill_ids.add(fill_id)

        previous_qty = self.filled_qty_qc
        previous_notional = (self.avg_fill_price_cc or 0) * previous_qty
        self.filled_qty_qc = previous_qty + qty_qc
        if self.filled_qty_qc > 0:
            self.avg_fill_price_cc = (previous_notional + price_cc * qty_qc) // self.filled_qty_qc
        self.fees_micros += fee_micros

        if self.filled_qty_qc >= self.requested_qty_qc:
            self.transition(OrderState.FILLED, reason="fully filled", strict=False)
        else:
            self.transition(OrderState.PARTIALLY_FILLED, reason="partial fill", strict=False)
        return True

    # -- derived ---------------------------------------------------------

    @property
    def remaining_qty_qc(self) -> int:
        return max(0, self.requested_qty_qc - self.filled_qty_qc)

    @property
    def is_working(self) -> bool:
        """True while this order could still consume exposure at the exchange."""
        return self.state.is_working

    @property
    def exposure_qty_qc(self) -> int:
        """Quantity that could still be added to a position by this order."""
        return self.remaining_qty_qc if self.is_working else 0

    @property
    def signed_remaining_qc(self) -> int:
        """Remaining quantity, signed by direction. Used for perp exposure."""
        remaining = self.exposure_qty_qc
        if self.book_side is not None:
            return remaining if self.book_side is BookSide.BID else -remaining
        return remaining if self.outcome_side is OutcomeSide.YES else -remaining

    def slippage_cc(self) -> int | None:
        """Difference between the average fill and the limit we asked for."""
        if self.avg_fill_price_cc is None or self.limit_price_cc is None:
            return None
        # Paying more than the limit on a buy, or receiving less on a sell, is
        # positive slippage.
        buying = (
            self.book_side is BookSide.BID
            if self.book_side is not None
            else self.outcome_side is OutcomeSide.YES
        )
        if buying:
            return self.avg_fill_price_cc - self.limit_price_cc
        return self.limit_price_cc - self.avg_fill_price_cc

    def to_row(self) -> dict[str, Any]:
        """Fields for the ``orders`` table."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "decision_id": self.decision_id,
            "correlation_id": self.correlation_id,
            "market_type": self.market_type.value,
            "ticker": self.ticker,
            "strategy_id": self.strategy_id,
            "state": self.state.value,
            "order_type": self.order_type.value,
            "book_side": (
                self.book_side.value
                if self.book_side
                else (self.outcome_side.to_book_side().value if self.outcome_side else "bid")
            ),
            "outcome_side": self.outcome_side.value if self.outcome_side else None,
            "time_in_force": self.time_in_force.value,
            "reduce_only": self.reduce_only,
            "post_only": self.post_only,
            "limit_price_cc": self.limit_price_cc,
            "requested_qty_qc": self.requested_qty_qc,
            "filled_qty_qc": self.filled_qty_qc,
            "remaining_qty_qc": self.remaining_qty_qc,
            "avg_fill_price_cc": self.avg_fill_price_cc,
            "fees_micros": self.fees_micros,
            "submitted_ts": self.submitted_ts,
            "acknowledged_ts": self.acknowledged_ts,
            "terminal_ts": self.terminal_ts,
            "submit_latency_ms": self.submit_latency_ms,
            "slippage_cc": self.slippage_cc(),
            "reject_reason": self.reject_reason,
            "environment": self.environment,
            "state_history": self.state_history,
            "raw_response": self.raw_response,
        }
