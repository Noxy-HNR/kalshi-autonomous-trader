"""Enumerations shared across domains.

Event contracts and perpetuals are deliberately *not* merged into one generic
instrument model. What they do share is a small vocabulary -- order sides, order
states, lifecycle states -- and that vocabulary lives here so both domains speak
the same language to the portfolio, audit and dashboard layers.
"""

from __future__ import annotations

from enum import Enum


class MarketType(str, Enum):
    """Which Kalshi trading domain an object belongs to.

    This is the top-level routing key: it selects the strategy set, the risk
    engine, the execution engine and the portfolio book that will handle an
    object. There is no "both" value by design.
    """

    EVENT = "event"
    PERP = "perp"


class BookSide(str, Enum):
    """Side of the Kalshi order book.

    Kalshi is migrating to ``book_side``/``outcome_side`` as the canonical
    direction fields; ``bid`` is equivalent to outcome side ``yes`` and ``ask``
    to outcome side ``no``. We store the book vocabulary because perpetuals only
    ever speak it.
    """

    BID = "bid"
    ASK = "ask"


class OutcomeSide(str, Enum):
    """Event-contract outcome side. ``buy-yes`` and ``sell-no`` both produce YES."""

    YES = "yes"
    NO = "no"

    def to_book_side(self) -> BookSide:
        return BookSide.BID if self is OutcomeSide.YES else BookSide.ASK


class OrderAction(str, Enum):
    """Legacy event-contract action. Retained because the REST API still takes it."""

    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    """Order lifetime. ``GTT`` is internal to Kalshi and is not a valid API value."""

    FILL_OR_KILL = "fill_or_kill"
    GOOD_TILL_CANCELED = "good_till_canceled"
    IMMEDIATE_OR_CANCEL = "immediate_or_cancel"


class SelfTradePrevention(str, Enum):
    TAKER_AT_CROSS = "taker_at_cross"
    MAKER = "maker"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderState(str, Enum):
    """Internal order state machine.

    This is *our* state, not Kalshi's. It exists because an HTTP 200 does not
    mean an order rests on the book, and a network timeout does not mean it does
    not. ``SUBMITTED`` is the ambiguous state: the request left the process and
    we do not yet know its fate. Only reconciliation may move an order out of
    ``SUBMITTED`` when the response was never seen.
    """

    CREATED = "created"                    # constructed locally, nothing sent
    RISK_REJECTED = "risk_rejected"        # our own risk engine refused it
    SUBMITTED = "submitted"                # request sent, outcome unknown
    ACKNOWLEDGED = "acknowledged"          # exchange confirmed it rests
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    REJECTED = "rejected"                  # exchange refused it
    EXPIRED = "expired"
    UNKNOWN = "unknown"                    # reconciliation could not classify it

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_ORDER_STATES

    @property
    def is_working(self) -> bool:
        """True when the order may still consume exposure at the exchange."""
        return self in _WORKING_ORDER_STATES


_TERMINAL_ORDER_STATES = frozenset(
    {
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.RISK_REJECTED,
        OrderState.EXPIRED,
    }
)

# An order in any of these states might still be filled by the exchange, so its
# notional must be counted against exposure limits.
_WORKING_ORDER_STATES = frozenset(
    {
        OrderState.SUBMITTED,
        OrderState.ACKNOWLEDGED,
        OrderState.PARTIALLY_FILLED,
        OrderState.CANCEL_REQUESTED,
        OrderState.UNKNOWN,
    }
)


class StrategyState(str, Enum):
    """Strategy lifecycle. Mode D gates every transition toward ``LIVE``.

    The ordering here is the promotion ladder. A strategy may always be demoted
    (to ``PAUSED`` or ``RETIRED``) but may only be promoted one rung at a time
    and only when the gate for that rung passes.
    """

    DISCOVERY = "discovery"
    DEVELOPMENT = "development"
    BACKTEST = "backtest"
    OUT_OF_SAMPLE = "out_of_sample"
    WALK_FORWARD = "walk_forward"
    ROBUSTNESS = "robustness"
    PAPER = "paper"
    PROBATION = "probation"
    LIVE = "live"
    MONITORING = "monitoring"
    REVIEW = "review"
    PAUSED = "paused"
    RETIRED = "retired"


class OperatingMode(str, Enum):
    """Platform autonomy mode, selected from the dashboard.

    ``GATED_DISCOVERY`` (Mode D) is the recommended default: the system may
    discover and validate strategies autonomously, but nothing reaches live
    capital without clearing every lifecycle gate.
    """

    FULL_AUTONOMOUS = "full_autonomous"          # Mode A
    HUMAN_STRATEGIES = "human_strategies"        # Mode B
    AUTONOMOUS_DISCOVERY = "autonomous_discovery"  # Mode C
    GATED_DISCOVERY = "gated_discovery"          # Mode D (default)


class TradingEnvironment(str, Enum):
    """Where orders actually go.

    ``PAPER`` never contacts the exchange order endpoints at all -- it simulates
    fills against live market data. ``DEMO`` sends real orders to Kalshi's demo
    exchange. ``LIVE`` sends real orders with real money.
    """

    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"

    @property
    def is_real_money(self) -> bool:
        return self is TradingEnvironment.LIVE


class KillSwitchScope(str, Enum):
    """Independent halt domains. Each persists across restarts."""

    GLOBAL = "global"
    EVENT = "event"
    PERP = "perp"
    STRATEGY = "strategy"
    RISK = "risk"


class RiskDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    RESIZED = "resized"


class DataQuality(str, Enum):
    """Provenance/quality tag stored with every ingested record."""

    OK = "ok"
    STALE = "stale"
    PARTIAL = "partial"
    SUSPECT = "suspect"
    SYNTHETIC = "synthetic"  # generated data; never mixed with real history
