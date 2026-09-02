"""The backtest engine, and the information-time guarantee it is built around.

Everything else in this platform is validated against this module, so the thing
it must get right is not speed -- it is refusing to know things.

Information time is a hard guarantee, not a convention
------------------------------------------------------
A simulation at time ``T`` must never see data whose availability timestamp is
later than ``T``. That is enforced twice, on the principle that the first
defence is code somebody can change without knowing why it is there:

1. **The feed cannot yield the future.** :class:`EventFeed` sorts on
   ``available_ms`` and hands out only rows that are already available. There is
   no method that returns a row by index, by ticker, or "the next one" -- the
   only accessor takes ``now_ms`` and filters on it. A strategy therefore cannot
   reach a future row even by mistake, because nothing in the object graph
   points at one.

2. **The guard checks what the strategy is handed.** Every feature, every model
   probability, and the market snapshot itself pass through
   :class:`LeakageDetector` immediately before
   :meth:`kat.strategies.base.Strategy.generate` is called. The check is on both
   timestamps: ``available_ms <= now`` (we could not have held it yet) *and*
   ``info_ts_ms <= now`` (it describes the future). The second catches the case
   the first cannot -- a value with an honest-looking availability stamp that
   was computed from a forward-looking window, which is what a research pipeline
   joining a feature matrix on the wrong side of a bar boundary produces.

In strict mode a violation raises :class:`LeakageDetected` and the run stops. A
backtest that has leaked is not a backtest with a caveat; it is a number with no
meaning, and continuing to compute metrics on it would be the fabrication this
module exists to prevent.

The pipeline is the real one
----------------------------
Intents go ``Strategy -> portfolio allocation -> real risk engine -> execution``.
The risk engines imported here are :class:`kat.risk.event_risk.EventRiskEngine`
and :class:`kat.risk.perp_risk.PerpRiskEngine` -- the same objects the live
engine uses, with no simulator-specific variant and no bypass. A backtest that
reimplemented risk would be measuring a strategy the platform will never run,
and in the perpetual domain it would be measuring one the 1.00x leverage
invariant would have refused.

Rounding
--------
The ledger values a fill at ``price_cc * qty_qc`` using the average price the
fill simulator already rounded *against* us, rather than the exact multi-level
walk total. The two differ by at most one micro-dollar per centi-contract, and
always in the direction that costs us money.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

from kat.common.enums import DataQuality, MarketType, OutcomeSide
from kat.common.errors import DataQualityError
from kat.common.fixedpoint import CC_PER_DOLLAR, QC_PER_CONTRACT, micros_to_decimal
from kat.common.ids import new_id
from kat.common.timeutil import SimulatedClock, from_ms
from kat.config.settings import AppSettings, EventRiskSettings, PerpRiskSettings, get_settings
from kat.db.models import BacktestRun
from kat.features.spec import FeatureVector
from kat.kalshi.fees import FeeSchedule, estimate_fee_micros
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketSnapshot, MarketState
from kat.observability.logging import get_logger
from kat.risk.event_risk import EventAccountState, EventPositionState, EventRiskEngine
from kat.risk.perp_risk import PerpAccountState, PerpPositionState, PerpRiskEngine
from kat.risk.types import OrderIntent, RiskVerdict
from kat.simulation.costs import PESSIMISTIC, CostModel
from kat.simulation.fills import FillSimulator, SimFill
from kat.simulation.metrics import BacktestMetrics, compute_metrics
from kat.strategies.base import Strategy, StrategyContext
from kat.strategies.common import TradingStrategy

log = get_logger(__name__)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "EquityPoint",
    "EventFeed",
    "EventKind",
    "LeakageDetected",
    "LeakageDetector",
    "LeakageViolation",
    "ModelPrediction",
    "PredictionRecord",
    "SimEvent",
    "TimedFeature",
    "TradeRecord",
    "book_event",
    "feature_event",
    "settlement_event",
    "status_event",
    "ticker_event",
    "trade_event",
]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class EventKind(str, Enum):
    """What one row of the historical feed represents."""

    BOOK = "book"
    TRADE = "trade"
    TICKER = "ticker"
    FEATURE = "feature"
    SETTLEMENT = "settlement"
    STATUS = "status"


@dataclass(frozen=True, slots=True)
class SimEvent:
    """One observation, with the two timestamps that make replay honest.

    ``info_ts_ms`` is when the observation happened; ``available_ms`` is the
    earliest moment this platform could have acted on it. They differ by the
    transport delay, and the simulation clock is driven by the *second* one.

    An event whose availability precedes its own information time is rejected at
    construction. That combination is not a pessimistic assumption or an edge
    case -- it is a claim to have received a message before it was sent.
    """

    available_ms: int
    info_ts_ms: int
    kind: EventKind
    ticker: str
    payload: dict[str, Any] = field(default_factory=dict)
    quality: DataQuality = DataQuality.OK
    seq: int = 0
    event_ticker: str = ""

    def __post_init__(self) -> None:
        if self.available_ms < self.info_ts_ms:
            raise DataQualityError(
                f"{self.kind.value} event for {self.ticker} claims to be available at "
                f"{self.available_ms} but describes information from {self.info_ts_ms}"
            )

    @property
    def lag_ms(self) -> int:
        return self.available_ms - self.info_ts_ms


def book_event(
    ticker: str,
    *,
    info_ts_ms: int,
    lag_ms: int = 0,
    bids: Sequence[tuple[int, int]] | None = None,
    asks: Sequence[tuple[int, int]] | None = None,
    yes_levels: Sequence[tuple[int, int]] | None = None,
    no_levels: Sequence[tuple[int, int]] | None = None,
    quality: DataQuality = DataQuality.OK,
    seq: int = 0,
    event_ticker: str = "",
) -> SimEvent:
    """A full book snapshot.

    Either two-sided levels (perpetuals) or Kalshi's two bid arrays (event
    contracts) may be given; the conversion from NO bids to YES asks is done by
    :meth:`kat.marketdata.book.OrderBook.from_event_contract_levels`, which is
    the only place in the platform that reflection happens.
    """
    payload: dict[str, Any] = {}
    if yes_levels is not None or no_levels is not None:
        payload["yes_levels"] = [list(level) for level in (yes_levels or ())]
        payload["no_levels"] = [list(level) for level in (no_levels or ())]
    else:
        payload["bids"] = [list(level) for level in (bids or ())]
        payload["asks"] = [list(level) for level in (asks or ())]
    return SimEvent(
        available_ms=info_ts_ms + lag_ms,
        info_ts_ms=info_ts_ms,
        kind=EventKind.BOOK,
        ticker=ticker,
        payload=payload,
        quality=quality,
        seq=seq,
        event_ticker=event_ticker,
    )


def trade_event(
    ticker: str,
    *,
    info_ts_ms: int,
    price_cc: int,
    qty_qc: int,
    lag_ms: int = 0,
    quality: DataQuality = DataQuality.OK,
    seq: int = 0,
) -> SimEvent:
    """A print. Prices are YES-denominated for event contracts."""
    return SimEvent(
        available_ms=info_ts_ms + lag_ms,
        info_ts_ms=info_ts_ms,
        kind=EventKind.TRADE,
        ticker=ticker,
        payload={"price_cc": price_cc, "qty_qc": qty_qc},
        quality=quality,
        seq=seq,
    )


def ticker_event(
    ticker: str,
    *,
    info_ts_ms: int,
    lag_ms: int = 0,
    mark_price_cc: int | None = None,
    volume_qc: int | None = None,
    open_interest_qc: int | None = None,
    funding_rate: float | None = None,
    next_funding_ms: int | None = None,
    quality: DataQuality = DataQuality.OK,
    seq: int = 0,
) -> SimEvent:
    return SimEvent(
        available_ms=info_ts_ms + lag_ms,
        info_ts_ms=info_ts_ms,
        kind=EventKind.TICKER,
        ticker=ticker,
        payload={
            "mark_price_cc": mark_price_cc,
            "volume_qc": volume_qc,
            "open_interest_qc": open_interest_qc,
            "funding_rate": funding_rate,
            "next_funding_ms": next_funding_ms,
        },
        quality=quality,
        seq=seq,
    )


def feature_event(
    ticker: str,
    *,
    info_ts_ms: int,
    values: dict[str, float],
    lag_ms: int,
    quality: DataQuality = DataQuality.OK,
    seq: int = 0,
) -> SimEvent:
    """Pre-computed features arriving through the feed.

    ``lag_ms`` is mandatory: it is the availability lag of the source
    (:mod:`kat.features.spec` publishes the per-source constants), and defaulting
    it to zero would silently assert that every feature is knowable the instant
    its inputs happened.
    """
    return SimEvent(
        available_ms=info_ts_ms + lag_ms,
        info_ts_ms=info_ts_ms,
        kind=EventKind.FEATURE,
        ticker=ticker,
        payload={"values": dict(values)},
        quality=quality,
        seq=seq,
    )


def settlement_event(
    ticker: str,
    *,
    info_ts_ms: int,
    outcome_yes: bool,
    lag_ms: int = 0,
    quality: DataQuality = DataQuality.OK,
    seq: int = 0,
) -> SimEvent:
    return SimEvent(
        available_ms=info_ts_ms + lag_ms,
        info_ts_ms=info_ts_ms,
        kind=EventKind.SETTLEMENT,
        ticker=ticker,
        payload={"outcome_yes": bool(outcome_yes)},
        quality=quality,
        seq=seq,
    )


def status_event(
    ticker: str, *, info_ts_ms: int, status: str, lag_ms: int = 0,
    quality: DataQuality = DataQuality.OK, seq: int = 0,
) -> SimEvent:
    return SimEvent(
        available_ms=info_ts_ms + lag_ms,
        info_ts_ms=info_ts_ms,
        kind=EventKind.STATUS,
        ticker=ticker,
        payload={"status": status},
        quality=quality,
        seq=seq,
    )


class EventFeed:
    """A historical feed that structurally cannot hand out the future.

    The only accessor is :meth:`drain_until`, which takes the current simulation
    time. There is deliberately no indexing, no ``events`` attribute and no
    lookahead helper: code that wanted to cheat would have to add one, which is
    a visible change rather than an invisible mistake.
    """

    __slots__ = ("_cursor", "_events", "_synthetic", "name")

    def __init__(self, events: Iterable[SimEvent], *, name: str = "") -> None:
        ordered = sorted(events, key=lambda e: (e.available_ms, e.seq))
        qualities = {e.quality for e in ordered}
        synthetic = DataQuality.SYNTHETIC in qualities
        if synthetic and len(qualities) > 1:
            # Blending generated and recorded history produces a result that is
            # neither, and nothing downstream could tell which parts were real.
            raise DataQualityError(
                "refusing to build a feed that mixes synthetic and real observations: "
                f"qualities present are {sorted(q.value for q in qualities)}"
            )
        self._events = ordered
        self._cursor = 0
        self._synthetic = synthetic
        self.name = name

    def __len__(self) -> int:
        return len(self._events)

    @property
    def is_synthetic(self) -> bool:
        return self._synthetic

    @property
    def remaining(self) -> int:
        return len(self._events) - self._cursor

    @property
    def start_ms(self) -> int:
        if not self._events:
            raise DataQualityError("cannot start a backtest on an empty feed")
        return self._events[0].available_ms

    @property
    def end_ms(self) -> int:
        if not self._events:
            raise DataQualityError("cannot end a backtest on an empty feed")
        return self._events[-1].available_ms

    def peek_ms(self) -> int | None:
        """Availability time of the next row, or ``None`` when exhausted."""
        if self._cursor >= len(self._events):
            return None
        return self._events[self._cursor].available_ms

    def drain_until(self, now_ms: int) -> list[SimEvent]:
        """Every row available at or before ``now_ms``, in order.

        This is the only way to read the feed, and the filter is unconditional.
        """
        out: list[SimEvent] = []
        while self._cursor < len(self._events):
            event = self._events[self._cursor]
            if event.available_ms > now_ms:
                break
            out.append(event)
            self._cursor += 1
        return out

    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted({e.ticker for e in self._events}))


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------


class LeakageDetected(DataQualityError):
    """A strategy was handed information from after the simulation's clock.

    Deliberately fatal in strict mode. A leaked backtest is not a result with a
    caveat attached; it is a measurement of a strategy that cannot exist.
    """


@dataclass(frozen=True, slots=True)
class LeakageViolation:
    """One provable look-ahead, with enough detail to find its source."""

    kind: str
    name: str
    ticker: str
    now_ms: int
    info_ts_ms: int
    available_ms: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "ticker": self.ticker,
            "now_ms": self.now_ms,
            "info_ts_ms": self.info_ts_ms,
            "available_ms": self.available_ms,
            "ahead_ms": max(self.info_ts_ms, self.available_ms) - self.now_ms,
            "detail": self.detail,
        }


class LeakageDetector:
    """Checks everything handed to a strategy against the simulation clock.

    Two independent conditions are tested, because they fail differently:

    ``available_ms > now``
        We could not yet have held the value. This catches an ordering bug in
        the feed or a provider that reached past the cursor.

    ``info_ts_ms > now``
        The value describes the future, whatever its availability stamp claims.
        This is the one that catches a forward-looking window -- a "next five
        minutes" column joined onto the current row -- which is invisible to any
        check that only compares availability.
    """

    __slots__ = ("checks", "strict", "violations")

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict
        self.checks = 0
        self.violations: list[LeakageViolation] = []

    def _fail(self, violation: LeakageViolation) -> None:
        self.violations.append(violation)
        log.error("simulation.leakage_detected", **violation.to_dict())
        if self.strict:
            raise LeakageDetected(
                f"{violation.kind} {violation.name!r} on {violation.ticker} is "
                f"{violation.to_dict()['ahead_ms']}ms ahead of simulation time "
                f"{violation.now_ms}: {violation.detail}"
            )

    def check(
        self,
        *,
        kind: str,
        name: str,
        ticker: str,
        info_ts_ms: int,
        available_ms: int,
        now_ms: int,
    ) -> bool:
        self.checks += 1
        if available_ms > now_ms:
            self._fail(
                LeakageViolation(
                    kind=kind, name=name, ticker=ticker, now_ms=now_ms,
                    info_ts_ms=info_ts_ms, available_ms=available_ms,
                    detail="value could not have been held yet",
                )
            )
            return False
        if info_ts_ms > now_ms:
            self._fail(
                LeakageViolation(
                    kind=kind, name=name, ticker=ticker, now_ms=now_ms,
                    info_ts_ms=info_ts_ms, available_ms=available_ms,
                    detail="value is derived from an observation in the future",
                )
            )
            return False
        return True

    def check_snapshot(self, snap: MarketSnapshot, now_ms: int) -> bool:
        return self.check(
            kind="market", name="snapshot", ticker=snap.ticker,
            info_ts_ms=snap.updated_ms, available_ms=snap.updated_ms, now_ms=now_ms,
        )

    def check_vector(self, vector: FeatureVector, now_ms: int) -> bool:
        """Validate a :class:`kat.features.spec.FeatureVector` value by value."""
        ok = True
        for name, fv in vector.values.items():
            ok &= self.check(
                kind="feature", name=name, ticker=vector.ticker,
                info_ts_ms=fv.info_ts_ms, available_ms=fv.available_ms, now_ms=now_ms,
            )
        return ok

    @property
    def clean(self) -> bool:
        return not self.violations

    def report(self) -> dict[str, Any]:
        return {
            "strict": self.strict,
            "checks": self.checks,
            "violations": len(self.violations),
            "clean": self.clean,
            "detail": [v.to_dict() for v in self.violations[:20]],
        }


# ---------------------------------------------------------------------------
# Inputs a strategy may be given
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimedFeature:
    """A feature value with the timestamps the leakage guard needs."""

    name: str
    value: float
    info_ts_ms: int
    available_ms: int


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """A model's probability, carrying its own information time."""

    probability: float
    info_ts_ms: int
    available_ms: int
    model_id: str | None = None
    model_version: str | None = None
    #: What the probability is a probability of. A strategy that reads the
    #: number without this is not reading a probability, it is reading a float --
    #: and a backtest that omits it measures a strategy standing aside.
    target: str | None = None
    #: The unconditional rate of the event this model predicts. Without it a
    #: rare-event probability read against a 0.5 prior is a constant, not a
    #: signal: at a 2.4% base rate every output is below a half.
    base_rate: float | None = None


FeatureProvider = Callable[[str, int], "Sequence[TimedFeature] | FeatureVector | None"]
ModelProvider = Callable[[str, int], "ModelPrediction | None"]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityPoint:
    ts_ms: int
    equity_micros: int
    cash_micros: int
    exposure_micros: int
    open_positions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "equity_micros": self.equity_micros,
            "cash_micros": self.cash_micros,
            "exposure_micros": self.exposure_micros,
            "open_positions": self.open_positions,
        }


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One closed round trip: capital committed, then released.

    Recorded when a position is reduced, not when an order is sent, so a trade
    exists only once its result is known.
    """

    ticker: str
    market_type: MarketType
    strategy_id: str | None
    entry_ms: int
    exit_ms: int
    qty_qc: int
    entry_price_cc: int
    exit_price_cc: int
    is_long: bool
    gross_pnl_micros: int
    fees_micros: int
    net_pnl_micros: int
    cost_micros: int
    exit_reason: str = "close"

    @property
    def holding_ms(self) -> int:
        return self.exit_ms - self.entry_ms

    @property
    def is_win(self) -> bool:
        return self.net_pnl_micros > 0

    @property
    def return_on_cost(self) -> float:
        return 0.0 if self.cost_micros <= 0 else self.net_pnl_micros / self.cost_micros

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "strategy_id": self.strategy_id,
            "entry_ms": self.entry_ms,
            "exit_ms": self.exit_ms,
            "holding_ms": self.holding_ms,
            "qty_qc": self.qty_qc,
            "entry_price_cc": self.entry_price_cc,
            "exit_price_cc": self.exit_price_cc,
            "is_long": self.is_long,
            "gross_pnl_micros": self.gross_pnl_micros,
            "fees_micros": self.fees_micros,
            "net_pnl_micros": self.net_pnl_micros,
            "net_pnl_usd": str(micros_to_decimal(self.net_pnl_micros)),
            "exit_reason": self.exit_reason,
        }


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """A probability asserted at decision time, for calibration scoring."""

    ticker: str
    ts_ms: int
    probability: float
    source: str
    strategy_id: str | None = None


# ---------------------------------------------------------------------------
# Ledgers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _EventPos:
    ticker: str
    event_ticker: str = ""
    qty_qc: int = 0            # signed: + YES held, - NO held
    basis_micros: int = 0      # capital committed to the open position
    fees_micros: int = 0       # entry-side fees still attached to it
    opened_ms: int = 0

    @property
    def avg_cost_cc(self) -> int:
        if self.qty_qc == 0:
            return 0
        return self.basis_micros // abs(self.qty_qc)


class EventLedger:
    """Cash, positions and realised profit for the event-contract domain.

    Event contracts are fully collateralised, so the ledger is a cash ledger: a
    YES buy at $0.40 removes $0.40 from cash and adds $0.40 of committed
    capital, and a NO buy at $0.60 does the same with the NO price. Nothing is
    ever financed, which is why there is no margin concept here at all.
    """

    def __init__(self, *, initial_capital_micros: int) -> None:
        self.initial_capital_micros = initial_capital_micros
        self.cash_micros = initial_capital_micros
        self.positions: dict[str, _EventPos] = {}
        self.trades: list[TradeRecord] = []
        self.realized_pnl_micros = 0
        self.fees_micros = 0
        self.peak_equity_micros = initial_capital_micros
        self._realized_today_micros = 0
        self._today: str = ""

    # -- fills ---------------------------------------------------------------

    def apply_fill(self, fill: SimFill, *, event_ticker: str = "") -> None:
        pos = self.positions.get(fill.ticker)
        if pos is None:
            pos = _EventPos(ticker=fill.ticker, event_ticker=event_ticker, opened_ms=fill.ts_ms)
            self.positions[fill.ticker] = pos

        self.cash_micros -= fill.fee_micros
        self.fees_micros += fill.fee_micros

        direction = 1 if fill.is_buy else -1
        qty = fill.qty_qc
        price = fill.price_cc

        closing = 0
        if pos.qty_qc != 0 and (pos.qty_qc > 0) != (direction > 0):
            closing = min(qty, abs(pos.qty_qc))

        if closing:
            self._close(pos, fill, closing, price)
            qty -= closing

        if qty > 0:
            # The contract we are buying is YES at ``price`` or NO at 1 - price.
            cost_cc = price if direction > 0 else CC_PER_DOLLAR - price
            cost = cost_cc * qty
            if pos.qty_qc == 0:
                pos.opened_ms = fill.ts_ms
                pos.event_ticker = pos.event_ticker or event_ticker
            self.cash_micros -= cost
            pos.basis_micros += cost
            pos.fees_micros += fill.fee_micros if not closing else 0
            pos.qty_qc += direction * qty
        elif closing:
            # Exit fee belongs to the trade just closed, already accounted there.
            pass

        if pos.qty_qc == 0 and pos.basis_micros == 0:
            self.positions.pop(fill.ticker, None)

    def _close(self, pos: _EventPos, fill: SimFill, qty: int, price: int) -> None:
        """Release capital committed to ``qty`` contracts and book the result."""
        held = abs(pos.qty_qc)
        is_long = pos.qty_qc > 0
        # A long holds YES, worth the YES price; a short holds NO, worth 1 - it.
        proceeds_cc = price if is_long else CC_PER_DOLLAR - price
        proceeds = proceeds_cc * qty

        released = pos.basis_micros if qty == held else pos.basis_micros * qty // held
        entry_fees = pos.fees_micros if qty == held else pos.fees_micros * qty // held
        exit_fee = fill.fee_micros

        gross = proceeds - released
        fees = entry_fees + exit_fee
        net = gross - fees

        self.cash_micros += proceeds
        pos.basis_micros -= released
        pos.fees_micros -= entry_fees
        pos.qty_qc -= qty if is_long else -qty
        self.realized_pnl_micros += net
        self._add_realized_today(fill.ts_ms, net)

        self.trades.append(
            TradeRecord(
                ticker=pos.ticker,
                market_type=MarketType.EVENT,
                strategy_id=fill.strategy_id,
                entry_ms=pos.opened_ms,
                exit_ms=fill.ts_ms,
                qty_qc=qty,
                entry_price_cc=released // qty if qty else 0,
                exit_price_cc=proceeds_cc,
                is_long=is_long,
                gross_pnl_micros=gross,
                fees_micros=fees,
                net_pnl_micros=net,
                cost_micros=released,
            )
        )

    def settle(self, ticker: str, *, outcome_yes: bool, now_ms: int) -> None:
        """Resolve an expired market. Kalshi charges no fee on settlement."""
        pos = self.positions.pop(ticker, None)
        if pos is None or pos.qty_qc == 0:
            return
        is_long = pos.qty_qc > 0
        won = outcome_yes if is_long else not outcome_yes
        qty = abs(pos.qty_qc)
        proceeds_cc = CC_PER_DOLLAR if won else 0
        proceeds = proceeds_cc * qty

        gross = proceeds - pos.basis_micros
        net = gross - pos.fees_micros
        self.cash_micros += proceeds
        self.realized_pnl_micros += net
        self._add_realized_today(now_ms, net)
        self.trades.append(
            TradeRecord(
                ticker=ticker,
                market_type=MarketType.EVENT,
                strategy_id=None,
                entry_ms=pos.opened_ms,
                exit_ms=now_ms,
                qty_qc=qty,
                entry_price_cc=pos.basis_micros // qty,
                exit_price_cc=proceeds_cc,
                is_long=is_long,
                gross_pnl_micros=gross,
                fees_micros=pos.fees_micros,
                net_pnl_micros=net,
                cost_micros=pos.basis_micros,
                exit_reason="settlement",
            )
        )

    # -- valuation -----------------------------------------------------------

    def position_value_micros(self, pos: _EventPos, snap: MarketSnapshot | None) -> int:
        """Liquidation value of a position: what we could get out at, now.

        A long is marked at the best bid and a short at the NO bid (``1 - best
        ask``). Marking at the mid would credit half the spread we have not
        earned, on every open position, at every equity sample.
        """
        if pos.qty_qc == 0:
            return 0
        qty = abs(pos.qty_qc)
        book = snap.book if snap else None
        if book is not None and not book.is_crossed:
            if pos.qty_qc > 0 and book.best_bid_cc is not None:
                return book.best_bid_cc * qty
            if pos.qty_qc < 0 and book.best_ask_cc is not None:
                return (CC_PER_DOLLAR - book.best_ask_cc) * qty
        reference = snap.reference_price_cc() if snap else None
        if reference is not None:
            return (reference if pos.qty_qc > 0 else CC_PER_DOLLAR - reference) * qty
        # No market at all: hold it at cost rather than invent a mark.
        return pos.basis_micros

    def portfolio_value_micros(self, market: MarketState) -> int:
        return sum(
            self.position_value_micros(p, market.get(p.ticker)) for p in self.positions.values()
        )

    def equity_micros(self, market: MarketState) -> int:
        return self.cash_micros + self.portfolio_value_micros(market)

    def exposure_micros(self) -> int:
        return sum(p.basis_micros for p in self.positions.values())

    def _add_realized_today(self, now_ms: int, amount: int) -> None:
        day = from_ms(now_ms).strftime("%Y-%m-%d")
        if day != self._today:
            self._today = day
            self._realized_today_micros = 0
        self._realized_today_micros += amount

    # -- risk-engine input ---------------------------------------------------

    def account_state(
        self, *, market: MarketState, fills: FillSimulator, now_ms: int, orders_this_minute: int
    ) -> EventAccountState:
        state = EventAccountState(
            balance_micros=self.cash_micros,
            portfolio_value_micros=self.portfolio_value_micros(market),
            updated_at_ms=now_ms,
            realized_pnl_today_micros=self._realized_today_micros,
            orders_this_minute=orders_this_minute,
        )
        for pos in self.positions.values():
            snap = market.get(pos.ticker)
            value = self.position_value_micros(pos, snap)
            state.positions[pos.ticker] = EventPositionState(
                ticker=pos.ticker,
                event_ticker=pos.event_ticker,
                position_qc=pos.qty_qc,
                avg_entry_cc=pos.avg_cost_cc or None,
                exposure_micros=pos.basis_micros,
                pending_exposure_micros=fills.pending_exposure_micros(pos.ticker),
                unrealized_pnl_micros=value - pos.basis_micros,
            )
        # Markets with working orders but no position still consume capital.
        for order in fills.working_orders():
            if order.ticker in state.positions:
                continue
            state.positions[order.ticker] = EventPositionState(
                ticker=order.ticker,
                pending_exposure_micros=fills.pending_exposure_micros(order.ticker),
            )
        self.peak_equity_micros = max(self.peak_equity_micros, state.equity_micros)
        state.peak_equity_micros = self.peak_equity_micros
        return state


@dataclass(slots=True)
class _PerpPos:
    ticker: str
    qty_qc: int = 0                 # signed
    entry_notional_micros: int = 0  # sum of price * size for the open side
    fees_micros: int = 0
    opened_ms: int = 0

    @property
    def avg_entry_cc(self) -> int:
        if self.qty_qc == 0:
            return 0
        return self.entry_notional_micros // abs(self.qty_qc)


class PerpLedger:
    """Realised cash and marked positions for the perpetual domain.

    Equity is cash plus unrealised mark-to-market, which is exactly the
    denominator :mod:`kat.risk.invariants` divides by. It is deliberately not
    smoothed or lagged: the leverage invariant must be applied against the same
    equity the exchange would use to liquidate us.
    """

    def __init__(self, *, initial_capital_micros: int) -> None:
        self.initial_capital_micros = initial_capital_micros
        self.cash_micros = initial_capital_micros
        self.positions: dict[str, _PerpPos] = {}
        self.trades: list[TradeRecord] = []
        self.realized_pnl_micros = 0
        self.fees_micros = 0
        self.peak_equity_micros = initial_capital_micros
        self._realized_today_micros = 0
        self._today = ""

    def apply_fill(self, fill: SimFill) -> None:
        pos = self.positions.get(fill.ticker)
        if pos is None:
            pos = _PerpPos(ticker=fill.ticker, opened_ms=fill.ts_ms)
            self.positions[fill.ticker] = pos

        self.cash_micros -= fill.fee_micros
        self.fees_micros += fill.fee_micros

        direction = 1 if fill.is_buy else -1
        qty = fill.qty_qc
        price = fill.price_cc

        closing = 0
        if pos.qty_qc != 0 and (pos.qty_qc > 0) != (direction > 0):
            closing = min(qty, abs(pos.qty_qc))

        if closing:
            held = abs(pos.qty_qc)
            is_long = pos.qty_qc > 0
            released = (
                pos.entry_notional_micros
                if closing == held
                else pos.entry_notional_micros * closing // held
            )
            entry_fees = (
                pos.fees_micros if closing == held else pos.fees_micros * closing // held
            )
            exit_value = price * closing
            gross = exit_value - released if is_long else released - exit_value
            fees = entry_fees + fill.fee_micros
            net = gross - fees

            self.cash_micros += gross
            pos.entry_notional_micros -= released
            pos.fees_micros -= entry_fees
            pos.qty_qc -= closing if is_long else -closing
            self.realized_pnl_micros += net
            self._add_realized_today(fill.ts_ms, net)
            self.trades.append(
                TradeRecord(
                    ticker=fill.ticker,
                    market_type=MarketType.PERP,
                    strategy_id=fill.strategy_id,
                    entry_ms=pos.opened_ms,
                    exit_ms=fill.ts_ms,
                    qty_qc=closing,
                    entry_price_cc=released // closing if closing else 0,
                    exit_price_cc=price,
                    is_long=is_long,
                    gross_pnl_micros=gross,
                    fees_micros=fees,
                    net_pnl_micros=net,
                    cost_micros=released,
                )
            )
            qty -= closing

        if qty > 0:
            if pos.qty_qc == 0:
                pos.opened_ms = fill.ts_ms
            pos.entry_notional_micros += price * qty
            pos.fees_micros += 0 if closing else fill.fee_micros
            pos.qty_qc += direction * qty

        if pos.qty_qc == 0:
            self.positions.pop(fill.ticker, None)

    def _mark_cc(self, ticker: str, market: MarketState) -> int | None:
        snap = market.get(ticker)
        return snap.reference_price_cc() if snap else None

    def unrealized_micros(self, market: MarketState) -> int:
        total = 0
        for pos in self.positions.values():
            mark = self._mark_cc(pos.ticker, market)
            if mark is None:
                continue
            value = mark * abs(pos.qty_qc)
            total += (
                value - pos.entry_notional_micros
                if pos.qty_qc > 0
                else pos.entry_notional_micros - value
            )
        return total

    def equity_micros(self, market: MarketState) -> int:
        return self.cash_micros + self.unrealized_micros(market)

    def exposure_micros(self, market: MarketState) -> int:
        total = 0
        for pos in self.positions.values():
            mark = self._mark_cc(pos.ticker, market)
            if mark is not None:
                total += abs(pos.qty_qc) * mark
        return total

    def _add_realized_today(self, now_ms: int, amount: int) -> None:
        day = from_ms(now_ms).strftime("%Y-%m-%d")
        if day != self._today:
            self._today = day
            self._realized_today_micros = 0
        self._realized_today_micros += amount

    def account_state(
        self, *, market: MarketState, fills: FillSimulator, now_ms: int
    ) -> PerpAccountState:
        equity = self.equity_micros(market)
        state = PerpAccountState(
            equity_micros=max(0, equity),
            equity_source="simulated_ledger",
            equity_is_known=True,
            updated_at_ms=now_ms,
            realized_pnl_today_micros=self._realized_today_micros,
        )
        tickers = set(self.positions) | {o.ticker for o in fills.working_orders()}
        for ticker in tickers:
            pos = self.positions.get(ticker)
            buy_qc, sell_qc = fills.pending_qty_qc(ticker)
            state.positions[ticker] = PerpPositionState(
                ticker=ticker,
                position_qc=pos.qty_qc if pos else 0,
                mark_price_cc=self._mark_cc(ticker, market),
                entry_price_cc=pos.avg_entry_cc if pos else None,
                pending_buy_qc=buy_qc,
                pending_sell_qc=sell_qc,
            )
        self.peak_equity_micros = max(self.peak_equity_micros, equity)
        state.peak_equity_micros = self.peak_equity_micros
        return state


# ---------------------------------------------------------------------------
# Configuration and result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Everything that decides what a run measures.

    Recorded verbatim on the result. Two runs whose configs differ are two
    different experiments, and a metric quoted without its config is not a
    reproducible claim.
    """

    market_type: MarketType = MarketType.EVENT
    initial_capital_micros: int = 200_000_000  # $200: the small-account case
    costs: CostModel = PESSIMISTIC
    #: How often strategies are consulted. Not every tick: a strategy that is
    #: asked on every message will trade the noise between messages.
    decision_interval_ms: int = 5_000
    #: Equity sampling period. Also the return period Sharpe is computed on.
    equity_sample_interval_ms: int = 60_000
    #: Simulated time from the first event before any decision is taken. The
    #: strategy's own ``warmup_ms`` is added on top of this.
    warmup_ms: int = 0
    strict_leakage: bool = True
    seed: int = 0
    #: Series fee multiplier passed to strategies. ``None`` means unknown, which
    #: makes both the strategy and the cost model use the inexact schedule.
    fee_multiplier: float | None = None
    #: Flatten every open position against the final book instead of leaving it
    #: open. Off by default: a strategy that ends the run holding a position has
    #: not produced a result for it, and closing it at a favourable mark is a
    #: way to manufacture one.
    flatten_at_end: bool = False
    kind: str = "backtest"
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_type": self.market_type.value,
            "initial_capital_usd": str(micros_to_decimal(self.initial_capital_micros)),
            "decision_interval_ms": self.decision_interval_ms,
            "equity_sample_interval_ms": self.equity_sample_interval_ms,
            "warmup_ms": self.warmup_ms,
            "strict_leakage": self.strict_leakage,
            "seed": self.seed,
            "fee_multiplier": self.fee_multiplier,
            "flatten_at_end": self.flatten_at_end,
            "kind": self.kind,
            "label": self.label,
            "costs": self.costs.to_dict(),
        }


@dataclass(slots=True)
class BacktestResult:
    """Everything one run produced, including what it refused to do.

    The rejection counts matter as much as the profit: a strategy whose intents
    were almost all refused by the risk engine has not been validated, it has
    been prevented, and the two look identical if only the equity curve is read.
    """

    run_id: str
    config: BacktestConfig
    strategy_ids: tuple[str, ...]
    market_type: MarketType
    start_ms: int
    end_ms: int
    is_synthetic_data: bool
    equity_curve: list[EquityPoint]
    trades: list[TradeRecord]
    fills: list[SimFill]
    metrics: BacktestMetrics
    leakage: dict[str, Any]
    cost_report: dict[str, Any]
    risk_decisions: dict[str, int]
    intents_generated: int
    orders_submitted: int
    decisions_taken: int
    skipped_stale: int
    strategy_health: dict[str, Any]
    predictions: list[PredictionRecord] = field(default_factory=list)
    settlements: dict[str, bool] = field(default_factory=dict)

    @property
    def initial_equity_micros(self) -> int:
        return self.equity_curve[0].equity_micros if self.equity_curve else 0

    @property
    def final_equity_micros(self) -> int:
        return self.equity_curve[-1].equity_micros if self.equity_curve else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy_ids": list(self.strategy_ids),
            "market_type": self.market_type.value,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "is_synthetic_data": self.is_synthetic_data,
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "leakage": self.leakage,
            "costs": self.cost_report,
            "risk_decisions": dict(sorted(self.risk_decisions.items())),
            "intents_generated": self.intents_generated,
            "orders_submitted": self.orders_submitted,
            "decisions_taken": self.decisions_taken,
            "skipped_stale": self.skipped_stale,
            "trade_count": len(self.trades),
            "strategy_health": self.strategy_health,
        }

    def to_backtest_run(self, *, git_commit: str | None = None) -> BacktestRun:
        """Build the persistable row. ``is_synthetic_data`` travels with it."""
        return BacktestRun(
            run_id=self.run_id,
            strategy_id=self.strategy_ids[0] if self.strategy_ids else "unknown",
            kind=self.config.kind,
            market_type=self.market_type.value,
            start_ts=from_ms(self.start_ms),
            end_ts=from_ms(self.end_ms),
            initial_capital_micros=self.initial_equity_micros,
            final_capital_micros=self.final_equity_micros,
            metrics=self.metrics.to_dict(),
            equity_curve=[p.to_dict() for p in self.equity_curve],
            trade_count=len(self.trades),
            is_synthetic_data=self.is_synthetic_data,
            cost_model=self.config.costs.to_dict(),
            git_commit=git_commit,
        )


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Drives a simulated clock over a feed and measures what a strategy does.

    One instance runs once. State accumulates in the ledger, the fill simulator
    and the leakage detector, and re-running would blend two experiments.
    """

    def __init__(
        self,
        *,
        feed: EventFeed,
        strategies: Sequence[Strategy],
        config: BacktestConfig | None = None,
        settings: AppSettings | None = None,
        event_risk_settings: EventRiskSettings | None = None,
        perp_risk_settings: PerpRiskSettings | None = None,
        feature_provider: FeatureProvider | None = None,
        model_provider: ModelProvider | None = None,
        allocated_capital_micros: int | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("a backtest needs at least one strategy")
        self.config = config or BacktestConfig()
        self.feed = feed
        self.strategies = list(strategies)
        for strategy in self.strategies:
            if strategy.market_type is not self.config.market_type:
                raise ValueError(
                    f"strategy {strategy.strategy_id} is a "
                    f"{strategy.market_type.value} strategy but the run is configured for "
                    f"{self.config.market_type.value}"
                )

        settings = settings or get_settings()
        self.clock = SimulatedClock(feed.start_ms)
        self.market = MarketState(self.config.market_type, max_age_ms=30_000)
        self.leakage = LeakageDetector(strict=self.config.strict_leakage)
        self.fills = FillSimulator(costs=self.config.costs, seed=self.config.seed)

        # The real risk engines. Not a simulator-specific variant, and not a
        # reimplementation -- the same classes the live engine constructs.
        self.event_risk = EventRiskEngine(event_risk_settings or settings.event_risk)
        self.perp_risk = PerpRiskEngine(perp_risk_settings or settings.perp_risk)

        self.event_ledger = EventLedger(initial_capital_micros=self.config.initial_capital_micros)
        self.perp_ledger = PerpLedger(initial_capital_micros=self.config.initial_capital_micros)

        self._feature_provider = feature_provider
        self._model_provider = model_provider
        self._feed_features: dict[str, dict[str, TimedFeature]] = {}
        self._event_tickers: dict[str, str] = {}
        self._settlements: dict[str, bool] = {}

        # The portfolio layer's contribution in a backtest: capital is split
        # evenly across strategies. Real allocation lives in kat.portfolio; a
        # strategy's own budget is a preference either way, and the risk engine
        # has the final word on size.
        capital = (
            allocated_capital_micros
            if allocated_capital_micros is not None
            else self.config.initial_capital_micros // len(self.strategies)
        )
        for strategy in self.strategies:
            strategy.allocated_capital_micros = capital

        self.equity_curve: list[EquityPoint] = []
        self.predictions: list[PredictionRecord] = []
        self.risk_decisions: dict[str, int] = {}
        self.intents_generated = 0
        self.orders_submitted = 0
        self.decisions_taken = 0
        self.skipped_stale = 0
        self._order_times: deque[int] = deque()
        self._next_decision_ms = 0
        self._next_sample_ms = 0
        self._ran = False

    # -- public --------------------------------------------------------------

    def run(self) -> BacktestResult:
        if self._ran:
            raise RuntimeError("a BacktestEngine runs once; construct a new one for another run")
        self._ran = True

        start_ms = self.feed.start_ms
        warmup = self.config.warmup_ms + max(s.warmup_ms() for s in self.strategies)
        self._next_decision_ms = start_ms + warmup
        self._next_sample_ms = start_ms
        now_ms = start_ms

        while True:
            next_event_ms = self.feed.peek_ms()
            next_arrival_ms = self._next_arrival_ms()
            candidates = [t for t in (next_event_ms, next_arrival_ms) if t is not None]
            if not candidates:
                break
            now_ms = max(now_ms, min(candidates))
            self.clock.set(now_ms)

            for event in self.feed.drain_until(now_ms):
                self._apply_event(event, now_ms)

            for fill in self.fills.step(now_ms=now_ms, book_for=self.market.book):
                self._apply_fill(fill)

            self._sample_equity(now_ms)

            if now_ms >= self._next_decision_ms:
                self._decide(now_ms)
                self._next_decision_ms = now_ms + self.config.decision_interval_ms

        if self.config.flatten_at_end:
            self._flatten(now_ms)
        self.fills.finalise()
        self._sample_equity(now_ms, force=True)

        return self._build_result(start_ms, now_ms)

    # -- event application ---------------------------------------------------

    def _apply_event(self, event: SimEvent, now_ms: int) -> None:
        if event.available_ms > now_ms:  # pragma: no cover - feed guarantees this
            raise LeakageDetected(
                f"feed yielded a {event.kind.value} event available at "
                f"{event.available_ms} while the clock reads {now_ms}"
            )

        if event.event_ticker:
            self._event_tickers[event.ticker] = event.event_ticker

        if event.kind is EventKind.BOOK:
            book = self._build_book(event)
            self.market.apply_snapshot(event.ticker, book, now_ms=now_ms, seq=event.seq)
            snap = self.market.ensure(event.ticker)
            if event.event_ticker:
                snap.event_ticker = event.event_ticker
        elif event.kind is EventKind.TRADE:
            price = int(event.payload["price_cc"])
            qty = int(event.payload["qty_qc"])
            self.market.apply_trade(event.ticker, price_cc=price, qty_qc=qty, now_ms=now_ms)
            for fill in self.fills.on_trade(
                ticker=event.ticker, price_cc=price, qty_qc=qty, now_ms=now_ms
            ):
                self._apply_fill(fill)
        elif event.kind is EventKind.TICKER:
            payload = event.payload
            self.market.apply_ticker(
                event.ticker,
                now_ms=now_ms,
                mark_price_cc=payload.get("mark_price_cc"),
                volume_qc=payload.get("volume_qc"),
                open_interest_qc=payload.get("open_interest_qc"),
                funding_rate=payload.get("funding_rate"),
                next_funding_ms=payload.get("next_funding_ms"),
            )
        elif event.kind is EventKind.FEATURE:
            store = self._feed_features.setdefault(event.ticker, {})
            for name, value in event.payload["values"].items():
                store[name] = TimedFeature(
                    name=name,
                    value=float(value),
                    info_ts_ms=event.info_ts_ms,
                    available_ms=event.available_ms,
                )
        elif event.kind is EventKind.STATUS:
            self.market.set_status(event.ticker, str(event.payload["status"]), now_ms=now_ms)
        elif event.kind is EventKind.SETTLEMENT:
            outcome = bool(event.payload["outcome_yes"])
            self._settlements[event.ticker] = outcome
            self.fills.cancel_all(event.ticker)
            if self.config.market_type is MarketType.EVENT:
                self.event_ledger.settle(event.ticker, outcome_yes=outcome, now_ms=now_ms)

    @staticmethod
    def _build_book(event: SimEvent) -> OrderBook:
        payload = event.payload
        if "yes_levels" in payload:
            return OrderBook.from_event_contract_levels(
                event.ticker,
                yes_levels=[(int(p), int(q)) for p, q in payload["yes_levels"]],
                no_levels=[(int(p), int(q)) for p, q in payload["no_levels"]],
                seq=event.seq,
                ts_ms=event.available_ms,
            )
        return OrderBook.from_two_sided_levels(
            event.ticker,
            bid_levels=[(int(p), int(q)) for p, q in payload.get("bids", [])],
            ask_levels=[(int(p), int(q)) for p, q in payload.get("asks", [])],
            seq=event.seq,
            ts_ms=event.available_ms,
        )

    def _next_arrival_ms(self) -> int | None:
        pending = [o.arrival_ms for o in self.fills.working_orders() if o.filled_qc == 0]
        arrivals = [ms for ms in pending if ms > self.clock.now_ms()]
        return min(arrivals) if arrivals else None

    def _apply_fill(self, fill: SimFill) -> None:
        if self.config.market_type is MarketType.EVENT:
            self.event_ledger.apply_fill(
                fill, event_ticker=self._event_tickers.get(fill.ticker, "")
            )
        else:
            self.perp_ledger.apply_fill(fill)
        for strategy in self.strategies:
            if strategy.strategy_id == fill.strategy_id:
                strategy.on_fill(fill.ticker, fill.qty_qc, fill.price_cc)

    # -- decisions -----------------------------------------------------------

    def _decide(self, now_ms: int) -> None:
        self.decisions_taken += 1
        for strategy in self.strategies:
            for ticker in self.market.tickers():
                if self._settlements.get(ticker) is not None:
                    continue
                snap = self.market.get(ticker)
                if snap is None:
                    continue
                if snap.status in ("paused", "closed", "settled"):
                    continue
                if not self.market.is_fresh(
                    ticker, now_ms=now_ms, max_age_ms=strategy.metadata.max_data_age_ms
                ):
                    self.skipped_stale += 1
                    continue
                if strategy.metadata.requires_book and (
                    snap.book is None or snap.book.is_crossed
                ):
                    self.skipped_stale += 1
                    continue

                ctx = self._build_context(strategy, snap, now_ms)
                if ctx is None:
                    continue
                if isinstance(strategy, TradingStrategy):
                    strategy.begin(ctx)
                intents = strategy.generate(ctx)
                for intent in intents:
                    self.intents_generated += 1
                    self._route(intent, snap, now_ms, strategy)

    def _build_context(
        self, strategy: Strategy, snap: MarketSnapshot, now_ms: int
    ) -> StrategyContext | None:
        """Assemble the strategy's view, checking every timestamp on the way in.

        Returns ``None`` when a leak was detected and the detector is not
        strict: the decision is skipped rather than taken on tainted input.
        """
        if not self.leakage.check_snapshot(snap, now_ms):
            return None

        features: dict[str, float] = {}
        for tf in self._feed_features.get(snap.ticker, {}).values():
            if self.leakage.check(
                kind="feature", name=tf.name, ticker=snap.ticker,
                info_ts_ms=tf.info_ts_ms, available_ms=tf.available_ms, now_ms=now_ms,
            ):
                features[tf.name] = tf.value
            else:
                return None

        if self._feature_provider is not None:
            supplied = self._feature_provider(snap.ticker, now_ms)
            if isinstance(supplied, FeatureVector):
                if not self.leakage.check_vector(supplied, now_ms):
                    return None
                features.update(supplied.as_dict())
            elif supplied:
                for tf in supplied:
                    if not self.leakage.check(
                        kind="feature", name=tf.name, ticker=snap.ticker,
                        info_ts_ms=tf.info_ts_ms, available_ms=tf.available_ms, now_ms=now_ms,
                    ):
                        return None
                    features[tf.name] = tf.value

        probability = model_id = model_version = model_target = None
        model_base_rate = None
        if self._model_provider is not None:
            prediction = self._model_provider(snap.ticker, now_ms)
            if prediction is not None:
                if not self.leakage.check(
                    kind="model", name=prediction.model_id or "model", ticker=snap.ticker,
                    info_ts_ms=prediction.info_ts_ms, available_ms=prediction.available_ms,
                    now_ms=now_ms,
                ):
                    return None
                probability = prediction.probability
                model_id = prediction.model_id
                model_version = prediction.model_version
                model_target = prediction.target
                model_base_rate = prediction.base_rate
                self.predictions.append(
                    PredictionRecord(
                        ticker=snap.ticker, ts_ms=now_ms, probability=probability,
                        source="model", strategy_id=strategy.strategy_id,
                    )
                )

        position_qc = 0
        position_age_ms: int | None = None
        if self.config.market_type is MarketType.EVENT:
            pos = self.event_ledger.positions.get(snap.ticker)
            position_qc = pos.qty_qc if pos else 0
            if pos is not None and pos.qty_qc and pos.opened_ms:
                position_age_ms = max(0, now_ms - pos.opened_ms)
        else:
            perp_pos = self.perp_ledger.positions.get(snap.ticker)
            position_qc = perp_pos.qty_qc if perp_pos else 0

        return StrategyContext(
            now_ms=now_ms,
            market=snap,
            features=features,
            model_probability=probability,
            model_target=model_target,
            model_base_rate=model_base_rate,
            model_id=model_id,
            model_version=model_version,
            position_qc=position_qc,
            position_age_ms=position_age_ms,
            resting_qty_qc=sum(
                o.remaining_qc for o in self.fills.working_orders(snap.ticker)
            ),
            fee_multiplier=self.config.fee_multiplier,
            # Supplied here for the same reason it is supplied live: on a delta
            # feed, silence on one market means the book has not changed, and
            # only something that sees every market at once can tell that from a
            # dropped connection. Leaving it unset made every backtest stricter
            # than the platform it is supposed to model -- 1,091,876 book events
            # produced one intent, with 229,761 strategy-level stale refusals,
            # so nothing could be validated against it.
            feed_live=self._feed_live(now_ms),
        )

    def _feed_live(self, now_ms: int) -> bool:
        """Whether the replayed feed is delivering at this simulated instant.

        The same test the live loop makes, against the same budget, on the same
        MarketState clock -- which in a replay advances with the events rather
        than with the wall clock.
        """
        age = self.market.feed_age_ms(now_ms=now_ms)
        return age is not None and age <= self.market.feed_max_age_ms

    def _route(
        self, intent: OrderIntent, snap: MarketSnapshot, now_ms: int, strategy: Strategy
    ) -> None:
        """Strategy -> risk engine -> execution. There is no other path."""
        fees = self._entry_fee_micros(intent)
        # Deliberately separate domains: the two ledgers return unrelated account
        # types and each is handed only to its own risk engine.
        state: Any
        if intent.market_type is MarketType.EVENT:
            state = self.event_ledger.account_state(
                market=self.market, fills=self.fills, now_ms=now_ms,
                orders_this_minute=self._orders_in_last_minute(now_ms),
            )
            verdict = self.event_risk.evaluate(
                intent, state, book=snap.book, now_ms=now_ms,
                event_ticker=snap.event_ticker or self._event_tickers.get(intent.ticker, ""),
                estimated_fees_micros=fees,
            )
        else:
            state = self.perp_ledger.account_state(
                market=self.market, fills=self.fills, now_ms=now_ms
            )
            verdict = self.perp_risk.evaluate(
                intent, state, book=snap.book, now_ms=now_ms, estimated_fees_micros=fees,
                funding_rate_bps_per_hour=(
                    Decimal(str(snap.funding_rate)) if snap.funding_rate is not None else None
                ),
            )

        code = verdict.reason_code if not verdict.approved else verdict.decision.value
        self.risk_decisions[code] = self.risk_decisions.get(code, 0) + 1
        if intent.fair_value_cc is not None:
            self.predictions.append(
                PredictionRecord(
                    ticker=intent.ticker, ts_ms=now_ms,
                    probability=intent.fair_value_cc / CC_PER_DOLLAR,
                    source="strategy_fair_value", strategy_id=strategy.strategy_id,
                )
            )
        if not verdict.approved:
            return

        order = self.fills.submit(
            verdict, now_ms=now_ms, reference_price_cc=self._reference_price_cc(verdict, snap)
        )
        if order is not None:
            self.orders_submitted += 1
            self._order_times.append(now_ms)

    @staticmethod
    def _reference_price_cc(verdict: RiskVerdict, snap: MarketSnapshot) -> int:
        """Touch price on the side the order intends to take, at decision time.

        Slippage is measured against this, so it captures both the book walk and
        whatever the book did during the order's flight.
        """
        intent = verdict.intent
        buying_yes = (
            intent.is_buy
            if intent.market_type is MarketType.PERP
            else intent.outcome_side is not OutcomeSide.NO
        )
        touch = snap.best_ask_cc if buying_yes else snap.best_bid_cc
        if touch is not None:
            return touch
        return intent.limit_price_cc if buying_yes else CC_PER_DOLLAR - intent.limit_price_cc

    def _entry_fee_micros(self, intent: OrderIntent) -> int:
        """Entry-leg fee estimate for the risk engine.

        Only the entry leg: a strategy's ``expected_value_micros`` has already
        been charged a full round trip by :mod:`kat.strategies.common`, and the
        risk engine subtracts this figure from it again. Charging the round trip
        here would double-count the exit and reject trades that clear their
        costs.
        """
        schedule = FeeSchedule.from_series("quadratic", self.config.fee_multiplier)
        return estimate_fee_micros(
            price_cc=intent.limit_price_cc, qty_qc=intent.qty_qc, schedule=schedule, is_taker=True
        ).fee_micros

    def _orders_in_last_minute(self, now_ms: int) -> int:
        while self._order_times and now_ms - self._order_times[0] > 60_000:
            self._order_times.popleft()
        return len(self._order_times)

    # -- accounting ----------------------------------------------------------

    def _sample_equity(self, now_ms: int, *, force: bool = False) -> None:
        if not force and now_ms < self._next_sample_ms:
            return
        if self.config.market_type is MarketType.EVENT:
            equity = self.event_ledger.equity_micros(self.market)
            cash = self.event_ledger.cash_micros
            exposure = self.event_ledger.exposure_micros()
            open_positions = len(self.event_ledger.positions)
        else:
            equity = self.perp_ledger.equity_micros(self.market)
            cash = self.perp_ledger.cash_micros
            exposure = self.perp_ledger.exposure_micros(self.market)
            open_positions = len(self.perp_ledger.positions)

        if self.equity_curve and self.equity_curve[-1].ts_ms == now_ms:
            self.equity_curve[-1] = EquityPoint(now_ms, equity, cash, exposure, open_positions)
        else:
            self.equity_curve.append(
                EquityPoint(now_ms, equity, cash, exposure, open_positions)
            )
        if not force:
            interval = self.config.equity_sample_interval_ms
            self._next_sample_ms = ((now_ms // interval) + 1) * interval

    def _flatten(self, now_ms: int) -> None:
        """Close open positions against the final book, at the price on it.

        Used only when the caller asked for it. The exit walks the book like any
        other order, so an oversized position pays for the levels it consumes.
        """
        self.fills.cancel_all()
        if self.config.market_type is MarketType.EVENT:
            positions = [(p.ticker, p.qty_qc) for p in self.event_ledger.positions.values()]
        else:
            positions = [(p.ticker, p.qty_qc) for p in self.perp_ledger.positions.values()]

        for ticker, qty in positions:
            if qty == 0:
                continue
            book = self.market.book(ticker)
            if book is None:
                continue
            is_buy = qty < 0
            size = (abs(qty) // QC_PER_CONTRACT) * QC_PER_CONTRACT
            if size <= 0:
                continue
            walked = book.walk(size, is_buy=is_buy)
            if walked is None:
                continue
            filled, notional, _ = walked
            if filled <= 0:
                continue
            price = -(-notional // filled) if is_buy else notional // filled
            fee_price = (
                CC_PER_DOLLAR - price
                if self.config.market_type is MarketType.EVENT and not is_buy
                else price
            )
            fill = SimFill(
                order_id=new_id("flat"),
                ticker=ticker,
                market_type=self.config.market_type,
                ts_ms=now_ms,
                is_buy=is_buy,
                qty_qc=filled,
                price_cc=price,
                notional_micros=notional,
                fee_micros=self.config.costs.fee_micros(
                    price_cc=fee_price, qty_qc=filled, is_taker=True
                ),
                is_taker=True,
                fee_price_cc=fee_price,
                slippage_cc=0,
                latency_ms=0,
                decision_ms=now_ms,
                strategy_id=None,
            )
            self.fills.fills.append(fill)
            self._apply_fill(fill)

    def _build_result(self, start_ms: int, end_ms: int) -> BacktestResult:
        ledger_trades: Sequence[Any] = (
            self.event_ledger.trades
            if self.config.market_type is MarketType.EVENT
            else self.perp_ledger.trades
        )
        pairs = self._calibration_pairs()
        metrics = compute_metrics(
            equity_curve=[(p.ts_ms, p.equity_micros) for p in self.equity_curve],
            trades=ledger_trades,
            period_ms=self.config.equity_sample_interval_ms,
            exposure=[(p.ts_ms, p.exposure_micros, p.equity_micros) for p in self.equity_curve],
            avg_slippage_cc=self.fills.report.avg_slippage_cc,
            fill_count=self.fills.report.fill_count,
            maker_fill_pct=self.fills.report.maker_fill_pct,
            total_fees_micros=self.fills.report.fees_micros,
            probabilities=[p for p, _ in pairs],
            outcomes=[o for _, o in pairs],
            seed=self.config.seed,
        )
        return BacktestResult(
            run_id=new_id("bt"),
            config=self.config,
            strategy_ids=tuple(s.strategy_id for s in self.strategies),
            market_type=self.config.market_type,
            start_ms=start_ms,
            end_ms=end_ms,
            is_synthetic_data=self.feed.is_synthetic,
            equity_curve=list(self.equity_curve),
            trades=list(ledger_trades),
            fills=list(self.fills.fills),
            metrics=metrics,
            leakage=self.leakage.report(),
            cost_report=self.fills.stats(),
            risk_decisions=dict(self.risk_decisions),
            intents_generated=self.intents_generated,
            orders_submitted=self.orders_submitted,
            decisions_taken=self.decisions_taken,
            skipped_stale=self.skipped_stale,
            strategy_health={s.strategy_id: s.health() for s in self.strategies},
            predictions=list(self.predictions),
            settlements=dict(self._settlements),
        )

    def _calibration_pairs(self) -> list[tuple[float, int]]:
        """Predicted YES probabilities paired with the outcome that occurred.

        Only markets that actually settled contribute. A prediction on a market
        whose outcome we never observed is not evidence about calibration, and
        scoring it against the last traded price would be scoring the model
        against itself.
        """
        pairs: list[tuple[float, int]] = []
        for record in self.predictions:
            outcome = self._settlements.get(record.ticker)
            if outcome is None:
                continue
            pairs.append((record.probability, 1 if outcome else 0))
        return pairs
