"""Typed models for the Kalshi perpetuals (margin) API.

Differences from event contracts that this module has to get right:

* **Pricing is decimal dollars, always.** There are no integer-cent fields on
  the margin API; ``UseDollars`` is permanently on.
* **Timestamps are Unix milliseconds with an ``_ms`` suffix**, not RFC3339.
* **Positions are signed** -- positive long, negative short -- and both
  directions consume exposure.
* **Balance carries margin state**: ``account_equity``, ``maintenance_margin``,
  ``initial_margin``, ``resting_orders_margin`` and ``available_balance``, with
  documented cases where several of them are zero.

Leverage note
-------------
``GetMarginRiskResponse.account_leverage`` is
``total_position_notional / total_maintenance_margin`` -- a margin-utilisation
ratio, not the equity-based leverage this platform caps at 1.00x. It is parsed
and stored as :attr:`PerpRiskSnapshot.exchange_account_leverage` for
cross-checking and display, and is never used as the invariant. See
:mod:`kat.risk.invariants`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kat.common.enums import BookSide, OrderState
from kat.common.fixedpoint import (
    Rounding,
    parse_count_qc,
    parse_dollars_micros,
    parse_price_cc,
)
from kat.common.timeutil import from_ms, parse_rfc3339
from kat.marketdata.book import OrderBook

__all__ = [
    "PerpBalance",
    "PerpFill",
    "PerpFundingRate",
    "PerpMarket",
    "PerpOrder",
    "PerpPositionSnapshot",
    "PerpRiskParameters",
    "PerpRiskPosition",
    "PerpRiskSnapshot",
    "PerpSubaccountBalance",
    "parse_margin_orderbook",
]


def _price(value: Any, *, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return parse_price_cc(value, rounding=Rounding.HALF_EVEN, field="price")
    except ValueError:
        return default


def _count(value: Any, *, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return parse_count_qc(value, rounding=Rounding.HALF_EVEN, field="count")
    except ValueError:
        return default


def _money(value: Any, *, rounding: str = Rounding.HALF_EVEN, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return parse_dollars_micros(value, rounding=rounding, field="amount")
    except ValueError:
        return default


def _ts(payload: dict[str, Any], *keys: str) -> datetime | None:
    """Read a margin timestamp, preferring ``*_ms`` integer fields."""
    for key in keys:
        value = payload.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, int | float):
            return from_ms(int(value))
        if isinstance(value, str):
            try:
                return parse_rfc3339(value)
            except ValueError:
                continue
    return None


@dataclass(slots=True)
class PerpMarket:
    """A perpetual market.

    ``contract_size`` is retained because notional is strictly
    ``|qty| * price * contract_size``. Kalshi's own risk endpoint documents
    ``position_notional`` as ``|qty| * mark_price``, which implies a contract
    size of 1 for current markets -- but the field exists, so we carry it and
    let :mod:`kat.risk.perp_risk` take the more conservative of the two figures
    rather than assuming they always agree.
    """

    ticker: str
    status: str
    title: str | None = None
    contract_size: str | None = None
    tick_size_cc: int | None = None
    fractional_trading_enabled: bool = False
    exchange_index: int | None = None
    mark_price_cc: int | None = None
    last_price_cc: int | None = None
    bid_cc: int | None = None
    ask_cc: int | None = None
    volume_qc: int | None = None
    open_interest_qc: int | None = None
    # Kalshi's own estimate of achievable leverage (1 / margin_rate). Recorded
    # for display; it is not an input to our limit.
    exchange_leverage_estimate: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> PerpMarket:
        return cls(
            ticker=payload.get("ticker", ""),
            status=str(payload.get("status", "unknown")),
            title=payload.get("title"),
            contract_size=payload.get("contract_size"),
            tick_size_cc=_price(payload.get("tick_size")),
            fractional_trading_enabled=bool(payload.get("fractional_trading_enabled")),
            exchange_index=payload.get("exchange_index"),
            mark_price_cc=_price(
                payload.get("mark_price") or payload.get("settlement_mark_price")
            ),
            last_price_cc=_price(payload.get("last_price")),
            bid_cc=_price(payload.get("bid") or payload.get("yes_bid")),
            ask_cc=_price(payload.get("ask") or payload.get("yes_ask")),
            volume_qc=_count(payload.get("volume"), default=0) or None,
            open_interest_qc=_count(payload.get("open_interest"), default=0) or None,
            exchange_leverage_estimate=payload.get("leverage_estimate"),
            raw=payload,
        )

    @property
    def is_tradeable(self) -> bool:
        return self.status in ("active", "open")

    def contract_multiplier_micros(self) -> int:
        """Contract size scaled to micro-units, defaulting to exactly 1.0.

        A missing or unparseable contract size falls back to 1.0 rather than 0:
        a zero multiplier would compute every position's notional as zero and
        make the leverage check vacuous, which is the one failure mode this
        platform must not have.
        """
        if not self.contract_size:
            return 1_000_000
        value = _money(self.contract_size, rounding=Rounding.CEIL)
        if not value or value <= 0:
            return 1_000_000
        return value


@dataclass(slots=True)
class PerpOrder:
    order_id: str
    client_order_id: str
    ticker: str
    book_side: BookSide | None
    price_cc: int | None
    fill_qc: int
    remaining_qc: int
    last_update_reason: str | None
    created_time: datetime | None
    last_update_time: datetime | None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> PerpOrder:
        side = payload.get("side")
        return cls(
            order_id=payload.get("order_id", ""),
            client_order_id=payload.get("client_order_id", ""),
            ticker=payload.get("ticker", ""),
            book_side=BookSide(side) if side in ("bid", "ask") else None,
            price_cc=_price(payload.get("price")),
            fill_qc=_count(payload.get("fill_count")),
            remaining_qc=_count(payload.get("remaining_count")),
            last_update_reason=payload.get("last_update_reason"),
            created_time=_ts(payload, "created_ts_ms", "created_time"),
            last_update_time=_ts(payload, "last_updated_ts_ms", "last_update_time"),
            raw=payload,
        )

    def to_internal_state(self) -> OrderState:
        """Map the margin order's reported state onto our state machine.

        The margin API expresses order state through ``last_update_reason``
        rather than an explicit status field, so this reads that. Anything
        unrecognised becomes ``UNKNOWN``, which still counts against exposure.
        """
        reason = (self.last_update_reason or "").lower()
        if self.remaining_qc == 0 and self.fill_qc > 0:
            return OrderState.FILLED
        if "cancel" in reason:
            return OrderState.CANCELLED
        if "reject" in reason:
            return OrderState.REJECTED
        if "expire" in reason:
            return OrderState.EXPIRED
        if self.fill_qc > 0:
            return OrderState.PARTIALLY_FILLED
        if self.remaining_qc > 0:
            return OrderState.ACKNOWLEDGED
        return OrderState.UNKNOWN


@dataclass(slots=True)
class PerpFill:
    fill_id: str
    order_id: str
    ticker: str
    book_side: BookSide | None
    count_qc: int
    price_cc: int | None
    is_taker: bool | None
    fee_micros: int
    ts: datetime | None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> PerpFill:
        side = payload.get("side")
        return cls(
            fill_id=payload.get("fill_id") or payload.get("trade_id", ""),
            order_id=payload.get("order_id", ""),
            ticker=payload.get("ticker", ""),
            book_side=BookSide(side) if side in ("bid", "ask") else None,
            count_qc=_count(payload.get("count")),
            price_cc=_price(payload.get("price")),
            is_taker=payload.get("is_taker"),
            fee_micros=_money(payload.get("fee"), rounding=Rounding.CEIL, default=0) or 0,
            ts=_ts(payload, "ts_ms", "created_time"),
            raw=payload,
        )


@dataclass(slots=True)
class PerpPositionSnapshot:
    """A perpetual position. ``position_qc`` is signed: + long, - short."""

    ticker: str
    subaccount: int
    position_qc: int
    entry_price_cc: int | None
    unrealized_pnl_micros: int
    margin_used_micros: int | None
    fees_micros: int
    roe: float | None
    is_portfolio: bool
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> PerpPositionSnapshot:
        return cls(
            ticker=payload.get("market_ticker", ""),
            subaccount=int(payload.get("subaccount", 0)),
            position_qc=_count(payload.get("position")),
            entry_price_cc=_price(payload.get("entry_price")),
            unrealized_pnl_micros=_money(payload.get("unrealized_pnl"), default=0) or 0,
            margin_used_micros=_money(payload.get("margin_used"), rounding=Rounding.CEIL),
            fees_micros=_money(payload.get("fees"), rounding=Rounding.CEIL, default=0) or 0,
            roe=payload.get("roe"),
            is_portfolio=bool(payload.get("is_portfolio")),
            raw=payload,
        )

    @property
    def is_long(self) -> bool:
        return self.position_qc > 0

    @property
    def is_flat(self) -> bool:
        return self.position_qc == 0


@dataclass(slots=True)
class PerpSubaccountBalance:
    """Per-subaccount margin balance.

    Kalshi documents ``account_equity`` and ``initial_margin`` as 0 for
    self-clearing members, and ``available_balance`` as 0 unless
    ``compute_available_balance=true`` was requested. Those zeros are *absences*,
    not real values, so :attr:`equity_is_reported` distinguishes them -- a zero
    equity must never be silently used as the denominator of a leverage check.
    """

    subaccount: int
    position_value_micros: int
    account_equity_micros: int
    maintenance_margin_micros: int
    initial_margin_micros: int
    resting_orders_margin_micros: int
    available_balance_micros: int
    equity_is_reported: bool
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> PerpSubaccountBalance:
        equity_raw = payload.get("account_equity")
        equity = _money(equity_raw, rounding=Rounding.FLOOR, default=0) or 0
        return cls(
            subaccount=int(payload.get("subaccount", 0)),
            # Position value can be negative (a losing short); do not floor to 0.
            position_value_micros=_money(payload.get("position_value"), default=0) or 0,
            account_equity_micros=equity,
            maintenance_margin_micros=_money(
                payload.get("maintenance_margin"), rounding=Rounding.CEIL, default=0
            ) or 0,
            initial_margin_micros=_money(
                payload.get("initial_margin"), rounding=Rounding.CEIL, default=0
            ) or 0,
            resting_orders_margin_micros=_money(
                payload.get("resting_orders_margin"), rounding=Rounding.CEIL, default=0
            ) or 0,
            available_balance_micros=_money(
                payload.get("available_balance"), rounding=Rounding.FLOOR, default=0
            ) or 0,
            equity_is_reported=equity_raw not in (None, "") and equity > 0,
            raw=payload,
        )


@dataclass(slots=True)
class PerpBalance:
    """Whole margin account balance across subaccounts."""

    settled_funds_micros: int
    subaccounts: list[PerpSubaccountBalance] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> PerpBalance:
        return cls(
            settled_funds_micros=_money(
                payload.get("settled_funds"), rounding=Rounding.FLOOR, default=0
            ) or 0,
            subaccounts=[
                PerpSubaccountBalance.parse(s) for s in payload.get("subaccount_balances") or []
            ],
            raw=payload,
        )

    def subaccount(self, number: int = 0) -> PerpSubaccountBalance | None:
        for s in self.subaccounts:
            if s.subaccount == number:
                return s
        return None

    @property
    def total_position_value_micros(self) -> int:
        return sum(s.position_value_micros for s in self.subaccounts)

    @property
    def total_maintenance_margin_micros(self) -> int:
        return sum(s.maintenance_margin_micros for s in self.subaccounts)


@dataclass(slots=True)
class PerpRiskPosition:
    subaccount: int
    ticker: str
    position_qc: int
    mark_price_cc: int | None
    position_notional_micros: int
    maintenance_margin_micros: int | None
    position_leverage: float | None
    estimated_liquidation_price_cc: int | None
    is_portfolio: bool

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> PerpRiskPosition:
        return cls(
            subaccount=int(payload.get("subaccount", 0)),
            ticker=payload.get("market_ticker", ""),
            position_qc=_count(payload.get("position")),
            mark_price_cc=_price(payload.get("mark_price")),
            position_notional_micros=_money(
                payload.get("position_notional"), rounding=Rounding.CEIL, default=0
            ) or 0,
            maintenance_margin_micros=_money(
                payload.get("maintenance_margin_required"), rounding=Rounding.CEIL
            ),
            position_leverage=payload.get("position_leverage"),
            estimated_liquidation_price_cc=_price(payload.get("estimated_liquidation_price")),
            is_portfolio=bool(payload.get("is_portfolio")),
        )


@dataclass(slots=True)
class PerpRiskSnapshot:
    """``GET /margin/risk``: the exchange's view of our margin risk.

    Used as an independent cross-check on locally computed exposure, never as
    the leverage limit itself.
    """

    total_position_notional_micros: int
    total_maintenance_margin_micros: int
    exchange_account_leverage: float | None
    positions: list[PerpRiskPosition] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> PerpRiskSnapshot:
        return cls(
            total_position_notional_micros=_money(
                payload.get("total_position_notional"), rounding=Rounding.CEIL, default=0
            ) or 0,
            total_maintenance_margin_micros=_money(
                payload.get("total_maintenance_margin"), rounding=Rounding.CEIL, default=0
            ) or 0,
            # NOTE: notional / maintenance-margin, NOT notional / equity.
            exchange_account_leverage=payload.get("account_leverage"),
            positions=[PerpRiskPosition.parse(p) for p in payload.get("positions") or []],
            raw=payload,
        )


@dataclass(slots=True)
class PerpRiskParameters:
    """System-wide margin parameters from ``GET /margin/risk_parameters``."""

    liquidation_margin_ratio_threshold: float
    queue_entry_margin_ratio_threshold: float
    initial_margin_multiplier: dict[str, float] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> PerpRiskParameters:
        return cls(
            liquidation_margin_ratio_threshold=float(
                payload.get("liquidation_margin_ratio_threshold", 0.0)
            ),
            queue_entry_margin_ratio_threshold=float(
                payload.get("queue_entry_margin_ratio_threshold", 0.0)
            ),
            initial_margin_multiplier=payload.get("initial_margin_multiplier") or {},
        )


@dataclass(slots=True)
class PerpFundingRate:
    ticker: str
    rate: float | None
    next_funding_ts: datetime | None
    ts: datetime | None
    is_estimate: bool
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any], *, is_estimate: bool = False) -> PerpFundingRate:
        rate = payload.get("funding_rate")
        if isinstance(rate, dict):
            rate = rate.get("rate")
        return cls(
            ticker=payload.get("ticker") or payload.get("market_ticker", ""),
            rate=float(rate) if rate is not None else None,
            next_funding_ts=_ts(payload, "next_funding_time_ms", "next_funding_time"),
            ts=_ts(payload, "ts_ms", "ts"),
            is_estimate=is_estimate,
            raw=payload,
        )


def parse_margin_orderbook(ticker: str, payload: dict[str, Any], *, ts_ms: int = 0) -> OrderBook:
    """Parse a margin orderbook, which is a conventional two-sided book.

    Unlike event contracts there is no YES/NO reflection here: ``bids`` and
    ``asks`` mean what they say.
    """
    book = payload.get("orderbook") or {}
    return OrderBook.from_two_sided_levels(
        ticker,
        _levels(book.get("bids")),
        _levels(book.get("asks")),
        ts_ms=ts_ms,
    )


def _levels(raw: Any) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for level in raw or []:
        try:
            price = parse_price_cc(level[0], rounding=Rounding.HALF_EVEN, field="level_price")
            size = parse_count_qc(level[1], rounding=Rounding.HALF_EVEN, field="level_size")
        except (ValueError, IndexError, TypeError):
            continue
        if size > 0:
            out.append((price, size))
    return out
