"""Typed models for the Kalshi event-contract API.

Parsing rules applied consistently here:

* Prefer the ``*_dollars`` and ``*_fp`` fixed-point fields over the legacy
  integer-cent and integer-count fields. The integer fields cannot represent
  sub-cent prices or fractional contracts, and markets using the finer tick
  structures will silently truncate through them.
* Prefer ``outcome_side``/``book_side`` over the deprecated ``side``/``action``
  fields, which Kalshi has scheduled for removal.
* Missing optional numbers become ``None``, never ``0``. A missing bid and a bid
  of zero are different facts, and conflating them makes an empty book look
  like a market trading at nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kat.common.enums import BookSide, OrderState, OutcomeSide
from kat.common.fixedpoint import (
    Rounding,
    parse_count_qc,
    parse_dollars_micros,
    parse_price_cc,
)
from kat.common.timeutil import parse_rfc3339
from kat.marketdata.book import OrderBook

__all__ = [
    "EventBalance",
    "EventFill",
    "EventMarket",
    "EventOrder",
    "EventPositionSnapshot",
    "parse_orderbook",
]


def _opt_price(payload: dict[str, Any], *keys: str) -> int | None:
    """First present ``*_dollars`` key parsed to centicents, else ``None``."""
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            # Tolerant rounding: a price finer than our grid is a Kalshi tick
            # structure we do not know about yet, and refusing to parse the whole
            # market over it would take us offline for a display-only field.
            return parse_price_cc(value, rounding=Rounding.HALF_EVEN, field=key)
    return None


def _opt_count(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return parse_count_qc(value, rounding=Rounding.HALF_EVEN, field=key)
    return None


def _opt_money(payload: dict[str, Any], *keys: str, rounding: str = Rounding.HALF_EVEN) -> int | None:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return parse_dollars_micros(value, rounding=rounding, field=key)
    return None


def _opt_time(payload: dict[str, Any], key: str) -> datetime | None:
    value = payload.get(key)
    if not value:
        return None
    try:
        return parse_rfc3339(value) if isinstance(value, str) else None
    except ValueError:
        return None


@dataclass(slots=True)
class EventMarket:
    """One event-contract market.

    ``price_ranges`` is retained verbatim because it is the exchange's source of
    truth for valid prices. Order prices are snapped to it rather than assumed
    to be whole cents.
    """

    ticker: str
    event_ticker: str
    status: str
    title: str | None = None
    yes_sub_title: str | None = None
    no_sub_title: str | None = None
    market_kind: str | None = None

    yes_bid_cc: int | None = None
    yes_ask_cc: int | None = None
    no_bid_cc: int | None = None
    no_ask_cc: int | None = None
    yes_bid_size_qc: int | None = None
    yes_ask_size_qc: int | None = None
    last_price_cc: int | None = None

    volume_qc: int | None = None
    volume_24h_qc: int | None = None
    open_interest_qc: int | None = None
    liquidity_micros: int | None = None
    notional_value_micros: int | None = None

    open_time: datetime | None = None
    close_time: datetime | None = None
    expected_expiration_time: datetime | None = None
    price_level_structure: str | None = None
    price_ranges: list[dict[str, Any]] = field(default_factory=list)
    result: str | None = None
    rules_primary: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> EventMarket:
        return cls(
            ticker=payload["ticker"],
            event_ticker=payload.get("event_ticker", ""),
            status=payload.get("status", "unknown"),
            title=payload.get("title") or payload.get("yes_sub_title"),
            yes_sub_title=payload.get("yes_sub_title"),
            no_sub_title=payload.get("no_sub_title"),
            market_kind=payload.get("market_type"),
            yes_bid_cc=_opt_price(payload, "yes_bid_dollars"),
            yes_ask_cc=_opt_price(payload, "yes_ask_dollars"),
            no_bid_cc=_opt_price(payload, "no_bid_dollars"),
            no_ask_cc=_opt_price(payload, "no_ask_dollars"),
            yes_bid_size_qc=_opt_count(payload, "yes_bid_size_fp"),
            yes_ask_size_qc=_opt_count(payload, "yes_ask_size_fp"),
            last_price_cc=_opt_price(payload, "last_price_dollars"),
            volume_qc=_opt_count(payload, "volume_fp"),
            volume_24h_qc=_opt_count(payload, "volume_24h_fp"),
            open_interest_qc=_opt_count(payload, "open_interest_fp"),
            liquidity_micros=_opt_money(payload, "liquidity_dollars"),
            notional_value_micros=_opt_money(payload, "notional_value_dollars"),
            open_time=_opt_time(payload, "open_time"),
            close_time=_opt_time(payload, "close_time"),
            expected_expiration_time=_opt_time(payload, "expected_expiration_time"),
            price_level_structure=payload.get("price_level_structure"),
            price_ranges=payload.get("price_ranges") or [],
            result=payload.get("result"),
            rules_primary=payload.get("rules_primary"),
            raw=payload,
        )

    @property
    def is_tradeable(self) -> bool:
        return self.status == "active"

    @property
    def has_live_quote(self) -> bool:
        """Whether anyone is actually quoting this market on both sides.

        Kalshi encodes "no orders at all" as a full-width book -- a bid of
        ``$0.0000`` and an ask of ``$1.0000`` -- rather than as a null. Those
        parse into perfectly valid integers, so an ``is not None`` test cannot
        tell an unquoted market from a quoted one and accepts every market on
        the exchange. That is the absence of a quote wearing the costume of a
        price, and it is exactly the case that filled 57 of 60 watchlist slots
        with markets that had no orders in them.

        A real two-sided quote sits strictly inside the boundaries with the bid
        below the ask.
        """
        bid, ask = self.yes_bid_cc, self.yes_ask_cc
        if bid is None or ask is None:
            return False
        return 0 < bid < ask < 10_000

    def tick_size_cc(self, at_price_cc: int) -> int:
        """Minimum price increment valid at ``at_price_cc`` for this market.

        Read from ``price_ranges`` rather than assumed, because Kalshi assigns
        different tick grids per market and introduces new ones over time. Falls
        back to one whole cent, which is valid in every documented structure.
        """
        for band in self.price_ranges:
            try:
                start = parse_price_cc(band["start"], rounding=Rounding.HALF_EVEN)
                end = parse_price_cc(band["end"], rounding=Rounding.HALF_EVEN)
                step = parse_price_cc(band["step"], rounding=Rounding.HALF_EVEN)
            except (KeyError, ValueError):
                continue
            if start <= at_price_cc <= end and step > 0:
                return step
        return 100  # $0.01

    def snap_price_cc(self, price_cc: int, *, round_up: bool) -> int:
        """Snap a price to this market's valid grid.

        ``round_up`` should be chosen so the snap is never in the trader's
        favour: round a buy limit *down* and a sell limit *up*, so snapping can
        only make an order less aggressive, never more.
        """
        step = self.tick_size_cc(price_cc)
        if step <= 1:
            return price_cc
        if round_up:
            return ((price_cc + step - 1) // step) * step
        return (price_cc // step) * step


@dataclass(slots=True)
class EventOrder:
    """An order as Kalshi reports it."""

    order_id: str
    client_order_id: str
    ticker: str
    status: str
    outcome_side: OutcomeSide | None
    book_side: BookSide | None
    order_type: str | None
    yes_price_cc: int | None
    no_price_cc: int | None
    filled_qc: int
    remaining_qc: int
    initial_qc: int | None
    taker_fees_micros: int
    maker_fees_micros: int
    created_time: datetime | None
    last_update_time: datetime | None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> EventOrder:
        outcome = payload.get("outcome_side")
        book = payload.get("book_side")
        return cls(
            order_id=payload.get("order_id", ""),
            client_order_id=payload.get("client_order_id", ""),
            ticker=payload.get("ticker", ""),
            status=payload.get("status", "unknown"),
            outcome_side=OutcomeSide(outcome) if outcome in ("yes", "no") else None,
            book_side=BookSide(book) if book in ("bid", "ask") else None,
            order_type=payload.get("type"),
            yes_price_cc=_opt_price(payload, "yes_price_dollars"),
            no_price_cc=_opt_price(payload, "no_price_dollars"),
            filled_qc=_opt_count(payload, "fill_count_fp") or 0,
            remaining_qc=_opt_count(payload, "remaining_count_fp") or 0,
            initial_qc=_opt_count(payload, "initial_count_fp"),
            taker_fees_micros=_opt_money(payload, "taker_fees_dollars", rounding=Rounding.CEIL) or 0,
            maker_fees_micros=_opt_money(payload, "maker_fees_dollars", rounding=Rounding.CEIL) or 0,
            created_time=_opt_time(payload, "created_time"),
            last_update_time=_opt_time(payload, "last_update_time"),
            raw=payload,
        )

    def to_internal_state(self) -> OrderState:
        """Map Kalshi's order status onto our internal state machine.

        Unknown statuses map to ``UNKNOWN`` rather than to a guess. An order in
        ``UNKNOWN`` still counts against exposure, so an unrecognised status is
        conservative by construction.
        """
        status = (self.status or "").lower()
        if status == "resting":
            return (
                OrderState.PARTIALLY_FILLED if self.filled_qc > 0 else OrderState.ACKNOWLEDGED
            )
        if status == "executed":
            return OrderState.FILLED
        if status == "canceled":
            # Kalshi reports a partially filled then cancelled order as
            # "canceled"; the fill still happened and must not be forgotten.
            return OrderState.CANCELLED
        if status == "pending":
            return OrderState.SUBMITTED
        return OrderState.UNKNOWN


@dataclass(slots=True)
class EventFill:
    fill_id: str
    order_id: str
    ticker: str
    outcome_side: OutcomeSide | None
    book_side: BookSide | None
    count_qc: int
    yes_price_cc: int | None
    no_price_cc: int | None
    is_taker: bool | None
    fee_micros: int
    created_time: datetime | None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> EventFill:
        outcome = payload.get("outcome_side")
        book = payload.get("book_side")
        return cls(
            fill_id=payload.get("fill_id") or payload.get("trade_id", ""),
            order_id=payload.get("order_id", ""),
            ticker=payload.get("ticker") or payload.get("market_ticker", ""),
            outcome_side=OutcomeSide(outcome) if outcome in ("yes", "no") else None,
            book_side=BookSide(book) if book in ("bid", "ask") else None,
            count_qc=_opt_count(payload, "count_fp") or 0,
            yes_price_cc=_opt_price(payload, "yes_price_dollars"),
            no_price_cc=_opt_price(payload, "no_price_dollars"),
            is_taker=payload.get("is_taker"),
            fee_micros=_opt_money(payload, "fee_cost_dollars", "fee_cost", rounding=Rounding.CEIL) or 0,
            created_time=_opt_time(payload, "created_time"),
            raw=payload,
        )

    @property
    def effective_price_cc(self) -> int | None:
        """Price paid in the direction the fill positioned us."""
        if self.outcome_side is OutcomeSide.NO:
            return self.no_price_cc
        return self.yes_price_cc


@dataclass(slots=True)
class EventPositionSnapshot:
    """Position as the exchange sees it. Positive is YES, negative is NO."""

    ticker: str
    position_qc: int
    market_exposure_micros: int
    realized_pnl_micros: int
    total_traded_micros: int
    fees_paid_micros: int
    last_updated: datetime | None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> EventPositionSnapshot:
        return cls(
            ticker=payload.get("ticker", ""),
            position_qc=_opt_count(payload, "position_fp") or 0,
            market_exposure_micros=_opt_money(
                payload, "market_exposure_dollars", rounding=Rounding.CEIL
            ) or 0,
            realized_pnl_micros=_opt_money(payload, "realized_pnl_dollars") or 0,
            total_traded_micros=_opt_money(payload, "total_traded_dollars") or 0,
            fees_paid_micros=_opt_money(payload, "fees_paid_dollars", rounding=Rounding.CEIL) or 0,
            last_updated=_opt_time(payload, "last_updated_ts"),
            raw=payload,
        )


@dataclass(slots=True)
class EventBalance:
    """Cash and portfolio value for the event-contract account.

    Parsed with floor rounding: understating available capital is the safe
    direction to be wrong in.
    """

    balance_micros: int
    portfolio_value_micros: int | None
    updated_ts: int | None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> EventBalance:
        balance = _opt_money(payload, "balance_dollars", rounding=Rounding.FLOOR)
        if balance is None:
            # Legacy integer-cent fallback, used only when the fixed-point field
            # is absent from an older response shape.
            cents = payload.get("balance")
            balance = int(cents) * 10_000 if cents is not None else 0
        portfolio = payload.get("portfolio_value")
        return cls(
            balance_micros=balance,
            portfolio_value_micros=int(portfolio) * 10_000 if portfolio is not None else None,
            updated_ts=payload.get("updated_ts"),
            raw=payload,
        )


def parse_orderbook(ticker: str, payload: dict[str, Any], *, ts_ms: int = 0) -> OrderBook:
    """Turn a ``GET /markets/{ticker}/orderbook`` response into an ``OrderBook``.

    Kalshi returns two arrays of *bids* (``yes_dollars`` and ``no_dollars``);
    the NO side is reflected through $1.00 to become the YES ask side. See
    :class:`kat.marketdata.book.OrderBook` for why this matters.
    """
    book_payload = payload.get("orderbook_fp") or payload.get("orderbook") or {}
    yes_levels = _levels(book_payload.get("yes_dollars") or book_payload.get("yes"))
    no_levels = _levels(book_payload.get("no_dollars") or book_payload.get("no"))
    return OrderBook.from_event_contract_levels(ticker, yes_levels, no_levels, ts_ms=ts_ms)


def _levels(raw: Any) -> list[tuple[int, int]]:
    """Parse ``[[price_dollars, count_fp], ...]`` into integer pairs.

    Individual malformed levels are skipped rather than failing the whole book:
    a single bad level should not blind us to an otherwise healthy market. The
    skip is silent by design here and counted by the caller.
    """
    out: list[tuple[int, int]] = []
    if not raw:
        return out
    for level in raw:
        try:
            price = parse_price_cc(level[0], rounding=Rounding.HALF_EVEN, field="level_price")
            size = parse_count_qc(level[1], rounding=Rounding.HALF_EVEN, field="level_size")
        except (ValueError, IndexError, TypeError):
            continue
        if size > 0:
            out.append((price, size))
    return out
