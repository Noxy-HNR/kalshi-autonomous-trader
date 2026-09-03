"""Polymarket US market and book types, mapped onto the platform's scales.

The exchange quotes decimal dollars as strings and this platform holds prices as
integer centicents, so every conversion happens here rather than at each call
site. The scales line up without loss: a Polymarket tick is $0.001, which is
exactly 10 centicents, and the platform's price scale is 1e-4 dollars.

That tick is worth pausing on. Kalshi's minimum increment is a whole cent, which
on a three-cent contract is a third of its notional and is the single fact that
made every cheap market on that board untradeable. A tenth of a cent is a tenth
of that obstacle, and it is why the same strategies deserve re-measuring here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.marketdata.book import OrderBook
from kat.polymarket.fees import PolymarketFeeSchedule

__all__ = ["PolymarketMarket", "parse_book", "price_cc", "qty_qc"]


def price_cc(value: Any) -> int | None:
    """A decimal-dollar string to centicents, or None when absent.

    Prices arrive either bare (``"0.1540"``) or wrapped in a money object
    (``{"value": "0.1540", "currency": "USD"}``); both shapes appear in the same
    payload, so both are accepted here rather than at each reader.
    """
    if isinstance(value, dict):
        value = value.get("value")
    if value in (None, ""):
        return None
    try:
        return round(float(value) * CC_PER_DOLLAR)
    except (TypeError, ValueError):
        return None


def qty_qc(value: Any) -> int:
    """Contracts to centi-contracts. Polymarket quotes whole-contract decimals."""
    if value in (None, ""):
        return 0
    try:
        return round(float(value) * 100)
    except (TypeError, ValueError):
        return 0


@dataclass(slots=True)
class PolymarketMarket:
    """One tradeable outcome, with the terms needed to price a trade in it."""

    slug: str
    market_id: str
    question: str
    category: str
    status: str
    closed: bool
    active: bool
    end_date: str | None
    tick_size_cc: int
    min_trade_qc: int
    fees: PolymarketFeeSchedule
    outcomes: list[tuple[str, int | None]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> PolymarketMarket:
        sides = payload.get("marketSides") or []
        tick = price_cc(payload.get("orderPriceMinTickSize"))
        return cls(
            slug=str(payload.get("slug") or ""),
            market_id=str(payload.get("id") or ""),
            question=str(payload.get("question") or ""),
            category=str(payload.get("category") or ""),
            status=str(payload.get("status") or ""),
            closed=bool(payload.get("closed")),
            active=bool(payload.get("active")),
            end_date=payload.get("endDate"),
            # A market that does not publish a tick is not assumed to have the
            # finest one on the venue; the coarser cent is the safe reading.
            tick_size_cc=tick if tick else 100,
            min_trade_qc=qty_qc(payload.get("minimumTradeQty") or 1),
            fees=PolymarketFeeSchedule.from_market(payload),
            outcomes=[
                (str(s.get("description") or ""), price_cc(s.get("price")))
                for s in sides
            ],
            raw=payload,
        )

    @property
    def is_tradeable(self) -> bool:
        return self.active and not self.closed


def parse_book(slug: str, payload: dict[str, Any], *, ts_ms: int = 0) -> OrderBook | None:
    """Build the platform's book from a ``/v1/markets/{slug}/book`` response.

    Returns None when the book is one-sided or empty. A one-sided book cannot be
    priced, and inventing the missing half is how a strategy ends up quoting
    against a market that is not there -- the same refusal the Kalshi path makes.

    The payload nests everything under ``marketData``; reading the top level
    finds nothing and silently reports an empty book, which is how the first
    survey written against this API concluded there were no two-sided markets on
    the entire venue.
    """
    data = payload.get("marketData") or payload
    bids, asks = [], []
    for level in data.get("bids") or []:
        p, q = price_cc(level.get("px")), qty_qc(level.get("qty"))
        if p is not None and q > 0:
            bids.append((p, q))
    for level in data.get("offers") or []:
        p, q = price_cc(level.get("px")), qty_qc(level.get("qty"))
        if p is not None and q > 0:
            asks.append((p, q))
    if not bids or not asks:
        return None
    bids.sort(key=lambda lv: -lv[0])
    asks.sort(key=lambda lv: lv[0])
    return OrderBook.from_two_sided_levels(
        slug, bid_levels=bids, ask_levels=asks, ts_ms=ts_ms
    )
