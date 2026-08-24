"""Market-data ingestion.

Feeds :class:`~kat.marketdata.state.MarketState` from Kalshi's WebSocket, and
persists what it sees through the async writer without ever putting the database
between a message and the in-memory state.

Order of operations per message::

    message -> update in-memory state (synchronous, microseconds)
             -> enqueue a row for persistence (non-blocking)

Never the reverse. The trading loop reads the same in-memory state, so a slow
disk delays history, not decisions.

Snapshot-then-delta
-------------------
Kalshi sends an ``orderbook_snapshot`` before any ``orderbook_delta``. A delta
that arrives for a market we hold no snapshot for is dropped and a fresh
snapshot requested, rather than applied to a partial book. The same happens when
a delta leaves the book crossed, which means a message was missed: the book is
discarded rather than trusted.

Watchlist
---------
The platform cannot subscribe to every Kalshi market and should not try. The
watchlist is chosen by liquidity and time-to-close and is refreshed on a slow
cycle, with subscriptions updated incrementally rather than torn down.
"""

from __future__ import annotations

import asyncio
from typing import Any

from kat.common.enums import MarketType
from kat.common.fixedpoint import Rounding, parse_count_qc, parse_price_cc
from kat.common.timeutil import Clock
from kat.db.models import Quote, TradePrint
from kat.db.writer import AsyncWriter
from kat.kalshi.event_contracts.rest import EventContractsREST
from kat.kalshi.perps.rest import PerpsREST
from kat.kalshi.ws import KalshiWebSocket
from kat.marketdata.state import MarketState
from kat.observability.logging import get_logger
from kat.observability.metrics import M_WS_MESSAGES, metrics

log = get_logger(__name__)

__all__ = ["MarketDataIngestor", "select_event_watchlist"]

#: How many markets to track per domain. Bounded because every subscription
#: costs bandwidth, memory and message-processing time, and a strategy cannot
#: sensibly monitor thousands of markets on a sub-$250 account.
DEFAULT_WATCHLIST_SIZE = 60


class MarketDataIngestor:
    """Drives one domain's WebSocket into one :class:`MarketState`."""

    def __init__(
        self,
        *,
        market_type: MarketType,
        ws: KalshiWebSocket,
        state: MarketState,
        clock: Clock,
        writer: AsyncWriter | None = None,
        persist_quotes: bool = True,
    ) -> None:
        self.market_type = market_type
        self.ws = ws
        self.state = state
        self.clock = clock
        self.writer = writer
        self.persist_quotes = persist_quotes
        self.watchlist: list[str] = []
        self._subscription: Any = None
        self.dropped_deltas = 0
        self.snapshots_applied = 0
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.ws.on("orderbook_snapshot", self._on_snapshot)
        self.ws.on("orderbook_delta", self._on_delta)
        self.ws.on("trade", self._on_trade)
        self.ws.on("ticker", self._on_ticker)
        self.ws.on("ticker_v2", self._on_ticker)
        self.ws.on("market_lifecycle_v2", self._on_lifecycle)

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        await self.ws.start()
        await self.ws.wait_connected(timeout_s=30.0)

    async def stop(self) -> None:
        await self.ws.stop()

    async def set_watchlist(self, tickers: list[str]) -> None:
        """Subscribe to ``tickers``, updating incrementally where possible."""
        new = list(dict.fromkeys(tickers))[:DEFAULT_WATCHLIST_SIZE]
        if not new:
            return

        if self._subscription is None:
            self._subscription = await self.ws.subscribe(
                ["orderbook_delta", "ticker", "trade"], market_tickers=new
            )
            self.watchlist = new
            log.info(
                "ingestion.subscribed",
                domain=self.market_type.value, markets=len(new),
            )
            return

        added = [t for t in new if t not in self.watchlist]
        removed = [t for t in self.watchlist if t not in new]
        if added or removed:
            await self.ws.update_subscription(self._subscription, add=added, remove=removed)
            for ticker in removed:
                self.state.drop(ticker)
            self.watchlist = new
            log.info(
                "ingestion.watchlist_updated",
                domain=self.market_type.value, added=len(added), removed=len(removed),
            )

    def healthy(self) -> bool:
        return self.ws.connected

    # -- handlers ------------------------------------------------------------

    def _on_snapshot(self, message: dict[str, Any]) -> None:
        payload = message.get("msg") or {}
        ticker = payload.get("market_ticker") or payload.get("ticker")
        if not ticker:
            return
        now_ms = self.clock.now_ms()
        book = self._book_from_snapshot(ticker, payload)
        if book is None:
            return
        self.state.apply_snapshot(ticker, book, now_ms=now_ms, seq=message.get("seq"))
        self.snapshots_applied += 1
        self._persist_quote(ticker, now_ms)

    def _book_from_snapshot(self, ticker: str, payload: dict[str, Any]):
        from kat.marketdata.book import OrderBook

        if self.market_type is MarketType.EVENT:
            return OrderBook.from_event_contract_levels(
                ticker,
                _levels(payload.get("yes") or payload.get("yes_dollars")),
                _levels(payload.get("no") or payload.get("no_dollars")),
            )
        return OrderBook.from_two_sided_levels(
            ticker, _levels(payload.get("bids")), _levels(payload.get("asks"))
        )

    def _on_delta(self, message: dict[str, Any]) -> None:
        payload = message.get("msg") or {}
        ticker = payload.get("market_ticker") or payload.get("ticker")
        if not ticker:
            return
        try:
            price_cc = parse_price_cc(
                payload.get("price_dollars") or payload.get("price"),
                rounding=Rounding.HALF_EVEN,
            )
            delta_qc = parse_count_qc(
                payload.get("delta_fp") or payload.get("delta"), rounding=Rounding.HALF_EVEN
            )
        except (ValueError, TypeError):
            return

        now_ms = self.clock.now_ms()
        if self.market_type is MarketType.EVENT:
            side = (payload.get("side") or "yes").lower()
            applied = self.state.apply_event_book_delta(
                ticker, price_cc=price_cc, delta_qc=delta_qc,
                is_yes=side == "yes", now_ms=now_ms, seq=message.get("seq"),
            )
        else:
            side = (payload.get("side") or "bid").lower()
            applied = self.state.apply_book_delta(
                ticker, price_cc=price_cc, delta_qc=delta_qc,
                is_bid=side == "bid", now_ms=now_ms, seq=message.get("seq"),
            )

        if not applied:
            # Either no snapshot yet, or the delta left the book crossed. Both
            # mean the local book cannot be trusted; ask for a fresh snapshot
            # rather than continuing to apply deltas to it.
            self.dropped_deltas += 1
            if self._subscription is not None:
                task = asyncio.create_task(
                    self.ws.request_snapshot(self._subscription, [ticker])
                )
                task.add_done_callback(lambda t: t.exception())
            return
        self._persist_quote(ticker, now_ms)

    def _on_trade(self, message: dict[str, Any]) -> None:
        payload = message.get("msg") or {}
        ticker = payload.get("market_ticker") or payload.get("ticker")
        if not ticker:
            return
        try:
            price_cc = parse_price_cc(
                payload.get("yes_price_dollars") or payload.get("price"),
                rounding=Rounding.HALF_EVEN,
            )
            qty_qc = parse_count_qc(
                payload.get("count_fp") or payload.get("count"), rounding=Rounding.HALF_EVEN
            )
        except (ValueError, TypeError):
            return

        now_ms = self.clock.now_ms()
        self.state.apply_trade(ticker, price_cc=price_cc, qty_qc=qty_qc, now_ms=now_ms)
        if self.writer is not None:
            self.writer.enqueue(
                TradePrint(
                    trade_id=payload.get("trade_id"),
                    ticker=ticker,
                    market_type=self.market_type.value,
                    ts=self.clock.now(),
                    price_cc=price_cc,
                    count_qc=qty_qc,
                    taker_side=payload.get("taker_side"),
                    source="kalshi_ws",
                    source_ts=self.clock.now(),
                )
            )

    def _on_ticker(self, message: dict[str, Any]) -> None:
        payload = message.get("msg") or {}
        ticker = payload.get("market_ticker") or payload.get("ticker")
        if not ticker:
            return

        def price(*keys: str) -> int | None:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    value = value.get("price") or value.get("value")
                if value not in (None, ""):
                    try:
                        return parse_price_cc(value, rounding=Rounding.HALF_EVEN)
                    except (ValueError, TypeError):
                        continue
            return None

        funding = payload.get("funding_rate")
        rate = funding.get("rate") if isinstance(funding, dict) else funding
        next_funding = (
            funding.get("next_funding_time_ms") if isinstance(funding, dict) else None
        )

        self.state.apply_ticker(
            ticker,
            now_ms=self.clock.now_ms(),
            mark_price_cc=price("settlement_mark_price", "mark_price", "reference_price"),
            last_cc=price("price_dollars", "last_price", "price"),
            funding_rate=float(rate) if rate is not None else None,
            next_funding_ms=int(next_funding) if next_funding else None,
        )

    def _on_lifecycle(self, message: dict[str, Any]) -> None:
        payload = message.get("msg") or {}
        ticker = payload.get("market_ticker") or payload.get("ticker")
        status = payload.get("new_status") or payload.get("status")
        if ticker and status:
            self.state.set_status(ticker, str(status), now_ms=self.clock.now_ms())

    # -- persistence ---------------------------------------------------------

    def _persist_quote(self, ticker: str, now_ms: int) -> None:
        if not self.persist_quotes or self.writer is None:
            return
        snap = self.state.get(ticker)
        if snap is None or snap.book is None:
            return
        book = snap.book
        self.writer.enqueue(
            Quote(
                ticker=ticker,
                market_type=self.market_type.value,
                ts=self.clock.now(),
                bid_cc=book.best_bid_cc,
                ask_cc=book.best_ask_cc,
                bid_size_qc=book.best_bid_size_qc,
                ask_size_qc=book.best_ask_size_qc,
                last_cc=snap.last_trade_cc,
                mark_cc=snap.mark_price_cc,
                volume_qc=snap.volume_qc,
                open_interest_qc=snap.open_interest_qc,
                source="kalshi_ws",
                source_ts=self.clock.now(),
            )
        )

    def stats(self) -> dict[str, Any]:
        return {
            "domain": self.market_type.value,
            "watchlist": len(self.watchlist),
            "snapshots_applied": self.snapshots_applied,
            "dropped_deltas": self.dropped_deltas,
            "websocket": self.ws.stats(),
            "messages": metrics.get_counter(M_WS_MESSAGES, domain=self.market_type.value),
        }


def _levels(raw: Any) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for level in raw or []:
        try:
            price = parse_price_cc(level[0], rounding=Rounding.HALF_EVEN)
            size = parse_count_qc(level[1], rounding=Rounding.HALF_EVEN)
        except (ValueError, IndexError, TypeError):
            continue
        if size > 0:
            out.append((price, size))
    return out


async def select_event_watchlist(
    rest: EventContractsREST, *, limit: int = DEFAULT_WATCHLIST_SIZE
) -> list[str]:
    """Choose which event markets to follow.

    Ranked by 24-hour volume, because liquidity is what makes an edge
    realisable: a large mispricing in a market that cannot absorb five contracts
    is not an opportunity. Markets already closed or without a live quote are
    excluded outright.
    """
    markets = await rest.get_markets(status="open", limit=200, max_pages=5)
    tradeable = [
        m
        for m in markets
        if m.is_tradeable and m.yes_bid_cc is not None and m.yes_ask_cc is not None
    ]
    tradeable.sort(key=lambda m: m.volume_24h_qc or 0, reverse=True)
    chosen = [m.ticker for m in tradeable[:limit]]
    log.info(
        "ingestion.watchlist_selected",
        domain="event", considered=len(markets), tradeable=len(tradeable), chosen=len(chosen),
    )
    return chosen


async def select_perp_watchlist(
    rest: PerpsREST, *, limit: int = DEFAULT_WATCHLIST_SIZE
) -> list[str]:
    """Choose which perpetual markets to follow."""
    markets = await rest.get_markets(limit=200, max_pages=3)
    tradeable = [m for m in markets if m.is_tradeable]
    tradeable.sort(key=lambda m: m.volume_qc or 0, reverse=True)
    chosen = [m.ticker for m in tradeable[:limit]]
    log.info(
        "ingestion.watchlist_selected",
        domain="perp", considered=len(markets), chosen=len(chosen),
    )
    return chosen
