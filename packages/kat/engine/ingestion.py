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
from datetime import UTC, datetime
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
        #: Last persisted top-of-book per ticker, so an unchanged quote is not
        #: written again.
        self._last_quote: dict[str, tuple] = {}
        self.quotes_persisted = 0
        self.quotes_unchanged = 0
        self.definitions_attached = 0
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

    def set_definitions(self, definitions: dict[str, dict[str, Any]]) -> None:
        """Attach each market's own text to its snapshot.

        A market's title, resolution rules and close time are *definitions*: the
        websocket never sends them, so without this step every snapshot carries
        an empty metadata dict. That is not a cosmetic gap -- the news matcher
        scores an article against a market's title and rules, so with both blank
        it can never match anything, and the whole news subsystem produced zero
        signals in every engine run while working perfectly in isolation.

        The term features need ``close_ts_ms`` for the same reason.

        Status is the same problem in a smaller costume. It is only ever set by a
        ``market_lifecycle`` message, which the exchange sends when a status
        *changes* -- so a market that was already open when we subscribed never
        sends one and its status stayed empty for the life of the process. The
        definition already carries it, so it is filled in here and left alone
        afterwards: a live lifecycle message is newer than this snapshot of the
        listing and must win.
        """
        attached = 0
        for ticker, definition in definitions.items():
            snapshot = self.state.get(ticker)
            if snapshot is None:
                snapshot = self.state.ensure(
                    ticker, event_ticker=str(definition.get("event_ticker", ""))
                )
            snapshot.metadata.update(definition)
            status = definition.get("status")
            if status and snapshot.status is None:
                snapshot.status = str(status)
            attached += 1
        self.definitions_attached = attached
        log.info(
            "ingestion.definitions_attached",
            domain=self.market_type.value,
            markets=attached,
            with_title=sum(1 for d in definitions.values() if d.get("title")),
            with_rules=sum(1 for d in definitions.values() if d.get("rules_primary")),
        )

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
                self._last_quote.pop(ticker, None)
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
            # Kalshi names these three different ways across schema versions, and
            # matching none of them is silent: `_levels(None)` is an empty book,
            # `from_event_contract_levels` accepts it, and the snapshot applies
            # as a market with no orders on either side. The book then rebuilds
            # itself from whatever deltas happen to arrive next, which is a
            # partial, usually one-sided view of a market that is in fact quoted
            # on both sides.
            #
            # Live, that showed as 4 of 60 markets having a midpoint, ages of
            # twenty-five minutes, and a dashboard that looked like nothing was
            # being captured. The current production websocket sends
            # `yes_dollars_fp` / `no_dollars_fp`; the older names are kept
            # because demo still sends them.
            return OrderBook.from_event_contract_levels(
                ticker,
                _levels(_first(payload, "yes_dollars_fp", "yes_dollars", "yes")),
                _levels(_first(payload, "no_dollars_fp", "no_dollars", "no")),
            )
        return OrderBook.from_two_sided_levels(
            ticker,
            _levels(_first(payload, "bids_fp", "bids")),
            _levels(_first(payload, "asks_fp", "asks")),
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
        """Store a quote row when the quote actually changed.

        A ``Quote`` row records the top of book. Most order-book deltas do not
        touch it -- a size change seven levels deep moves nothing we store -- so
        writing one per delta produced a stream of byte-identical rows: roughly
        110 a second against 57 demo markets, which is about 9.5 million rows a
        day recording almost nothing, plus the event-loop and database work to
        put them there while the websocket was falling behind.

        Deduplicating on the stored fields keeps the series exactly as
        informative and costs a tuple comparison.
        """
        if not self.persist_quotes or self.writer is None:
            return
        snap = self.state.get(ticker)
        if snap is None or snap.book is None:
            return
        book = snap.book

        fingerprint = (
            book.best_bid_cc, book.best_ask_cc,
            book.best_bid_size_qc, book.best_ask_size_qc,
            snap.last_trade_cc, snap.mark_price_cc,
            snap.volume_qc, snap.open_interest_qc,
        )
        if self._last_quote.get(ticker) == fingerprint:
            self.quotes_unchanged += 1
            return
        self._last_quote[ticker] = fingerprint
        self.quotes_persisted += 1

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
            "definitions_attached": self.definitions_attached,
            "quotes_persisted": self.quotes_persisted,
            "quotes_unchanged": self.quotes_unchanged,
            "websocket": self.ws.stats(),
            "messages": metrics.get_counter(M_WS_MESSAGES, domain=self.market_type.value),
        }


def _first(payload: dict[str, Any], *names: str) -> Any:
    """The first key present in ``payload``, or None.

    Named lookups rather than a chain of ``or`` because an empty list is a
    legitimate value that ``or`` would skip past to the next name -- a market
    genuinely quoted on one side only would then be read with the wrong schema.
    """
    for name in names:
        if name in payload:
            return payload[name]
    return None


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


#: Definitions for the markets the last selection chose, keyed by ticker.
#: Populated by :func:`select_event_watchlist` and read by the caller, which
#: hands them to :meth:`MarketDataIngestor.set_definitions`.
_LAST_DEFINITIONS: dict[str, dict[str, Any]] = {}


#: How far out a market may settle and still be worth a subscription slot.
#:
#: The module has always said the watchlist is chosen by "liquidity and
#: time-to-close" and only ever used liquidity. That gap put the 2028
#: presidential nomination markets at the top of the list: genuinely the most
#: traded contracts on the exchange, quoted a tenth of a cent wide, and almost
#: perfectly still -- two of sixty watched markets repriced even four times in
#: half an hour. A model learning from that has nothing to learn, and a strategy
#: waiting for a dislocation waits forever.
#:
#: A year is deliberately loose. It excludes the multi-year political contracts
#: without starving the watchlist: 2,911 markets inside a three-cent spread
#: settle within a year, against 9 within a week.
_WATCH_MAX_DAYS_TO_CLOSE = 365

#: How wide a market may be quoted and still be worth a subscription slot.
#: Deliberately looser than any strategy's own limit: this decides what is worth
#: watching, not what is worth trading, and a market that drifts in and out of
#: tradeable is exactly one to keep an eye on.
_WATCH_MAX_SPREAD_CC = 1_000


def _closes_within(market: Any, days: int) -> bool:
    """Whether this market settles inside the horizon.

    A market with no close time is kept: an unknown settlement date is not
    evidence of a distant one, and dropping it would silently narrow the
    watchlist on a schema change.
    """
    close = getattr(market, "close_time", None)
    if close is None:
        return True
    return (close - datetime.now(UTC)).days <= days


def _spread_cc(market: Any) -> int | None:
    bid, ask = market.yes_bid_cc, market.yes_ask_cc
    if bid is None or ask is None:
        return None
    return ask - bid


def last_event_definitions() -> dict[str, dict[str, Any]]:
    """Market text from the most recent watchlist selection."""
    return dict(_LAST_DEFINITIONS)


async def select_event_watchlist(
    rest: EventContractsREST, *, limit: int = DEFAULT_WATCHLIST_SIZE
) -> list[str]:
    """Choose which event markets to follow.

    Ranked by 24-hour volume, because liquidity is what makes an edge
    realisable: a large mispricing in a market that cannot absorb five contracts
    is not an opportunity. Markets already closed or without a live quote are
    excluded outright.

    Enumerated through events, not through ``GET /markets``
    ------------------------------------------------------
    Both list the same universe. Only one of them lists it usefully.

    On production, ``/markets`` is dominated by provisional single-market shards
    that carry no quote: scanning 30,000 of them turned up four quoted markets
    and not one with any volume, so this selector picked two zero-volume shards
    and the platform watched them for hours. Nothing failed -- every request
    answered 200 -- and it was indistinguishable from a quiet exchange.

    ``/events?with_nested_markets=true`` returns 14,433 markets at the same
    moment on the same key, of which 12,421 are quoted and 1,903 sit inside a
    three-cent spread. It never showed up on demo, where the market count is
    small enough that ordering does not matter.
    """
    markets = await rest.get_markets_via_events(status="open", limit=200, max_pages=20)
    active = [m for m in markets if m.is_tradeable]
    # `has_live_quote`, not `is not None`: Kalshi reports an unquoted market as a
    # $0.00/$1.00 book rather than as a null, so the null test passed literally
    # every market and spent the subscription budget on markets with no orders
    # in them. See EventMarket.has_live_quote.
    tradeable = [m for m in active if m.has_live_quote]

    # Volume alone picks markets that already traded, not markets that trade.
    #
    # Ranking on 24-hour volume filled the watchlist with 2028 nomination
    # longshots: enormous cumulative volume, a one-cent book on one side only,
    # and no update for twenty-five minutes. Twenty-nine of sixty were stale at
    # any moment and eight had a two-sided book, so the features had almost
    # nothing to compute on and the strategies had almost nothing to consider.
    #
    # A market has to be *quotable* before its volume means anything, so the
    # spread is a filter rather than a tiebreak. The bound is deliberately
    # looser than any strategy's own limit -- this decides what is worth
    # watching, and a market drifting in and out of tradeable is worth watching
    # -- but it excludes the ones no strategy could ever act on.
    quotable = [
        m
        for m in tradeable
        if _spread_cc(m) is not None
        and _spread_cc(m) <= _WATCH_MAX_SPREAD_CC
        and _closes_within(m, _WATCH_MAX_DAYS_TO_CLOSE)
    ]
    if not quotable:
        # Never widen silently: an empty watchlist is a fact worth logging, and
        # falling back to the unfiltered list would hide the exchange being shut.
        log.warning(
            "ingestion.no_quotable_markets",
            domain="event",
            quoted=len(tradeable),
            max_spread_cc=_WATCH_MAX_SPREAD_CC,
            note="watching the widest-quoted markets instead; nothing may be tradeable",
        )
        quotable = tradeable
    quotable.sort(key=lambda m: (m.volume_24h_qc or 0, -(_spread_cc(m) or 10_000)), reverse=True)
    picked = quotable[:limit]
    chosen = [m.ticker for m in picked]
    # Kept rather than discarded: these objects already carry the title, rules
    # and close time that the news matcher and the term features need, and
    # nothing else in the system ever fetches them.
    _LAST_DEFINITIONS.clear()
    _LAST_DEFINITIONS.update({
        m.ticker: {
            "title": m.title or "",
            "rules_primary": m.rules_primary or "",
            "event_ticker": m.event_ticker or "",
            "close_ts_ms": int(m.close_time.timestamp() * 1000) if m.close_time else None,
            "status": m.status or "",
        }
        for m in picked
    })
    log.info(
        "ingestion.watchlist_selected",
        domain="event",
        considered=len(markets),
        active=len(active),
        quoted=len(tradeable),
        unquoted=len(active) - len(tradeable),
        chosen=len(chosen),
    )
    if not chosen and active:
        # Worth saying out loud rather than subscribing to nothing in silence:
        # on the demo exchange this is the normal state of most of the board.
        log.warning(
            "ingestion.no_quoted_markets",
            domain="event",
            active=len(active),
            note="no market has orders on both sides; nothing to subscribe to",
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
