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
The platform cannot subscribe to every Kalshi market and should not try, and
which few it picks decides more about whether it can make money than any
strategy does. Markets are ranked by how far their last move went *relative to
what capturing it costs* -- the spread plus two fees -- then filtered for a
quote, a spread, a settlement date inside a year, and enough volume to trade
against. Refreshed on a slow cycle, with subscriptions updated incrementally
rather than torn down.

Ranking by volume was the earlier answer and it was the wrong question. Volume
says a market traded; it does not say it moved, and it says nothing about the
toll. Measured over a week of banked quotes, every market the platform was
watching cleared its own round-trip cost in 0.0% of thirty-minute windows: moves
of 0 to 12 centicents against costs of 136 to 466. No model and no amount of
tuning wins there, because the arithmetic is decided before any opinion is
formed.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from kat.common.enums import MarketType
from kat.common.fixedpoint import (
    CC_PER_DOLLAR,
    QC_PER_CONTRACT,
    Rounding,
    parse_count_qc,
    parse_price_cc,
)
from kat.common.timeutil import Clock
from kat.db.models import Quote, TradePrint
from kat.db.writer import AsyncWriter
from kat.kalshi.event_contracts.rest import EventContractsREST
from kat.kalshi.fees import FeeSchedule, realised_fee_micros
from kat.kalshi.perps.rest import PerpsREST
from kat.kalshi.ws import KalshiWebSocket
from kat.marketdata.state import MarketState
from kat.observability.logging import get_logger
from kat.observability.metrics import M_WS_MESSAGES, metrics

log = get_logger(__name__)

__all__ = ["MarketDataIngestor", "load_fee_schedules", "select_event_watchlist"]

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
        #: Top-of-book changes seen per ticker since the last watchlist refresh.
        #: Reset by :meth:`take_movement`, which the refresh consults.
        self.quote_changes: dict[str, int] = {}
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

        # Anything in the state we are not subscribed to can never update again,
        # so it would sit there being permanently stale and being counted as a
        # market the trading loop examined. These arrive from set_definitions,
        # which ensures a snapshot for every market it is given -- and the
        # watchlist selector hands back definitions for every *candidate* it
        # considered, not only the ones that won a slot.
        orphans = [t for t in self.state.tickers() if t not in set(new)]
        for ticker in orphans:
            self.state.drop(ticker)
            self._last_quote.pop(ticker, None)
        if orphans:
            log.info(
                "ingestion.dropped_unsubscribed",
                domain=self.market_type.value, markets=len(orphans),
            )
            log.info(
                "ingestion.watchlist_updated",
                domain=self.market_type.value, added=len(added), removed=len(removed),
            )

    def take_movement(self) -> dict[str, int]:
        """Top-of-book changes per ticker since the last call, and reset.

        Consumed rather than read so each refresh judges the interval it just
        lived through. A cumulative count would let a market that was busy this
        morning hold its slot all evening, which is the behaviour being fixed.
        """
        movement = self.quote_changes
        self.quote_changes = {}
        return movement

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
                _first(payload, "price_dollars", "price"),
                rounding=Rounding.HALF_EVEN,
            )
            delta_qc = parse_count_qc(
                _first(payload, "delta_fp", "delta"), rounding=Rounding.HALF_EVEN
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
                _first(payload, "yes_price_dollars", "price"),
                rounding=Rounding.HALF_EVEN,
            )
            qty_qc = parse_count_qc(
                _first(payload, "count_fp", "count"), rounding=Rounding.HALF_EVEN
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
        # The dedup above is already deciding "did this market actually move",
        # once per message, and then throwing the answer away per ticker. Keeping
        # it is free and is the only first-hand evidence the platform has about
        # which of its subscriptions are worth holding: volume and spread come
        # from a REST listing and describe the market's past, this describes what
        # it did while we were watching.
        self.quote_changes[ticker] = self.quote_changes.get(ticker, 0) + 1

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

#: How much a market must have traded in a day to be worth watching.
#:
#: Movement is necessary and not sufficient: a market that jumped on four
#: contracts has moved, but there is nobody there to trade against, and a market
#: that ticks once an hour yields no realised volatility for a strategy to read.
#: This is now the *ranking* key as well as a floor, so the bar itself only has
#: to exclude the genuinely empty.
_MIN_WATCH_VOLUME_QC = 5_000

#: How far a market's last move must go, as a multiple of what capturing it
#: costs, before it is worth a subscription slot.
#:
#: One would be break-even on the last move alone, which is too strict to be
#: useful as a filter and too generous to be a promise. This is deliberately
#: modest: it is a screen against markets that are arithmetically unwinnable,
#: not a prediction that the next move will pay.
_MIN_MOVE_VS_COST = 0.5

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


def _round_trip_cost_cc(market: Any) -> int | None:
    """What entering and leaving one contract costs, in centicents.

    The spread is paid once across the round trip, and the fee twice. The fee is
    the part that is easy to forget and impossible to outrun: it is quadratic in
    price, so it is 175cc a side at $0.50 and 14cc at $0.98. A market's price
    level therefore decides most of its own tradeability, before any strategy has
    an opinion about it.
    """
    spread = _spread_cc(market)
    bid, ask = market.yes_bid_cc, market.yes_ask_cc
    if spread is None or bid is None or ask is None:
        return None
    mid = (bid + ask) // 2
    if mid <= 0 or mid >= CC_PER_DOLLAR:
        return None
    # The realised fee, not the estimate. The estimate pads an unknown schedule
    # by 50% so an unguessable fee cannot make a marginal trade look profitable,
    # which belongs to a trading decision. This is asking what the venue
    # actually charges, and inflating it here would reject markets that are
    # genuinely affordable.
    fee_micros = realised_fee_micros(price_cc=mid, qty_qc=QC_PER_CONTRACT)
    # micros to centicents: 1e-6 USD against 1e-4 USD.
    return spread + 2 * (fee_micros // 100)


def _move_vs_cost(market: Any) -> float | None:
    """A market's most recent realised move, as a multiple of its round trip.

    Below 1.0 the exchange takes more than the market moved, and no strategy --
    no model, no amount of tuning -- can make money there. This is arithmetic
    about the venue rather than a view about the market, which is why it belongs
    in the decision about what is worth watching at all.

    Measured over a week of banked quotes: every market the platform was
    watching scored 0.0% of 30-minute windows clearing its own cost, with moves
    of 0 to 12cc against costs of 136 to 466cc. The ones that did clear it were
    never selected, because the ranking asked for volume and a spread cap and
    those are not the same question.
    """
    last, previous = market.last_price_cc, market.previous_price_cc
    cost = _round_trip_cost_cc(market)
    if not last or not previous or not cost:
        # A zero previous price is the exchange saying it has none, not a market
        # that traded at nothing: 9,752 of 26,479 live markets report exactly 0.
        # Read literally it manufactures a full-dollar move out of an absent
        # number, and since the score is a ratio, those markets then sort to the
        # top of the watchlist -- the missing value becoming the most attractive
        # one, which is the failure this codebase refuses everywhere else. A
        # live contract cannot trade at $0.0000 anyway.
        return None
    return abs(last - previous) / cost


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
        if (spread := _spread_cc(m)) is not None
        and spread <= _WATCH_MAX_SPREAD_CC
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
    # Filtered on whether a move can pay for itself, then ranked by activity.
    #
    # Volume-first put the platform in markets that could not produce a winning
    # trade at any skill level: a fortnight of banked quotes has every watched
    # market clearing its own round-trip cost in 0.0% of 30-minute windows.
    # Volume says a market traded; it does not say it moved, and it says nothing
    # at all about the toll. A market whose last move was smaller than the cost
    # of capturing it is not a hard market, it is an unwinnable one.
    #
    # Volume stays as the tiebreak, because among markets that do move, the one
    # with more depth is the one an order can actually get into.
    # Two different questions, and each was the wrong single answer.
    #
    # Volume-first watched markets that tick constantly and cannot pay: 0 of 60
    # stale, and 0.0% of thirty-minute windows clearing their own cost.
    # Move-vs-cost-first watched markets that can pay and barely tick: 46 of 60
    # stale, no realised volatility computable, and again no intents. A market
    # has to clear its costs *and* be alive; economics decides eligibility,
    # activity decides priority.
    quotable.sort(key=lambda m: float(m.volume_24h_qc or 0), reverse=True)
    tradeable = [
        m
        for m in quotable
        if (_move_vs_cost(m) or 0.0) >= _MIN_MOVE_VS_COST
        and (m.volume_24h_qc or 0) >= _MIN_WATCH_VOLUME_QC
    ]
    if tradeable:
        quotable = tradeable
    else:
        log.warning(
            "ingestion.no_market_covers_its_costs",
            domain="event",
            considered=len(quotable),
            min_move_vs_cost=_MIN_MOVE_VS_COST,
            note="watching the best of a bad board; expect very few intents",
        )
    picked = quotable[:limit]
    chosen = [m.ticker for m in picked]
    # Kept rather than discarded: these objects already carry the title, rules
    # and close time that the news matcher and the term features need, and
    # nothing else in the system ever fetches them.
    _LAST_DEFINITIONS.clear()
    _LAST_DEFINITIONS.update({
        m.ticker: {
            "title": m.title or "",
            # What this market is a market *about*: on a multi-outcome event
            # every sibling shares a ticker prefix and differs only here, so
            # "Chelsea" is what separates one Premier League contract from the
            # nineteen next to it. Without it a position reads as a ticker.
            "yes_sub_title": m.yes_sub_title or "",
            "no_sub_title": m.no_sub_title or "",
            "rules_primary": m.rules_primary or "",
            "event_ticker": m.event_ticker or "",
            # Carried so the fee terms can be fetched for these markets and only
            # these: the series endpoint is per-series, and the watchlist is a
            # few dozen markets drawn from a handful of them.
            "series_ticker": m.series_ticker or "",
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


#: Fee terms per series, keyed by series ticker. Fetched once and kept for the
#: life of the process: a series' fee schedule is a property of the product, not
#: of the market's state, and it does not change while the platform is running.
_SERIES_SCHEDULES: dict[str, FeeSchedule] = {}


async def load_fee_schedules(
    rest: EventContractsREST,
    definitions: dict[str, dict[str, Any]],
    *,
    into: dict[str, FeeSchedule],
) -> int:
    """Fill ``into`` with the fee terms for each market in ``definitions``.

    Nothing was calling this. ``FeeSchedule.from_series`` existed, ``get_series``
    existed, and the ``fee_schedules`` dicts on the trading loop and the
    arbitrage scanner existed -- but no code ever connected them, so the mapping
    stayed empty for the life of every run and every fee estimate in the platform
    fell back to an unknown schedule.

    That is not a neutral default. An unknown schedule is marked inexact, which
    inflates the estimate by 50% so an unguessable fee cannot make a marginal
    trade look profitable. Correct when the terms really are unknown; permanently
    wrong when they are one request away. Every strategy costed its round trip
    half again over the odds and declined trades on an edge it did have.

    Fetched per *series*, not per market: sixty watched markets come from a
    handful of series, and the terms are a property of the series. Failures are
    logged and skipped rather than raised -- a schedule we could not fetch leaves
    the market on the inexact fallback, which is exactly the conservative
    behaviour that was already in place.
    """
    wanted: dict[str, list[str]] = {}
    for ticker, definition in definitions.items():
        series = str(definition.get("series_ticker") or "").strip()
        if series:
            wanted.setdefault(series, []).append(ticker)

    fetched = 0
    for series in wanted:
        if series in _SERIES_SCHEDULES:
            continue
        try:
            payload = await rest.get_series(series)
        except Exception as exc:
            log.warning(
                "ingestion.fee_schedule_unavailable",
                series=series,
                error=str(exc),
                note="markets in this series keep the inexact fallback",
            )
            continue
        _SERIES_SCHEDULES[series] = FeeSchedule.from_exchange_series(payload)
        fetched += 1

    applied = 0
    for series, tickers in wanted.items():
        schedule = _SERIES_SCHEDULES.get(series)
        if schedule is None:
            continue
        for ticker in tickers:
            into[ticker] = schedule
            applied += 1

    # Markets that have left the watchlist keep no slot here: the mapping is read
    # by ticker on the trading path and an entry for a market nobody watches is a
    # slow leak across a long-running process.
    for ticker in list(into):
        if ticker not in definitions:
            del into[ticker]

    if wanted:
        log.info(
            "ingestion.fee_schedules_loaded",
            series_known=len(_SERIES_SCHEDULES),
            series_fetched=fetched,
            markets_priced=applied,
            markets_without=len(definitions) - applied,
        )
    return applied
