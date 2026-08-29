"""In-memory market state -- the hot path.

Every trading decision reads from here and nowhere else. There is no database
query between a websocket message and an order, by design: persistence happens
asynchronously behind :class:`~kat.db.writer.AsyncWriter`, and the authoritative
view of "what is the market right now" lives in this process's memory.

Freshness is a first-class property. Each market records when it was last
updated, and :meth:`MarketState.is_fresh` is consulted before any decision. Data
older than the configured budget does not produce a cautious decision -- it
produces no decision at all, because a stale book is a book that may be
crossed, empty or gone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kat.common.enums import MarketType
from kat.marketdata.book import OrderBook
from kat.observability.logging import get_logger
from kat.observability.metrics import M_MARKET_DATA_AGE_MS, metrics

log = get_logger(__name__)

__all__ = ["MarketSnapshot", "MarketState", "funding_rate_bps_per_hour"]


def funding_rate_per_hour(
    rate: float | None, interval_hours: float = 1.0
) -> float | None:
    """Convert a raw Kalshi funding rate into a raw fraction per hour.

    The raw rate is a fraction charged over one funding interval, so the
    conversion is ``rate / interval_hours``. This is the single division;
    :func:`funding_rate_bps_per_hour` is defined in terms of it, so the two can
    never drift apart.

    Returns ``None`` when the rate is unknown, which callers must treat as "no
    funding information" rather than as a rate of zero.
    """
    if rate is None:
        return None
    if interval_hours <= 0:
        return None
    return rate / interval_hours


def funding_rate_bps_per_hour(
    rate: float | None, interval_hours: float = 1.0
) -> float | None:
    """Convert a raw Kalshi funding rate into basis points per hour.

    The raw rate is a fraction charged over one funding interval, so the
    conversion is ``rate * 10_000 / interval_hours``. This function is the only
    place that conversion is performed: a funding rate that means one thing in
    the strategy and another in the risk engine is a silent, expensive bug, and
    the fix is to have exactly one definition of the units.

    Returns ``None`` when the rate is unknown, which callers must treat as
    "no funding information" rather than as a rate of zero.
    """
    per_hour = funding_rate_per_hour(rate, interval_hours)
    return None if per_hour is None else per_hour * 10_000.0


@dataclass(slots=True)
class MarketSnapshot:
    """Everything known about one market at one instant."""

    ticker: str
    market_type: MarketType
    book: OrderBook | None = None
    last_trade_cc: int | None = None
    mark_price_cc: int | None = None          # perpetuals only
    volume_qc: int | None = None
    open_interest_qc: int | None = None
    status: str | None = None
    event_ticker: str = ""

    # Milliseconds since the epoch, from the platform clock.
    updated_ms: int = 0
    book_updated_ms: int = 0
    trade_updated_ms: int = 0

    # --- Funding, for perpetuals -------------------------------------------
    #
    # UNITS MATTER HERE AND ARE EASY TO GET WRONG.
    #
    # ``funding_rate`` is the **raw rate exactly as Kalshi publishes it**: a
    # signed fraction charged over one funding interval. 0.0001 means one basis
    # point per interval, not per hour and not annualised. It is stored raw
    # rather than pre-converted because a derived value in a shared field is how
    # two modules end up disagreeing about what the number means -- which is
    # precisely the bug this comment exists to prevent.
    #
    # Anything that wants a per-hour figure must call
    # :func:`funding_rate_bps_per_hour`, which is the single place the
    # conversion happens.
    funding_rate: float | None = None
    next_funding_ms: int | None = None
    #: Length of one funding interval in hours. Kalshi's perpetuals settle
    #: funding on a fixed schedule; the default reflects the common 1-hour
    #: interval and is overridden from market metadata when available.
    funding_interval_hours: float = 1.0

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_bid_cc(self) -> int | None:
        return self.book.best_bid_cc if self.book else None

    @property
    def best_ask_cc(self) -> int | None:
        return self.book.best_ask_cc if self.book else None

    @property
    def mid_cc(self) -> int | None:
        return self.book.mid_cc if self.book else None

    @property
    def spread_cc(self) -> int | None:
        return self.book.spread_cc if self.book else None

    def reference_price_cc(self) -> int | None:
        """Best available single price for this market.

        Preference order is mark (perps, exchange-authoritative), then the book
        midpoint, then the last trade. The last trade is the weakest of the
        three -- it can be arbitrarily stale -- so it is only used when there is
        no live book at all.
        """
        if self.mark_price_cc is not None:
            return self.mark_price_cc
        mid = self.mid_cc
        if mid is not None:
            return mid
        return self.last_trade_cc


class MarketState:
    """Registry of live market snapshots for one domain.

    Deliberately synchronous and lock-free. It is mutated only from the single
    asyncio event loop that owns the websocket feeds, so there is no contention
    to protect against, and adding a lock would put one on the critical path for
    nothing.
    """

    def __init__(
        self,
        market_type: MarketType,
        *,
        max_age_ms: int = 5_000,
        quiet_max_age_ms: int = 300_000,
    ) -> None:
        self.market_type = market_type
        self.max_age_ms = max_age_ms
        #: How old a *quiet* market's book may be while the feed is proven live.
        #: See :meth:`is_fresh` for why this is a different number.
        self.quiet_max_age_ms = quiet_max_age_ms
        self._markets: dict[str, MarketSnapshot] = {}
        self.messages_applied = 0
        self.books_resynced = 0
        #: When this feed last delivered anything, on any ticker. This is the
        #: evidence that silence on one market means "nothing happened" rather
        #: than "we are disconnected".
        self.last_message_ms = 0

    # -- access --------------------------------------------------------------

    def get(self, ticker: str) -> MarketSnapshot | None:
        return self._markets.get(ticker)

    def book(self, ticker: str) -> OrderBook | None:
        snap = self._markets.get(ticker)
        return snap.book if snap else None

    def ensure(self, ticker: str, *, event_ticker: str = "") -> MarketSnapshot:
        snap = self._markets.get(ticker)
        if snap is None:
            snap = MarketSnapshot(
                ticker=ticker, market_type=self.market_type, event_ticker=event_ticker
            )
            self._markets[ticker] = snap
        return snap

    def tickers(self) -> list[str]:
        return list(self._markets)

    def __len__(self) -> int:
        return len(self._markets)

    # -- freshness -----------------------------------------------------------

    def age_ms(self, ticker: str, *, now_ms: int) -> int | None:
        snap = self._markets.get(ticker)
        if snap is None or snap.updated_ms == 0:
            return None
        return now_ms - snap.updated_ms

    def feed_age_ms(self, *, now_ms: int) -> int | None:
        """How long since this feed delivered anything at all, on any ticker.

        None before the first message. This is a property of the *connection*,
        not of any one market, and it is what distinguishes the two kinds of
        silence in :meth:`is_fresh`.
        """
        if self.last_message_ms == 0:
            return None
        return now_ms - self.last_message_ms

    def is_fresh(self, ticker: str, *, now_ms: int, max_age_ms: int | None = None) -> bool:
        """Whether this market is fresh enough to trade on.

        A market that has never been updated is not fresh. That is not the same
        as "unknown": it is a definite refusal, because the alternative is
        pricing against a book we have never seen.

        Two kinds of silence
        --------------------
        This is a delta feed. The exchange sends a message when the book
        *changes*, so a market that has sent nothing for a minute is usually
        telling us the book is exactly what we already hold -- not that our copy
        has decayed. Treating that as stale is what a snapshot-polling feed would
        require, and it is wrong here: measured against a five-second budget, a
        thin market that reprices every thirty seconds is never fresh, and the
        platform declines every market it watches, forever. That is not caution;
        it is a system that cannot trade.

        The silence that *does* mean danger is silence across the whole feed,
        because that is indistinguishable from being disconnected. So the test
        is in two parts:

        * the ticker updated within ``max_age_ms`` -- unambiguously current; or
        * the **feed** delivered something within ``max_age_ms`` (so we are
          demonstrably still connected and receiving) and this ticker updated
          within ``quiet_max_age_ms``.

        The second clause is bounded rather than open-ended because a live feed
        does not prove *this* subscription is live: a per-ticker subscription can
        be dropped silently while others keep flowing, and after
        ``quiet_max_age_ms`` we stop assuming otherwise.

        The failure mode that matters is preserved exactly. If the connection
        drops, ``feed_age_ms`` grows past the budget and every market in this
        domain goes stale at once, which is the correct and conservative answer.
        """
        age = self.age_ms(ticker, now_ms=now_ms)
        if age is None:
            return False
        budget = max_age_ms if max_age_ms is not None else self.max_age_ms
        if age <= budget:
            return True
        feed_age = self.feed_age_ms(now_ms=now_ms)
        if feed_age is None or feed_age > budget:
            return False
        return age <= self.quiet_max_age_ms

    def stale_tickers(self, *, now_ms: int) -> list[str]:
        return [t for t in self._markets if not self.is_fresh(t, now_ms=now_ms)]

    # -- mutation ------------------------------------------------------------

    def apply_snapshot(
        self, ticker: str, book: OrderBook, *, now_ms: int, seq: int | None = None
    ) -> None:
        """Replace the book wholesale from an authoritative snapshot."""
        snap = self.ensure(ticker)
        book.seq = seq
        book.ts_ms = now_ms
        snap.book = book
        snap.book_updated_ms = now_ms
        snap.updated_ms = now_ms
        self.messages_applied += 1
        self.last_message_ms = now_ms
        metrics.gauge(M_MARKET_DATA_AGE_MS, 0, domain=self.market_type.value)

    def apply_book_delta(
        self,
        ticker: str,
        *,
        price_cc: int,
        delta_qc: int,
        is_bid: bool,
        now_ms: int,
        seq: int | None = None,
    ) -> bool:
        """Apply one incremental level change.

        Returns ``False`` when the delta could not be applied safely -- either
        because there is no book to apply it to, or because applying it left the
        book crossed. A crossed book means a message was missed or misapplied,
        so the book is dropped rather than trusted; the caller requests a fresh
        snapshot.
        """
        snap = self._markets.get(ticker)
        if snap is None or snap.book is None:
            return False

        snap.book.apply_delta(price_cc, delta_qc, is_bid=is_bid)
        if snap.book.is_crossed:
            log.warning(
                "marketdata.crossed_book_after_delta",
                ticker=ticker, domain=self.market_type.value,
                bid=snap.book.best_bid_cc, ask=snap.book.best_ask_cc,
            )
            snap.book = None
            self.books_resynced += 1
            return False

        snap.book.seq = seq
        snap.book.ts_ms = now_ms
        snap.book_updated_ms = now_ms
        snap.updated_ms = now_ms
        self.messages_applied += 1
        self.last_message_ms = now_ms
        return True

    def apply_event_book_delta(
        self,
        ticker: str,
        *,
        price_cc: int,
        delta_qc: int,
        is_yes: bool,
        now_ms: int,
        seq: int | None = None,
    ) -> bool:
        """Event-contract delta, reflecting NO changes onto the ask side."""
        # A NO level at price p is a YES ask at 1 - p; see marketdata.book.
        if is_yes:
            return self.apply_book_delta(
                ticker, price_cc=price_cc, delta_qc=delta_qc, is_bid=True,
                now_ms=now_ms, seq=seq,
            )
        return self.apply_book_delta(
            ticker, price_cc=10_000 - price_cc, delta_qc=delta_qc, is_bid=False,
            now_ms=now_ms, seq=seq,
        )

    def apply_trade(self, ticker: str, *, price_cc: int, qty_qc: int, now_ms: int) -> None:
        snap = self.ensure(ticker)
        snap.last_trade_cc = price_cc
        snap.volume_qc = (snap.volume_qc or 0) + qty_qc
        snap.trade_updated_ms = now_ms
        snap.updated_ms = now_ms
        self.messages_applied += 1
        self.last_message_ms = now_ms

    def apply_ticker(
        self,
        ticker: str,
        *,
        now_ms: int,
        mark_price_cc: int | None = None,
        last_cc: int | None = None,
        volume_qc: int | None = None,
        open_interest_qc: int | None = None,
        funding_rate: float | None = None,
        next_funding_ms: int | None = None,
    ) -> None:
        snap = self.ensure(ticker)
        if mark_price_cc is not None:
            snap.mark_price_cc = mark_price_cc
        if last_cc is not None:
            snap.last_trade_cc = last_cc
        if volume_qc is not None:
            snap.volume_qc = volume_qc
        if open_interest_qc is not None:
            snap.open_interest_qc = open_interest_qc
        if funding_rate is not None:
            snap.funding_rate = funding_rate
        if next_funding_ms is not None:
            snap.next_funding_ms = next_funding_ms
        snap.updated_ms = now_ms
        self.messages_applied += 1
        self.last_message_ms = now_ms

    def set_status(self, ticker: str, status: str, *, now_ms: int) -> None:
        snap = self.ensure(ticker)
        snap.status = status
        snap.updated_ms = now_ms

    def drop(self, ticker: str) -> None:
        self._markets.pop(ticker, None)

    # -- reporting -----------------------------------------------------------

    def stats(self, *, now_ms: int) -> dict[str, Any]:
        ages = [
            now_ms - s.updated_ms for s in self._markets.values() if s.updated_ms
        ]
        return {
            "domain": self.market_type.value,
            "markets": len(self._markets),
            "messages_applied": self.messages_applied,
            # The feed-level liveness that decides whether a quiet market
            # counts as current. None before the first message arrives.
            "feed_age_ms": self.feed_age_ms(now_ms=now_ms),
            "quiet_max_age_ms": self.quiet_max_age_ms,
            "books_resynced": self.books_resynced,
            "with_book": sum(1 for s in self._markets.values() if s.book is not None),
            "max_age_ms": max(ages) if ages else None,
            "median_age_ms": sorted(ages)[len(ages) // 2] if ages else None,
            "stale_count": sum(1 for a in ages if a > self.max_age_ms),
        }
