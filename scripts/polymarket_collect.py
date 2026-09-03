"""Bank Polymarket US order books, because nobody else will.

The survey shows the maker economics on this venue are positive before adverse
selection -- all 355 live two-sided books, against 169 of the same books under
Kalshi's schedule. Whether they survive adverse selection is the question that
killed passive quoting on both Kalshi instruments, and it cannot be answered
from a snapshot: a resting order is reached only when the market trades
*through* it, which is a narrower and more hostile subset of the flow than any
survey sees.

Answering it needs history, and this exchange does not sell any. Its candles
carry ``open/high/low/close/volume`` and no bid or ask, and the endpoint serving
them is authenticated anyway. The Kalshi research leaned on candles that
included both sides of the book -- that is what let a strategy be backtested
against history that already existed. Here there is no such shortcut, so the
history has to be made.

This makes it. Books are polled and written into ``quotes``, the same table the
Kalshi ingester fills, so ``price_baseline.py``, the fill simulator and the
walk-forward splits read them without knowing which venue they came from.

Cost of being wrong about the poll rate
---------------------------------------
The public gateway allows 20 requests a second per IP and asks that WebSockets
be preferred over polling -- but the market-data socket is authenticated, so
polling is the only door open without an account. This stays well under the
limit and backs off on 429 rather than retrying into it, because the failure
mode of being impolite to a public endpoint is losing access to it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import re
import signal
import sys
from typing import Any

from kat.common.errors import ExchangeError
from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.config.settings import AppSettings
from kat.observability.logging import configure_logging, get_logger
from kat.polymarket.rest import PolymarketREST

log = get_logger(__name__)

#: How often to re-list open markets. They resolve and appear through the day,
#: but not fast enough to be worth a request every cycle.
REFRESH_MARKETS_S = 1_800

#: Requests per second. A quarter of the published limit: the endpoint is free
#: and unauthenticated, and there is no version of this worth being rude for.
RATE_LIMIT_PER_S = 5.0

#: How long to sleep after a 429 before trying anything again.
BACKOFF_S = 30.0

#: Books buffered before a write. A full sweep of the board takes thirteen
#: minutes at a polite rate, and holding every row until the end of it means a
#: stop anywhere in that window discards the whole pass -- which is exactly what
#: the first run of this collector did: four thousand markets polled, nothing
#: banked.
FLUSH_EVERY = 50



async def _open_markets(rest: PolymarketREST) -> list[Any]:
    markets = await rest.get_open_markets(max_pages=8)
    return [m for m in markets if m.is_tradeable]


async def _flush(conn: Any, rows: list[tuple[Any, ...]]) -> int:
    """Write the buffer and clear it. Returns how many rows landed.

    A write failure clears the buffer rather than retaining it: the rows are a
    sample of a book that has already moved on, and holding them would grow
    without bound while re-inserting stale quotes under a fresh timestamp.
    """
    if not rows:
        return 0
    try:
        await _write(conn, rows)
    except Exception as exc:
        log.error("polymarket.write_failed", error=str(exc), rows=len(rows))
        rows.clear()
        return 0
    count = len(rows)
    rows.clear()
    return count


async def _write(conn: Any, rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    await conn.executemany(
        """
        insert into quotes
          (ticker, market_type, ts, bid_cc, ask_cc, bid_size_qc, ask_size_qc,
           source, source_ts, ingested_ts, quality)
        values ($1, 'event', $2, $3, $4, $5, $6, 'polymarket', $2, now(), 'ok')
        """,
        rows,
    )


async def collect(
    conn: Any, rest: PolymarketREST, *, interval_s: float,
    stop: asyncio.Event, max_cycles: int | None = None,
) -> None:
    markets: list[Any] = []
    quotable: set[str] = set()
    listed_at = 0.0
    loop = asyncio.get_running_loop()
    cycles = written = 0

    while not stop.is_set():
        started = loop.time()
        if not markets or started - listed_at > REFRESH_MARKETS_S:
            try:
                markets = await _open_markets(rest)
                listed_at = started
                # A fresh listing invalidates what we knew about which markets
                # carry a book, so the next pass sweeps all of them again.
                quotable = set()
                log.info("polymarket.markets_listed", count=len(markets))
            except ExchangeError as exc:
                log.warning("polymarket.list_failed", error=str(exc))
                await asyncio.wait([asyncio.create_task(stop.wait())], timeout=BACKOFF_S)
                continue

        # The venue lists about 2,000 open markets and roughly 355 have a book
        # at all. Polling every one at a polite rate takes thirteen minutes and
        # spends most of it confirming that long-dated sports futures still have
        # no bids, so the full sweep runs only when the listing is refreshed and
        # every cycle in between revisits what was quotable last time.
        rows: list[tuple[Any, ...]] = []
        found = 0
        gap = 1.0 / RATE_LIMIT_PER_S
        sweeping = not quotable
        pass_over = markets if sweeping else [
            m for m in markets if m.slug in quotable
        ]
        for market in pass_over:
            if stop.is_set():
                break
            try:
                book = await rest.get_book(market.slug)
            except ExchangeError as exc:
                if "429" in str(exc):
                    log.warning("polymarket.rate_limited", backoff_s=BACKOFF_S)
                    await asyncio.wait(
                        [asyncio.create_task(stop.wait())], timeout=BACKOFF_S
                    )
                    break
                continue
            await asyncio.sleep(gap)
            if book is None:
                quotable.discard(market.slug)
                continue
            bid, ask = book.best_bid_cc, book.best_ask_cc
            # A crossed or nonsensical book is a bad read, not a trading
            # opportunity; banking it would put a fake arbitrage in the history.
            if bid is None or ask is None or not 0 < bid < ask < CC_PER_DOLLAR:
                continue
            quotable.add(market.slug)
            found += 1
            rows.append((
                market.slug,
                dt.datetime.now(dt.UTC),
                bid, ask,
                book.best_bid_size_qc, book.best_ask_size_qc,
            ))
            if len(rows) >= FLUSH_EVERY:
                written += await _flush(conn, rows)

        written += await _flush(conn, rows)

        cycles += 1
        log.info(
            "polymarket.cycle",
            cycle=cycles, books=found, written_total=written,
            polled=len(pass_over), sweeping=sweeping,
            elapsed_s=round(loop.time() - started, 1),
        )
        if max_cycles is not None and cycles >= max_cycles:
            return
        remaining = interval_s - (loop.time() - started)
        if remaining > 0:
            await asyncio.wait([asyncio.create_task(stop.wait())], timeout=remaining)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-s", type=float, default=60.0,
                        help="seconds between full passes over the open board")
    parser.add_argument("--once", action="store_true",
                        help="one pass and exit, for checking it works")
    args = parser.parse_args()

    configure_logging("INFO")
    import asyncpg

    dsn = re.sub(r"^postgresql\+\w+", "postgresql", str(AppSettings().database.url))
    conn = await asyncpg.connect(dsn)
    rest = PolymarketREST()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Windows has no signal handlers on the loop; the KeyboardInterrupt
        # path below catches it there instead.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    try:
        await collect(
            conn, rest,
            interval_s=args.interval_s, stop=stop,
            max_cycles=1 if args.once else None,
        )
    except KeyboardInterrupt:
        stop.set()
    finally:
        await conn.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
