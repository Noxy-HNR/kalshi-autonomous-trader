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
from kat.common.fixedpoint import CC_PER_DOLLAR, QC_PER_CONTRACT
from kat.config.settings import AppSettings
from kat.observability.logging import configure_logging, get_logger
from kat.polymarket.fees import polymarket_fee_micros
from kat.polymarket.rest import PolymarketREST

log = get_logger(__name__)

#: How often to re-list the board and re-pick what to follow.
#:
#: This is not just a listing refresh: it invalidates the follow set, and
#: rebuilding that means pricing every market's book again -- a thirteen-minute
#: sweep. At the half-hour it was first set to, nearly half of all runtime went
#: on re-sweeping at coarse resolution instead of following closely, which
#: defeats the point of following anything. Six hours amortises the sweep to
#: about 3% of the time, and the composition of a prediction-market board does
#: not turn over faster than that.
REFRESH_MARKETS_S = 21_600

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

#: Postgres advisory-lock key, so only one collector polls the board at a time.
#: Arbitrary but fixed; it only has to not collide with another lock in this
#: database.
_LOCK_KEY = 0x504D_424B

#: How many markets to actually follow. This is the difference between data that
#: answers the question and data that does not.
#:
#: The first version polled every quotable market -- 3,538 of them. At a polite
#: five requests a second that is a twelve-minute pass, so each market was
#: sampled about once every twenty-six minutes: 105,408 snapshots that cannot see
#: what they were collected to see. Adverse selection happens in the seconds
#: after a resting order is filled, and a book observed twice an hour cannot
#: distinguish a quote that was lifted on the way up from one that was run over.
#:
#: Following 120 markets instead gives a pass every twenty-four seconds. Fewer
#: markets, but each one with a history fine enough to measure.
DEFAULT_FOLLOW = 120



async def _open_markets(rest: PolymarketREST) -> list[Any]:
    markets = await rest.get_open_markets(max_pages=8)
    return [m for m in markets if m.is_tradeable]


def _maker_edge_cc(market: Any, book: Any) -> int | None:
    """What one passive round trip earns here: the spread, plus both rebates.

    Signed fees, so the rebate arrives negative and subtracting it adds income.
    None when the book cannot be priced.
    """
    bid, ask = book.best_bid_cc, book.best_ask_cc
    if bid is None or ask is None or not 0 < bid < ask < CC_PER_DOLLAR:
        return None
    mid = (bid + ask) // 2
    fee = polymarket_fee_micros(
        price_cc=mid, qty_qc=QC_PER_CONTRACT, schedule=market.fees, is_taker=False
    ) // 100
    return (ask - bid) - 2 * fee


def _rank(candidates: list[tuple[Any, int, int]], limit: int) -> list[str]:
    """Pick the markets worth following closely.

    Ranked on reachability rather than on edge alone. A wide spread on a book
    nobody trades into pays nothing, and the Kalshi work spent a long time
    learning that the binding constraint is the queue ahead of a resting order,
    not the spread in front of it. So the sort is by edge per contract of queue:
    what the round trip pays, divided by how much has to trade before we are
    reached.
    """
    scored = [
        (edge / max(queue, QC_PER_CONTRACT), market.slug)
        for market, edge, queue in candidates
        if edge > 0
    ]
    scored.sort(reverse=True)
    return [slug for _, slug in scored[:limit]]


class _Db:
    """A Postgres connection that survives losing the server.

    The collector is meant to run for weeks and the first attempt ran for
    thirteen hours: one connection, opened at startup, never checked. A restart
    of the database, a network blip or an idle timeout ends the process, and
    because it was launched by the scheduler with nowhere to write, it ended
    silently. Reconnecting is cheap and the alternative is a data set with holes
    exactly where something interesting happened.

    The advisory lock is re-taken on every reconnect, because a session-scoped
    lock dies with the session that held it. If some other collector claimed it
    in the meantime, this one stops rather than quietly polling alongside it.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self.conn: Any = None

    async def connect(self) -> bool:
        import asyncpg

        self.conn = await asyncpg.connect(self._dsn)
        if await self.conn.fetchval("select pg_try_advisory_lock($1)", _LOCK_KEY):
            return True
        await self.conn.close()
        self.conn = None
        return False

    async def reconnect(self) -> bool:
        with contextlib.suppress(Exception):
            if self.conn is not None:
                await self.conn.close()
        self.conn = None
        try:
            return await self.connect()
        except Exception as exc:
            log.error("polymarket.reconnect_failed", error=str(exc))
            return False

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            if self.conn is not None:
                await self.conn.close()


async def _flush(db: _Db, rows: list[tuple[Any, ...]]) -> int:
    """Write the buffer and clear it. Returns how many rows landed.

    A write failure clears the buffer rather than retaining it: the rows are a
    sample of a book that has already moved on, and holding them would grow
    without bound while re-inserting stale quotes under a fresh timestamp. One
    reconnect is attempted first, because the common failure is a connection
    that died between passes rather than a row the server refuses.
    """
    if not rows:
        return 0
    for attempt in (1, 2):
        try:
            await _write(db.conn, rows)
        except Exception as exc:
            log.error(
                "polymarket.write_failed",
                error=str(exc), rows=len(rows), attempt=attempt,
            )
            if attempt == 1 and await db.reconnect():
                continue
            rows.clear()
            return 0
        count = len(rows)
        rows.clear()
        return count
    return 0


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
    db: _Db, rest: PolymarketREST, *, interval_s: float,
    stop: asyncio.Event, max_cycles: int | None = None,
    follow: int = DEFAULT_FOLLOW,
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

        # A sweep looks at everything and picks what to follow; every cycle
        # after it revisits only those, which is what buys the time resolution.
        # Polling the whole board takes twelve minutes and samples each market
        # twice an hour -- too coarse to see the thing this exists to measure.
        rows: list[tuple[Any, ...]] = []
        found = 0
        gap = 1.0 / RATE_LIMIT_PER_S
        sweeping = not quotable
        candidates: list[tuple[Any, int, int]] = []
        pass_over = markets if sweeping else [
            m for m in markets if m.slug in quotable
        ]
        next_at = loop.time()
        for market in pass_over:
            if stop.is_set():
                break
            # Pace to a rate, rather than sleeping a fixed gap after each
            # request. Sleeping afterwards adds the round trip to the gap, so a
            # 5/s budget ran at 2.4/s and a pass predicted at 24 seconds took
            # 50. Waiting only for the remainder spends the network time instead
            # of adding to it, and cannot exceed the rate whatever the latency.
            delay = next_at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            next_at = loop.time() + gap
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
            if book is None:
                quotable.discard(market.slug)
                continue
            bid, ask = book.best_bid_cc, book.best_ask_cc
            # A crossed or nonsensical book is a bad read, not a trading
            # opportunity; banking it would put a fake arbitrage in the history.
            if bid is None or ask is None or not 0 < bid < ask < CC_PER_DOLLAR:
                continue
            found += 1
            if sweeping:
                edge = _maker_edge_cc(market, book)
                if edge is not None:
                    candidates.append((
                        market, edge,
                        min(book.best_bid_size_qc, book.best_ask_size_qc),
                    ))
            else:
                quotable.add(market.slug)
            rows.append((
                market.slug,
                dt.datetime.now(dt.UTC),
                bid, ask,
                book.best_bid_size_qc, book.best_ask_size_qc,
            ))
            if len(rows) >= FLUSH_EVERY:
                written += await _flush(db, rows)

        written += await _flush(db, rows)

        if sweeping:
            quotable = set(_rank(candidates, follow))
            log.info(
                "polymarket.following",
                followed=len(quotable), quotable=len(candidates),
                # A budget, not a measurement. What a pass actually costs is
                # reported per cycle as elapsed_s, and only that one is real.
                paced_seconds_per_pass=round(len(quotable) / RATE_LIMIT_PER_S, 1),
            )

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
    parser.add_argument("--follow", type=int, default=DEFAULT_FOLLOW,
                        help="how many markets to follow closely; fewer markets "
                             "means finer time resolution on each")
    args = parser.parse_args()

    # To a file, not just stdout. Run from the scheduler there is no console to
    # capture, so the first version of this died after thirteen hours and left
    # nothing behind but a task exit code of 1 -- which says something failed and
    # nothing about what.
    configure_logging("INFO", log_dir="logs", filename="polymarket.log")

    dsn = re.sub(r"^postgresql\+\w+", "postgresql", str(AppSettings().database.url))

    # One collector at a time, enforced where both would meet rather than by
    # remembering not to start a second. A scheduled task and a hand-started run
    # will otherwise both poll the whole board: twice the rows, twice the load on
    # a free public endpoint, and no more information than one pass. The lock is
    # session-scoped, so it releases on a kill, which a pid file would not.
    db = _Db(dsn)
    if not await db.connect():
        log.warning("polymarket.already_collecting", lock=_LOCK_KEY)
        return 0

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
            db, rest,
            interval_s=args.interval_s, stop=stop,
            max_cycles=1 if args.once else None,
            follow=args.follow,
        )
    except KeyboardInterrupt:
        stop.set()
    finally:
        await db.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
