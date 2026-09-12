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
import json
import re
import signal
import sys
from dataclasses import dataclass, field
from typing import Any

from kat.common.errors import ExchangeError
from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.config.settings import AppSettings
from kat.observability.logging import configure_logging, get_logger
from kat.polymarket.rest import PolymarketREST
from kat.polymarket.selection import maker_edge_cc, rank_tradeable

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
#: Following 60 markets instead gives a pass every twenty-four seconds. Fewer
#: markets, but each one with a history fine enough to measure.
#:
#: Sixty rather than a hundred and twenty because each followed market costs two
#: requests, not one: the book for the queue ahead of a resting order, and the
#: bbo for the cumulative traded count. The venue publishes no trades endpoint,
#: so flow has to be differenced out of that counter -- and a banked book with no
#: flow beside it cannot be replayed at all, because a resting order is filled by
#: the market trading through it. Half the markets, all of the data.
DEFAULT_FOLLOW = 60



async def _open_markets(rest: PolymarketREST) -> list[Any]:
    markets = await rest.get_open_markets(max_pages=8)
    return [m for m in markets if m.is_tradeable]


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
           last_cc, volume_qc,
           source, source_ts, ingested_ts, quality)
        values ($1, 'event', $2, $3, $4, $5, $6, $7, $8,
                'polymarket', $2, now(), 'ok')
        """,
        [r[:8] for r in rows],
    )
    # The whole book, not just the touch. Banking one level made every replayed
    # book 235 times thinner than the real one -- live books carry a median of
    # 494,100 centi-contracts across their levels against 2,100 at the touch --
    # and the risk engine's min_book_depth_qc of 1,000 then refused 72,467 of
    # 101,379 intents. That read as "the venue is too thin to trade" when it was
    # "the history is one level deep".
    await conn.executemany(
        """
        insert into orderbook_snapshots
          (ticker, market_type, ts, bids, asks, depth_levels,
           source, source_ts, ingested_ts, quality)
        values ($1, 'event', $2, $3, $4, $5,
                'polymarket', $2, now(), 'ok')
        """,
        [
            (r[0], r[1], json.dumps(r[8]), json.dumps(r[9]), len(r[8]))
            for r in rows if r[8] and r[9]
        ],
    )


@dataclass
class FollowState:
    """What a crash must not throw away.

    The listing and the follow set used to be locals, and the supervisor
    re-enters :func:`collect` after a crash -- so every crash discarded the
    chosen markets and forced a fresh sweep of about 4,000 books, 25 to 30
    minutes at the polite rate, before fine sampling resumed. Three crashes on
    2026-09-11 cost an hour and a half of exactly the resolution the follow set
    exists to buy.

    Only a *completed* sweep fills ``quotable``, so a crash partway through one
    still leaves nothing chosen and the next attempt starts the sweep over: a
    ranking of the first few markets on the board is not a follow set.
    """

    markets: list[Any] = field(default_factory=list)
    quotable: set[str] = field(default_factory=set)
    listed_at: float = 0.0
    #: Cumulative across attempts, so the cycle number in the log keeps counting
    #: through a crash instead of restarting at 1 and reading like a new run.
    cycles: int = 0
    written: int = 0


async def collect(
    db: _Db, rest: PolymarketREST, *, interval_s: float,
    stop: asyncio.Event, max_cycles: int | None = None,
    follow: int = DEFAULT_FOLLOW, state: FollowState | None = None,
) -> None:
    state = FollowState() if state is None else state
    loop = asyncio.get_running_loop()
    done = 0  # cycles in *this* call, which is what max_cycles bounds

    while not stop.is_set():
        started = loop.time()
        if not state.markets or started - state.listed_at > REFRESH_MARKETS_S:
            try:
                state.markets = await _open_markets(rest)
                state.listed_at = started
                # A fresh listing invalidates what we knew about which markets
                # carry a book, so the next pass sweeps all of them again.
                state.quotable = set()
                log.info("polymarket.markets_listed", count=len(state.markets))
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
        sweeping = not state.quotable
        candidates: list[tuple[Any, int, int, int | None]] = []
        pass_over = state.markets if sweeping else [
            m for m in state.markets if m.slug in state.quotable
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
                state.quotable.discard(market.slug)
                continue
            bid, ask = book.best_bid_cc, book.best_ask_cc
            # A crossed or nonsensical book is a bad read, not a trading
            # opportunity; banking it would put a fake arbitrage in the history.
            if bid is None or ask is None or not 0 < bid < ask < CC_PER_DOLLAR:
                continue
            found += 1

            # Cumulative traded count: differenced into per-interval flow while
            # following, and read during a sweep to tell a market that is quiet
            # from one that has never traded at all. The sweep costs twice as
            # long for it, which at once per six hours is worth paying --
            # ranking without it produced a follow set with zero trades in 297
            # intervals, on a venue doing 1,473 contracts an hour.
            shares = last = None
            delay = next_at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            next_at = loop.time() + gap
            with contextlib.suppress(ExchangeError):
                shares, last = await rest.get_flow(market.slug)

            if sweeping:
                edge = maker_edge_cc(market, book)
                if edge is not None:
                    candidates.append((
                        market, edge,
                        min(book.best_bid_size_qc, book.best_ask_size_qc),
                        shares,
                    ))
            else:
                state.quotable.add(market.slug)
            rows.append((
                market.slug,
                dt.datetime.now(dt.UTC),
                bid, ask,
                book.best_bid_size_qc, book.best_ask_size_qc,
                last, shares,
                [[p, book.bids[p]] for p in book.bid_prices[:10]],
                [[p, book.asks[p]] for p in book.ask_prices[:10]],
            ))
            if len(rows) >= FLUSH_EVERY:
                state.written += await _flush(db, rows)

        state.written += await _flush(db, rows)

        if sweeping:
            state.quotable = set(rank_tradeable(candidates, follow))
            log.info(
                "polymarket.following",
                followed=len(state.quotable), quotable=len(candidates),
                ever_traded=sum(1 for c in candidates if c[3]),
                # A budget, not a measurement: two requests per followed
                # market -- the book for the queue, the bbo for the traded
                # counter -- so the pacing floor is twice the naive one. What a
                # pass actually costs is elapsed_s on each cycle, and only that
                # one is measured.
                paced_seconds_per_pass=round(
                    2 * len(state.quotable) / RATE_LIMIT_PER_S, 1
                ),
            )

        state.cycles += 1
        done += 1
        log.info(
            "polymarket.cycle",
            cycle=state.cycles, books=found, written_total=state.written,
            polled=len(pass_over), sweeping=sweeping,
            elapsed_s=round(loop.time() - started, 1),
        )
        if max_cycles is not None and done >= max_cycles:
            return
        remaining = interval_s - (loop.time() - started)
        if remaining > 0:
            await asyncio.wait([asyncio.create_task(stop.wait())], timeout=remaining)


async def supervise(
    db: _Db, rest: PolymarketREST, *, stop: asyncio.Event, interval_s: float,
    max_cycles: int | None = None, follow: int = DEFAULT_FOLLOW,
) -> int:
    """Run the collector, re-entering it after a crash and keeping its state.

    Supervise rather than exit. Twice this has run for most of a day and then
    stopped with exit code 1 and nothing in the log, because an unhandled
    exception's traceback goes to *stderr* -- and the scheduler discards that,
    so fixing the log file fixed only the half of the problem that was already
    being written down. The alternative, letting the process die and relying on
    the task's restart policy, was tried and did not fire: the task showed Ready
    with result 1 and stayed that way for fourteen hours while nothing collected.

    The state outlives the attempt, which is the point: re-entering with a fresh
    one is what turned each of three crashes on 2026-09-11 into half an hour of
    re-sweeping before fine sampling resumed.
    """
    state = FollowState()
    attempt = 0
    while not stop.is_set():
        try:
            await collect(
                db, rest, interval_s=interval_s, stop=stop,
                max_cycles=max_cycles, follow=follow, state=state,
            )
            return 0
        except KeyboardInterrupt:
            stop.set()
            return 0
        except Exception as exc:
            attempt += 1
            log.exception(
                "polymarket.collect_crashed",
                attempt=attempt,
                # Named in the line itself, not only in the traceback: a
                # ConnectionResetError has an empty str(), and for months the
                # only record of a crash was the word "crashed".
                error=str(exc) or type(exc).__name__,
                error_type=type(exc).__name__,
                followed=len(state.quotable),
            )
            if not await db.reconnect():
                log.error("polymarket.giving_up", reason="lock lost or db down")
                return 1
            await asyncio.wait([asyncio.create_task(stop.wait())], timeout=BACKOFF_S)
    return 0


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
        return await supervise(
            db, rest,
            stop=stop, interval_s=args.interval_s,
            max_cycles=1 if args.once else None,
            follow=args.follow,
        )
    finally:
        await db.close()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
    except BaseException:
        # Last resort. Anything that escapes main() would otherwise reach only
        # stderr, which the scheduler throws away -- the exact reason two
        # multi-hour runs ended with an exit code and no explanation.
        log.exception("polymarket.fatal")
        raise
