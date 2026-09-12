"""Background batched writer -- the reason the database is never on the hot path.

Producers (market-data ingestion, the execution engine, the risk engine) call
:meth:`AsyncWriter.enqueue`, which is a non-blocking put onto a bounded queue and
returns in microseconds. A single consumer task drains the queue in batches and
commits them.

Bounded, and what happens when it fills
---------------------------------------
The queue has a size limit. When it is full we drop the *oldest* non-critical
rows and count the drops, rather than blocking the producer or growing until the
process runs out of memory. Losing a quote row under extreme load is acceptable;
stalling an order submission because a disk is slow is not.

Rows tagged ``critical=True`` (orders, fills, decisions, risk events, audit) are
never evicted to make room for anything else. Their backlog is allowed to grow
past the nominal limit, and a backlog that is not draining makes this component
*unhealthy*, which stops new trading through the ordinary health gate -- because
a platform that cannot record what it is doing should not be doing more of it.
Only past a hard ceiling, where the alternative is exhausting memory, are
critical rows dropped, and then loudly.

Why a batch costs round trips, not rows
---------------------------------------
On 2026-09-11 this writer was alive and committing about 1,500 rows an hour
against 70,000 being produced, and dropped 2.1 million rows over the days either
side of a reboot. Nothing had died: ``_flush_once`` awaited ``session.merge()``
once per row, which is one SELECT round trip per row, and each round trip needs
its own turn of the event loop. The trading loop had grown to a 1.7-second cycle
against a 1.0-second interval, so there was no idle time left and the writer got
roughly one round trip per cycle. The queue filled, and every subsequent
``enqueue`` then scanned all 20,000 queued items looking for something evictable
and logged a line when it failed -- which took more loop time, which slowed the
writer further.

So a batch now primes the session's identity map with one indexed lookup per
*table* and merges only the rows that are actually already stored. The cost of a
batch is a handful of round trips whatever its size, eviction is O(1), and drops
are summarised rather than narrated line by line.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import Counter, deque
from typing import Any

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select
from sqlalchemy.exc import DataError, IntegrityError

from kat.config.settings import get_settings
from kat.db.base import Base
from kat.db.engine import Database, get_database
from kat.observability.logging import get_logger
from kat.observability.metrics import M_DB_DROPS, M_DB_QUEUE_DEPTH, M_DB_WRITES, metrics

log = get_logger(__name__)

__all__ = ["AsyncWriter", "get_writer", "reset_writer_singleton"]


#: Shortest gap between attempts to restart a writer task that died. If it
#: dies immediately every time, the restarts must not become the new flood.
_REVIVE_INTERVAL_S = 30.0

#: How often a continuing flood of drops is summarised. The incident produced
#: 757,042 individual ``critical_queue_full`` error lines -- three quarters of
#: everything in the log -- which buried every other event and cost loop time to
#: write. One line to open the burst and one summary per interval says strictly
#: more, because the summary carries counts per table.
_DROP_REPORT_INTERVAL_S = 10.0

#: A queue with rows in it and no commit for this long is not slow, it is stuck.
#: Well above a normal flush (half a second) and above the slowest batch seen
#: while healthy, so ordinary congestion does not trip it.
_STALL_ALERT_S = 60.0

#: How far the critical backlog may exceed the nominal queue size before rows
#: are dropped. Critical rows are small and the health gate stops new work long
#: before this; the ceiling exists only so an unreachable database cannot turn
#: into an unbounded allocation.
_CRITICAL_CEILING_FACTOR = 5

#: Identity lookups are chunked so a batch cannot build a bind-parameter list
#: longer than SQLite will accept.
_PRIME_CHUNK = 200


class _Item:
    __slots__ = ("critical", "obj", "queued_at", "seq")

    def __init__(self, obj: Base, critical: bool, seq: int, queued_at: float) -> None:
        self.obj = obj
        self.critical = critical
        #: Enqueue order, kept explicitly because critical and non-critical rows
        #: live in separate queues and must still be written in the order they
        #: were produced: an order's transitions, or a decision and the
        #: execution result merged over it, are only correct in sequence.
        self.seq = seq
        self.queued_at = queued_at


class AsyncWriter:
    """Batching, non-blocking persistence for ORM objects."""

    def __init__(self, database: Database | None = None) -> None:
        settings = get_settings().database
        self._db = database or get_database()
        self._max_size = settings.write_queue_size
        self._batch_size = settings.write_batch_size
        self._flush_interval = settings.write_flush_interval_s

        # Two queues rather than one, so evicting the oldest non-critical row is
        # a popleft instead of a scan of the whole queue. Order between them is
        # restored by sequence number when a batch is taken.
        self._critical: deque[_Item] = deque()
        self._normal: deque[_Item] = deque()
        self._seq = 0
        self._not_empty = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._dropped = 0
        self._written = 0
        # Rows that could not be committed are retried a bounded number of times
        # before being logged in full and abandoned, so a single poison row
        # cannot wedge the writer forever.
        self._retry_batch: list[_Item] = []
        self._retry_attempts = 0
        self._revived_at = 0.0
        self._revivals = 0

        # Progress, which is what "healthy" has to mean for this component. The
        # old flag recorded what start() had been asked to do; it stayed True
        # through sixty-six hours of dropping everything.
        self._created_at = time.monotonic()
        self._last_commit_at: float | None = None
        self._commits: deque[tuple[float, int]] = deque()
        self._critical_drops: deque[float] = deque()
        self._healthy = True

        # Drop accounting, reported in summaries rather than per row.
        self._drop_window: Counter[tuple[bool, str]] = Counter()
        self._burst_open = False
        self._last_drop_at = 0.0
        self._last_drop_report_at = 0.0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="db-writer")
        log.info("db_writer.started", queue_size=self._max_size, batch=self._batch_size)

    async def stop(self, *, drain: bool = True, timeout_s: float = 10.0) -> None:
        """Stop the writer, optionally flushing what is still queued.

        ``drain=True`` on graceful shutdown means a clean exit does not lose the
        last few seconds of audit trail.
        """
        if not self._running:
            return
        self._running = False
        self._not_empty.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=timeout_s)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        if drain:
            try:
                await asyncio.wait_for(self._flush_all(), timeout=timeout_s)
            except (TimeoutError, Exception) as exc:
                log.warning("db_writer.drain_incomplete", error=str(exc))
        log.info("db_writer.stopped", written=self._written, dropped=self._dropped)

    # -- producer side -------------------------------------------------------

    def enqueue(self, obj: Base, *, critical: bool = False) -> bool:
        """Queue one row. Non-blocking; returns False if the row was dropped.

        Callers on the trading path ignore the return value by design -- there is
        nothing useful they could do about a full queue, and checking it would
        add a branch to the hot path for no benefit. Drops are counted, reported
        in summaries, and make this component unhealthy.
        """
        # A dead writer is the one failure this queue cannot absorb, so it is
        # checked here -- on the only path that always runs -- rather than by a
        # health check nobody reads. Restarting is safe: start() is idempotent
        # on _running, which is cleared first so it takes effect.
        if self._running and self._task is not None and self._task.done():
            self._revive()

        if self.depth >= self._max_size and not self._make_room(obj, critical=critical):
            return False

        self._seq += 1
        item = _Item(obj, critical, self._seq, time.monotonic())
        (self._critical if critical else self._normal).append(item)
        if not self._not_empty.is_set():
            self._not_empty.set()
        return True

    def _make_room(self, obj: Base, *, critical: bool) -> bool:
        """Make space for one row, or refuse it. Returns whether to enqueue.

        A critical row is never refused for want of space a non-critical row is
        occupying, and is never evicted to make room for one.
        """
        if self._normal:
            evicted = self._normal.popleft()
            self._note_drop(type(evicted.obj).__name__, critical=False)
            return True
        if not critical:
            # Nothing but critical rows is queued: this row is the cheapest
            # thing in the system to lose.
            self._note_drop(type(obj).__name__, critical=False)
            return False
        if len(self._critical) >= self._max_size * _CRITICAL_CEILING_FACTOR:
            self._note_drop("critical_queue_full", critical=True)
            return False
        # Over the nominal size but under the ceiling: the audit trail is worth
        # more than the limit, and the health gate has already stopped trading.
        return True

    def enqueue_many(self, objs: list[Base], *, critical: bool = False) -> int:
        return sum(1 for obj in objs if self.enqueue(obj, critical=critical))

    # -- drop accounting -----------------------------------------------------

    def _note_drop(self, kind: str, *, critical: bool) -> None:
        now = time.monotonic()
        self._dropped += 1
        self._last_drop_at = now
        metrics.increment(M_DB_DROPS, kind="critical_queue_full" if critical else kind)
        if critical:
            self._critical_drops.append(now)
            self._trim(self._critical_drops, now)

        if not self._burst_open:
            self._burst_open = True
            self._last_drop_report_at = now
            log.error(
                "db_writer.dropping",
                first=kind, critical=critical,
                queued=self.depth, dropped_total=self._dropped,
                note="further drops are summarised, not logged individually",
            )
            return

        self._drop_window[(critical, kind)] += 1
        if now - self._last_drop_report_at >= _DROP_REPORT_INTERVAL_S:
            self._report_drops(now)

    def _report_drops(self, now: float | None = None) -> None:
        """Emit one summary per interval, and close a burst that has ended."""
        now = time.monotonic() if now is None else now
        if not self._burst_open:
            return
        if now - self._last_drop_report_at < _DROP_REPORT_INTERVAL_S:
            return
        if self._drop_window:
            log.error(
                "db_writer.drop_summary",
                window_s=round(now - self._last_drop_report_at, 1),
                rows=sum(self._drop_window.values()),
                by_kind={
                    f"{'critical' if crit else 'evicted'}:{kind}": count
                    for (crit, kind), count in sorted(
                        self._drop_window.items(), key=lambda kv: -kv[1]
                    )
                },
                queued=self.depth, dropped_total=self._dropped,
            )
            self._drop_window.clear()
            self._last_drop_report_at = now
            return
        # An interval with nothing in it means the flood stopped; the next drop
        # opens a new burst and says so.
        self._burst_open = False

    @staticmethod
    def _trim(stamps: deque[float], now: float, window_s: float = 60.0) -> None:
        while stamps and now - stamps[0] > window_s:
            stamps.popleft()

    # -- consumer side -------------------------------------------------------

    async def _run(self) -> None:
        """Drain the queue until stopped.

        Wrapped so that *any* exit is recorded. This task was created
        fire-and-forget and nothing held a reference to it, so when it stopped
        the engine kept running with ``_running`` still True, ``stats()`` still
        reporting "running", and every row silently dropped once the queue
        filled. A CancelledError is a BaseException and slipped past the
        ``except Exception`` inside the loop without leaving a line behind.
        """
        try:
            await self._drain_forever()
        except asyncio.CancelledError:
            log.warning("db_writer.cancelled", queued=self.depth)
            raise
        except BaseException as exc:
            log.critical(
                "db_writer.died",
                error=str(exc), error_type=type(exc).__name__,
                queued=self.depth, exc_info=True,
            )
            raise

    async def _drain_forever(self) -> None:
        while self._running:
            # A timeout simply means the flush interval elapsed with nothing
            # queued, which is the common idle case rather than an error.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._not_empty.wait(), timeout=self._flush_interval)
            self._not_empty.clear()
            metrics.gauge(M_DB_QUEUE_DEPTH, self.depth)
            # Here rather than only on the drop path, so the tail of a burst is
            # still reported once the drops stop.
            self._report_drops()
            try:
                await self._flush_once()
            except Exception as exc:
                log.error("db_writer.flush_failed", error=str(exc), exc_info=True)
                await asyncio.sleep(min(5.0, self._flush_interval * 4))

    def _revive(self) -> None:
        """Restart a writer task that stopped without being asked to.

        Rate-limited, because if it dies immediately on every attempt the
        restarts must not become the new flood. Whatever killed it is already
        logged by :meth:`_run`; this only ensures the next row has somewhere to
        go.
        """
        now = time.monotonic()
        if now - self._revived_at < _REVIVE_INTERVAL_S:
            return
        self._revived_at = now
        self._revivals += 1
        log.error(
            "db_writer.reviving",
            revivals=self._revivals, queued=self.depth, dropped=self._dropped,
        )
        self._running = False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._running = True
        self._task = loop.create_task(self._run(), name="db-writer")

    async def _take_batch(self) -> list[_Item]:
        """Up to one batch of rows, in the order they were enqueued."""
        batch: list[_Item]
        if self._retry_batch:
            batch, self._retry_batch = self._retry_batch, []
            return batch
        batch = []
        while len(batch) < self._batch_size and (self._critical or self._normal):
            if not self._normal:
                batch.append(self._critical.popleft())
            elif not self._critical:
                batch.append(self._normal.popleft())
            elif self._critical[0].seq <= self._normal[0].seq:
                batch.append(self._critical.popleft())
            else:
                batch.append(self._normal.popleft())
        return batch

    async def _flush_once(self) -> None:
        batch = await self._take_batch()
        if not batch:
            return
        try:
            async with self._db.session() as session:
                await self._write_batch(session, batch)
            self._record_commit(len(batch))
            self._retry_attempts = 0
            metrics.increment(M_DB_WRITES, len(batch))
        except (IntegrityError, DataError) as exc:
            # A row the database refuses will not become acceptable on a retry:
            # it conflicts with what is stored, or it does not fit the column.
            # Re-sending the same batch three times just loses the same 500 rows
            # more slowly. The batch is replayed row by row instead, so the one
            # offending row is dropped and the other 499 are written.
            await self._flush_individually(batch, exc)
        except Exception as exc:
            self._retry_attempts += 1
            if self._retry_attempts <= 3:
                self._retry_batch = batch
                log.warning(
                    "db_writer.batch_retry", attempt=self._retry_attempts, rows=len(batch),
                    error=str(exc), error_type=type(exc).__name__,
                )
            else:
                # Abandon the batch, but log every critical row in full so the
                # information survives in the log file even though the write
                # failed. Non-critical rows are simply counted.
                critical = [i for i in batch if i.critical]
                for item in critical:
                    log.error(
                        "db_writer.row_abandoned",
                        table=type(item.obj).__tablename__,
                        row=_safe_dict(item.obj),
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                self._dropped += len(batch)
                metrics.increment(M_DB_DROPS, len(batch), kind="write_failed")
                self._retry_attempts = 0
                log.error(
                    "db_writer.batch_abandoned",
                    rows=len(batch), error=str(exc), error_type=type(exc).__name__,
                )

    async def _write_batch(self, session: Any, batch: list[_Item]) -> None:
        """Write a batch in round trips set by the number of tables, not rows.

        ``session.merge()`` is correct but asks the database whether each row
        exists, one row at a time. Asking once per table instead -- and merging
        only the rows that came back -- is the same question with the same
        answer, in one round trip rather than five hundred.
        """
        # The primed rows are held for the whole batch, not just looked at. The
        # session's identity map keeps only weak references, so rows nobody
        # holds are collected the moment the priming loop ends -- and then every
        # merge below misses the map and asks the database again, which is the
        # exact cost this method exists to avoid.
        stored = await self._prime(session, batch)
        pending: dict[tuple[Any, tuple[Any, ...]], Base] = {}

        for item in batch:
            obj = item.obj
            key = _identity(obj)
            if key is None:
                # A surrogate key the database has not assigned yet: there is
                # nothing to merge against, and `add` avoids the SELECT that
                # `merge` would issue per row.
                session.add(obj)
                continue
            if _is_composite(type(obj)):
                # Rare, and not worth a second lookup shape: these tables see a
                # handful of rows a day between them.
                await session.merge(obj)
                continue
            if key in pending:
                # The same row enqueued twice in one batch -- a decision and the
                # execution result merged over it, an order and its next
                # transition. Applying the later attributes to the instance
                # already staged keeps merge's "last write wins, field by field"
                # semantics without a flush per duplicate.
                _copy_set_columns(obj, pending[key])
                continue
            if key in stored:
                await session.merge(obj)  # identity map hit: no SELECT
                continue
            clone = _clone(obj)
            session.add(clone)
            pending[key] = clone

    async def _prime(
        self, session: Any, batch: list[_Item]
    ) -> dict[tuple[Any, tuple[Any, ...]], Base]:
        """Load the rows in this batch that already exist, one query per table.

        Returns them by identity rather than as a set of keys, because the
        caller has to keep them alive: the identity map they were loaded into
        holds weak references, and a row that is collected is a row the next
        ``merge`` has to fetch all over again.
        """
        wanted: dict[type[Base], set[Any]] = {}
        for item in batch:
            key = _identity(item.obj)
            if key is None or _is_composite(type(item.obj)):
                continue
            wanted.setdefault(type(item.obj), set()).add(key[1][0])

        stored: dict[tuple[Any, tuple[Any, ...]], Base] = {}
        for cls, values in wanted.items():
            attr = getattr(cls, _pk_keys(cls)[0])
            ordered = list(values)
            for start in range(0, len(ordered), _PRIME_CHUNK):
                chunk = ordered[start : start + _PRIME_CHUNK]
                rows = (
                    await session.execute(select(cls).where(attr.in_(chunk)))
                ).scalars().all()
                for row in rows:
                    identity = _identity(row)
                    if identity is not None:
                        stored[identity] = row
        return stored

    async def _flush_individually(self, batch: list[_Item], cause: Exception) -> None:
        """Write a conflicted batch one row at a time.

        Slower by design and only reached after a constraint violation. The
        alternative is discarding every row that happened to share a batch with
        a duplicate, which is how a restart that re-ingests six hours of news
        silently loses six hours of everything else.
        """
        written = 0
        conflicted: dict[str, int] = {}
        failed: list[_Item] = []

        for item in batch:
            try:
                async with self._db.session() as session:
                    if _has_primary_key(item.obj):
                        await session.merge(item.obj)
                    else:
                        session.add(item.obj)
                written += 1
            except IntegrityError as exc:
                if not _is_duplicate(exc):
                    # Not every integrity error is a replay. A NOT NULL or
                    # foreign-key violation is a defect in what we built, and
                    # counting it as "already stored" is how a row that never
                    # once reached the database gets reported as a duplicate.
                    failed.append(item)
                    log.error(
                        "db_writer.row_rejected",
                        table=type(item.obj).__tablename__,
                        row=_safe_dict(item.obj),
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                    continue
                # Already stored. For an append-only row with a natural key this
                # is the normal outcome of a replay, not an error.
                table = type(item.obj).__tablename__
                conflicted[table] = conflicted.get(table, 0) + 1
            except DataError as exc:
                # The row does not fit the schema. Unlike a duplicate this is a
                # genuine defect -- a column too narrow, or a value that should
                # have been bounded upstream -- so it is reported, not counted
                # as a benign replay.
                failed.append(item)
                log.error(
                    "db_writer.row_rejected",
                    table=type(item.obj).__tablename__,
                    row=_safe_dict(item.obj),
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            except Exception as exc:
                failed.append(item)
                if item.critical:
                    log.error(
                        "db_writer.row_abandoned",
                        table=type(item.obj).__tablename__,
                        row=_safe_dict(item.obj),
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )

        if written:
            self._record_commit(written)
            metrics.increment(M_DB_WRITES, written)
        self._dropped += len(failed)
        self._retry_attempts = 0
        if failed:
            metrics.increment(M_DB_DROPS, len(failed), kind="write_failed")
        if conflicted:
            metrics.increment(M_DB_DROPS, sum(conflicted.values()), kind="already_stored")

        log.info(
            "db_writer.batch_split",
            rows=len(batch),
            written=written,
            already_stored=conflicted or None,
            failed=len(failed),
            cause=type(cause).__name__,
        )

    def _record_commit(self, rows: int) -> None:
        now = time.monotonic()
        self._written += rows
        self._last_commit_at = now
        self._commits.append((now, rows))
        while self._commits and now - self._commits[0][0] > 60.0:
            self._commits.popleft()

    async def flush(self) -> None:
        """Drain the queue and return once everything pending has been attempted.

        Public because callers legitimately need a checkpoint -- a test that has
        just enqueued rows, or a shutdown path that wants the queue empty before
        it reports clean. It is not on the hot path: the background loop flushes
        on its own interval and nothing on the order path waits for this.
        """
        await self._flush_all()

    async def _flush_all(self) -> None:
        while self._critical or self._normal or self._retry_batch:
            before = self.depth + len(self._retry_batch)
            await self._flush_once()
            if self.depth + len(self._retry_batch) >= before:
                break  # not making progress; stop rather than spin

    # -- introspection -------------------------------------------------------

    @property
    def depth(self) -> int:
        return len(self._critical) + len(self._normal)

    def _seconds_since_commit(self) -> float:
        since = self._last_commit_at if self._last_commit_at is not None else self._created_at
        return time.monotonic() - since

    def _oldest_queued_age_s(self) -> float | None:
        heads = [queue[0].queued_at for queue in (self._critical, self._normal) if queue]
        return None if not heads else time.monotonic() - min(heads)

    def assess(self) -> tuple[bool, str]:
        """Whether rows are actually reaching the database, and if not, why."""
        if self._running and (self._task is None or self._task.done()):
            return False, "the drain task is not running"
        now = time.monotonic()
        self._trim(self._critical_drops, now)
        if self._critical_drops:
            return False, (
                f"{len(self._critical_drops)} critical rows dropped in the last minute"
            )
        queued = self.depth
        if queued and self._seconds_since_commit() > _STALL_ALERT_S:
            return False, (
                f"{queued:,} rows queued and nothing committed for "
                f"{self._seconds_since_commit():.0f}s"
            )
        return True, ""

    def healthy(self) -> bool:
        """Health probe for the component registry. Logs each transition once."""
        ok, detail = self.assess()
        if ok != self._healthy:
            self._healthy = ok
            if ok:
                log.info("db_writer.recovered", **self.stats())
            else:
                log.error("db_writer.stalled", detail=detail, **self.stats())
        return ok

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        self._trim(self._critical_drops, now)
        age = self._oldest_queued_age_s()
        return {
            "queued": self.depth,
            "queued_critical": len(self._critical),
            "capacity": self._max_size,
            "written": self._written,
            "dropped": self._dropped,
            # Whether the task is actually alive, not merely whether it was
            # asked to be. These disagreed for sixty-six hours.
            "running": self._running and self._task is not None
            and not self._task.done(),
            "requested_running": self._running,
            "revivals": self._revivals,
            "retrying": bool(self._retry_batch),
            # Progress, which is the only thing that distinguishes a writer that
            # is working from one that is merely present.
            "seconds_since_commit": (
                round(self._seconds_since_commit(), 1)
                if self._last_commit_at is not None
                else None
            ),
            "committed_last_minute": sum(rows for _, rows in self._commits),
            "critical_drops_last_minute": len(self._critical_drops),
            "oldest_queued_age_s": round(age, 1) if age is not None else None,
        }


def _pk_keys(cls: type[Base]) -> list[str]:
    """Mapped attribute names of the primary-key columns, in order."""
    mapper = sa_inspect(cls)
    return [mapper.get_property_by_column(column).key for column in mapper.primary_key]


def _is_composite(cls: type[Base]) -> bool:
    return len(sa_inspect(cls).primary_key) > 1


def _identity(obj: Base) -> tuple[Any, tuple[Any, ...]] | None:
    """``(class, key)`` when every primary-key value is present, else None.

    Autoincrement surrogate keys are ``None`` until the database assigns them,
    which is precisely the case that must be inserted rather than merged.
    """
    values = tuple(getattr(obj, key, None) for key in _pk_keys(type(obj)))
    if any(value is None for value in values):
        return None
    return (type(obj), values)


def _has_primary_key(obj: Base) -> bool:
    return _identity(obj) is not None


def _is_duplicate(exc: IntegrityError) -> bool:
    """Whether this integrity error means "already stored" rather than "wrong".

    Only a uniqueness violation is the benign replay the split path is built
    for. PostgreSQL says so in SQLSTATE 23505; SQLite says so in the message.
    Anything else -- a NOT NULL column, a foreign key with no parent -- is a row
    that is never going to be accepted, and must be reported rather than tallied
    as a duplicate that was already safely written.
    """
    orig = getattr(exc, "orig", None)
    sqlstate = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
    if sqlstate:
        return str(sqlstate) == "23505"
    text = str(orig or exc).lower()
    return "unique constraint" in text or "duplicate key" in text


def _column_state(obj: Base) -> dict[str, Any]:
    """The column attributes actually set on this instance.

    ``merge`` copies only what the source has set, leaving other columns as the
    database has them. Anything reproducing merge has to make the same
    distinction between "set to None" and "never set".
    """
    mapper = sa_inspect(type(obj))
    present = sa_inspect(obj).dict
    return {attr.key: present[attr.key] for attr in mapper.column_attrs if attr.key in present}


def _clone(obj: Base) -> Base:
    """A detached copy carrying the same set columns.

    The caller's instance is never added to a session: ``merge`` never attached
    it either, and a producer that reuses or mutates a row object afterwards
    must not find it quietly bound to a transaction that has since closed.
    """
    return type(obj)(**_column_state(obj))


def _copy_set_columns(src: Base, dst: Base) -> None:
    for key, value in _column_state(src).items():
        setattr(dst, key, value)


def _safe_dict(obj: Base) -> dict[str, Any]:
    try:
        return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in obj.to_dict().items()}
    except Exception:
        return {"repr": repr(obj)}


_writer: AsyncWriter | None = None


def get_writer() -> AsyncWriter:
    global _writer
    if _writer is None:
        _writer = AsyncWriter()
    return _writer


def reset_writer_singleton() -> None:
    global _writer
    _writer = None
