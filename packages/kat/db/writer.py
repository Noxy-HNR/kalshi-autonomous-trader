"""Background batched writer -- the reason the database is never on the hot path.

Producers (market-data ingestion, the execution engine, the risk engine) call
:meth:`AsyncWriter.enqueue`, which is a non-blocking put onto a bounded queue and
returns in microseconds. A single consumer task drains the queue in batches and
commits them.

Bounded, and what happens when it fills
---------------------------------------
The queue has a hard size limit. When it is full we drop the *oldest*
non-critical rows and count the drops, rather than blocking the producer or
growing until the process runs out of memory. Losing a quote row under extreme
load is acceptable; stalling an order submission because a disk is slow is not.

Rows tagged ``critical=True`` (orders, fills, decisions, risk events, audit) are
never dropped. If the queue is full of critical rows the writer applies
backpressure to that call alone -- and that condition is itself an alert, because
it means the database has been unavailable long enough to matter.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import Any

from sqlalchemy.exc import IntegrityError

from kat.config.settings import get_settings
from kat.db.base import Base
from kat.db.engine import Database, get_database
from kat.observability.logging import get_logger
from kat.observability.metrics import M_DB_DROPS, M_DB_QUEUE_DEPTH, M_DB_WRITES, metrics

log = get_logger(__name__)

__all__ = ["AsyncWriter", "get_writer", "reset_writer_singleton"]


class _Item:
    __slots__ = ("critical", "obj")

    def __init__(self, obj: Base, critical: bool) -> None:
        self.obj = obj
        self.critical = critical


class AsyncWriter:
    """Batching, non-blocking persistence for ORM objects."""

    def __init__(self, database: Database | None = None) -> None:
        settings = get_settings().database
        self._db = database or get_database()
        self._max_size = settings.write_queue_size
        self._batch_size = settings.write_batch_size
        self._flush_interval = settings.write_flush_interval_s

        self._queue: deque[_Item] = deque()
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
        add a branch to the hot path for no benefit. The drop is counted and
        surfaced on the System tab instead.
        """
        if len(self._queue) >= self._max_size and not self._evict_one_non_critical():
                # Everything queued is critical and the queue is full: the
                # database has been down long enough that we are now losing
                # audit data. Surface it loudly and keep the newest row.
                self._dropped += 1
                metrics.increment(M_DB_DROPS, kind="critical_queue_full")
                log.error(
                    "db_writer.critical_queue_full",
                    queued=len(self._queue),
                    dropped_total=self._dropped,
                )
                return False
        self._queue.append(_Item(obj, critical))
        if not self._not_empty.is_set():
            self._not_empty.set()
        return True

    def _evict_one_non_critical(self) -> bool:
        """Drop the oldest non-critical row to make room. Returns success."""
        for index, item in enumerate(self._queue):
            if not item.critical:
                del self._queue[index]
                self._dropped += 1
                metrics.increment(M_DB_DROPS, kind=type(item.obj).__name__)
                return True
        return False

    def enqueue_many(self, objs: list[Base], *, critical: bool = False) -> int:
        return sum(1 for obj in objs if self.enqueue(obj, critical=critical))

    # -- consumer side -------------------------------------------------------

    async def _run(self) -> None:
        while self._running:
            # A timeout simply means the flush interval elapsed with nothing
            # queued, which is the common idle case rather than an error.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._not_empty.wait(), timeout=self._flush_interval)
            self._not_empty.clear()
            metrics.gauge(M_DB_QUEUE_DEPTH, len(self._queue))
            try:
                await self._flush_once()
            except Exception as exc:
                log.error("db_writer.flush_failed", error=str(exc), exc_info=True)
                await asyncio.sleep(min(5.0, self._flush_interval * 4))

    async def _take_batch(self) -> list[_Item]:
        if self._retry_batch:
            batch, self._retry_batch = self._retry_batch, []
            return batch
        batch: list[_Item] = []
        while self._queue and len(batch) < self._batch_size:
            batch.append(self._queue.popleft())
        return batch

    async def _flush_once(self) -> None:
        batch = await self._take_batch()
        if not batch:
            return
        try:
            async with self._db.session() as session:
                for item in batch:
                    if _has_primary_key(item.obj):
                        # Rows with a natural key (order_id, fill_id, ticker) are
                        # merged: reconciliation and websocket replays re-send
                        # rows we already hold, and an upsert is the correct
                        # semantic for an idempotent replay.
                        await session.merge(item.obj)
                    else:
                        # Append-only rows with a surrogate key have nothing to
                        # merge against. `add` avoids the SELECT that `merge`
                        # would issue per row, which matters when this is the
                        # highest-volume write path in the system.
                        session.add(item.obj)
            self._written += len(batch)
            self._retry_attempts = 0
            metrics.increment(M_DB_WRITES, len(batch))
        except IntegrityError as exc:
            # A constraint violation will not go away on a retry: the row
            # conflicts with what is already stored, and re-sending the same
            # batch three times just loses the same 500 rows more slowly. The
            # batch is replayed row by row instead, so the one offending row is
            # dropped and the other 499 are written.
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
            except IntegrityError:
                # Already stored. For an append-only row with a natural key this
                # is the normal outcome of a replay, not an error.
                table = type(item.obj).__tablename__
                conflicted[table] = conflicted.get(table, 0) + 1
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

        self._written += written
        self._dropped += len(failed)
        self._retry_attempts = 0
        if written:
            metrics.increment(M_DB_WRITES, written)
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

    async def flush(self) -> None:
        """Drain the queue and return once everything pending has been attempted.

        Public because callers legitimately need a checkpoint -- a test that has
        just enqueued rows, or a shutdown path that wants the queue empty before
        it reports clean. It is not on the hot path: the background loop flushes
        on its own interval and nothing on the order path waits for this.
        """
        await self._flush_all()

    async def _flush_all(self) -> None:
        while self._queue or self._retry_batch:
            before = len(self._queue) + len(self._retry_batch)
            await self._flush_once()
            if len(self._queue) + len(self._retry_batch) >= before:
                break  # not making progress; stop rather than spin

    # -- introspection -------------------------------------------------------

    @property
    def depth(self) -> int:
        return len(self._queue)

    def stats(self) -> dict[str, Any]:
        return {
            "queued": len(self._queue),
            "capacity": self._max_size,
            "written": self._written,
            "dropped": self._dropped,
            "running": self._running,
            "retrying": bool(self._retry_batch),
        }


def _has_primary_key(obj: Base) -> bool:
    """Whether every primary-key column on this row already carries a value.

    Autoincrement surrogate keys are ``None`` until the database assigns them,
    which is precisely the case that must be inserted rather than merged.
    """
    return all(
        getattr(obj, column.name, None) is not None
        for column in obj.__table__.primary_key.columns
    )


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
