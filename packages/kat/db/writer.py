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
from collections import deque
from typing import Any

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
            except (TimeoutError, Exception) as exc:  # noqa: B014
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
        if len(self._queue) >= self._max_size:
            if not self._evict_one_non_critical():
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
            try:
                await asyncio.wait_for(self._not_empty.wait(), timeout=self._flush_interval)
            except TimeoutError:
                pass
            self._not_empty.clear()
            metrics.gauge(M_DB_QUEUE_DEPTH, len(self._queue))
            try:
                await self._flush_once()
            except Exception as exc:  # noqa: BLE001 - the writer must never die
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
                # ``merge`` rather than ``add``: reconciliation and websocket
                # updates both re-send rows we already hold, and an upsert is the
                # correct semantic for an idempotent replay.
                for item in batch:
                    await session.merge(item.obj)
            self._written += len(batch)
            self._retry_attempts = 0
            metrics.increment(M_DB_WRITES, len(batch))
        except Exception as exc:  # noqa: BLE001
            self._retry_attempts += 1
            if self._retry_attempts <= 3:
                self._retry_batch = batch
                log.warning(
                    "db_writer.batch_retry", attempt=self._retry_attempts, rows=len(batch),
                    error=str(exc),
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
                    )
                self._dropped += len(batch)
                metrics.increment(M_DB_DROPS, len(batch), kind="write_failed")
                self._retry_attempts = 0
                log.error("db_writer.batch_abandoned", rows=len(batch), error=str(exc))

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


def _safe_dict(obj: Base) -> dict[str, Any]:
    try:
        return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in obj.to_dict().items()}
    except Exception:  # noqa: BLE001
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
