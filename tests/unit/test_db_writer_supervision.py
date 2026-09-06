"""A writer that stops working must say so, and must not fail quietly.

Written from a real failure. The drain task was created fire-and-forget, nothing
held a reference to check on it, and ``_running`` stayed True after the task
stopped -- so ``stats()`` kept reporting "running" while every row was dropped
once the queue filled.

It ran that way for sixty-six hours and lost 2,084,222 rows. The only symptom was
42,262 identical ``critical_queue_full`` errors, which read as a slow database
rather than an absent writer, and the engine went on trading and deciding
throughout with no persisted record of any of it.

``asyncio.CancelledError`` derives from BaseException, so it passed straight
through the ``except Exception`` guarding the flush and left nothing behind.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from kat.db.models import AuditLog
from kat.db.writer import AsyncWriter

pytestmark = pytest.mark.asyncio


async def _kill(writer: AsyncWriter) -> asyncio.Task:
    """Stop the drain task the way a stray cancellation would."""
    task = writer._task
    assert task is not None
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return task


class TestStatsTellTheTruth:
    async def test_running_means_the_task_is_alive(self, sqlite_db):
        """Not merely that start() was called. These disagreed for 66 hours."""
        writer = AsyncWriter(sqlite_db)
        await writer.start()
        assert writer.stats()["running"] is True

        await _kill(writer)

        assert writer.stats()["running"] is False
        assert writer.stats()["requested_running"] is True
        await writer.stop(drain=False)


class TestADeadWriterIsRevived:
    async def test_the_next_row_restarts_it(self, sqlite_db):
        """Checked on the producer path, the only one that always runs.

        A health check nobody reads would not have caught this: what everyone
        eventually noticed was the queue filling, sixty-six hours late.
        """
        writer = AsyncWriter(sqlite_db)
        await writer.start()
        dead = await _kill(writer)

        writer.enqueue(AuditLog(actor="t", action="revive", subject="x"), critical=True)

        assert writer._task is not dead
        assert writer.stats()["revivals"] == 1
        await writer.stop(drain=False)

    async def test_rows_are_written_again_after_a_revival(self, sqlite_db):
        """The point of reviving is not the counter, it is that data lands."""
        from sqlalchemy import func, select

        writer = AsyncWriter(sqlite_db)
        await writer.start()
        await _kill(writer)

        writer.enqueue(AuditLog(actor="t", action="after", subject="x"), critical=True)
        await asyncio.sleep(1.0)
        await writer.stop()

        async with sqlite_db.session() as session:
            count = (
                await session.execute(select(func.count()).select_from(AuditLog))
            ).scalar()
        assert count >= 1

    async def test_revival_is_rate_limited(self, sqlite_db):
        """If it dies instantly every time, the restarts must not become the
        flood that the dropped rows already were."""
        writer = AsyncWriter(sqlite_db)
        await writer.start()
        for _ in range(5):
            await _kill(writer)
            writer.enqueue(AuditLog(actor="t", action="spam", subject="x"))

        assert writer.stats()["revivals"] == 1
        await writer.stop(drain=False)

    async def test_a_healthy_writer_is_never_revived(self, sqlite_db):
        writer = AsyncWriter(sqlite_db)
        await writer.start()
        for _ in range(20):
            writer.enqueue(AuditLog(actor="t", action="ok", subject="x"))
        assert writer.stats()["revivals"] == 0
        await writer.stop(drain=False)
