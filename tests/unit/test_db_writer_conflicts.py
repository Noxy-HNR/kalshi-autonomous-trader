"""One duplicate row must not destroy the batch it travelled in.

The failure this pins: a restart re-ingests the last six hours of news, every
one of those articles already exists, and the unique constraint on
``article_id`` fails the whole 500-row batch. Retrying the identical batch three
times loses the same 500 rows more slowly, and 499 of them had nothing wrong
with them.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from kat.config.settings import DatabaseSettings
from kat.db.base import Base
from kat.db.engine import Database
from kat.db.models import NewsArticle
from kat.db.writer import AsyncWriter


@pytest.fixture
async def database(tmp_path):
    db = Database(DatabaseSettings(url=f"sqlite+aiosqlite:///{tmp_path / 'w.db'}"))
    await db.connect()
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield db
    await db.disconnect()


def article(n: int) -> NewsArticle:
    return NewsArticle(
        article_id=f"art_{n:04d}",
        content_hash=f"{n:064d}",
        title=f"headline {n}",
        source=f"src_{n % 3}",
        source_reliability=0.8,
        quality="ok",
    )


class TestDuplicateHandling:
    async def test_a_duplicate_does_not_take_the_rest_of_the_batch_with_it(self, database):
        writer = AsyncWriter(database)
        await writer.start()

        writer.enqueue(article(1))
        await writer.flush()

        # A replay: one row already stored, plus nine that are not.
        for n in range(1, 11):
            writer.enqueue(article(n))
        await writer.flush()
        await writer.stop()

        async with database.session() as session:
            rows = (await session.execute(select(NewsArticle.article_id))).scalars().all()

        assert len(rows) == 10, "the nine new rows must survive the one duplicate"
        assert len(set(rows)) == 10, "and the duplicate must not have been stored twice"

    async def test_the_writer_keeps_working_after_a_conflict(self, database):
        writer = AsyncWriter(database)
        await writer.start()
        writer.enqueue(article(1))
        await writer.flush()
        writer.enqueue(article(1))
        await writer.flush()
        writer.enqueue(article(2))
        await writer.flush()
        await writer.stop()

        async with database.session() as session:
            count = len((await session.execute(select(NewsArticle.id))).scalars().all())
        assert count == 2
