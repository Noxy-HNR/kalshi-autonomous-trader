"""Which markets have been quoted is remembered, not re-counted from the whole table.

Every fifteen minutes the settlement poller asked the database for every distinct
ticker in ``quotes``. That table is the largest in the schema and only grows, so
the question cost more every time it was asked. It now reads only what arrived
since the previous pass, with an overlap for rows written late, and keeps the set
it has already seen.

Run against a real database, because the property is about the query.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import delete

from kat.common.timeutil import utcnow_aware
from kat.db.models import Quote
from kat.engine.settlements import SettlementRecorder


def _quote(ticker: str, *, age: timedelta) -> Quote:
    return Quote(ticker=ticker, market_type="event", ts=utcnow_aware() - age, source="test")


async def test_a_market_seen_once_is_remembered_after_its_rows_are_gone(sqlite_db):
    recorder = SettlementRecorder(rest=None, writer=None, database=sqlite_db)  # type: ignore[arg-type]
    async with sqlite_db.session() as session:
        session.add(_quote("OLD", age=timedelta(days=2)))

    assert await recorder._pending() == ["OLD"]

    async with sqlite_db.session() as session:
        await session.execute(delete(Quote))

    # Not re-read from the table -- it is not there any more -- but still owed an outcome.
    assert await recorder._pending() == ["OLD"]


async def test_a_row_written_late_inside_the_overlap_is_still_found(sqlite_db):
    """Rows carry the time they were observed, not the time they landed."""
    recorder = SettlementRecorder(rest=None, writer=None, database=sqlite_db)  # type: ignore[arg-type]
    async with sqlite_db.session() as session:
        session.add(_quote("FIRST", age=timedelta(minutes=1)))
    await recorder._pending()

    async with sqlite_db.session() as session:
        session.add(_quote("LATE", age=timedelta(minutes=30)))

    assert "LATE" in await recorder._pending()


async def test_a_row_that_lands_after_its_window_is_caught_by_the_full_rescan(sqlite_db, monkeypatch):
    """A stalled writer can hold a quote past the overlap; the rescan still finds it."""
    import kat.engine.settlements as settlements

    monkeypatch.setattr(settlements, "_FULL_RESCAN_EVERY", 3)
    recorder = SettlementRecorder(rest=None, writer=None, database=sqlite_db)  # type: ignore[arg-type]
    async with sqlite_db.session() as session:
        session.add(_quote("FIRST", age=timedelta(minutes=1)))
    await recorder._pending()  # pass 0: full

    async with sqlite_db.session() as session:
        session.add(_quote("STALLED", age=timedelta(hours=5)))  # observed long before it landed

    assert "STALLED" not in await recorder._pending()  # pass 1: incremental, outside the overlap
    assert "STALLED" not in await recorder._pending()  # pass 2: incremental
    assert "STALLED" in await recorder._pending()      # pass 3: full rescan
