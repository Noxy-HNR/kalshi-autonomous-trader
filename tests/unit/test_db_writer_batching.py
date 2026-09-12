"""A batch must cost round trips, not rows -- and a stuck writer must say so.

Written from the 2026-09-11 incident. The writer was alive and had committed
about 1,500 rows in an hour while 70,000 were produced: ``_flush_once`` asked the
database whether each row already existed, one row at a time, and each of those
round trips needed its own turn of an event loop that the trading loop had
saturated. The queue filled with critical rows, every further enqueue scanned all
20,000 of them looking for something evictable, and every failure wrote an error
line. 757,042 of those lines and 2.1 million dropped rows later, ``stats()`` still
reported the writer as running, because nothing measured whether rows were
landing.

So the tests here are about cost and about truthfulness: the number of statements
a batch emits must not grow with the batch, the queue must never be walked, the
drops must be counted rather than narrated, and "healthy" must mean rows are
reaching the database.
"""

from __future__ import annotations

import asyncio
import math
import time

import pytest
from sqlalchemy import event, select

from kat.config.settings import DatabaseSettings
from kat.db.base import Base
from kat.db.engine import Database
from kat.db.models import AuditLog, Decision, Order, RiskEvent
from kat.db.writer import _PRIME_CHUNK, AsyncWriter

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def database(tmp_path):
    db = Database(DatabaseSettings(url=f"sqlite+aiosqlite:///{tmp_path / 'w.db'}"))
    await db.connect()
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield db
    await db.disconnect()


class _Statements:
    """Counts the SQL actually sent, which is the thing that cost us an hour."""

    def __init__(self, database: Database) -> None:
        self.seen: list[str] = []
        self._engine = database.engine.sync_engine
        event.listen(self._engine, "before_cursor_execute", self._record)

    def _record(self, conn, cursor, statement, parameters, context, executemany) -> None:
        self.seen.append(statement.split()[0].upper())

    def stop(self) -> None:
        event.remove(self._engine, "before_cursor_execute", self._record)

    def count(self, verb: str) -> int:
        return sum(1 for s in self.seen if s == verb)


def decision(n: int, **kw) -> Decision:
    return Decision(
        decision_id=f"dec_{n:05d}",
        market_type="event",
        ticker=f"TICK-{n % 7}",
        decision="approved",
        reason_code="APPROVED",
        **kw,
    )


def order(order_id: str, state: str, **kw) -> Order:
    return Order(
        order_id=order_id,
        client_order_id=f"cli_{order_id}",
        market_type="event",
        ticker="TICK-1",
        state=state,
        order_type="limit",
        book_side="yes",
        requested_qty_qc=100,
        **kw,
    )


def risk_event(reason: str) -> RiskEvent:
    return RiskEvent(market_type="event", reason_code=reason)


class TestRoundTripsDoNotScaleWithRows:
    async def test_a_ten_times_bigger_batch_does_not_cost_ten_times_the_queries(
        self, database
    ):
        """The defect, stated as a measurement.

        Five hundred rows used to mean five hundred SELECTs, one per row, each
        waiting for the event loop. The batch is one question per table now.
        """
        writer = AsyncWriter(database)
        writer._batch_size = 1000

        small = _Statements(database)
        for n in range(50):
            writer.enqueue(decision(n), critical=True)
        await writer.flush()
        small.stop()

        large = _Statements(database)
        for n in range(1000, 1500):
            writer.enqueue(decision(n), critical=True)
        await writer.flush()
        large.stop()

        # Lookups are bounded by chunks of keys, not by rows: ten times the rows
        # costs at most ceil(500/200) queries instead of ten times the queries.
        assert small.count("SELECT") <= 1
        assert large.count("SELECT") <= math.ceil(500 / _PRIME_CHUNK), large.seen
        assert large.count("SELECT") * 100 < 500, "nowhere near one lookup per row"

    async def test_rows_already_stored_still_cost_one_lookup(self, database):
        writer = AsyncWriter(database)
        writer._batch_size = 1000

        for n in range(200):
            writer.enqueue(decision(n), critical=True)
        await writer.flush()

        counted = _Statements(database)
        for n in range(200):
            writer.enqueue(decision(n, strategy_id="second-pass"), critical=True)
        await writer.flush()
        counted.stop()

        assert counted.count("SELECT") <= 2
        async with database.session() as session:
            row = await session.get(Decision, "dec_00007")
        assert row is not None and row.strategy_id == "second-pass"

    async def test_several_tables_cost_one_lookup_each(self, database):
        writer = AsyncWriter(database)
        writer._batch_size = 1000

        counted = _Statements(database)
        for n in range(100):
            writer.enqueue(decision(n), critical=True)
            writer.enqueue(order(f"ord_{n}", "created"), critical=True)
            writer.enqueue(risk_event("ORDER_RATE_LIMIT"), critical=True)
        await writer.flush()
        counted.stop()

        # decisions and orders are looked up; risk_events has a surrogate key and
        # nothing to look up at all.
        assert counted.count("SELECT") <= 3, counted.seen


class TestMergeSemanticsArePreserved:
    async def test_a_partial_update_leaves_other_columns_alone(self, database):
        """merge copies what the caller set, not what it left unset."""
        writer = AsyncWriter(database)
        writer.enqueue(decision(1, strategy_id="first", model_id="m1"), critical=True)
        await writer.flush()

        writer.enqueue(decision(1, model_id="m2"), critical=True)
        await writer.flush()

        async with database.session() as session:
            row = await session.get(Decision, "dec_00001")
        assert row.model_id == "m2"
        assert row.strategy_id == "first", "an unset column must not be nulled"

    async def test_the_same_row_twice_in_one_batch_applies_in_order(self, database):
        """A decision and the execution result merged over it, in one batch."""
        writer = AsyncWriter(database)
        writer._batch_size = 1000
        writer.enqueue(decision(2, strategy_id="s", approved_qty_qc=100), critical=True)
        writer.enqueue(decision(2, order_id="ord_2"), critical=True)
        await writer.flush()

        async with database.session() as session:
            row = await session.get(Decision, "dec_00002")
        assert row.order_id == "ord_2"
        assert row.approved_qty_qc == 100
        assert row.strategy_id == "s"

    async def test_order_transitions_in_one_batch_end_in_the_last_state(self, database):
        writer = AsyncWriter(database)
        writer._batch_size = 1000
        for state in ("created", "submitted", "acknowledged"):
            writer.enqueue(order("ord_a", state), critical=True)
        await writer.flush()

        async with database.session() as session:
            row = await session.get(Order, "ord_a")
        assert row.state == "acknowledged"

    async def test_a_row_stored_by_an_earlier_batch_is_updated_not_duplicated(
        self, database
    ):
        writer = AsyncWriter(database)
        writer.enqueue(order("ord_b", "created"), critical=True)
        await writer.flush()
        writer.enqueue(order("ord_b", "filled"), critical=True)
        await writer.flush()

        async with database.session() as session:
            rows = (await session.execute(select(Order))).scalars().all()
        assert len(rows) == 1 and rows[0].state == "filled"

    async def test_the_callers_instance_is_never_attached_to_a_session(self, database):
        """merge never bound the caller's row object to a transaction.

        A producer that reuses or mutates the instance afterwards must not find
        it quietly persistent in a session that has since closed.
        """
        from sqlalchemy import inspect as sa_inspect

        writer = AsyncWriter(database)
        row = decision(3)
        writer.enqueue(row, critical=True)
        await writer.flush()

        state = sa_inspect(row)
        assert state.session is None
        assert state.key is None, "the caller's object must still be transient"

    async def test_surrogate_key_rows_are_inserted_each_time(self, database):
        writer = AsyncWriter(database)
        for _ in range(5):
            writer.enqueue(risk_event("INSUFFICIENT_EDGE"), critical=True)
        await writer.flush()

        async with database.session() as session:
            rows = (await session.execute(select(RiskEvent))).scalars().all()
        assert len(rows) == 5


class TestQueueOrderAndEviction:
    async def test_a_batch_comes_back_in_enqueue_order(self, database):
        """Critical and non-critical live in separate queues; order is global.

        An order's transitions are only correct in sequence, and a decision's
        execution result must not be written before the decision.
        """
        writer = AsyncWriter(database)
        writer.enqueue(decision(1), critical=True)
        writer.enqueue(AuditLog(actor="a", action="x", subject="s"))
        writer.enqueue(decision(2), critical=True)
        writer.enqueue(AuditLog(actor="b", action="y", subject="s"))

        batch = await writer._take_batch()
        assert [item.seq for item in batch] == [1, 2, 3, 4]
        assert [type(item.obj).__name__ for item in batch] == [
            "Decision", "AuditLog", "Decision", "AuditLog",
        ]

    async def test_the_oldest_non_critical_row_makes_room(self, database):
        writer = AsyncWriter(database)
        writer._max_size = 3
        for n in range(3):
            writer.enqueue(decision(n))  # non-critical
        writer.enqueue(decision(99), critical=True)

        assert writer.depth == 3
        assert writer.stats()["dropped"] == 1
        remaining = [item.obj.decision_id for item in await writer._take_batch()]
        assert "dec_00000" not in remaining, "the oldest non-critical row goes first"
        assert "dec_00099" in remaining

    async def test_critical_rows_are_not_evicted_and_may_exceed_the_size(self, database):
        """The audit trail is worth more than the nominal limit."""
        writer = AsyncWriter(database)
        writer._max_size = 4
        for n in range(10):
            assert writer.enqueue(decision(n), critical=True) is True

        assert writer.depth == 10
        assert writer.stats()["dropped"] == 0

    async def test_beyond_the_hard_ceiling_critical_rows_are_refused(self, database):
        writer = AsyncWriter(database)
        writer._max_size = 4  # ceiling is five times this
        for n in range(20):
            writer.enqueue(decision(n), critical=True)

        assert writer.enqueue(decision(999), critical=True) is False
        assert writer.stats()["critical_drops_last_minute"] == 1

    async def test_a_non_critical_row_is_refused_when_only_critical_rows_are_queued(
        self, database
    ):
        writer = AsyncWriter(database)
        writer._max_size = 2
        writer.enqueue(decision(1), critical=True)
        writer.enqueue(decision(2), critical=True)

        assert writer.enqueue(AuditLog(actor="a", action="x", subject="s")) is False
        assert writer.stats()["critical_drops_last_minute"] == 0


class TestDropsAreSummarised:
    async def test_a_flood_produces_one_line_then_summaries(self, database, monkeypatch):
        """757,042 identical error lines is not reporting, it is noise."""
        import kat.db.writer as module

        lines: list[tuple[str, dict]] = []

        class _Log:
            def __getattr__(self, level):
                def record(event, **kw):
                    lines.append((event, kw))
                return record

        monkeypatch.setattr(module, "log", _Log())

        writer = AsyncWriter(database)
        writer._max_size = 2
        writer.enqueue(decision(1))
        writer.enqueue(decision(2))
        for n in range(500):
            writer.enqueue(decision(100 + n))

        opened = [line for line in lines if line[0] == "db_writer.dropping"]
        summaries = [line for line in lines if line[0] == "db_writer.drop_summary"]
        assert len(opened) == 1, "one line opens the burst"
        assert summaries == [], "nothing to summarise until the interval elapses"

        writer._last_drop_report_at -= module._DROP_REPORT_INTERVAL_S + 1
        writer.enqueue(decision(9999))
        summaries = [line for line in lines if line[0] == "db_writer.drop_summary"]
        assert len(summaries) == 1
        assert summaries[0][1]["rows"] >= 500
        assert summaries[0][1]["by_kind"] == {"evicted:Decision": summaries[0][1]["rows"]}

    async def test_evicted_non_critical_rows_are_reported_too(self, database, monkeypatch):
        """These were silent: Kalshi quotes, trade prints and balance snapshots
        all stopped at the same millisecond and nothing said why."""
        import kat.db.writer as module

        lines: list[tuple[str, dict]] = []

        class _Log:
            def __getattr__(self, level):
                def record(event, **kw):
                    lines.append((event, kw))
                return record

        monkeypatch.setattr(module, "log", _Log())

        writer = AsyncWriter(database)
        writer._max_size = 1
        writer.enqueue(AuditLog(actor="a", action="x", subject="s"))
        writer.enqueue(AuditLog(actor="b", action="y", subject="s"))

        assert [line[0] for line in lines].count("db_writer.dropping") == 1
        assert lines[0][1]["critical"] is False


class TestHealthMeansRowsAreLanding:
    async def test_a_new_writer_with_nothing_to_do_is_healthy(self, database):
        writer = AsyncWriter(database)
        assert writer.healthy() is True

    async def test_a_backlog_that_is_not_draining_is_unhealthy(self, database):
        """The incident exactly: alive, running, committing a trickle."""
        writer = AsyncWriter(database)
        writer.enqueue(decision(1), critical=True)
        writer._created_at = time.monotonic() - 120

        ok, detail = writer.assess()
        assert ok is False
        assert "queued" in detail and "committed" in detail
        assert writer.stats()["running"] is False  # never started, and says so

    async def test_committing_again_clears_it(self, database):
        writer = AsyncWriter(database)
        writer.enqueue(decision(1), critical=True)
        writer._created_at = time.monotonic() - 120
        assert writer.healthy() is False

        await writer.flush()
        assert writer.healthy() is True
        assert writer.stats()["committed_last_minute"] == 1

    async def test_a_dropped_critical_row_is_unhealthy_on_its_own(self, database):
        writer = AsyncWriter(database)
        writer._max_size = 1
        for n in range(10):
            writer.enqueue(decision(n), critical=True)
        writer.enqueue(decision(99), critical=True)  # past the ceiling

        assert writer.assess()[0] is False
        assert "critical rows dropped" in writer.assess()[1]

    async def test_a_busy_writer_that_keeps_up_stays_healthy(self, database):
        writer = AsyncWriter(database)
        await writer.start()
        try:
            for n in range(300):
                writer.enqueue(decision(n), critical=True)
            await asyncio.sleep(1.0)
            assert writer.healthy() is True
            stats = writer.stats()
            assert stats["committed_last_minute"] >= 300
            assert stats["seconds_since_commit"] is not None
        finally:
            await writer.stop(drain=False)

    async def test_stats_report_the_oldest_queued_row(self, database):
        writer = AsyncWriter(database)
        writer.enqueue(decision(1), critical=True)
        assert writer.stats()["oldest_queued_age_s"] is not None
        await writer.flush()
        assert writer.stats()["oldest_queued_age_s"] is None
