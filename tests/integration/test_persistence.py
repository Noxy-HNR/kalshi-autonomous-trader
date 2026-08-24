"""Database schema and the async writer, against a real engine.

Real SQL, real tables, real commits -- no mocks. SQLite stands in for PostgreSQL
by default so this suite runs anywhere; the ``integration`` marked tests below
additionally run against PostgreSQL when ``KAT_TEST_POSTGRES_URL`` is set, which
is what CI does.

The regression these tests exist for: every append-only table uses an
auto-incrementing surrogate key, and SQLite only auto-increments a column
declared exactly ``INTEGER PRIMARY KEY``. A ``BIGINT PRIMARY KEY`` is an ordinary
column there, so inserts failed a NOT NULL constraint and the writer silently
abandoned every audit row after exhausting its retries. It surfaced only as a log
line, which is exactly the kind of failure a test has to catch.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from kat.config.settings import DatabaseSettings  # noqa: F401 - used by the PostgreSQL fixture
from kat.db.base import Base
from kat.db.engine import Database
from kat.db.models import (
    ApiEvent,
    AuditLog,
    Candle,
    Decision,
    Fill,
    Order,
    Quote,
    RiskEvent,
    TradePrint,
)
from kat.db.writer import AsyncWriter

# Every table whose primary key is an auto-incrementing surrogate. These are the
# ones the dialect difference breaks.
AUTOINCREMENT_TABLES = [
    t for t in Base.metadata.sorted_tables
    if len(t.primary_key.columns) == 1
    and next(iter(t.primary_key.columns)).autoincrement is True
]


async def _writer_for(db: Database) -> AsyncWriter:
    writer = AsyncWriter(db)
    await writer.start()
    return writer


class TestSchema:
    async def test_schema_creates(self, sqlite_db: Database):
        assert len(Base.metadata.tables) >= 35

    def test_every_autoincrement_key_is_dialect_safe(self):
        """SQLite needs INTEGER, PostgreSQL wants BIGINT. Both must be produced."""
        from sqlalchemy.dialects import postgresql, sqlite

        assert AUTOINCREMENT_TABLES, "expected append-only tables with surrogate keys"
        for table in AUTOINCREMENT_TABLES:
            column = next(iter(table.primary_key.columns))
            sqlite_type = column.type.compile(dialect=sqlite.dialect())
            pg_type = column.type.compile(dialect=postgresql.dialect())
            assert sqlite_type == "INTEGER", (
                f"{table.name}.{column.name} compiles to {sqlite_type} on SQLite; "
                "only INTEGER PRIMARY KEY auto-increments there"
            )
            assert pg_type == "BIGINT", (
                f"{table.name}.{column.name} compiles to {pg_type} on PostgreSQL; "
                "market-data tables need the 64-bit width"
            )


class TestWriter:
    async def test_append_only_rows_commit(self, sqlite_db: Database):
        """The regression: these inserts used to fail and be silently dropped."""
        writer = await _writer_for(sqlite_db)
        now = datetime.now(UTC)

        writer.enqueue(
            AuditLog(actor="test", action="unit_test", subject="s", detail={"ok": True}),
            critical=True,
        )
        writer.enqueue(
            Quote(ticker="T", market_type="event", ts=now, bid_cc=4000, ask_cc=4100, source="test")
        )
        writer.enqueue(
            TradePrint(
                ticker="T", market_type="event", ts=now, price_cc=4050,
                count_qc=100, source="test",
            )
        )
        writer.enqueue(
            Candle(
                ticker="T", market_type="event", period_s=60, ts=now,
                open_cc=4000, high_cc=4100, low_cc=3900, close_cc=4050, source="test",
            )
        )
        writer.enqueue(
            RiskEvent(market_type="perp", reason_code="LEVERAGE_LIMIT", message="test"),
            critical=True,
        )
        writer.enqueue(ApiEvent(domain="event", method="GET", path="/markets", status=200))

        await asyncio.sleep(1.0)
        stats = writer.stats()
        await writer.stop()

        assert stats["dropped"] == 0, "rows were dropped; the writer abandoned them"
        assert stats["written"] == 6

        async with sqlite_db.session() as session:
            for model, expected in (
                (AuditLog, 1), (Quote, 1), (TradePrint, 1),
                (Candle, 1), (RiskEvent, 1), (ApiEvent, 1),
            ):
                count = (await session.execute(select(func.count()).select_from(model))).scalar()
                assert count == expected, f"{model.__tablename__} has {count} rows, expected {expected}"

    async def test_natural_key_rows_are_upserted_not_duplicated(self, sqlite_db: Database):
        """A websocket replay re-sends rows we hold; that must not duplicate them."""
        writer = await _writer_for(sqlite_db)
        for state in ("submitted", "acknowledged", "filled"):
            writer.enqueue(
                Order(
                    order_id="ord-1",
                    client_order_id="cli-1",
                    market_type="event",
                    ticker="T",
                    state=state,
                    order_type="limit",
                    book_side="bid",
                    requested_qty_qc=1000,
                ),
                critical=True,
            )
        await asyncio.sleep(1.0)
        await writer.stop()

        async with sqlite_db.session() as session:
            rows = (await session.execute(select(Order))).scalars().all()
        assert len(rows) == 1
        assert rows[0].state == "filled"

    async def test_duplicate_fill_ids_collapse(self, sqlite_db: Database):
        writer = await _writer_for(sqlite_db)
        for _ in range(3):
            writer.enqueue(
                Fill(
                    fill_id="f-1", order_id="ord-1", market_type="event", ticker="T",
                    ts=datetime.now(UTC), price_cc=4000, count_qc=100,
                ),
                critical=True,
            )
        await asyncio.sleep(1.0)
        await writer.stop()

        async with sqlite_db.session() as session:
            count = (await session.execute(select(func.count()).select_from(Fill))).scalar()
        assert count == 1

    async def test_enqueue_is_non_blocking(self, sqlite_db: Database):
        """The trading path must never wait on a database round-trip."""
        writer = await _writer_for(sqlite_db)
        start = asyncio.get_running_loop().time()
        for i in range(5_000):
            writer.enqueue(
                Quote(
                    ticker=f"T{i}", market_type="event", ts=datetime.now(UTC),
                    bid_cc=4000, source="bench",
                )
            )
        elapsed_ms = (asyncio.get_running_loop().time() - start) * 1000
        await writer.stop(drain=False, timeout_s=2.0)
        # Generous bound: the point is that it is memory-speed, not IO-speed.
        assert elapsed_ms < 500, f"5000 enqueues took {elapsed_ms:.0f}ms; not off the hot path"

    async def test_full_queue_drops_non_critical_before_critical(self, sqlite_db: Database):
        """Under pressure, a quote is expendable and an audit row is not.

        The bound is shrunk on the instance rather than through settings, whose
        validator enforces a sane floor of 100 -- a real queue that small would
        be a misconfiguration, but the eviction policy still needs exercising.
        """
        writer = AsyncWriter(sqlite_db)
        writer._max_size = 10  # noqa: SLF001 - deliberately exercising the bound

        for i in range(10):
            writer.enqueue(
                Quote(ticker=f"Q{i}", market_type="event", ts=datetime.now(UTC), source="t")
            )
        # Queue is full of non-critical rows; a critical row must displace one.
        assert writer.enqueue(AuditLog(actor="t", action="critical"), critical=True) is True
        assert writer.stats()["dropped"] == 1

    async def test_decision_records_round_trip(self, sqlite_db: Database):
        """The explainability record must survive storage intact."""
        writer = await _writer_for(sqlite_db)
        leverage = {
            "equity_usd": "200.000000",
            "projected_leverage": "1.000025",
            "max_leverage": "1.00",
            "permitted": False,
        }
        writer.enqueue(
            Decision(
                decision_id="dec-1",
                market_type="perp",
                ticker="P",
                decision="rejected",
                reason_code="LEVERAGE_WOULD_EXCEED_1X",
                reason="projected exposure exceeds equity",
                leverage_assessment=leverage,
                environment="paper",
            ),
            critical=True,
        )
        await asyncio.sleep(1.0)
        await writer.stop()

        async with sqlite_db.session() as session:
            row = (await session.execute(select(Decision))).scalar_one()
        assert row.leverage_assessment == leverage
        assert row.reason_code == "LEVERAGE_WOULD_EXCEED_1X"


@pytest.mark.integration
class TestPostgres:
    """The same guarantees against real PostgreSQL.

    Skipped unless ``KAT_TEST_POSTGRES_URL`` is set. CI sets it against a
    PostgreSQL 17 service container.
    """

    @pytest.fixture
    async def pg(self):
        url = os.environ.get("KAT_TEST_POSTGRES_URL")
        if not url:
            pytest.skip("set KAT_TEST_POSTGRES_URL to run PostgreSQL integration tests")
        db = Database(DatabaseSettings(url=url))
        await db.connect()
        async with db.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        try:
            yield db
        finally:
            await db.disconnect()

    async def test_schema_creates_on_postgres(self, pg: Database):
        async with pg.session() as session:
            count = (await session.execute(select(func.count()).select_from(AuditLog))).scalar()
        assert count == 0

    async def test_append_only_rows_commit_on_postgres(self, pg: Database):
        writer = AsyncWriter(pg)
        await writer.start()
        writer.enqueue(AuditLog(actor="test", action="pg_test"), critical=True)
        writer.enqueue(
            Quote(ticker="T", market_type="event", ts=datetime.now(UTC), bid_cc=1, source="t")
        )
        await asyncio.sleep(1.0)
        stats = writer.stats()
        await writer.stop()
        assert stats["dropped"] == 0
        assert stats["written"] == 2
