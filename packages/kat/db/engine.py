"""Async database engine and session management.

The trading path never awaits this module. Reads happen at start-up (state
recovery) and in the dashboard API; writes go through
:class:`kat.db.writer.AsyncWriter`, which batches them off the critical path.

Both PostgreSQL and SQLite are supported from the same models. PostgreSQL is the
production store; SQLite exists so the full test suite -- including the
integration and failure-injection tests -- runs on a laptop or in CI with no
database server at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from kat.config.settings import DatabaseSettings, get_settings
from kat.db.base import Base
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["Database", "get_database", "reset_database_singleton"]


class Database:
    """Owns the engine and session factory for one process."""

    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self.settings = settings or get_settings().database
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None
        self._healthy = False
        self._lock = asyncio.Lock()
        self.timescale_available = False

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        """Create the engine and verify the connection actually works.

        Verification is a real round-trip, not merely constructing the engine:
        SQLAlchemy creates engines lazily, so an unreachable database would
        otherwise surface at the first write rather than at start-up.
        """
        async with self._lock:
            if self._engine is not None:
                return

            kwargs: dict[str, Any] = {"echo": self.settings.echo, "future": True}
            if not self.settings.is_sqlite:
                kwargs.update(
                    pool_size=self.settings.pool_size,
                    max_overflow=self.settings.max_overflow,
                    pool_timeout=self.settings.pool_timeout_s,
                    pool_pre_ping=True,  # survives an idle-timeout or a DB restart
                    pool_recycle=1800,
                )

            engine = create_async_engine(self.settings.url, **kwargs)
            self._engine = engine
            self._sessionmaker = async_sessionmaker(
                engine, expire_on_commit=False, class_=AsyncSession
            )

            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
                if self.settings.is_sqlite:
                    # WAL lets the background writer commit while readers read.
                    await conn.execute(text("PRAGMA journal_mode=WAL"))
                    await conn.execute(text("PRAGMA synchronous=NORMAL"))
                    await conn.execute(text("PRAGMA foreign_keys=ON"))
                else:
                    row = await conn.execute(
                        text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
                    )
                    self.timescale_available = row.first() is not None

            self._healthy = True
            log.info(
                "database.connected",
                dialect=engine.dialect.name,
                timescale=self.timescale_available,
            )

    async def disconnect(self) -> None:
        async with self._lock:
            if self._engine is not None:
                await self._engine.dispose()
                log.info("database.disconnected")
            self._engine = None
            self._sessionmaker = None
            self._healthy = False

    async def create_all(self) -> None:
        """Create the schema from the models.

        Used by tests and by first-run bootstrap. Production schema changes go
        through Alembic migrations, which are generated from these same models.
        """
        engine = self._require_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("database.schema_created", tables=len(Base.metadata.tables))

    # -- access --------------------------------------------------------------

    def _require_engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("Database.connect() has not been called")
        return self._engine

    @property
    def engine(self) -> AsyncEngine:
        return self._require_engine()

    @property
    def healthy(self) -> bool:
        return self._healthy

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Transactional session. Commits on success, rolls back on error."""
        if self._sessionmaker is None:
            raise RuntimeError("Database.connect() has not been called")
        session = self._sessionmaker()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def ping(self) -> bool:
        """Health probe used by the System tab and the supervisor.

        Marks the database unhealthy on failure rather than raising, so a
        transient outage degrades the platform instead of crashing it. The
        trading engine reacts to ``healthy`` going false by refusing to promote
        anything to live, but continues to manage existing positions.
        """
        try:
            engine = self._require_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            self._healthy = True
            return True
        except Exception as exc:  # noqa: BLE001 - a probe must not propagate
            self._healthy = False
            log.warning("database.ping_failed", error=str(exc))
            return False


_database: Database | None = None


def get_database() -> Database:
    global _database
    if _database is None:
        _database = Database()
    return _database


def reset_database_singleton() -> None:
    """Test helper: drop the process-wide instance so a new URL takes effect."""
    global _database
    _database = None
