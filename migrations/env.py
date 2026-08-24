"""Alembic environment.

The database URL comes from the application's own settings rather than from
``alembic.ini``. That is deliberate: a migration must run against exactly the
database the platform will use, and a URL duplicated in two files is a URL that
will eventually disagree with itself.

TimescaleDB is used opportunistically. Migrations detect the extension and create
hypertables when it is present; on plain PostgreSQL -- which includes every
Windows deployment, since Timescale does not ship Windows binaries -- the same
tables are created with BRIN indexes on their time columns instead. The schema is
identical either way, so a database can move between the two.
"""

from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "packages") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "packages"))

from kat.config.settings import get_settings  # noqa: E402
from kat.db.base import Base  # noqa: E402
from kat.db.models import HYPERTABLES  # noqa: E402,F401 - imported for its side effect

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# One source of truth for the URL.
config.set_main_option("sqlalchemy.url", get_settings().database.url)

target_metadata = Base.metadata


def include_object(
    obj: object, name: str | None, type_: str, reflected: bool, compare_to: object
) -> bool:
    """Filter what autogenerate is allowed to notice.

    The time-series indexes created by migration ``0002`` are chosen at migration
    time from what the server supports -- BRIN on plain PostgreSQL, hypertable
    chunks under TimescaleDB -- so they deliberately do not appear in the model
    metadata. Without this filter, ``alembic check`` would report them as drift
    on every run and generate a migration that drops them, which is exactly
    backwards.
    """
    if type_ == "index" and name:
        if name.startswith("brin_"):
            return False
        # TimescaleDB creates its own chunk indexes under an internal schema.
        if name.startswith("_hyper_") or name.startswith("_timescaledb"):
            return False
    if type_ == "table" and name and name.startswith("_timescaledb"):
        return False
    return True


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live connection, for review."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Type changes are easy to miss by eye and expensive to discover in
        # production, so autogenerate is told to look for them.
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
