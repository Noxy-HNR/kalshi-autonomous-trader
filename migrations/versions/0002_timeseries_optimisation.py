"""Time-series optimisation: TimescaleDB hypertables where available, BRIN otherwise.

Revision ID: 0002_timeseries
Revises: 91daec4c8145
Create Date: generated

The six highest-cardinality tables accumulate indefinitely and are almost always
queried by a time range. Two ways to make that efficient:

* **TimescaleDB hypertables**, which partition by time transparently. Preferred
  where the extension exists.
* **BRIN indexes**, which exploit the fact that rows are inserted in roughly
  time order, so a block-range summary is enough to skip most of the table.
  Far smaller than a B-tree and nearly as effective for append-only data.

Which one runs is decided at migration time by looking for the extension, so the
same migration produces a working schema on both. This matters concretely:
**TimescaleDB does not ship Windows binaries**, so the initial desktop deployment
takes the BRIN path and the Linux server migration takes the hypertable path,
with no schema divergence between them.

SQLite gets neither and needs neither -- it is used for tests and first-run
development, where the tables never grow large enough for it to matter.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_timeseries"
down_revision = "91daec4c8145"
branch_labels = None
depends_on = None

#: (table, time column). These are the tables that grow without bound.
TIME_SERIES_TABLES: tuple[tuple[str, str], ...] = (
    ("quotes", "ts"),
    ("orderbook_snapshots", "ts"),
    ("trade_prints", "ts"),
    ("candles", "ts"),
    ("predictions", "ts"),
    ("api_events", "ts"),
)


def _timescale_available(connection: sa.Connection) -> bool:
    return (
        connection.execute(
            sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'")
        ).first()
        is not None
    )


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # SQLite: nothing to do. The tables exist and are small by construction.
        return

    if _timescale_available(bind):
        op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
        for table, column in TIME_SERIES_TABLES:
            # migrate_data moves the rows that are already there; if_not_exists
            # keeps the migration re-runnable.
            op.execute(
                sa.text(
                    f"SELECT create_hypertable('{table}', '{column}', "
                    "if_not_exists => TRUE, migrate_data => TRUE)"
                )
            )
        return

    # No TimescaleDB (the Windows case). BRIN is the right fallback for
    # append-only, time-ordered data: a fraction of the size of a B-tree and
    # nearly as selective for range scans.
    for table, column in TIME_SERIES_TABLES:
        op.execute(
            sa.text(
                f"CREATE INDEX IF NOT EXISTS brin_{table}_{column} "
                f"ON {table} USING BRIN ({column}) WITH (pages_per_range = 32)"
            )
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for table, column in TIME_SERIES_TABLES:
        op.execute(sa.text(f"DROP INDEX IF EXISTS brin_{table}_{column}"))
    # Hypertables are deliberately not reverted to plain tables: doing so would
    # rewrite every row, and a downgrade should not be the most expensive
    # operation in the system. Drop and recreate the database instead if that is
    # genuinely wanted.
