"""Declarative base and shared column conventions.

Conventions used by every table in this schema:

* **Money is ``BIGINT`` micro-dollars.** Never ``FLOAT``, never ``NUMERIC`` on
  the hot path. This mirrors :mod:`kat.common.fixedpoint` exactly, so a value
  round-trips through the database without changing.
* **Prices are ``INTEGER`` centicents** and **counts are ``BIGINT``
  centi-contracts**, again matching the in-memory representation.
* **Time is ``TIMESTAMPTZ``.** Two timestamps are recorded on ingested data:
  when the source says it happened, and when we received it. Backtests read the
  first and check it against the second to detect look-ahead.
* **Natural keys are preserved.** Kalshi's ``order_id``, ``fill_id`` and
  tickers are stored as given, so reconciliation is a join and not a heuristic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Integer, MetaData, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = ["Base", "IngestMixin", "TimestampMixin", "money", "price_cc", "qty_qc", "utcnow"]

# Explicit constraint naming so Alembic can autogenerate reversible migrations.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    def to_dict(self) -> dict[str, Any]:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


def utcnow() -> datetime:
    return datetime.now(UTC)


def money(**kw: Any) -> Mapped[int]:
    """A monetary column in micro-dollars (1e-6 USD)."""
    return mapped_column(BigInteger, **kw)


def price_cc(**kw: Any) -> Mapped[int]:
    """A price column in centicents (1e-4 USD)."""
    return mapped_column(Integer, **kw)


def qty_qc(**kw: Any) -> Mapped[int]:
    """A contract-count column in centi-contracts (1e-2 contracts)."""
    return mapped_column(BigInteger, **kw)


class TimestampMixin:
    """``created_at`` / ``updated_at`` maintained by the database."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class IngestMixin:
    """Provenance carried by every externally sourced record.

    ``source_ts`` is when the event happened according to the source;
    ``ingested_ts`` is when this process learned of it. Keeping both is what
    makes information-time integrity checkable rather than assumed: a backtest
    at time T may only use rows whose ``ingested_ts <= T``, even if their
    ``source_ts`` is much earlier.
    """

    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ingested_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    quality: Mapped[str] = mapped_column(String(16), nullable=False, default="ok")
