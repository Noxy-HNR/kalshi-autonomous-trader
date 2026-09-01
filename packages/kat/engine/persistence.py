"""Making the paper account survive a restart.

A paper account had no memory. Every start seeded it from
``paper_starting_balance_usd`` and every restart therefore erased whatever the
strategies had done -- positions, cost bases, realised P&L, the peak equity that
the drawdown halt measures against, all of it back to $200.00.

That was tolerable while the platform only ran when someone was watching. It
stopped being tolerable the moment it was put under a scheduled task that
restarts it on failure and again at every logon, because the question the paper
run exists to answer is whether the strategies make money *consistently*, and a
record that resets on every restart can never answer it. The ``event_positions``
and ``balance_snapshots`` tables have been in the schema since the beginning and
nothing ever wrote a row to either.

What is restored and what is not
--------------------------------
Only ``PAPER`` restores. Demo and live rebuild positions from the exchange
through the reconciler, and a second source for the same number is how two
sources start disagreeing.

``realized_pnl_today_micros`` is restored only when the snapshot is from the same
UTC day, because it feeds the daily-loss halt: carrying yesterday's loss into
today would halt trading on a budget that has already reset, and resetting a
loss taken an hour ago would hand back a budget that has not.

``peak_equity_micros`` is always restored and never lowered. It is the
denominator of the drawdown halt, so re-seeding it from the restart equity would
quietly forgive every drawdown a restart happened to sit inside -- which, under a
supervisor that restarts on failure, is exactly the drawdown most worth keeping.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from kat.common.enums import MarketType, TradingEnvironment
from kat.db.engine import Database
from kat.db.models import BalanceSnapshot
from kat.db.models import EventPosition as EventPositionRow
from kat.db.writer import AsyncWriter
from kat.observability.logging import get_logger
from kat.risk.event_risk import EventAccountState, EventPositionState

log = get_logger(__name__)

__all__ = ["restore_event_account", "snapshot_event_account"]


def _day(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%d")


def snapshot_event_account(
    state: EventAccountState,
    *,
    writer: AsyncWriter | None,
    environment: TradingEnvironment,
    now: datetime,
) -> int:
    """Write the account's current shape to the database.

    Positions are keyed by ticker, which the writer treats as a natural key and
    merges, so this is an upsert per market rather than an ever-growing log. The
    balance row has a surrogate key and is appended, which is what makes it an
    equity curve.

    A position that has gone flat is still written. ``position_qc = 0`` with a
    realised P&L is the record of a round trip that closed, and dropping the row
    would erase the only evidence the trade ever happened.
    """
    if writer is None:
        return 0

    rows = 0
    for ticker, position in state.positions.items():
        writer.enqueue(
            EventPositionRow(
                ticker=ticker,
                position_qc=position.position_qc,
                avg_entry_cc=position.avg_entry_cc,
                exposure_micros=position.exposure_micros,
                realized_pnl_micros=0,
                unrealized_pnl_micros=position.unrealized_pnl_micros,
                fees_paid_micros=0,
                environment=environment.value,
                last_reconciled_ts=now,
            )
        )
        rows += 1

    writer.enqueue(
        BalanceSnapshot(
            ts=now,
            market_type=MarketType.EVENT.value,
            environment=environment.value,
            balance_micros=state.balance_micros,
            portfolio_value_micros=state.portfolio_value_micros,
            equity_micros=state.equity_micros,
            total_exposure_micros=state.total_exposure_micros(),
            equity_source="paper_local",
            # Two fields the table has no column for, and both are risk inputs
            # rather than reporting: the drawdown halt reads the peak, the
            # daily-loss halt reads today's realised P&L.
            raw={
                "peak_equity_micros": state.peak_equity_micros,
                "realized_pnl_today_micros": state.realized_pnl_today_micros,
                "day": _day(now),
            },
        )
    )
    return rows + 1


async def restore_event_account(
    db: Database,
    *,
    environment: TradingEnvironment,
    now: datetime,
    seed_micros: int,
) -> tuple[EventAccountState | None, dict[str, Any]]:
    """Rebuild the paper account from its last snapshot.

    Returns ``(None, detail)`` when there is nothing to restore, which is the
    first run and any run whose snapshot cannot be read. The caller then seeds as
    it always did: an account that cannot be recovered is started fresh rather
    than started wrong.
    """
    detail: dict[str, Any] = {"restored": False}
    if environment is not TradingEnvironment.PAPER:
        detail["reason"] = "only paper restores; the exchange is the source elsewhere"
        return None, detail

    try:
        async with db.session() as session:
            snapshot = (
                await session.execute(
                    select(BalanceSnapshot)
                    .where(
                        BalanceSnapshot.market_type == MarketType.EVENT.value,
                        BalanceSnapshot.environment == environment.value,
                    )
                    .order_by(BalanceSnapshot.ts.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if snapshot is None:
                detail["reason"] = "no previous snapshot; this is a first run"
                return None, detail
            positions = list(
                (
                    await session.execute(
                        select(EventPositionRow).where(
                            EventPositionRow.environment == environment.value
                        )
                    )
                ).scalars()
            )
    except Exception as exc:
        # A restore that fails must not stop the platform starting. Seeding is a
        # worse record, not a broken one.
        log.warning("paper.restore_failed", error=str(exc))
        detail["reason"] = f"{type(exc).__name__}: {exc}"
        return None, detail

    balance = snapshot.balance_micros
    if balance is None:
        detail["reason"] = "snapshot carries no balance"
        return None, detail

    raw = snapshot.raw if isinstance(snapshot.raw, dict) else {}
    state = EventAccountState(
        balance_micros=int(balance),
        portfolio_value_micros=int(snapshot.portfolio_value_micros or 0),
        updated_at_ms=int(now.timestamp() * 1000),
    )
    for row in positions:
        if row.position_qc == 0 and not row.exposure_micros:
            # Flat and costing nothing: kept in the table as the record of a
            # closed round trip, but it is not a holding to restore.
            continue
        state.positions[row.ticker] = EventPositionState(
            ticker=row.ticker,
            position_qc=int(row.position_qc),
            avg_entry_cc=row.avg_entry_cc,
            exposure_micros=int(row.exposure_micros or 0),
            unrealized_pnl_micros=int(row.unrealized_pnl_micros or 0),
        )

    # The drawdown halt measures against this, so it is never allowed to fall to
    # whatever equity a restart happened to land on.
    state.peak_equity_micros = max(
        int(raw.get("peak_equity_micros") or 0),
        state.equity_micros,
        int(seed_micros),
    )

    same_day = raw.get("day") == _day(now)
    state.realized_pnl_today_micros = (
        int(raw.get("realized_pnl_today_micros") or 0) if same_day else 0
    )

    detail.update(
        restored=True,
        balance_micros=state.balance_micros,
        equity_micros=state.equity_micros,
        positions=len(state.positions),
        peak_equity_micros=state.peak_equity_micros,
        carried_daily_pnl=same_day,
        snapshot_ts=snapshot.ts.isoformat() if snapshot.ts else None,
    )
    return state, detail
