"""The paper account has to outlive the process.

Every start seeded the account from ``paper_starting_balance_usd``, so every
restart erased whatever the strategies had done: positions, cost bases, realised
P&L, and the peak equity the drawdown halt measures against, all back to $200.

That was survivable while the platform ran only when someone was watching. Under
a scheduled task that restarts on failure and again at every logon it is fatal to
the entire point of paper trading, which is to find out whether the strategies
make money *consistently*. A record that resets on restart cannot answer that.

``event_positions`` and ``balance_snapshots`` had been in the schema from the
beginning with nothing ever writing a row to either.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from kat.common.enums import TradingEnvironment
from kat.db.models import BalanceSnapshot
from kat.db.models import EventPosition as EventPositionRow
from kat.engine.persistence import restore_event_account, snapshot_event_account
from kat.risk.event_risk import EventAccountState, EventPositionState

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
SEED = 200_000_000


class _Writer:
    """The async writer, reduced to the one call this uses."""

    def __init__(self) -> None:
        self.rows: list[object] = []

    def enqueue(self, obj: object, *, critical: bool = False) -> bool:
        self.rows.append(obj)
        return True


def _account() -> EventAccountState:
    state = EventAccountState(
        balance_micros=142_000_000,
        portfolio_value_micros=55_000_000,
        realized_pnl_today_micros=-3_000_000,
        updated_at_ms=int(NOW.timestamp() * 1000),
    )
    state.peak_equity_micros = 210_000_000
    state.positions["KXPL-27-CFC"] = EventPositionState(
        ticker="KXPL-27-CFC",
        position_qc=1_980,
        avg_entry_cc=2_000,
        exposure_micros=39_600_000,
        unrealized_pnl_micros=1_400_000,
    )
    return state


class TestWhatGetsWritten:
    def test_positions_and_a_balance_row_are_enqueued(self):
        writer = _Writer()

        written = snapshot_event_account(
            _account(), writer=writer, environment=TradingEnvironment.PAPER, now=NOW
        )

        assert written == 2
        positions = [r for r in writer.rows if isinstance(r, EventPositionRow)]
        balances = [r for r in writer.rows if isinstance(r, BalanceSnapshot)]
        assert len(positions) == 1
        assert len(balances) == 1
        assert positions[0].ticker == "KXPL-27-CFC"
        assert positions[0].avg_entry_cc == 2_000

    def test_the_two_risk_inputs_travel_in_the_raw_column(self):
        """The table has no column for either, and both are risk inputs."""
        writer = _Writer()
        snapshot_event_account(
            _account(), writer=writer, environment=TradingEnvironment.PAPER, now=NOW
        )

        raw = next(r for r in writer.rows if isinstance(r, BalanceSnapshot)).raw
        assert raw["peak_equity_micros"] == 210_000_000
        assert raw["realized_pnl_today_micros"] == -3_000_000
        assert raw["day"] == "2026-09-01"

    def test_no_writer_is_not_an_error(self):
        assert snapshot_event_account(
            _account(), writer=None, environment=TradingEnvironment.PAPER, now=NOW
        ) == 0


class TestARestartFindsTheAccount:
    """Against a real database, because merge-by-ticker is the whole mechanism."""

    async def _persist(self, db, state, *, now=NOW):
        rows: list[object] = []

        class _W:
            def enqueue(self, obj, *, critical=False):
                rows.append(obj)
                return True

        snapshot_event_account(
            state, writer=_W(), environment=TradingEnvironment.PAPER, now=now
        )
        async with db.session() as session:
            for row in rows:
                await session.merge(row) if isinstance(
                    row, EventPositionRow
                ) else session.add(row)

    async def test_balance_and_positions_come_back(self, sqlite_db):
        await self._persist(sqlite_db, _account())

        state, detail = await restore_event_account(
            sqlite_db,
            environment=TradingEnvironment.PAPER,
            now=NOW,
            seed_micros=SEED,
        )

        assert detail["restored"] is True
        assert state is not None
        assert state.balance_micros == 142_000_000
        assert state.positions["KXPL-27-CFC"].position_qc == 1_980
        assert state.positions["KXPL-27-CFC"].avg_entry_cc == 2_000

    async def test_the_peak_equity_is_never_lowered_by_a_restart(self, sqlite_db):
        """Otherwise a restart forgives the drawdown it happened to sit inside."""
        await self._persist(sqlite_db, _account())

        state, _ = await restore_event_account(
            sqlite_db, environment=TradingEnvironment.PAPER, now=NOW, seed_micros=SEED
        )

        assert state is not None
        assert state.peak_equity_micros == 210_000_000
        assert state.peak_equity_micros > state.equity_micros

    async def test_todays_realised_pnl_is_carried_within_the_same_day(self, sqlite_db):
        await self._persist(sqlite_db, _account())

        state, detail = await restore_event_account(
            sqlite_db,
            environment=TradingEnvironment.PAPER,
            now=NOW + timedelta(hours=2),
            seed_micros=SEED,
        )

        assert detail["carried_daily_pnl"] is True
        assert state is not None
        assert state.realized_pnl_today_micros == -3_000_000

    async def test_yesterdays_realised_pnl_is_not_carried_into_today(self, sqlite_db):
        """The daily-loss budget resets; carrying a loss across it halts on a
        budget that has already been given back."""
        await self._persist(sqlite_db, _account())

        state, detail = await restore_event_account(
            sqlite_db,
            environment=TradingEnvironment.PAPER,
            now=NOW + timedelta(days=1),
            seed_micros=SEED,
        )

        assert detail["carried_daily_pnl"] is False
        assert state is not None
        assert state.realized_pnl_today_micros == 0

    async def test_a_flat_position_is_not_restored_as_a_holding(self, sqlite_db):
        state = _account()
        state.positions["CLOSED"] = EventPositionState(
            ticker="CLOSED", position_qc=0, exposure_micros=0
        )
        await self._persist(sqlite_db, state)

        restored, _ = await restore_event_account(
            sqlite_db, environment=TradingEnvironment.PAPER, now=NOW, seed_micros=SEED
        )

        assert restored is not None
        assert "CLOSED" not in restored.positions
        assert "KXPL-27-CFC" in restored.positions

    async def test_a_position_is_upserted_rather_than_duplicated(self, sqlite_db):
        """Snapshots run every thirty seconds; the table must not grow per tick."""
        state = _account()
        for _ in range(3):
            await self._persist(sqlite_db, state)

        async with sqlite_db.session() as session:
            rows = list(
                (
                    await session.execute(
                        select(EventPositionRow).where(
                            EventPositionRow.ticker == "KXPL-27-CFC"
                        )
                    )
                ).scalars()
            )
        assert len(rows) == 1

    async def test_the_newest_snapshot_wins(self, sqlite_db):
        await self._persist(sqlite_db, _account())
        later = _account()
        later.balance_micros = 99_000_000
        await self._persist(sqlite_db, later, now=NOW + timedelta(minutes=5))

        state, _ = await restore_event_account(
            sqlite_db,
            environment=TradingEnvironment.PAPER,
            now=NOW + timedelta(minutes=6),
            seed_micros=SEED,
        )

        assert state is not None
        assert state.balance_micros == 99_000_000

    async def test_a_first_run_restores_nothing_and_says_so(self, sqlite_db):
        state, detail = await restore_event_account(
            sqlite_db, environment=TradingEnvironment.PAPER, now=NOW, seed_micros=SEED
        )

        assert state is None
        assert detail["restored"] is False
        assert "first run" in detail["reason"]

    async def test_live_never_restores_from_here(self, sqlite_db):
        """Demo and live rebuild from the exchange; a second source would
        eventually disagree with the first."""
        await self._persist(sqlite_db, _account())

        state, detail = await restore_event_account(
            sqlite_db, environment=TradingEnvironment.LIVE, now=NOW, seed_micros=SEED
        )

        assert state is None
        assert detail["restored"] is False
