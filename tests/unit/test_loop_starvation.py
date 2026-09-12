"""One thread, shared by everything, and what happens when one task keeps it.

Market-data ingestion, both trading loops, the news pipeline, the health loop
and every database round trip the writer makes run on the same event loop. On
2026-09-11 the trading loop's cycle grew to about 1.7 seconds against its
1.0-second interval, so ``sleep(max(0.0, interval - elapsed))`` slept for zero
and yielded exactly once per cycle. Every other task then got one turn per
cycle: the writer, which needs a turn per round trip, committed about 1,500 rows
an hour while 70,000 were produced, and a third of the watchlist read as stale.

Two things follow, and they are what these tests pin. A cycle must leave a floor
of idle time however long it took, and how long the loop is making everything
else wait has to be a number somebody can read.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from kat.common.enums import MarketType
from kat.common.timeutil import SimulatedClock
from kat.config.settings import EventRiskSettings
from kat.engine.trading import _MIN_IDLE_S, TradingLoop
from kat.marketdata.state import MarketState
from kat.observability.eventloop import LoopLagMonitor
from kat.risk.event_risk import EventRiskEngine

NOW_MS = 1_760_000_000_000


def _loop(interval_s: float = 1.0) -> TradingLoop:
    return TradingLoop(
        market_type=MarketType.EVENT,
        market_state=MarketState(MarketType.EVENT),
        risk_engine=EventRiskEngine(EventRiskSettings()),
        execution=None,  # type: ignore[arg-type]
        recorder=None,  # type: ignore[arg-type]
        clock=SimulatedClock(NOW_MS),
        interval_s=interval_s,
    )


class _Cycles:
    """Runs the loop for a fixed number of cycles, recording what it slept."""

    def __init__(self, loop: TradingLoop, *, cycle_s: float, cycles: int) -> None:
        self.loop = loop
        self.cycle_s = cycle_s
        self.remaining = cycles
        self.slept: list[float] = []

    async def run_cycle(self):
        from kat.engine.trading import CycleResult

        # Occupy the loop the way a real cycle does: synchronously.
        time.sleep(self.cycle_s)
        return CycleResult()

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.remaining -= 1
        if self.remaining <= 0:
            self.loop.stop()

    async def go(self, monkeypatch) -> None:
        monkeypatch.setattr(self.loop, "run_cycle", self.run_cycle)
        monkeypatch.setattr("kat.engine.trading.asyncio.sleep", self.sleep)
        await self.loop.run()


class TestACycleAlwaysLeavesIdleTime:
    async def test_an_overrunning_cycle_still_yields_the_floor(self, monkeypatch):
        """The defect: an overrun slept for zero, which yields once.

        One yield per cycle is one database round trip per cycle, and a batch
        needs several hundred.
        """
        loop = _loop(interval_s=0.05)
        cycles = _Cycles(loop, cycle_s=0.12, cycles=3)  # every cycle overruns
        await cycles.go(monkeypatch)

        assert cycles.slept, "the loop must sleep between cycles"
        assert all(s >= _MIN_IDLE_S for s in cycles.slept), cycles.slept

    async def test_a_cycle_inside_its_interval_still_paces_on_elapsed_time(
        self, monkeypatch
    ):
        """The floor must not become the interval: a fast cycle still waits."""
        loop = _loop(interval_s=1.0)
        cycles = _Cycles(loop, cycle_s=0.0, cycles=2)
        await cycles.go(monkeypatch)

        assert all(s > _MIN_IDLE_S for s in cycles.slept), cycles.slept
        assert all(s <= 1.0 for s in cycles.slept)

    async def test_overruns_are_counted_for_the_summary(self, monkeypatch):
        loop = _loop(interval_s=0.05)
        cycles = _Cycles(loop, cycle_s=0.12, cycles=3)
        await cycles.go(monkeypatch)

        assert loop._overruns == 3
        assert len(loop._cycle_times) == 3
        assert all(t >= 0.1 for t in loop._cycle_times)


class TestTheLagMonitorNamesTheStall:
    async def test_it_measures_a_loop_somebody_else_is_holding(self):
        """The number that would have said in one line what took a day to find."""
        monitor = LoopLagMonitor()
        watching = asyncio.create_task(monitor.run())
        await asyncio.sleep(0)
        time.sleep(0.75)  # a cycle that does not yield, three ticks long
        await asyncio.sleep(0.3)
        watching.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watching

        stats = monitor.stats()
        assert stats["samples"] >= 1
        # Generous: Windows timers are coarse, and the assertion is about the
        # stall being visible, not about its exact size.
        assert stats["max_ms"] >= 300, stats

    def test_an_unsampled_monitor_reports_zeroes_rather_than_nothing(self):
        assert LoopLagMonitor().stats()["samples"] == 0

    def test_the_worst_stall_survives_the_rolling_window(self):
        monitor = LoopLagMonitor(window=2)
        monitor.record(1.5)
        monitor.record(0.001)
        monitor.record(0.001)

        stats = monitor.stats()
        assert stats["max_ms"] < 100, "the window has moved past it"
        assert stats["worst_ever_ms"] >= 1500, "but the worst stall is still reported"
