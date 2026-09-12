"""How late the event loop is running its own callbacks.

Everything in this process shares one thread: market-data ingestion, both
trading loops, the news pipeline, the health loop, and every database round trip
the writer makes. When one of them holds the thread, the others do not run --
and nothing in the platform could say so.

That gap cost a day of data. The writer was committing about 1,500 rows an hour
against 70,000 produced, and the evidence needed to explain it had to be
reconstructed afterwards from Postgres process state and log timestamps: the
trading loop had grown to a 1.7-second cycle against a 1.0-second interval, so
every other task was getting one turn per cycle. A number that says "callbacks
are running 1.7 seconds late" would have named that in one line.

Deliberately an observer. Lag has benign spikes -- a garbage collection, a burst
of websocket frames -- and gating trading on it would trade a data problem for
an availability problem. What gates trading is whether the work is actually
happening, which each component answers for itself.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

from kat.observability.logging import get_logger
from kat.observability.metrics import M_LOOP_LAG_MS, metrics

log = get_logger(__name__)

__all__ = ["LoopLagMonitor"]

#: How often to sample. Frequent enough to catch a stall worth knowing about,
#: cheap enough to be free: one timer callback and a subtraction.
_TICK_S = 0.25

#: At most one line a minute. A saturated loop must not get a log line per
#: sample from the very thing that noticed it was saturated.
_REPORT_INTERVAL_S = 60.0

#: Above this the loop is not merely busy. A task waiting on a database round
#: trip gets one turn per quarter second at this lag, which is how a batch of
#: five hundred rows comes to take ten minutes.
_WARN_P99_S = 0.25


class LoopLagMonitor:
    """Measures how much later than scheduled this loop wakes a sleeper."""

    def __init__(self, *, window: int = 600) -> None:
        #: About two and a half minutes of samples at the tick rate.
        self._samples: deque[float] = deque(maxlen=window)
        self._worst = 0.0
        self._reported_at = 0.0

    async def run(self) -> None:
        self._reported_at = time.monotonic()
        while True:
            before = time.monotonic()
            await asyncio.sleep(_TICK_S)
            # Anything beyond the sleep we asked for is time the loop spent not
            # getting back to us.
            self.record(max(0.0, time.monotonic() - before - _TICK_S))
            self._maybe_report()

    def record(self, lag_s: float) -> None:
        self._samples.append(lag_s)
        self._worst = max(self._worst, lag_s)
        metrics.gauge(M_LOOP_LAG_MS, lag_s * 1000)

    def _maybe_report(self) -> None:
        now = time.monotonic()
        if now - self._reported_at < _REPORT_INTERVAL_S or not self._samples:
            return
        self._reported_at = now
        stats = self.stats()
        if stats["p99_ms"] >= _WARN_P99_S * 1000:
            log.warning("engine.loop_lag", **stats)
        else:
            log.info("engine.loop_lag", **stats)

    def stats(self) -> dict[str, Any]:
        if not self._samples:
            return {"samples": 0, "p50_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0,
                    "worst_ever_ms": round(self._worst * 1000, 1)}
        ordered = sorted(self._samples)
        return {
            "samples": len(ordered),
            "p50_ms": round(ordered[len(ordered) // 2] * 1000, 1),
            "p99_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))] * 1000, 1),
            "max_ms": round(ordered[-1] * 1000, 1),
            #: Kept across the window, because the worst stall is the one worth
            #: explaining and a rolling window forgets it.
            "worst_ever_ms": round(self._worst * 1000, 1),
        }
