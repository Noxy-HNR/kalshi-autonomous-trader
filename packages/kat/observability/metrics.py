"""In-process metrics.

Deliberately not Prometheus. The platform runs as a single long-lived process on
one machine; an in-memory registry read by the dashboard's ``/system`` endpoint
answers every question we actually have ("what is my execution latency right
now?") without adding a scrape endpoint, a time-series database and a second
deployment to operate.

Latency is recorded as a bounded reservoir rather than a running mean, because
the number that matters on an execution path is the tail, and a mean hides it.
"""

from __future__ import annotations

import threading
import time
from bisect import insort
from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["MetricsRegistry", "Timer", "metrics"]


@dataclass
class _Histogram:
    """Bounded sample reservoir with exact percentiles over the window."""

    window: int = 2_048
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=2_048))
    count: int = 0
    total: float = 0.0
    max_seen: float = 0.0

    def observe(self, value: float) -> None:
        self.samples.append(value)
        self.count += 1
        self.total += value
        if value > self.max_seen:
            self.max_seen = value

    def snapshot(self) -> dict[str, float | int]:
        if not self.samples:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        ordered: list[float] = []
        for value in self.samples:
            insort(ordered, value)
        n = len(ordered)

        def pct(p: float) -> float:
            # Nearest-rank percentile; exact for the retained window.
            idx = min(n - 1, max(0, int(round(p * (n - 1)))))
            return ordered[idx]

        return {
            "count": self.count,
            "mean": round(sum(ordered) / n, 4),
            "p50": round(pct(0.50), 4),
            "p95": round(pct(0.95), 4),
            "p99": round(pct(0.99), 4),
            "max": round(self.max_seen, 4),
        }


class MetricsRegistry:
    """Thread-safe registry of counters, gauges and latency histograms.

    Labels are folded into the metric name (``kalshi.rest.latency_ms|GET|/markets``)
    rather than modelled as dimensions. With a fixed, small set of call sites
    that is simpler and costs one dict lookup.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, _Histogram] = {}
        self._started_at = time.time()

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        parts = "|".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}|{parts}"

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def gauge(self, name: str, value: float, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._gauges[key] = float(value)

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            hist = self._histograms.get(key)
            if hist is None:
                hist = _Histogram()
                self._histograms[key] = hist
            hist.observe(value)

    def timer(self, name: str, **labels: str) -> Timer:
        """Context manager that records elapsed milliseconds into a histogram."""
        return Timer(self, name, labels)

    def get_counter(self, name: str, **labels: str) -> int:
        with self._lock:
            return self._counters.get(self._key(name, labels), 0)

    def get_gauge(self, name: str, **labels: str) -> float | None:
        with self._lock:
            return self._gauges.get(self._key(name, labels))

    def snapshot(self) -> dict[str, Any]:
        """Full point-in-time view, as served to the dashboard System tab."""
        with self._lock:
            return {
                "uptime_s": round(time.time() - self._started_at, 1),
                "counters": dict(sorted(self._counters.items())),
                "gauges": dict(sorted(self._gauges.items())),
                "histograms": {
                    key: hist.snapshot() for key, hist in sorted(self._histograms.items())
                },
            }

    def reset(self) -> None:
        """Test helper. Never called by the running platform."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


class Timer:
    """Measure a block in milliseconds using the monotonic clock.

    Monotonic, not wall clock: an NTP correction or a Windows resume must not be
    able to produce a negative or wildly inflated latency sample.
    """

    __slots__ = ("_labels", "_name", "_registry", "_start", "elapsed_ms")

    def __init__(self, registry: MetricsRegistry, name: str, labels: dict[str, str]) -> None:
        self._registry = registry
        self._name = name
        self._labels = labels
        self._start = 0
        self.elapsed_ms = 0.0

    def __enter__(self) -> Timer:
        self._start = time.monotonic_ns()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.monotonic_ns() - self._start) / 1_000_000.0
        self._registry.observe(self._name, self.elapsed_ms, **self._labels)


#: Process-wide registry. A singleton because metrics are inherently global and
#: threading one through every constructor buys nothing.
metrics = MetricsRegistry()


# --- canonical metric names -------------------------------------------------
# Defined here so producers and the dashboard cannot drift apart on spelling.

M_REST_LATENCY = "kalshi.rest.latency_ms"
M_REST_ERRORS = "kalshi.rest.errors"
M_REST_REQUESTS = "kalshi.rest.requests"
M_RATE_LIMITED = "kalshi.rest.rate_limited"
M_WS_MESSAGES = "kalshi.ws.messages"
M_WS_RECONNECTS = "kalshi.ws.reconnects"
M_WS_SEQUENCE_GAPS = "kalshi.ws.sequence_gaps"
M_WS_LAG_MS = "kalshi.ws.lag_ms"
M_MARKET_DATA_AGE_MS = "marketdata.age_ms"
M_DECISION_LATENCY = "engine.decision_latency_ms"
M_RISK_LATENCY = "risk.evaluation_latency_ms"
M_EXEC_LATENCY = "execution.submit_latency_ms"
M_INFERENCE_LATENCY = "model.inference_latency_ms"
M_ORDERS_SUBMITTED = "execution.orders_submitted"
M_ORDERS_FILLED = "execution.orders_filled"
M_ORDERS_REJECTED = "execution.orders_rejected"
M_RISK_REJECTIONS = "risk.rejections"
M_LEVERAGE_BLOCKS = "risk.leverage_blocks"
M_DB_QUEUE_DEPTH = "db.write_queue_depth"
M_DB_WRITES = "db.rows_written"
M_DB_DROPS = "db.rows_dropped"
