"""Time handling with an explicit, swappable clock.

Backtests must be able to run the *same* strategy code as live trading without
that code ever reading the wall clock. Every component therefore takes a
``Clock``; the live engine passes ``SystemClock`` and the simulator passes
``SimulatedClock``, which advances only when the simulation says so.

This is also where information-time integrity begins: a strategy asking "what
time is it?" during a backtest must be told the *simulation* time, or it can
trivially leak the future by comparing against real ``now``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime

__all__ = [
    "Clock",
    "SimulatedClock",
    "SystemClock",
    "from_ms",
    "from_unix_seconds",
    "isoformat",
    "parse_rfc3339",
    "to_ms",
]


class Clock(ABC):
    """Source of truth for "now" for one execution context."""

    @abstractmethod
    def now_ms(self) -> int:
        """Current time as Unix epoch milliseconds."""

    def now(self) -> datetime:
        """Current time as a timezone-aware UTC ``datetime``."""
        return from_ms(self.now_ms())

    def monotonic_ns(self) -> int:
        """Monotonic counter for latency measurement.

        Separate from ``now_ms`` because wall time can jump (NTP correction,
        Windows resume from sleep) and would produce negative latencies.
        """
        return time.monotonic_ns()


class SystemClock(Clock):
    """Wall clock. The only clock the live trading path may use."""

    __slots__ = ()

    def now_ms(self) -> int:
        return time.time_ns() // 1_000_000


class SimulatedClock(Clock):
    """Clock driven by the simulator.

    ``monotonic_ns`` is derived from simulated time so that latency budgets
    measured in a backtest reflect *simulated* latency, not the wall-clock cost
    of running the backtest.
    """

    __slots__ = ("_now_ms",)

    def __init__(self, start_ms: int) -> None:
        self._now_ms = int(start_ms)

    def now_ms(self) -> int:
        return self._now_ms

    def monotonic_ns(self) -> int:
        return self._now_ms * 1_000_000

    def set(self, ms: int) -> None:
        """Jump to an absolute simulated time. Never moves backwards."""
        ms = int(ms)
        if ms < self._now_ms:
            raise ValueError(f"simulated clock cannot move backwards: {ms} < {self._now_ms}")
        self._now_ms = ms

    def advance(self, delta_ms: int) -> None:
        if delta_ms < 0:
            raise ValueError("simulated clock cannot advance by a negative amount")
        self._now_ms += int(delta_ms)


# --- conversions -----------------------------------------------------------


def from_ms(ms: int) -> datetime:
    """Unix epoch milliseconds -> aware UTC datetime."""
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


def to_ms(dt: datetime) -> int:
    """Aware datetime -> Unix epoch milliseconds.

    Naive datetimes are rejected: silently assuming UTC for a local timestamp is
    exactly the kind of bug that shifts a news article across a decision
    boundary in a backtest.
    """
    if dt.tzinfo is None:
        raise ValueError("naive datetime is ambiguous; attach a timezone before converting")
    return int(dt.timestamp() * 1000)


def from_unix_seconds(seconds: int | float) -> datetime:
    """Kalshi event-contract payloads use Unix *seconds* in several places."""
    return datetime.fromtimestamp(float(seconds), tz=UTC)


def parse_rfc3339(value: str) -> datetime:
    """Parse the RFC3339 timestamps the event-contract API returns.

    Perpetual payloads use ``*_ms`` integer fields instead; that difference is
    handled in the per-domain model modules rather than hidden here.
    """
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def isoformat(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")
