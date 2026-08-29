"""Downtime must be reported, not merely survived.

The platform already knew a previous run had died. It did not know *when*, so a
five-second restart and a five-hour outage produced the same log line -- and the
second one means nobody was watching open positions overnight.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from kat.common.timeutil import utcnow_aware
from kat.engine.lifecycle import _HEARTBEAT_INTERVAL_MS, _seconds_since


def test_missing_timestamp_is_unknown_not_zero() -> None:
    """Zero would read as "no downtime", which is the favourable answer."""
    assert _seconds_since(None) is None
    assert _seconds_since("") is None


def test_unparseable_timestamp_is_unknown_not_zero() -> None:
    assert _seconds_since("not-a-timestamp") is None
    assert _seconds_since("2026-13-45T99:99:99") is None


def test_downtime_is_measured_from_the_last_heartbeat() -> None:
    seen = (utcnow_aware() - timedelta(hours=5, minutes=7)).isoformat()
    downtime = _seconds_since(seen)
    assert downtime is not None
    assert 18_000 < downtime < 18_900  # 5h07m, allowing for test execution time


def test_naive_timestamps_are_read_as_utc() -> None:
    """A marker written by an older build has no timezone on it."""
    naive = (utcnow_aware().replace(tzinfo=None) - timedelta(minutes=3)).isoformat()
    downtime = _seconds_since(naive)
    assert downtime is not None
    assert 150 < downtime < 250


def test_a_future_timestamp_does_not_produce_negative_downtime() -> None:
    """Clock skew between runs must not make an outage look like a head start."""
    ahead = (utcnow_aware() + timedelta(hours=1)).isoformat()
    assert _seconds_since(ahead) == 0.0


@pytest.mark.parametrize("elapsed_s", [0, 5, 29])
def test_heartbeat_does_not_write_more_often_than_its_interval(elapsed_s: int) -> None:
    assert elapsed_s * 1000 < _HEARTBEAT_INTERVAL_MS


def test_heartbeat_interval_bounds_the_reported_error() -> None:
    """The interval is the worst-case error in any reported downtime."""
    assert _HEARTBEAT_INTERVAL_MS <= 60_000, "a coarser heartbeat blurs short outages"
    assert _HEARTBEAT_INTERVAL_MS >= 10_000, "a finer one writes to the database for no gain"


def test_persisted_state_loads_only_once() -> None:
    """Two callers, one read of the shutdown marker.

    The entrypoint loads persisted state early so the database is up before any
    component writes to it, and ``Application.start`` loads it again as part of
    its fixed sequence. The second pass used to read the marker the first pass
    had just cleared, so ``clean_previous_shutdown`` came back False after a
    perfectly clean shutdown -- and the flag exists precisely to raise the bar
    for re-arming live trading after a crash.
    """
    import asyncio

    from kat.engine.lifecycle import Application

    app = Application.__new__(Application)
    app._persisted_state_loaded = False
    calls = 0

    async def _once() -> None:
        nonlocal calls
        calls += 1

    app._load_persisted_state_once = _once  # type: ignore[method-assign]

    async def _drive() -> None:
        await Application._load_persisted_state(app)
        await Application._load_persisted_state(app)
        await Application._load_persisted_state(app)

    asyncio.run(_drive())
    assert calls == 1, "the marker must be read and cleared exactly once per process"
