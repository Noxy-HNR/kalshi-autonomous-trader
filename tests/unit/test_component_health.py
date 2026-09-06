"""A component is healthy when its work is happening, not when a proxy says so.

Every custom health probe in this platform reports on something *adjacent* to
the work rather than on the work itself, and each of those proxies outlives the
task it stands for:

* the event and perpetual ingestors report ``ws.connected``, which stays true
  when the consuming task dies and the socket does not;
* the news pipeline reports on its last result, which stops changing at exactly
  the moment it stops running -- and its "not yet run" case reports healthy.

The platform has already paid for this shape once. The database writer's
``_running`` flag recorded what ``start()`` had been asked to do rather than
whether the drain task existed; the task stopped, the flag did not, and
2,084,222 rows were dropped over sixty-six hours while ``stats()`` reported the
writer as running.

So a dead task is unhealthy whatever the probe returns.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from kat.engine.lifecycle import ServiceComponent

pytestmark = pytest.mark.asyncio


async def _dead_component(*, health) -> ServiceComponent:
    """A component whose task has stopped, with the probe still cheerful."""
    started = asyncio.Event()

    async def run() -> None:
        started.set()
        await asyncio.sleep(3600)

    component = ServiceComponent("probe-me", run=run, health=health)
    await component.start()
    await started.wait()
    assert component.task is not None
    component.task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await component.task
    return component


class TestADeadTaskIsNeverHealthy:
    async def test_an_optimistic_probe_cannot_mask_a_dead_task(self):
        component = await _dead_component(health=lambda: True)
        assert await component.check_health() is False
        assert "not running" in component.status.detail

    async def test_an_async_probe_is_covered_too(self):
        async def always_fine() -> bool:
            return True

        component = await _dead_component(health=always_fine)
        assert await component.check_health() is False

    async def test_the_failure_is_counted_like_any_other(self):
        component = await _dead_component(health=lambda: True)
        await component.check_health()
        assert component.status.consecutive_failures == 1


class TestTheProbeStillGoverns:
    async def test_a_live_task_with_a_failing_probe_is_unhealthy(self):
        """The task being alive is necessary, not sufficient: an ingestor whose
        socket has dropped is not healthy merely because its loop is spinning."""
        async def run() -> None:
            await asyncio.sleep(3600)

        component = ServiceComponent("c", run=run, health=lambda: False)
        await component.start()
        try:
            assert await component.check_health() is False
        finally:
            await component.stop()

    async def test_a_live_task_with_a_passing_probe_is_healthy(self):
        async def run() -> None:
            await asyncio.sleep(3600)

        component = ServiceComponent("c", run=run, health=lambda: True)
        await component.start()
        try:
            assert await component.check_health() is True
            assert component.status.consecutive_failures == 0
        finally:
            await component.stop()

    async def test_a_raising_probe_is_unhealthy_and_says_why(self):
        async def run() -> None:
            await asyncio.sleep(3600)

        def boom() -> bool:
            raise RuntimeError("probe exploded")

        component = ServiceComponent("c", run=run, health=boom)
        await component.start()
        try:
            assert await component.check_health() is False
            assert "probe exploded" in component.status.detail
        finally:
            await component.stop()


class TestComponentsWithoutAProbe:
    async def test_a_component_with_no_task_is_healthy(self):
        """A start/stop-only component has nothing to outlive."""
        component = ServiceComponent("passive")
        assert await component.check_health() is True
