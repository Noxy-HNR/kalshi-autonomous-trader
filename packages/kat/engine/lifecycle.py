"""Application lifecycle.

This is the native long-lived process the platform runs as. No container, no
supervisor script, no shell wrapper: start-up, shutdown, reconnection, health,
recovery and state restoration are all handled inside the application, because a
trading system that needs an external babysitter to stay correct has moved its
safety properties somewhere it cannot see them.

The start-up sequence is fixed and its order is the point
--------------------------------------------------------

::

    verify configuration
      -> connect database
      -> apply/verify schema
      -> load persisted state (kill switches, mode, armed flag)
      -> connect Kalshi and authenticate
      -> determine API capabilities (is perps even enabled for this account?)
      -> query live account
      -> reconcile positions, orders, fills, balances
      -> rebuild in-memory state
      -> validate risk state
      -> start market-data ingestion
      -> start research infrastructure
      -> run health checks
      -> ONLY THEN permit trading

Trading is gated on the whole sequence completing. If reconciliation cannot prove
our state matches the exchange, the risk kill switch engages and the platform
runs in a read-only, observe-only mode until a human resolves it. It never
"resumes anyway".

Crash recovery
--------------
A clean shutdown writes a marker. Its absence at start-up means the previous run
died unexpectedly, which changes nothing about the sequence above -- reconciliation
runs identically either way -- but it is recorded, surfaced on the dashboard, and
raises the bar for automatically re-arming live trading.

The marker also carries a heartbeat, rewritten by the health loop while the
platform is up. The process can die in ways that run no Python at all -- the
machine sleeps, the OS terminates it, the power goes -- and none of those reach
the shutdown path. The heartbeat is what lets the *next* run say how long the
platform was down, which is the difference between knowing it restarted and
knowing nobody was watching the positions for five hours.
"""

from __future__ import annotations

import asyncio
import contextlib
import platform
import signal
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import select

from kat.common.enums import KillSwitchScope, MarketType, TradingEnvironment
from kat.common.errors import ConfigError
from kat.common.ids import use_correlation_id
from kat.common.timeutil import SystemClock, utcnow_aware
from kat.config.settings import AppSettings, get_settings
from kat.db.engine import Database, get_database
from kat.db.models import AuditLog, HealthEvent, SystemState
from kat.db.writer import AsyncWriter, get_writer
from kat.observability.eventloop import LoopLagMonitor
from kat.observability.logging import get_logger
from kat.observability.metrics import metrics
from kat.risk.killswitch import KillSwitchManager

log = get_logger(__name__)

__all__ = ["Application", "ComponentHealth", "Phase", "ServiceComponent"]

#: The database writer, registered as a component so that its *progress* -- not
#: merely its existence -- is one of the conditions for trading.
_WRITER_COMPONENT = "db-writer"

#: The event-loop lag monitor. Not critical: it observes, and what gates trading
#: is whether each component's own work is happening.
_LOOP_MONITOR_COMPONENT = "loop-lag"

#: Key under which the clean-shutdown marker is stored.
_SHUTDOWN_MARKER = "clean_shutdown"

#: How often the heartbeat is written. Thirty seconds bounds the error in a
#: reported downtime to thirty seconds, which is far below the resolution anyone
#: cares about when asking "was the platform up overnight?", while keeping this
#: to two writes a minute against a row nothing else contends for.
_HEARTBEAT_INTERVAL_MS = 30_000


def _seconds_since(timestamp: str | None) -> float | None:
    """Seconds between an ISO timestamp and now, or None if it cannot be read.

    Returns None rather than 0 for an unparseable or missing value: zero would
    read as "no downtime", which is the favourable answer, and an unknown gap is
    not a proven-absent one.
    """
    if not timestamp:
        return None
    try:
        seen = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=UTC)
    return max(0.0, (utcnow_aware() - seen).total_seconds())


class Phase(str, Enum):
    """Where the application is in its lifecycle."""

    CREATED = "created"
    STARTING = "starting"
    RECONCILING = "reconciling"
    RUNNING = "running"
    DEGRADED = "degraded"      # up, but not permitted to trade
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(slots=True)
class ComponentHealth:
    name: str
    healthy: bool = False
    detail: str = ""
    last_ok_ms: int = 0
    consecutive_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "healthy": self.healthy,
            "detail": self.detail,
            "last_ok_ms": self.last_ok_ms,
            "consecutive_failures": self.consecutive_failures,
        }


class ServiceComponent:
    """A long-lived background component owned by the application.

    Components are started in registration order and stopped in reverse, so a
    component may depend on anything registered before it. Each runs as its own
    supervised task: a crash restarts that component with backoff rather than
    taking the process down, because losing news ingestion should not stop the
    platform from managing open positions.
    """

    def __init__(
        self,
        name: str,
        *,
        start: Callable[[], Awaitable[None]] | None = None,
        stop: Callable[[], Awaitable[None]] | None = None,
        run: Callable[[], Awaitable[None]] | None = None,
        health: Callable[[], Awaitable[bool]] | Callable[[], bool] | None = None,
        critical: bool = False,
        restart_on_failure: bool = True,
    ) -> None:
        self.name = name
        self._start = start
        self._stop = stop
        self._run = run
        self._health = health
        #: A critical component failing puts the platform into DEGRADED, which
        #: halts trading. A non-critical one is logged and restarted.
        self.critical = critical
        self.restart_on_failure = restart_on_failure
        self.task: asyncio.Task[None] | None = None
        self.status = ComponentHealth(name=name)

    async def start(self) -> None:
        if self._start is not None:
            await self._start()
        if self._run is not None:
            self.task = asyncio.create_task(self._supervise(), name=f"component:{self.name}")
        self.status.healthy = True
        self.status.last_ok_ms = SystemClock().now_ms()

    async def _supervise(self) -> None:
        """Run the component, restarting it with backoff if it dies."""
        delay = 1.0
        while True:
            try:
                assert self._run is not None
                await self._run()
                return  # a component that returns cleanly is finished
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.status.healthy = False
                self.status.consecutive_failures += 1
                self.status.detail = f"{type(exc).__name__}: {exc}"
                log.error(
                    "component.crashed",
                    component=self.name,
                    critical=self.critical,
                    failures=self.status.consecutive_failures,
                    error=str(exc),
                    exc_info=True,
                )
                if not self.restart_on_failure:
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
            self.task = None
        if self._stop is not None:
            with contextlib.suppress(Exception):
                await self._stop()
        self.status.healthy = False

    async def check_health(self) -> bool:
        """Healthy means the work is happening, not that a probe likes what it
        sees.

        A dead task is unhealthy whatever the probe returns. That is not
        belt-and-braces: every custom probe here reports on a *proxy* for the
        work rather than on the work itself, and each proxy survives its task.
        The ingestors report ``ws.connected``, which stays true if the consuming
        task dies while the socket lives; the news pipeline reports on its last
        result, which stops changing precisely when it stops running.

        This is the shape of the failure that cost the platform sixty-six hours
        of persistence: the database writer's ``_running`` flag reported what
        start() had been asked to do rather than whether the drain task existed,
        and 2,084,222 rows were dropped while it said it was fine.
        """
        alive = self.task is None or not self.task.done()
        if self._health is None:
            ok = alive
        else:
            try:
                result = self._health()
                ok = await result if asyncio.iscoroutine(result) else bool(result)
            except Exception as exc:
                ok = False
                self.status.detail = f"health probe failed: {exc}"
            if ok and not alive:
                ok = False
                self.status.detail = (
                    "task is not running; the health probe reports on a proxy "
                    "that outlived it"
                )
        self.status.healthy = ok
        if ok:
            self.status.last_ok_ms = SystemClock().now_ms()
            self.status.consecutive_failures = 0
        else:
            self.status.consecutive_failures += 1
        return ok


@dataclass(slots=True)
class StartupReport:
    """What happened during start-up, and whether trading may begin."""

    phase: Phase = Phase.CREATED
    clean_previous_shutdown: bool | None = None
    #: How long the platform was not running, when that can be established.
    #:
    #: An unclean shutdown says the process died; this says when, and therefore
    #: for how long nobody was watching the positions. A five-hour gap and a
    #: five-second one are the same event without it.
    downtime_s: float | None = None
    last_seen_before_restart: str | None = None
    database_ready: bool = False
    kalshi_event_ready: bool = False
    kalshi_perp_ready: bool = False
    perps_enabled_for_account: bool | None = None
    reconciled: bool = False
    trading_permitted: bool = False
    blocking_reasons: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "clean_previous_shutdown": self.clean_previous_shutdown,
            "downtime_s": self.downtime_s,
            "last_seen_before_restart": self.last_seen_before_restart,
            "database_ready": self.database_ready,
            "kalshi_event_ready": self.kalshi_event_ready,
            "kalshi_perp_ready": self.kalshi_perp_ready,
            "perps_enabled_for_account": self.perps_enabled_for_account,
            "reconciled": self.reconciled,
            "trading_permitted": self.trading_permitted,
            "blocking_reasons": list(self.blocking_reasons),
            "components": list(self.components),
        }


class Application:
    """Owns the process: components, lifecycle, health, and the trading gate."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.clock = SystemClock()
        self.phase = Phase.CREATED
        self.database: Database = get_database()
        self.writer: AsyncWriter | None = None
        self.loop_lag: LoopLagMonitor | None = None
        self.kill_switches = KillSwitchManager()
        self.components: list[ServiceComponent] = []
        self.report = StartupReport()
        self._shutdown = asyncio.Event()
        self._last_heartbeat_ms = 0
        self._persisted_state_loaded = False
        self._started_at_iso = utcnow_aware().isoformat()
        self._health_task: asyncio.Task[None] | None = None
        self._started_ms = 0
        # Set only when the full start-up sequence completed successfully.
        self._trading_permitted = False
        #: Blocks that cannot be re-tested by a health probe and must therefore
        #: be cleared by whoever set them. Reconciliation is the motivating
        #: case: "we do not know what positions we hold" is not a condition a
        #: liveness check can disprove, and it must not lapse on its own.
        self._latched_blocks: dict[str, str] = {}

    # -- registration --------------------------------------------------------

    def register(self, component: ServiceComponent) -> None:
        """Register a component. Order matters: start order, reverse stop order."""
        if self.phase not in (Phase.CREATED, Phase.STARTING):
            raise RuntimeError("components must be registered before the application starts")
        self.components.append(component)

    # -- blocks --------------------------------------------------------------

    def add_block(self, key: str, reason: str) -> None:
        """Latch a condition that must stop trading until explicitly cleared.

        Use this rather than appending to ``report.blocking_reasons`` directly.
        A bare append is only read once, during start-up, and the health loop
        then recovers the application without ever knowing the reason existed.
        """
        if key not in self._latched_blocks:
            log.warning("app.block_latched", block=key, reason=reason)
        self._latched_blocks[key] = reason
        if reason not in self.report.blocking_reasons:
            self.report.blocking_reasons.append(reason)

    def clear_block(self, key: str) -> None:
        """Release a latched block, because the condition behind it is resolved."""
        reason = self._latched_blocks.pop(key, None)
        if reason is None:
            return
        log.info("app.block_cleared", block=key, reason=reason)
        with contextlib.suppress(ValueError):
            self.report.blocking_reasons.remove(reason)

    def current_blocks(self, unhealthy_critical: Sequence[str] = ()) -> list[str]:
        """Everything blocking trading *right now*.

        One definition, consulted by both start-up and the health loop. They
        used to disagree: start-up refused to permit trading for any of seven
        reasons, and the health loop re-permitted it five seconds later after
        checking two of them. An engaged event kill switch, a failed
        reconciliation, or an unreachable exchange were all silently cleared
        without anything having changed.
        """
        reasons: list[str] = []

        for status in self.kill_switches.all_statuses():
            if status.engaged:
                subject = f":{status.subject}" if status.subject else ""
                reasons.append(
                    f"kill switch engaged: {status.scope}{subject}"
                    f" ({status.reason or 'no reason recorded'})"
                )

        reasons.extend(
            f"critical component {name} is unhealthy" for name in unhealthy_critical
        )
        reasons.extend(self._latched_blocks.values())
        return reasons

    # -- the trading gate ----------------------------------------------------

    @property
    def trading_permitted(self) -> bool:
        """The single predicate the trading loop consults.

        False until the entire start-up sequence has completed, and false again
        the moment a critical component becomes unhealthy or a kill switch
        engages. Nothing sets this optimistically.
        """
        if not self._trading_permitted:
            return False
        if self.phase is not Phase.RUNNING:
            return False
        return not self.kill_switches.is_engaged(KillSwitchScope.GLOBAL)

    @property
    def can_submit_real_orders(self) -> bool:
        """Whether real money can move. Requires both switches AND a clean start."""
        return self.trading_permitted and self.settings.can_submit_real_orders

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> StartupReport:
        """Run the full start-up sequence. Never raises on a recoverable failure.

        A failure that prevents safe trading leaves the application ``DEGRADED``
        and running -- observable from the dashboard, ingesting data, but not
        trading -- rather than exiting. An operator can then see why.
        """
        with use_correlation_id():
            self.phase = Phase.STARTING
            self.report.phase = Phase.STARTING
            self._started_ms = self.clock.now_ms()
            log.info(
                "app.starting",
                version=_version(),
                mode=self.settings.operating_mode.value,
                environment=self.settings.environment.value,
                live_armed=self.settings.live_trading_armed,
                python=sys.version.split()[0],
                platform=platform.platform(),
            )

            try:
                await self._verify_configuration()
                await self._start_database()
                await self._load_persisted_state()
                self._register_loop_lag_monitor()
                await self._start_components()
                await self._run_health_checks()
            except Exception as exc:
                self.phase = Phase.FAILED
                self.report.phase = Phase.FAILED
                self.add_block("startup", f"{type(exc).__name__}: {exc}")
                log.error("app.startup_failed", error=str(exc), exc_info=True)
                return self.report

            # Trading is permitted only when nothing is blocking. The live
            # conditions are folded in here so start-up and the health loop
            # reach their verdict the same way.
            for reason in self.current_blocks():
                if reason not in self.report.blocking_reasons:
                    self.report.blocking_reasons.append(reason)

            if self.report.blocking_reasons:
                self.phase = Phase.DEGRADED
                self._trading_permitted = False
                log.warning(
                    "app.degraded_trading_disabled",
                    reasons=self.report.blocking_reasons,
                )
            else:
                self.phase = Phase.RUNNING
                self._trading_permitted = True
                log.info("app.running", environment=self.settings.environment.value)

            self.report.phase = self.phase
            self.report.trading_permitted = self.trading_permitted
            self.report.components = [c.name for c in self.components]

            self._health_task = asyncio.create_task(self._health_loop(), name="health-loop")
            self._audit(
                "app_started",
                subject="application",
                detail=self.report.to_dict(),
                severity="info" if self.trading_permitted else "warning",
            )
            return self.report

    async def _verify_configuration(self) -> None:
        settings = self.settings
        # The two live switches must agree. Settings validation already enforces
        # this at load; re-checking here means a programmatically constructed
        # AppSettings cannot slip past it.
        if settings.live_trading_armed and settings.environment is not TradingEnvironment.LIVE:
            raise ConfigError(
                "live_trading_armed is set without environment=live; refusing to start"
            )
        settings.log_path.mkdir(parents=True, exist_ok=True)
        settings.data_path.mkdir(parents=True, exist_ok=True)
        log.info("app.configuration_verified", config=settings.redacted())

    async def _start_database(self) -> None:
        try:
            await self.database.connect()
            self.report.database_ready = True
        except Exception as exc:
            # A platform without a database can still observe markets, but it
            # cannot audit, cannot persist state and cannot be trusted to
            # reconcile after a restart. That is not a trading state.
            self.add_block("database", f"database unavailable: {exc}")
            log.error("app.database_unavailable", error=str(exc))
            return

        self.writer = get_writer()
        await self.writer.start()
        self._register_writer_health()
        self.kill_switches = KillSwitchManager(self.database)

    def _register_writer_health(self) -> None:
        """Make the writer's *progress* a condition for trading.

        A writer that is not draining is not a degraded convenience; it is the
        audit trail stopping. On 2026-09-11 it committed roughly 1,500 rows an
        hour against 70,000 produced, for most of a day, and nothing noticed:
        health meant "the task exists" rather than "rows are landing". A
        platform that cannot record what it is doing should not be doing more of
        it, so this is a critical component and its failure degrades trading
        through the same gate as everything else -- positions are still managed,
        nothing new is opened.

        Registration is guarded because ``_start_database`` has two callers: the
        entrypoint runs it early so components are built against a live writer,
        and :meth:`start` runs it again as part of the fixed sequence.
        """
        if self.writer is None or any(c.name == _WRITER_COMPONENT for c in self.components):
            return
        writer = self.writer
        holder: list[ServiceComponent] = []

        def probe() -> bool:
            ok = writer.healthy()
            # The reason, not just the verdict: "23,104 rows queued and nothing
            # committed for 900s" is the whole diagnosis, and it belongs where
            # the System tab and the degraded-trading log line will read it.
            if holder:
                holder[0].status.detail = "" if ok else writer.assess()[1]
            return ok

        component = ServiceComponent(_WRITER_COMPONENT, health=probe, critical=True)
        holder.append(component)
        self.register(component)

    async def _load_persisted_state(self) -> None:
        """Restore kill switches, mode and the shutdown marker. Runs once.

        Idempotent because it has two callers -- the entrypoint runs it early so
        the database is up before any component that writes to it starts, and
        :meth:`start` runs it as part of the fixed sequence. Without the guard it
        ran twice, and the second pass read the marker the first pass had just
        cleared: ``clean_previous_shutdown`` was therefore False on every start,
        including the ones that shut down cleanly.

        That is worse than a duplicated log line. The flag exists to raise the
        bar for re-arming live trading after a crash, and a flag that is always
        set carries no information at all.
        """
        if self._persisted_state_loaded:
            return
        self._persisted_state_loaded = True
        await self._load_persisted_state_once()

    async def _load_persisted_state_once(self) -> None:
        """Restore kill switches and detect whether the last shutdown was clean."""
        await self.kill_switches.load()

        # Engaged switches are not appended here: they are re-derived live by
        # `current_blocks`, so a switch released while running stops blocking
        # and a switch engaged while running starts blocking, without either
        # depending on what was true at start-up.
        engaged = [s for s in self.kill_switches.all_statuses() if s.engaged]
        if engaged:
            log.warning(
                "app.kill_switches_engaged_at_startup",
                switches=[f"{s.scope}:{s.subject}" for s in engaged],
            )

        if not self.report.database_ready:
            return

        try:
            async with self.database.session() as session:
                row = (
                    await session.execute(
                        select(SystemState).where(SystemState.key == _SHUTDOWN_MARKER)
                    )
                ).scalar_one_or_none()
                previous = (row.value or {}) if row is not None else {}
                clean = bool(previous.get("clean"))
                self.report.clean_previous_shutdown = clean if row is not None else None
                # How long was nobody watching? The heartbeat is the last moment
                # the previous run is known to have been alive; for a clean stop
                # that is the shutdown itself, and for a kill it is up to one
                # heartbeat interval before the process died.
                self.report.last_seen_before_restart = previous.get("last_seen") or previous.get("at")
                self.report.downtime_s = _seconds_since(self.report.last_seen_before_restart)
                # Clear the marker: this run is now the one in progress.
                await session.merge(
                    SystemState(
                        key=_SHUTDOWN_MARKER,
                        value={"clean": False, "started_at": utcnow_aware().isoformat()},
                        updated_by="startup",
                    )
                )
        except Exception as exc:
            log.warning("app.shutdown_marker_unreadable", error=str(exc))
            return

        if self.report.clean_previous_shutdown is False:
            # Not blocking by itself -- reconciliation runs identically either
            # way -- but it is recorded and raises the bar for re-arming live.
            downtime = self.report.downtime_s
            log.warning(
                "app.previous_shutdown_was_unclean",
                note="reconciliation will determine actual state before trading",
                last_seen=self.report.last_seen_before_restart,
                downtime_s=round(downtime) if downtime is not None else None,
            )
            self._audit(
                "unclean_shutdown_detected",
                subject="application",
                detail={
                    "note": "previous run did not shut down cleanly",
                    "last_seen": self.report.last_seen_before_restart,
                    "downtime_s": round(downtime) if downtime is not None else None,
                },
                severity="warning",
            )

    def _register_loop_lag_monitor(self) -> None:
        """Watch how late this loop runs its callbacks.

        Registered here rather than by the entrypoint because it is not optional
        wiring: it is the one number that would have named the writer's collapse
        in a line instead of a reconstruction.
        """
        if any(c.name == _LOOP_MONITOR_COMPONENT for c in self.components):
            return
        self.loop_lag = LoopLagMonitor()
        self.register(
            ServiceComponent(
                _LOOP_MONITOR_COMPONENT, run=self.loop_lag.run, critical=False
            )
        )

    async def _start_components(self) -> None:
        for component in self.components:
            try:
                await component.start()
                log.info("component.started", component=component.name)
            except Exception as exc:
                component.status.healthy = False
                component.status.detail = str(exc)
                message = f"component {component.name} failed to start: {exc}"
                log.error("component.start_failed", component=component.name, error=str(exc))
                if component.critical:
                    self.add_block(f"component:{component.name}", message)

    async def _run_health_checks(self) -> None:
        for component in self.components:
            ok = await component.check_health()
            if not ok and component.critical:
                self.add_block(
                    f"component:{component.name}",
                    f"critical component {component.name} is unhealthy: "
                    f"{component.status.detail}",
                )

    # -- running -------------------------------------------------------------

    async def run_until_stopped(self) -> None:
        """Block until a shutdown signal arrives."""
        self._install_signal_handlers()
        await self._shutdown.wait()

    def request_shutdown(self, reason: str = "shutdown requested") -> None:
        log.info("app.shutdown_requested", reason=reason)
        self._shutdown.set()

    def _install_signal_handlers(self) -> None:
        """Handle Ctrl-C and service stop on both platforms.

        ``loop.add_signal_handler`` is not implemented on Windows, so the
        portable ``signal.signal`` path is used there. Both end at the same
        ``request_shutdown``.
        """
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_shutdown, f"signal {sig.name}")
            except (NotImplementedError, AttributeError, ValueError):
                with contextlib.suppress(Exception):
                    signal.signal(
                        sig, lambda s, _f: self.request_shutdown(f"signal {s}")
                    )

    async def _write_heartbeat(self) -> None:
        """Record that the platform was alive just now.

        The process can die in ways that run no Python at all -- the machine
        sleeps, the OS terminates it, the power goes. None of those reach the
        shutdown path, so the only way the *next* run can report how long the
        platform was down is if this one left a timestamp behind while it was
        healthy.

        Written from the health loop rather than on its own timer so it inherits
        that loop's meaning: a heartbeat is only laid down by a run that just
        passed its health checks and pinged its database.
        """
        now = self.clock.now_ms()
        if now - self._last_heartbeat_ms < _HEARTBEAT_INTERVAL_MS:
            return
        self._last_heartbeat_ms = now
        # Suppressed rather than logged: the database ping above already failed
        # loudly if the database is unreachable, and a heartbeat that logs its
        # own failure every thirty seconds would bury that.
        with contextlib.suppress(Exception):
            async with self.database.session() as session:
                await session.merge(
                    SystemState(
                        key=_SHUTDOWN_MARKER,
                        value={
                            "clean": False,
                            "started_at": self._started_at_iso,
                            "last_seen": utcnow_aware().isoformat(),
                        },
                        updated_by="heartbeat",
                    )
                )

    async def _health_loop(self) -> None:
        """Periodically probe components and gate trading on the result.

        Wrapped so that its own death is recorded. This loop is what notices
        every other component failing, so if it stops the platform loses not
        just one component but the ability to observe any of them -- and it
        keeps trading, because trading is gated on what this loop last decided.

        Nothing supervises this task; the heartbeat it writes every thirty
        seconds is the only external evidence it still runs. Making its exit
        loud is the cheap half of that, and the half that would have turned the
        database writer's sixty-six-hour silence into a single line at the top
        of the log.
        """
        try:
            await self._health_loop_forever()
        except asyncio.CancelledError:
            log.warning("app.health_loop_cancelled", phase=self.phase.value)
            raise
        except BaseException as exc:
            log.critical(
                "app.health_loop_died",
                error=str(exc), error_type=type(exc).__name__,
                phase=self.phase.value, exc_info=True,
            )
            raise

    async def _health_loop_forever(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(5.0)
            try:
                unhealthy_critical: list[str] = []
                for component in self.components:
                    was_healthy = component.status.healthy
                    ok = await component.check_health()
                    if was_healthy != ok:
                        self._record_health(component)
                    if not ok and component.critical:
                        unhealthy_critical.append(component.name)

                if self.database is not None and self.report.database_ready:
                    await self.database.ping()
                    await self._write_heartbeat()

                # The same predicate start-up used. Recovery that checked a
                # narrower set of conditions than the refusal did would undo the
                # refusal without anything having changed -- which is exactly
                # what it used to do, within five seconds, for an engaged kill
                # switch or a failed reconciliation.
                blocks = self.current_blocks(unhealthy_critical)
                self.report.blocking_reasons = list(blocks)

                if blocks and self.phase is Phase.RUNNING:
                    # Stop trading, keep observing. Existing positions still need
                    # to be managed, and the dashboard still needs to be truthful.
                    self.phase = Phase.DEGRADED
                    self._trading_permitted = False
                    log.error("app.degraded", reasons=blocks)
                    self._audit(
                        "trading_disabled_degraded",
                        subject="application",
                        detail={"reasons": blocks},
                        severity="critical",
                    )
                elif not blocks and self.phase is Phase.DEGRADED:
                    self.phase = Phase.RUNNING
                    self._trading_permitted = True
                    log.info("app.recovered")
                    self._audit(
                        "trading_re_enabled",
                        subject="application",
                        detail={"note": "every blocking condition cleared"},
                        severity="warning",
                    )

                metrics.gauge("app.trading_permitted", 1.0 if self.trading_permitted else 0.0)
                metrics.gauge("app.uptime_s", (self.clock.now_ms() - self._started_ms) / 1000)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("app.health_loop_error", error=str(exc), exc_info=True)

    def _record_health(self, component: ServiceComponent) -> None:
        if self.writer is None:
            return
        self.writer.enqueue(
            HealthEvent(
                component=component.name,
                status="healthy" if component.status.healthy else "unhealthy",
                message=component.status.detail or None,
                detail=component.status.to_dict(),
            )
        )

    # -- shutdown ------------------------------------------------------------

    async def stop(self, *, clean: bool = True) -> None:
        """Stop everything in reverse registration order.

        Ordering matters on the way down too: components that might still submit
        orders stop before the clients they submit through, so nothing tries to
        use a closed session on its way out.
        """
        if self.phase in (Phase.STOPPING, Phase.STOPPED):
            return
        self.phase = Phase.STOPPING
        self._trading_permitted = False
        log.info("app.stopping")

        if self._health_task is not None:
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_task
            self._health_task = None

        for component in reversed(self.components):
            try:
                await component.stop()
                log.info("component.stopped", component=component.name)
            except Exception as exc:
                log.error("component.stop_failed", component=component.name, error=str(exc))

        if clean and self.report.database_ready:
            with contextlib.suppress(Exception):
                async with self.database.session() as session:
                    await session.merge(
                        SystemState(
                            key=_SHUTDOWN_MARKER,
                            value={"clean": True, "at": utcnow_aware().isoformat()},
                            updated_by="shutdown",
                        )
                    )

        self._audit("app_stopped", subject="application", detail={"clean": clean})

        if self.writer is not None:
            # Drain: a clean exit should not lose the last seconds of audit trail.
            await self.writer.stop(drain=True)
        with contextlib.suppress(Exception):
            await self.database.disconnect()

        self.phase = Phase.STOPPED
        log.info("app.stopped", clean=clean)

    # -- reporting -----------------------------------------------------------

    def _audit(self, action: str, *, subject: str, detail: dict, severity: str = "info") -> None:
        if self.writer is None:
            return
        self.writer.enqueue(
            AuditLog(
                actor="system",
                action=action,
                subject=subject,
                detail=detail,
                severity=severity,
                environment=self.settings.environment.value,
            ),
            critical=True,
        )

    def status(self) -> dict[str, Any]:
        """Full status for the dashboard System tab."""
        return {
            "phase": self.phase.value,
            "version": _version(),
            "uptime_s": round((self.clock.now_ms() - self._started_ms) / 1000, 1)
            if self._started_ms
            else 0,
            "operating_mode": self.settings.operating_mode.value,
            "environment": self.settings.environment.value,
            "live_armed": self.settings.live_trading_armed,
            "trading_permitted": self.trading_permitted,
            "can_submit_real_orders": self.can_submit_real_orders,
            "database_healthy": self.database.healthy if self.database else False,
            "startup": self.report.to_dict(),
            "components": [c.status.to_dict() for c in self.components],
            "kill_switches": [
                {
                    "scope": s.scope,
                    "subject": s.subject,
                    "engaged": s.engaged,
                    "reason": s.reason,
                }
                for s in self.kill_switches.all_statuses()
            ],
            "writer": self.writer.stats() if self.writer else None,
            "loop_lag": self.loop_lag.stats() if self.loop_lag else None,
        }


def _version() -> str:
    from kat import __version__

    return __version__


# Domains the application knows about. Exposed so callers can iterate both
# without hard-coding the pair in several places.
DOMAINS: tuple[MarketType, ...] = (MarketType.EVENT, MarketType.PERP)
