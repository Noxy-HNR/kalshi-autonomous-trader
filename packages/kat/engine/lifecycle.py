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
"""

from __future__ import annotations

import asyncio
import contextlib
import platform
import signal
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
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
from kat.observability.logging import get_logger
from kat.observability.metrics import metrics
from kat.risk.killswitch import KillSwitchManager

log = get_logger(__name__)

__all__ = ["Application", "ComponentHealth", "Phase", "ServiceComponent"]

#: Key under which the clean-shutdown marker is stored.
_SHUTDOWN_MARKER = "clean_shutdown"


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
            except Exception as exc:  # noqa: BLE001 - supervision must catch everything
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
        if self._health is None:
            # No probe: healthy iff its task has not died.
            ok = self.task is None or not self.task.done()
        else:
            try:
                result = self._health()
                ok = await result if asyncio.iscoroutine(result) else bool(result)
            except Exception as exc:  # noqa: BLE001
                ok = False
                self.status.detail = f"health probe failed: {exc}"
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
        self.kill_switches = KillSwitchManager()
        self.components: list[ServiceComponent] = []
        self.report = StartupReport()
        self._shutdown = asyncio.Event()
        self._health_task: asyncio.Task[None] | None = None
        self._started_ms = 0
        # Set only when the full start-up sequence completed successfully.
        self._trading_permitted = False

    # -- registration --------------------------------------------------------

    def register(self, component: ServiceComponent) -> None:
        """Register a component. Order matters: start order, reverse stop order."""
        if self.phase not in (Phase.CREATED, Phase.STARTING):
            raise RuntimeError("components must be registered before the application starts")
        self.components.append(component)

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
        if self.kill_switches.is_engaged(KillSwitchScope.GLOBAL):
            return False
        return True

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
                await self._start_components()
                await self._run_health_checks()
            except Exception as exc:  # noqa: BLE001
                self.phase = Phase.FAILED
                self.report.phase = Phase.FAILED
                self.report.blocking_reasons.append(f"{type(exc).__name__}: {exc}")
                log.error("app.startup_failed", error=str(exc), exc_info=True)
                return self.report

            # Trading is permitted only when nothing is blocking.
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
        except Exception as exc:  # noqa: BLE001
            # A platform without a database can still observe markets, but it
            # cannot audit, cannot persist state and cannot be trusted to
            # reconcile after a restart. That is not a trading state.
            self.report.blocking_reasons.append(f"database unavailable: {exc}")
            log.error("app.database_unavailable", error=str(exc))
            return

        self.writer = get_writer()
        await self.writer.start()
        self.kill_switches = KillSwitchManager(self.database)

    async def _load_persisted_state(self) -> None:
        """Restore kill switches and detect whether the last shutdown was clean."""
        await self.kill_switches.load()

        engaged = [s for s in self.kill_switches.all_statuses() if s.engaged]
        if engaged:
            self.report.blocking_reasons.extend(
                f"kill switch engaged: {s.scope}{':' + s.subject if s.subject else ''}"
                f" ({s.reason or 'no reason recorded'})"
                for s in engaged
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
                clean = bool(row and (row.value or {}).get("clean"))
                self.report.clean_previous_shutdown = clean if row is not None else None
                # Clear the marker: this run is now the one in progress.
                await session.merge(
                    SystemState(key=_SHUTDOWN_MARKER, value={"clean": False}, updated_by="startup")
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("app.shutdown_marker_unreadable", error=str(exc))
            return

        if self.report.clean_previous_shutdown is False:
            # Not blocking by itself -- reconciliation runs identically either
            # way -- but it is recorded and raises the bar for re-arming live.
            log.warning(
                "app.previous_shutdown_was_unclean",
                note="reconciliation will determine actual state before trading",
            )
            self._audit(
                "unclean_shutdown_detected",
                subject="application",
                detail={"note": "previous run did not shut down cleanly"},
                severity="warning",
            )

    async def _start_components(self) -> None:
        for component in self.components:
            try:
                await component.start()
                log.info("component.started", component=component.name)
            except Exception as exc:  # noqa: BLE001
                component.status.healthy = False
                component.status.detail = str(exc)
                message = f"component {component.name} failed to start: {exc}"
                log.error("component.start_failed", component=component.name, error=str(exc))
                if component.critical:
                    self.report.blocking_reasons.append(message)

    async def _run_health_checks(self) -> None:
        for component in self.components:
            ok = await component.check_health()
            if not ok and component.critical:
                self.report.blocking_reasons.append(
                    f"critical component {component.name} is unhealthy: {component.status.detail}"
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

    async def _health_loop(self) -> None:
        """Periodically probe components and gate trading on the result."""
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

                if unhealthy_critical and self.phase is Phase.RUNNING:
                    # Stop trading, keep observing. Existing positions still need
                    # to be managed, and the dashboard still needs to be truthful.
                    self.phase = Phase.DEGRADED
                    self._trading_permitted = False
                    log.error("app.degraded", components=unhealthy_critical)
                    self._audit(
                        "trading_disabled_degraded",
                        subject="application",
                        detail={"unhealthy": unhealthy_critical},
                        severity="critical",
                    )
                elif not unhealthy_critical and self.phase is Phase.DEGRADED:
                    # Recovery does NOT automatically re-arm live trading if the
                    # reason we degraded was a reconciliation failure -- that
                    # requires the kill switch to be released deliberately.
                    if not self.kill_switches.is_engaged(KillSwitchScope.RISK):
                        self.phase = Phase.RUNNING
                        self._trading_permitted = True
                        log.info("app.recovered")

                metrics.gauge("app.trading_permitted", 1.0 if self.trading_permitted else 0.0)
                metrics.gauge("app.uptime_s", (self.clock.now_ms() - self._started_ms) / 1000)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
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
            except Exception as exc:  # noqa: BLE001
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
        }


def _version() -> str:
    from kat import __version__

    return __version__


# Domains the application knows about. Exposed so callers can iterate both
# without hard-coding the pair in several places.
DOMAINS: tuple[MarketType, ...] = (MarketType.EVENT, MarketType.PERP)
