"""The read model the dashboard sees.

The dashboard API runs **in the same process** as the trading engine rather than
as a separate service. That is a deliberate simplification: the data it needs is
the engine's own in-memory state, and putting a process boundary in the middle
would mean either polling the database (stale, and it puts the database back on
a path we removed it from) or inventing an IPC layer for one consumer.

What this module provides is a single object the API can read and the engine can
populate. It holds *references*, not copies, so a dashboard request sees live
state with no synchronisation step.

The one rule
------------
**Everything here is read-only from the API's perspective, except for the
explicitly enumerated control actions.** Those actions -- mode changes, kill
switches, pause/resume, emergency flatten, enabling a strategy -- go through the
same code paths as everything else. In particular:

* Emergency flatten constructs an ``OrderIntent`` and routes it through the risk
  engine. It is not exempt from evaluation; the risk engine simply recognises it
  as exposure-reducing and approves it.
* There is no control that raises a risk limit beyond its backend clamp, and
  none at all that touches perpetual leverage -- no such field or endpoint
  exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kat.common.enums import (
    KillSwitchScope,
    MarketType,
    OperatingMode,
    TradingEnvironment,
)
from kat.common.fixedpoint import micros_to_decimal
from kat.net.tailscale import remote_access_status
from kat.observability.logging import get_logger
from kat.observability.metrics import metrics
from kat.risk.invariants import MAX_PERP_LEVERAGE

#: system_state key holding operator-adjusted risk limits.
RISK_OVERRIDES_KEY = "risk_overrides"

if TYPE_CHECKING:
    from kat.engine.lifecycle import Application
    from kat.engine.trading import TradingLoop
    from kat.execution.reconcile import Reconciler
    from kat.marketdata.state import MarketState

log = get_logger(__name__)

__all__ = ["EngineContext", "get_context", "set_context"]


@dataclass
class EngineContext:
    """Live references to everything the dashboard reads."""

    app: Application | None = None
    loops: dict[MarketType, TradingLoop] = field(default_factory=dict)
    market_states: dict[MarketType, MarketState] = field(default_factory=dict)
    reconciler: Reconciler | None = None

    #: Subsystems that may be absent (perps not enabled, LLM disabled, etc.).
    strategy_registry: Any = None
    model_registry: Any = None
    arbitrage_scanner: Any = None
    news_pipeline: Any = None
    research_loop: Any = None
    portfolio_manager: Any = None
    #: Feature workers by domain. Held so the dashboard can show warm-up
    #: progress: several features need an hour of history before they compute at
    #: all, and until they do the strategies that use them decline every market.
    #: Without this, that is indistinguishable from a broken feature pipeline.
    feature_workers: dict[MarketType, Any] = field(default_factory=dict)

    # -- status --------------------------------------------------------------

    def overview(self) -> dict[str, Any]:
        """The Overview tab: equity, P&L, exposure, health, at a glance."""
        event = self._domain_summary(MarketType.EVENT)
        perp = self._domain_summary(MarketType.PERP)
        total_equity = (event.get("equity_micros") or 0) + (perp.get("equity_micros") or 0)
        return {
            "environment": self.app.settings.environment.value if self.app else "unknown",
            "operating_mode": self.app.settings.operating_mode.value if self.app else "unknown",
            "live_armed": self.app.settings.live_trading_armed if self.app else False,
            "can_submit_real_orders": self.app.can_submit_real_orders if self.app else False,
            "trading_permitted": self.app.trading_permitted if self.app else False,
            "phase": self.app.phase.value if self.app else "unknown",
            "total_equity_usd": str(micros_to_decimal(total_equity)),
            # Whether that number is money or a placeholder. In PAPER the equity
            # is a configured seed, and a dashboard showing it unlabelled next to
            # live production prices reads as an account balance -- which it is
            # not. The page needs to be able to say so.
            "equity_is_simulated": self.app.settings.environment is not TradingEnvironment.LIVE
            if self.app
            else True,
            "domains_available": [
                domain.value
                for domain, loop in self.loops.items()
                if loop.account_state is not None
            ],
            "event": event,
            "perp": perp,
            "health": self.health(),
        }

    def _domain_summary(self, market_type: MarketType) -> dict[str, Any]:
        loop = self.loops.get(market_type)
        if loop is None:
            return {"available": False}
        state = loop.account_state
        summary: dict[str, Any] = {
            "available": True,
            "strategies_active": len(loop.active_strategies()),
            "strategies_registered": len(loop.strategies),
            "markets_watched": len(loop.market_state),
            "open_orders": len(loop.execution.working_orders()),
            "cycles": loop.cycles,
            "last_cycle": loop.last_cycle.to_dict() if loop.last_cycle else None,
        }
        if state is None:
            summary["equity_micros"] = 0
            return summary

        if market_type is MarketType.EVENT:
            summary.update(
                equity_micros=state.equity_micros,
                equity_usd=str(micros_to_decimal(state.equity_micros)),
                balance_usd=str(micros_to_decimal(state.balance_micros)),
                exposure_usd=str(micros_to_decimal(state.total_exposure_micros())),
                open_positions=sum(1 for p in state.positions.values() if p.position_qc != 0),
                realized_pnl_today_usd=str(
                    micros_to_decimal(state.realized_pnl_today_micros)
                ),
                drawdown_pct=str(state.drawdown_pct()),
            )
        else:
            leverage = state.effective_leverage()
            summary.update(
                equity_micros=state.equity_micros,
                equity_usd=str(micros_to_decimal(state.equity_micros)),
                equity_source=state.equity_source,
                equity_is_known=state.equity_is_known,
                exposure_usd=str(micros_to_decimal(state.effective_exposure_micros())),
                pending_exposure_usd=str(
                    micros_to_decimal(state.total_pending_exposure_micros())
                ),
                # The three numbers the Perpetuals tab must make unmissable.
                effective_leverage=f"{leverage:.6f}" if leverage is not None else None,
                max_leverage=f"{MAX_PERP_LEVERAGE:.2f}",
                exchange_account_leverage=state.exchange_account_leverage,
                open_positions=sum(1 for p in state.positions.values() if p.position_qc != 0),
                realized_pnl_today_usd=str(
                    micros_to_decimal(state.realized_pnl_today_micros)
                ),
            )
        return summary

    def market_label(self, market_type: MarketType, ticker: str) -> dict[str, Any]:
        """How a market and a side should be named to a person.

        A ticker like ``KXPREMIERLEAGUE-27-CFC`` is precise and unreadable, and
        it is actively ambiguous on a multi-outcome event: twenty Premier League
        contracts share a prefix and differ only in the last few characters. The
        exchange already publishes what separates them.

        ``outcome`` is the thing this contract is about ("Chelsea"), which is the
        field that disambiguates siblings. ``question`` is the whole question
        ("Will Chelsea win the English Premier League?"). ``event`` groups the
        siblings so a reader can see that four of their positions are the same
        bet wearing different names.

        Every field degrades to the ticker rather than to nothing: a market whose
        definition has not arrived yet still has to be nameable.
        """
        state = self.market_states.get(market_type)
        snapshot = state.get(ticker) if state is not None else None
        meta = dict(snapshot.metadata) if snapshot is not None else {}

        outcome = str(meta.get("yes_sub_title") or "").strip()
        question = str(meta.get("title") or "").strip()
        event = str(meta.get("event_ticker") or "").strip()
        if not event and snapshot is not None:
            event = snapshot.event_ticker or ""

        return {
            "ticker": ticker,
            # Short enough for a table column, and the thing a person recognises.
            "name": outcome or question or ticker,
            "outcome": outcome or None,
            "question": question or None,
            "event_ticker": event or None,
        }

    def describe_side(
        self, market_type: MarketType, ticker: str, *, position_qc: int
    ) -> dict[str, Any]:
        """A position, said in words.

        Holding YES pays if the question resolves yes; holding NO pays if it does
        not. Stating that explicitly matters most on the NO side, where the
        ticker, the outcome name and the question all describe the thing you are
        betting *against* -- a position in "Democratic Party" that profits when
        Democrats lose reads as its own opposite until someone says so.
        """
        label = self.market_label(market_type, ticker)
        long_yes = position_qc > 0
        question = label["question"]
        pays_if = (question if long_yes else f"NOT: {question}") if question else None
        label.update(
            side="yes" if long_yes else "no",
            holding=(
                f"YES on {label['name']}" if long_yes else f"NO on {label['name']}"
            ),
            pays_if=pays_if,
        )
        return label

    def leverage_panel(self) -> dict[str, Any]:
        """The Perpetuals tab's leverage panel.

        Includes the projected leverage for every working order, so an operator
        can see not only where exposure is but where it would be if everything
        resting filled.
        """
        loop = self.loops.get(MarketType.PERP)
        state = loop.account_state if loop else None
        if state is None:
            return {
                "available": False,
                "max_leverage": f"{MAX_PERP_LEVERAGE:.2f}",
                "note": "perpetual account state is not available",
            }

        current = state.effective_exposure_micros()
        pending = state.total_pending_exposure_micros()
        equity = state.equity_micros

        def ratio(value: int) -> str | None:
            if not state.equity_is_known or equity <= 0:
                return None
            from decimal import Decimal

            return f"{Decimal(value) / Decimal(equity):.6f}"

        return {
            "available": True,
            "max_leverage": f"{MAX_PERP_LEVERAGE:.2f}",
            "equity_usd": str(micros_to_decimal(equity)),
            "equity_source": state.equity_source,
            "equity_is_known": state.equity_is_known,
            "current_exposure_usd": str(micros_to_decimal(current)),
            "pending_exposure_usd": str(micros_to_decimal(pending)),
            "current_leverage": ratio(current),
            "projected_leverage": ratio(current + pending),
            "headroom_usd": str(micros_to_decimal(max(0, equity - current - pending))),
            "exchange_account_leverage": state.exchange_account_leverage,
            "exchange_leverage_note": (
                "Kalshi's account_leverage is notional / maintenance-margin, not "
                "notional / equity. It is shown for cross-checking only and is "
                "never used as the limit."
            ),
            "blocked_violations": metrics.get_counter("risk.leverage_blocks"),
        }

    def health(self) -> dict[str, Any]:
        if self.app is None:
            return {"status": "not_started"}
        status = self.app.status()
        return {
            "phase": status["phase"],
            "trading_permitted": status["trading_permitted"],
            "database": status["database_healthy"],
            "components": status["components"],
            "kill_switches": status["kill_switches"],
            "writer": status["writer"],
        }

    def system(self) -> dict[str, Any]:
        """The System tab: process, queues, latency, data freshness."""
        import psutil

        process = psutil.Process()
        with process.oneshot():
            mem = process.memory_info()
            cpu = process.cpu_percent(interval=None)
        disk = psutil.disk_usage(".")

        now_ms = self.app.clock.now_ms() if self.app else 0
        return {
            "process": {
                "cpu_pct": cpu,
                "rss_mb": round(mem.rss / 1_048_576, 1),
                "threads": process.num_threads(),
                "open_files": _safe(lambda: len(process.open_files())),
                "connections": _safe(lambda: len(process.net_connections())),
            },
            "host": {
                "cpu_pct": psutil.cpu_percent(interval=None),
                "ram_pct": psutil.virtual_memory().percent,
                "disk_free_gb": round(disk.free / 1_073_741_824, 1),
                "disk_pct": disk.percent,
            },
            # Whether this dashboard is reachable from another device, and how.
            # Shown because "I set it up and it isn't working" is otherwise
            # indistinguishable from "it is working": both look like a page that
            # loads on the machine it runs on.
            "remote_access": remote_access_status(),
            # Warm-up progress. Several features need an hour of history before
            # they produce anything, and the strategies that read them decline
            # every market until they do -- which looks exactly like a broken
            # pipeline unless the counts are visible.
            "features": {
                domain.value: worker.stats()
                for domain, worker in self.feature_workers.items()
            },
            "market_data": {
                domain.value: state.stats(now_ms=now_ms)
                for domain, state in self.market_states.items()
            },
            "metrics": metrics.snapshot(),
            "app": self.app.status() if self.app else None,
        }

    # -- control actions -----------------------------------------------------

    async def set_operating_mode(self, mode: OperatingMode, *, actor: str = "dashboard") -> None:
        """Change autonomy mode. Never affects any risk limit."""
        if self.app is None:
            raise RuntimeError("application is not running")
        previous = self.app.settings.operating_mode
        object.__setattr__(self.app.settings, "operating_mode", mode)
        for loop in self.loops.values():
            loop.operating_mode = mode
        log.warning(
            "control.operating_mode_changed",
            actor=actor, previous=previous.value, mode=mode.value,
        )
        self.app._audit(
            "operating_mode_changed",
            subject="application",
            detail={"from": previous.value, "to": mode.value, "actor": actor},
            severity="warning",
        )

    async def set_risk_tunable(
        self, key: str, value: Any, *, actor: str = "dashboard"
    ) -> dict[str, Any]:
        """Adjust one risk limit on the running platform.

        Applied live, audited, and persisted, in that order. Persisting last is
        deliberate: a value that could not be applied must not be written down
        as though it had been.

        Raises :class:`~kat.risk.tunables.TunableError` for an unknown key or an
        out-of-range value. The perpetual leverage ceiling is not a key here and
        never can be -- there is no number behind it to set.
        """
        from kat.risk.tunables import apply_override, tunable_by_key

        if self.app is None:
            raise RuntimeError("application is not running")

        tunable = tunable_by_key(key)
        previous, applied = apply_override(self.app.settings, key, value)

        log.warning(
            "control.risk_tunable_changed",
            actor=actor, key=key, previous=str(previous), applied=str(applied),
            requested=str(value),
        )
        self.app._audit(
            "risk_tunable_changed",
            subject=key,
            detail={
                "key": key,
                "label": tunable.label,
                "from": str(previous),
                "to": str(applied),
                # Recorded separately because pydantic clamps on assignment, so
                # what was asked for and what took effect are not always equal.
                "requested": str(value),
                "actor": actor,
            },
            severity="warning",
        )
        await self._persist_risk_overrides()
        return {
            "key": key,
            "previous": str(previous),
            "value": str(applied),
            "requested": str(value),
            "clamped": str(applied) != str(value),
        }

    async def _persist_risk_overrides(self) -> None:
        """Write every non-default tunable so a restart does not revert them.

        A risk limit that quietly returns to its default overnight is worse than
        one that was never changed: the operator believes a bound is in force
        that is not.
        """

        from kat.config.settings import AppSettings
        from kat.db.models import SystemState
        from kat.risk.tunables import TUNABLES

        if self.app is None or self.app.database is None:
            return

        defaults = AppSettings()
        overrides = {
            t.key: str(t.read(self.app.settings))
            for t in TUNABLES
            if t.read(self.app.settings) != t.read(defaults)
        }
        try:
            async with self.app.database.session() as session:
                await session.merge(
                    SystemState(
                        key=RISK_OVERRIDES_KEY, value=overrides, updated_by="dashboard"
                    )
                )
        except Exception as exc:
            # The change is already live; failing to write it down is worth
            # shouting about but must not undo it.
            log.error("control.risk_overrides_not_persisted", error=str(exc))

    async def load_risk_overrides(self) -> int:
        """Re-apply persisted tunables at start-up. Returns how many were applied."""
        from sqlalchemy import select

        from kat.db.models import SystemState
        from kat.risk.tunables import TunableError, apply_override

        if self.app is None or self.app.database is None:
            return 0
        try:
            async with self.app.database.session() as session:
                row = (
                    await session.execute(
                        select(SystemState).where(SystemState.key == RISK_OVERRIDES_KEY)
                    )
                ).scalar_one_or_none()
        except Exception as exc:
            log.error("control.risk_overrides_unreadable", error=str(exc))
            return 0

        stored = (row.value if row else None) or {}
        applied = 0
        for key, value in stored.items():
            try:
                _, took = apply_override(self.app.settings, key, value)
                applied += 1
                log.info("control.risk_override_restored", key=key, value=str(took))
            except TunableError as exc:
                # A key that no longer exists, or a bound that has since
                # tightened. Skipped loudly rather than blocking start-up.
                log.warning("control.risk_override_discarded", key=key, error=str(exc))
        return applied

    async def engage_kill_switch(
        self, scope: KillSwitchScope, *, reason: str, actor: str = "dashboard", subject: str = ""
    ) -> None:
        if self.app is None:
            raise RuntimeError("application is not running")
        await self.app.kill_switches.engage(
            scope.value, reason=reason, actor=actor, subject=subject
        )

    async def release_kill_switch(
        self, scope: KillSwitchScope, *, actor: str = "dashboard", subject: str = ""
    ) -> None:
        if self.app is None:
            raise RuntimeError("application is not running")
        await self.app.kill_switches.release(scope.value, actor=actor, subject=subject)

    async def pause_trading(self, *, actor: str = "dashboard") -> None:
        await self.engage_kill_switch(
            KillSwitchScope.GLOBAL, reason="paused from dashboard", actor=actor
        )

    async def resume_trading(self, *, actor: str = "dashboard") -> None:
        await self.release_kill_switch(KillSwitchScope.GLOBAL, actor=actor)

    async def cancel_all_orders(self, market_type: MarketType | None = None) -> int:
        """Cancel working orders. Reduces risk, so it is always permitted."""
        total = 0
        for domain, loop in self.loops.items():
            if market_type is None or domain is market_type:
                total += await loop.execution.cancel_all()
        log.warning("control.cancel_all", cancelled=total, domain=str(market_type))
        return total

    def snapshot(self) -> dict[str, Any]:
        """Everything, for a single dashboard poll."""
        return {
            "overview": self.overview(),
            "leverage": self.leverage_panel(),
            "health": self.health(),
        }


def _safe(fn: Any) -> Any:
    """Some psutil probes need privileges we deliberately do not have."""
    try:
        return fn()
    except Exception:
        return None


_context: EngineContext | None = None


def get_context() -> EngineContext:
    """The process-wide context. Created empty so the API can start first."""
    global _context
    if _context is None:
        _context = EngineContext()
    return _context


def set_context(context: EngineContext) -> None:
    global _context
    _context = context
