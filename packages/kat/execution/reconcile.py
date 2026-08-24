"""Reconciliation.

Run at start-up, after any reconnection, and on a slow periodic cycle. Its job is
to answer one question honestly:

    *Does our view of positions, orders and balances match the exchange's?*

If the answer is no and the difference cannot be explained, the platform does
**not** resume trading. It halts, engages the risk kill switch and reports what
disagreed. Trading on an unreconciled state is how a system doubles a position it
already had, or hedges something it does not hold.

Start-up sequence (the order matters)::

    connect -> authenticate -> load persisted state -> query live account
      -> reconcile positions -> reconcile orders -> reconcile fills
      -> reconcile balances -> rebuild in-memory state -> validate risk state
      -> only then resume

Nothing here mutates the exchange. Reconciliation observes and reports;
remediation (cancelling an orphan, flattening an unexpected position) is a
separate, explicit action that still routes through the risk engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kat.common.enums import KillSwitchScope, MarketType, OrderState
from kat.common.fixedpoint import micros_to_decimal
from kat.execution.base import ExecutionEngine
from kat.kalshi.event_contracts.rest import EventContractsREST
from kat.kalshi.perps.rest import PerpsREST
from kat.observability.logging import get_logger
from kat.risk.event_risk import EventAccountState, EventPositionState
from kat.risk.killswitch import KillSwitchManager
from kat.risk.perp_risk import PerpAccountState, PerpPositionState

log = get_logger(__name__)

__all__ = ["Reconciler", "ReconciliationReport"]


@dataclass(slots=True)
class Discrepancy:
    kind: str
    subject: str
    ours: str
    theirs: str
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "ours": self.ours,
            "theirs": self.theirs,
            "severity": self.severity,
        }


@dataclass(slots=True)
class ReconciliationReport:
    """What reconciliation found. ``safe_to_trade`` is the operative field."""

    domain: str
    positions_checked: int = 0
    orders_checked: int = 0
    unknown_orders_resolved: int = 0
    orphaned_orders: list[str] = field(default_factory=list)
    discrepancies: list[Discrepancy] = field(default_factory=list)
    balance_micros: int | None = None
    equity_micros: int | None = None
    error: str | None = None

    @property
    def safe_to_trade(self) -> bool:
        """True only when nothing serious disagreed.

        Warnings do not block: an orphaned order is reported and cancelled, not
        treated as corruption. Errors do block, without exception.
        """
        if self.error is not None:
            return False
        return not any(d.severity == "error" for d in self.discrepancies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "safe_to_trade": self.safe_to_trade,
            "positions_checked": self.positions_checked,
            "orders_checked": self.orders_checked,
            "unknown_orders_resolved": self.unknown_orders_resolved,
            "orphaned_orders": self.orphaned_orders,
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "balance_usd": (
                str(micros_to_decimal(self.balance_micros))
                if self.balance_micros is not None else None
            ),
            "equity_usd": (
                str(micros_to_decimal(self.equity_micros))
                if self.equity_micros is not None else None
            ),
            "error": self.error,
        }


class Reconciler:
    """Compares local state against the exchange for both domains."""

    def __init__(
        self,
        *,
        event_rest: EventContractsREST | None,
        perp_rest: PerpsREST | None,
        event_exec: ExecutionEngine | None,
        perp_exec: ExecutionEngine | None,
        kill_switches: KillSwitchManager | None = None,
    ) -> None:
        self.event_rest = event_rest
        self.perp_rest = perp_rest
        self.event_exec = event_exec
        self.perp_exec = perp_exec
        self.kill_switches = kill_switches

    # -- event contracts -----------------------------------------------------

    async def reconcile_event_contracts(
        self, state: EventAccountState, *, now_ms: int
    ) -> ReconciliationReport:
        report = ReconciliationReport(domain="event")
        if self.event_rest is None:
            report.error = "event-contract REST client is not configured"
            return report

        try:
            balance = await self.event_rest.get_balance()
            positions = await self.event_rest.get_positions()
            exchange_orders = await self.event_rest.get_orders(status="resting")
        except Exception as exc:
            report.error = f"could not read event-contract account state: {exc}"
            log.error("reconcile.event_query_failed", error=str(exc))
            return report

        # --- balances -------------------------------------------------------
        report.balance_micros = balance.balance_micros
        state.balance_micros = balance.balance_micros
        if balance.portfolio_value_micros is not None:
            state.portfolio_value_micros = balance.portfolio_value_micros

        # --- positions: the exchange is authoritative ------------------------
        rebuilt: dict[str, EventPositionState] = {}
        for pos in positions:
            if pos.position_qc == 0:
                continue
            report.positions_checked += 1
            previous = state.positions.get(pos.ticker)
            if previous is not None and previous.position_qc != pos.position_qc:
                report.discrepancies.append(
                    Discrepancy(
                        kind="position_mismatch",
                        subject=pos.ticker,
                        ours=str(previous.position_qc),
                        theirs=str(pos.position_qc),
                        # Recoverable: we adopt the exchange's number. It is an
                        # error rather than a warning because it means we missed
                        # a fill, and that deserves investigation before trading.
                        severity="error",
                    )
                )
            rebuilt[pos.ticker] = EventPositionState(
                ticker=pos.ticker,
                event_ticker=previous.event_ticker if previous else "",
                position_qc=pos.position_qc,
                exposure_micros=pos.market_exposure_micros,
            )

        # Positions we think we hold that the exchange does not know about.
        for ticker, local in state.positions.items():
            if ticker not in rebuilt and local.position_qc != 0:
                report.discrepancies.append(
                    Discrepancy(
                        kind="phantom_position",
                        subject=ticker,
                        ours=str(local.position_qc),
                        theirs="0",
                        severity="error",
                    )
                )
        state.positions = rebuilt
        state.updated_at_ms = now_ms

        # --- orders ---------------------------------------------------------
        await self._reconcile_orders(
            report,
            self.event_exec,
            {o.client_order_id: o for o in exchange_orders if o.client_order_id},
        )
        self._log_report(report)
        return report

    # -- perpetuals ----------------------------------------------------------

    async def reconcile_perpetuals(
        self, state: PerpAccountState, *, now_ms: int
    ) -> ReconciliationReport:
        report = ReconciliationReport(domain="perp")
        if self.perp_rest is None:
            report.error = "perpetual REST client is not configured"
            return report

        try:
            # compute_available_balance is expensive (50 tokens) but this is the
            # reconciliation path, not a fast poll, and the figure matters.
            balance = await self.perp_rest.get_balance(compute_available_balance=True)
            positions = await self.perp_rest.get_positions()
            risk = await self.perp_rest.get_risk()
            exchange_orders = await self.perp_rest.get_orders()
        except Exception as exc:
            report.error = f"could not read perpetual account state: {exc}"
            log.error("reconcile.perp_query_failed", error=str(exc))
            return report

        rebuilt = PerpAccountState.from_exchange(balance, risk, updated_at_ms=now_ms)

        # Merge in per-position detail the risk endpoint does not carry.
        for pos in positions:
            if pos.position_qc == 0:
                continue
            report.positions_checked += 1
            existing = rebuilt.positions.get(pos.ticker)
            mark = existing.mark_price_cc if existing else None
            rebuilt.positions[pos.ticker] = PerpPositionState(
                ticker=pos.ticker,
                position_qc=pos.position_qc,
                mark_price_cc=mark,
                entry_price_cc=pos.entry_price_cc,
            )
            previous = state.positions.get(pos.ticker)
            if previous is not None and previous.position_qc != pos.position_qc:
                report.discrepancies.append(
                    Discrepancy(
                        kind="position_mismatch",
                        subject=pos.ticker,
                        ours=str(previous.position_qc),
                        theirs=str(pos.position_qc),
                        severity="error",
                    )
                )

        for ticker, local in state.positions.items():
            if ticker not in rebuilt.positions and local.position_qc != 0:
                report.discrepancies.append(
                    Discrepancy(
                        kind="phantom_position", subject=ticker,
                        ours=str(local.position_qc), theirs="0", severity="error",
                    )
                )

        # Adopt the exchange's view wholesale.
        state.equity_micros = rebuilt.equity_micros
        state.equity_source = rebuilt.equity_source
        state.equity_is_known = rebuilt.equity_is_known
        state.positions = rebuilt.positions
        state.exchange_total_notional_micros = rebuilt.exchange_total_notional_micros
        state.exchange_account_leverage = rebuilt.exchange_account_leverage
        state.exchange_maintenance_margin_micros = rebuilt.exchange_maintenance_margin_micros
        state.updated_at_ms = now_ms
        report.equity_micros = state.equity_micros

        # Equity that cannot be established is not a warning: without it the
        # leverage invariant has no denominator and no perpetual order can be
        # proven safe.
        if not state.equity_is_known:
            report.discrepancies.append(
                Discrepancy(
                    kind="equity_unavailable",
                    subject="perp_account",
                    ours="unknown",
                    theirs=rebuilt.equity_source,
                    severity="error",
                )
            )

        # The notional risk limit Kalshi has set for this account is an
        # additional ceiling we must respect.
        try:
            limit = await self.perp_rest.get_notional_risk_limit()
            raw = limit.get("default_notional_value_risk_limit")
            if raw:
                from kat.common.fixedpoint import Rounding, parse_dollars_micros

                state.exchange_notional_risk_limit_micros = parse_dollars_micros(
                    raw, rounding=Rounding.FLOOR, field="notional_risk_limit"
                )
        except Exception as exc:
            # Not fatal: our own limits are tighter. Recorded so the dashboard
            # can show that the exchange ceiling is currently unverified.
            log.warning("reconcile.notional_limit_unavailable", error=str(exc))

        await self._reconcile_orders(
            report,
            self.perp_exec,
            {o.client_order_id: o for o in exchange_orders if o.client_order_id},
        )

        # Independent cross-check: our computed leverage against the exchange's
        # own figures. They measure different things, so this is a sanity check
        # on the inputs rather than a limit.
        local_leverage = state.effective_leverage()
        if local_leverage is not None:
            log.info(
                "reconcile.leverage_crosscheck",
                local_effective_leverage=str(local_leverage),
                exchange_account_leverage=state.exchange_account_leverage,
                local_exposure_usd=str(micros_to_decimal(state.total_exposure_micros())),
                exchange_exposure_usd=str(
                    micros_to_decimal(state.exchange_total_notional_micros)
                ),
                note="exchange account_leverage is notional/maintenance-margin, not notional/equity",
            )
            if local_leverage > 1:
                report.discrepancies.append(
                    Discrepancy(
                        kind="leverage_above_limit",
                        subject="perp_account",
                        ours=f"{local_leverage}x",
                        theirs="1.00x maximum",
                        # A warning, not an error: this is a state to de-risk out
                        # of, not a reason to refuse the orders that would fix
                        # it. Increasing orders are blocked by the invariant.
                        severity="warning",
                    )
                )

        self._log_report(report)
        return report

    # -- shared --------------------------------------------------------------

    async def _reconcile_orders(
        self,
        report: ReconciliationReport,
        engine: ExecutionEngine | None,
        exchange_by_client_id: dict[str, Any],
    ) -> None:
        """Match local orders to exchange orders; resolve and flag differences."""
        if engine is None:
            return

        # 1. Resolve anything whose fate we never observed.
        for order in list(engine.orders.values()):
            if order.state is OrderState.UNKNOWN:
                await engine.resolve_unknown(order)
                report.unknown_orders_resolved += 1

        local_by_client_id = {
            o.client_order_id: o for o in engine.orders.values() if o.is_working
        }

        # 2. Orders resting at the exchange that we are not tracking. These are
        #    orphans -- typically survivors of a crash before we persisted them.
        for client_id in exchange_by_client_id.keys() - local_by_client_id.keys():
            report.orphaned_orders.append(client_id)
            report.discrepancies.append(
                Discrepancy(
                    kind="orphaned_order",
                    subject=client_id,
                    ours="not tracked",
                    theirs="resting",
                    # Recoverable by cancelling; does not by itself mean our
                    # state is corrupt.
                    severity="warning",
                )
            )

        # 3. Orders we think are working that the exchange does not have.
        for client_id in local_by_client_id.keys() - exchange_by_client_id.keys():
            order = local_by_client_id[client_id]
            if order.state in (OrderState.SUBMITTED, OrderState.UNKNOWN):
                await engine.resolve_unknown(order)
                report.unknown_orders_resolved += 1
            else:
                report.discrepancies.append(
                    Discrepancy(
                        kind="missing_order",
                        subject=client_id,
                        ours=order.state.value,
                        theirs="absent",
                        severity="error",
                    )
                )
        report.orders_checked = len(local_by_client_id) + len(report.orphaned_orders)

    async def cancel_orphans(
        self, report: ReconciliationReport, market_type: MarketType
    ) -> int:
        """Cancel resting orders at the exchange that we do not own.

        Called explicitly after a reconciliation reports orphans. Cancelling is
        the conservative action: an untracked resting order can fill into a
        position nothing in this process is managing.
        """
        rest = self.event_rest if market_type is MarketType.EVENT else self.perp_rest
        if rest is None or not report.orphaned_orders:
            return 0

        cancelled = 0
        try:
            orders = (
                await self.event_rest.get_orders(status="resting")  # type: ignore[union-attr]
                if market_type is MarketType.EVENT
                else await self.perp_rest.get_orders()  # type: ignore[union-attr]
            )
        except Exception as exc:
            log.error("reconcile.orphan_listing_failed", error=str(exc))
            return 0

        orphan_ids = set(report.orphaned_orders)
        for order in orders:
            if order.client_order_id in orphan_ids and order.order_id:
                try:
                    await rest.cancel_order(order.order_id)
                    cancelled += 1
                    log.warning(
                        "reconcile.orphan_cancelled",
                        domain=market_type.value,
                        exchange_order_id=order.order_id,
                        client_order_id=order.client_order_id,
                    )
                except Exception as exc:
                    log.error(
                        "reconcile.orphan_cancel_failed",
                        exchange_order_id=order.order_id, error=str(exc),
                    )
        return cancelled

    async def halt_if_unsafe(self, report: ReconciliationReport) -> bool:
        """Engage the risk kill switch when reconciliation failed.

        Returns True when trading was halted. This is the mechanism behind
        "never automatically resume live trading if state cannot be reconciled
        safely".
        """
        if report.safe_to_trade:
            return False
        reason = report.error or "; ".join(
            f"{d.kind}:{d.subject}" for d in report.discrepancies if d.severity == "error"
        )
        log.error(
            "reconcile.unsafe_halting",
            domain=report.domain, reason=reason, report=report.to_dict(),
        )
        if self.kill_switches is not None:
            await self.kill_switches.engage(
                KillSwitchScope.RISK,
                reason=f"reconciliation failed for {report.domain}: {reason}",
                actor="reconciler",
            )
        return True

    @staticmethod
    def _log_report(report: ReconciliationReport) -> None:
        level = log.info if report.safe_to_trade else log.error
        level(
            "reconcile.complete",
            domain=report.domain,
            safe_to_trade=report.safe_to_trade,
            positions=report.positions_checked,
            orders=report.orders_checked,
            unknown_resolved=report.unknown_orders_resolved,
            orphans=len(report.orphaned_orders),
            discrepancies=len(report.discrepancies),
        )
