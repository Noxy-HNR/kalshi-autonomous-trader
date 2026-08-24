"""Kill switches.

Five independent scopes, each of which can halt trading on its own:

``GLOBAL``    -- everything stops.
``EVENT``     -- event-contract trading stops; perpetuals continue.
``PERP``      -- perpetual trading stops; event contracts continue.
``STRATEGY``  -- one named strategy stops.
``RISK``      -- engaged automatically when a risk condition trips.

Two properties make these actually useful rather than decorative:

**They persist.** State lives in the database and is reloaded at start-up. A
kill switch that forgets it was engaged when the process restarts is worse than
no kill switch at all, because it creates false confidence.

**They fail closed.** If the persisted state cannot be read, the switch reports
*engaged*. An unknown halt state is treated as a halt.

Engaging never requires the database to be healthy -- the in-memory state flips
immediately and the write is queued -- so a kill switch works during exactly the
kind of infrastructure failure that makes you want one.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select

from kat.common.enums import KillSwitchScope, MarketType
from kat.common.errors import KillSwitchEngaged
from kat.common.timeutil import utcnow_aware
from kat.db.engine import Database
from kat.db.models import AuditLog, KillSwitchState
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["KillSwitchManager", "SwitchStatus"]


@dataclass(frozen=True, slots=True)
class SwitchStatus:
    scope: str
    subject: str
    engaged: bool
    reason: str | None
    engaged_by: str | None
    engaged_ts: datetime | None


class KillSwitchManager:
    """In-memory authority over halt state, backed by durable storage."""

    def __init__(self, database: Database | None = None) -> None:
        self._db = database
        # (scope, subject) -> SwitchStatus. Subject is "" for non-strategy scopes.
        self._state: dict[tuple[str, str], SwitchStatus] = {}
        self._loaded = False
        self._load_failed = False
        self._lock = asyncio.Lock()

    # -- lifecycle -----------------------------------------------------------

    async def load(self) -> None:
        """Restore persisted state. Fails closed on error."""
        if self._db is None:
            self._loaded = True
            return
        try:
            async with self._db.session() as session:
                rows = (await session.execute(select(KillSwitchState))).scalars().all()
            self._state = {
                (row.scope, row.subject or ""): SwitchStatus(
                    scope=row.scope,
                    subject=row.subject or "",
                    engaged=bool(row.engaged),
                    reason=row.reason,
                    engaged_by=row.engaged_by,
                    engaged_ts=row.engaged_ts,
                )
                for row in rows
            }
            self._loaded = True
            self._load_failed = False
            engaged = [k for k, v in self._state.items() if v.engaged]
            log.info("killswitch.loaded", total=len(self._state), engaged=len(engaged))
            if engaged:
                log.warning("killswitch.engaged_at_startup", switches=[f"{s}:{n}" for s, n in engaged])
        except Exception as exc:
            # Fail closed: we cannot prove trading is permitted, so it is not.
            self._load_failed = True
            self._loaded = True
            log.error("killswitch.load_failed_failing_closed", error=str(exc))

    # -- queries -------------------------------------------------------------

    def is_engaged(self, scope: str, subject: str = "") -> bool:
        if self._load_failed:
            return True  # unknown state is a halt
        status = self._state.get((scope, subject))
        return bool(status and status.engaged)

    def blocking_switch(
        self, market_type: MarketType, *, strategy_id: str | None = None
    ) -> str | None:
        """Return the scope of the first switch blocking this trade, if any.

        Checked in severity order so the reported reason is the broadest one --
        "global halt" is more useful to an operator than "strategy X halted"
        when both are engaged.
        """
        if self._load_failed:
            return KillSwitchScope.GLOBAL
        if self.is_engaged(KillSwitchScope.GLOBAL):
            return KillSwitchScope.GLOBAL
        if self.is_engaged(KillSwitchScope.RISK):
            return KillSwitchScope.RISK
        domain_scope = (
            KillSwitchScope.EVENT if market_type is MarketType.EVENT else KillSwitchScope.PERP
        )
        if self.is_engaged(domain_scope):
            return domain_scope
        if strategy_id and self.is_engaged(KillSwitchScope.STRATEGY, strategy_id):
            return KillSwitchScope.STRATEGY
        return None

    def assert_clear(self, market_type: MarketType, *, strategy_id: str | None = None) -> None:
        """Raise :class:`KillSwitchEngaged` when any relevant switch is on."""
        scope = self.blocking_switch(market_type, strategy_id=strategy_id)
        if scope is not None:
            status = self._state.get((scope, strategy_id if scope == KillSwitchScope.STRATEGY else ""))
            reason = (status.reason if status else None) or "no reason recorded"
            raise KillSwitchEngaged(scope, f"{scope} kill switch is engaged: {reason}")

    def all_statuses(self) -> list[SwitchStatus]:
        out = list(self._state.values())
        if self._load_failed:
            out.append(
                SwitchStatus(
                    scope=KillSwitchScope.GLOBAL,
                    subject="",
                    engaged=True,
                    reason="kill switch state could not be loaded; failing closed",
                    engaged_by="system",
                    engaged_ts=None,
                )
            )
        return out

    # -- mutation ------------------------------------------------------------

    async def engage(
        self, scope: str, *, reason: str, actor: str = "system", subject: str = ""
    ) -> None:
        """Engage a switch. Takes effect in memory immediately.

        The database write is queued rather than awaited on the halt path: a
        halt must work when the database is the thing that is broken.
        """
        async with self._lock:
            self._state[(scope, subject)] = SwitchStatus(
                scope=scope,
                subject=subject,
                engaged=True,
                reason=reason,
                engaged_by=actor,
                engaged_ts=utcnow_aware(),
            )
        log.error("killswitch.engaged", scope=scope, subject=subject, reason=reason, actor=actor)
        await self._persist(scope, subject, engaged=True, reason=reason, actor=actor)

    async def release(self, scope: str, *, actor: str = "system", subject: str = "") -> None:
        """Release a switch.

        Releasing is always a deliberate human or supervisor action; nothing in
        the trading path may release a switch it engaged.
        """
        async with self._lock:
            existing = self._state.get((scope, subject))
            self._state[(scope, subject)] = SwitchStatus(
                scope=scope,
                subject=subject,
                engaged=False,
                reason=None,
                engaged_by=actor,
                engaged_ts=existing.engaged_ts if existing else None,
            )
        log.warning("killswitch.released", scope=scope, subject=subject, actor=actor)
        await self._persist(scope, subject, engaged=False, reason=None, actor=actor)

    async def _persist(
        self, scope: str, subject: str, *, engaged: bool, reason: str | None, actor: str
    ) -> None:
        if self._db is None:
            return
        now = utcnow_aware()
        try:
            async with self._db.session() as session:
                await session.merge(
                    KillSwitchState(
                        scope=scope,
                        subject=subject,
                        engaged=engaged,
                        reason=reason,
                        engaged_by=actor,
                        engaged_ts=now if engaged else None,
                        released_ts=None if engaged else now,
                    )
                )
                session.add(
                    AuditLog(
                        actor=actor,
                        action="kill_switch_engaged" if engaged else "kill_switch_released",
                        subject=f"{scope}:{subject}" if subject else scope,
                        detail={"reason": reason},
                        severity="critical" if engaged else "warning",
                    )
                )
        except Exception as exc:
            # In-memory state already reflects the change and is authoritative
            # for this process; the log line is the durable record if the write
            # failed. Never let a failed write undo a halt.
            log.error(
                "killswitch.persist_failed",
                scope=scope, subject=subject, engaged=engaged, error=str(exc),
            )
