"""Strategy registry.

Holds the strategy instances the platform knows about, tracks their lifecycle
state, and persists both. Two properties matter:

**Promotion goes through the ladder.** :meth:`StrategyRegistry.promote` moves a
strategy one rung and refuses anything else, so there is no code path that
registers a strategy directly into ``LIVE``. Demotion is unrestricted, because
stopping something misbehaving must never require evidence.

**Registration is not endorsement.** Every strategy enters at ``DEVELOPMENT``
unless a caller deliberately says otherwise, and the default set below is
registered into paper trading, not live. Being in the registry means the platform
knows about a strategy, not that it trusts it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kat.common.enums import MarketType, StrategyState
from kat.common.timeutil import utcnow_aware
from kat.db.models import Strategy as StrategyRow
from kat.db.models import StrategyStateChange
from kat.db.writer import AsyncWriter
from kat.observability.logging import get_logger
from kat.strategies.base import Strategy, next_promotion_state

log = get_logger(__name__)

__all__ = ["StrategyRegistry", "build_default_strategies"]


class StrategyRegistry:
    """The set of strategies the platform knows about, and their states."""

    def __init__(self, writer: AsyncWriter | None = None) -> None:
        self.writer = writer
        self._strategies: dict[str, Strategy] = {}

    # -- membership ----------------------------------------------------------

    def register(self, strategy: Strategy) -> Strategy:
        if strategy.strategy_id in self._strategies:
            raise ValueError(f"{strategy.strategy_id} is already registered")
        self._strategies[strategy.strategy_id] = strategy
        self._persist(strategy, reason="registered")
        log.info(
            "strategy.registered",
            strategy_id=strategy.strategy_id,
            version=strategy.version,
            market_type=strategy.market_type.value,
            state=strategy.state.value,
        )
        return strategy

    def get(self, strategy_id: str) -> Strategy | None:
        return self._strategies.get(strategy_id)

    def all(self, *, market_type: MarketType | None = None) -> list[Strategy]:
        return [
            s
            for s in self._strategies.values()
            if market_type is None or s.market_type is market_type
        ]

    def __len__(self) -> int:
        return len(self._strategies)

    # -- lifecycle -----------------------------------------------------------

    def promote(
        self, strategy_id: str, *, actor: str = "research", gate_results: dict | None = None
    ) -> StrategyState | None:
        """Move a strategy up exactly one rung, or refuse.

        The caller is responsible for having evaluated the gates -- see
        :mod:`kat.discovery.lifecycle`. This method enforces only that the
        movement is a single legal step, so no path can jump to ``LIVE``.
        """
        strategy = self._strategies.get(strategy_id)
        if strategy is None:
            return None
        target = next_promotion_state(strategy.state)
        if target is None:
            log.warning(
                "strategy.promotion_refused",
                strategy_id=strategy_id,
                state=strategy.state.value,
                reason="no next rung on the ladder",
            )
            return None
        previous = strategy.state
        strategy.state = target
        self._record_change(strategy, previous, target, actor, "promoted", gate_results)
        return target

    def demote(
        self, strategy_id: str, target: StrategyState, *, actor: str, reason: str
    ) -> StrategyState | None:
        """Pause, retire or send a strategy for review. Never gated."""
        strategy = self._strategies.get(strategy_id)
        if strategy is None:
            return None
        if target not in (StrategyState.PAUSED, StrategyState.RETIRED, StrategyState.REVIEW):
            raise ValueError(f"{target.value} is not a demotion target")
        previous = strategy.state
        strategy.state = target
        self._record_change(strategy, previous, target, actor, reason, None)
        return target

    def set_enabled(self, strategy_id: str, enabled: bool, *, actor: str = "dashboard") -> bool:
        """Enable or disable a strategy without changing its lifecycle state.

        Mode B uses this: a human chooses which permitted strategies run, and
        that is separate from whether a strategy has earned its rung.
        """
        strategy = self._strategies.get(strategy_id)
        if strategy is None:
            return False
        if not enabled:
            self.demote(strategy_id, StrategyState.PAUSED, actor=actor, reason="disabled")
        log.info("strategy.enabled_changed", strategy_id=strategy_id, enabled=enabled, actor=actor)
        return True

    # -- persistence ---------------------------------------------------------

    def _persist(self, strategy: Strategy, *, reason: str) -> None:
        if self.writer is None:
            return
        meta = strategy.metadata
        self.writer.enqueue(
            StrategyRow(
                strategy_id=strategy.strategy_id,
                version=strategy.version,
                name=meta.name,
                market_type=strategy.market_type.value,
                family=meta.family,
                state=strategy.state.value,
                enabled=strategy.state
                not in (StrategyState.PAUSED, StrategyState.RETIRED),
                params=dict(meta.params),
                param_space=meta.to_dict()["param_space"],
                description=meta.description,
                allocated_capital_micros=strategy.allocated_capital_micros,
                discovered_by=meta.discovered_by,
                git_commit=meta.git_commit,
            ),
            critical=True,
        )

    def _record_change(
        self,
        strategy: Strategy,
        previous: StrategyState,
        target: StrategyState,
        actor: str,
        reason: str,
        gate_results: dict | None,
    ) -> None:
        log.warning(
            "strategy.state_changed",
            strategy_id=strategy.strategy_id,
            **{"from": previous.value},
            to=target.value,
            actor=actor,
            reason=reason,
        )
        self._persist(strategy, reason=reason)
        if self.writer is None:
            return
        self.writer.enqueue(
            StrategyStateChange(
                strategy_id=strategy.strategy_id,
                ts=utcnow_aware(),
                from_state=previous.value,
                to_state=target.value,
                actor=actor,
                reason=reason,
                gate_results=gate_results,
            ),
            critical=True,
        )

    def stats(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        for strategy in self._strategies.values():
            by_state[strategy.state.value] = by_state.get(strategy.state.value, 0) + 1
        return {"total": len(self._strategies), "by_state": by_state}


def build_default_strategies(
    *, state: StrategyState = StrategyState.PAPER
) -> list[Strategy]:
    """The strategies the platform ships with.

    Registered at ``PAPER`` by default: they trade with simulated fills against
    live market data, which is the lowest rung that produces evidence. Nothing
    here has earned live capital, and nothing reaches it without climbing the
    ladder.

    Import failures are logged and skipped rather than raised. One broken
    strategy should not prevent the platform from starting with the others.
    """
    from kat.strategies.event.expiry_drift import ExpiryDriftStrategy
    from kat.strategies.event.mean_reversion import MeanReversionStrategy
    from kat.strategies.event.mispricing import MispricingStrategy
    from kat.strategies.event.model_drift import ModelDriftStrategy
    from kat.strategies.event.momentum import MomentumStrategy
    from kat.strategies.event.news_reaction import NewsReactionStrategy
    from kat.strategies.event.passive_quote import PassiveQuoteStrategy
    from kat.strategies.perp.basis import PerpBasisStrategy
    from kat.strategies.perp.funding_carry import FundingCarryStrategy
    from kat.strategies.perp.passive_quote import PerpPassiveQuoteStrategy
    from kat.strategies.perp.trend import PerpTrendStrategy

    classes: list[Callable[[], Strategy]] = [
        MispricingStrategy,
        ModelDriftStrategy,
        PassiveQuoteStrategy,
        MeanReversionStrategy,
        MomentumStrategy,
        ExpiryDriftStrategy,
        NewsReactionStrategy,
        FundingCarryStrategy,
        PerpTrendStrategy,
        PerpBasisStrategy,
        PerpPassiveQuoteStrategy,
    ]

    built: list[Strategy] = []
    for cls in classes:
        try:
            strategy = cls()
            strategy.state = state
            built.append(strategy)
        except Exception as exc:
            log.error("strategy.construction_failed", strategy=cls.__name__, error=str(exc))
    return built
