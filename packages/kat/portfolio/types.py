"""Portfolio-layer contracts.

The portfolio sits between strategies and risk. Its job is *allocation*: given
several strategies competing for the same limited capital, decide how much each
one gets, and scale their intents accordingly.

It does not enforce safety. Every intent it emits still goes through a risk
engine, which may resize it further or refuse it outright. The separation is
deliberate -- allocation is an optimisation problem and risk is a constraint
problem, and merging them produces a system where making the optimiser more
aggressive quietly makes the constraints weaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from kat.common.enums import MarketType
from kat.common.fixedpoint import micros_to_decimal

__all__ = ["AllocationDecision", "StrategyAllocation", "StrategyScore"]


@dataclass(slots=True)
class StrategyScore:
    """The evidence behind one strategy's claim on capital.

    Every field is drawn from *out-of-sample or later* performance. Training
    performance is deliberately not an input: a strategy cannot buy capital with
    a good backtest.
    """

    strategy_id: str
    market_type: MarketType
    # Realised risk-adjusted return, out-of-sample or live.
    sharpe: float = 0.0
    expectancy_micros: int = 0
    win_rate: float = 0.0
    trade_count: int = 0
    max_drawdown_pct: float = 0.0
    # How much we believe the above, given sample size. Drives shrinkage.
    confidence: float = 0.0
    # Correlation with the rest of the portfolio, in [0, 1]. High correlation
    # reduces the marginal value of more capital in this strategy.
    correlation: float = 0.0
    # Recent degradation signals; any of these suppress allocation.
    degradation_flags: tuple[str, ...] = ()
    stage: str = "paper"

    @property
    def is_degraded(self) -> bool:
        return bool(self.degradation_flags)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "market_type": self.market_type.value,
            "sharpe": round(self.sharpe, 4),
            "expectancy_usd": str(micros_to_decimal(self.expectancy_micros)),
            "win_rate": round(self.win_rate, 4),
            "trade_count": self.trade_count,
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "confidence": round(self.confidence, 4),
            "correlation": round(self.correlation, 4),
            "degradation_flags": list(self.degradation_flags),
            "stage": self.stage,
        }


@dataclass(slots=True)
class StrategyAllocation:
    """Capital granted to one strategy, and why."""

    strategy_id: str
    market_type: MarketType
    allocated_micros: int
    weight: Decimal
    reason: str
    score: StrategyScore | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "market_type": self.market_type.value,
            "allocated_usd": str(micros_to_decimal(self.allocated_micros)),
            "weight": f"{self.weight:.6f}",
            "reason": self.reason,
            "score": self.score.to_dict() if self.score else None,
        }


@dataclass(slots=True)
class AllocationDecision:
    """The outcome of one allocation cycle for one domain."""

    market_type: MarketType
    total_capital_micros: int
    deployable_micros: int
    allocations: list[StrategyAllocation] = field(default_factory=list)
    unallocated_micros: int = 0
    reason: str = ""

    def for_strategy(self, strategy_id: str) -> StrategyAllocation | None:
        return next((a for a in self.allocations if a.strategy_id == strategy_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_type": self.market_type.value,
            "total_capital_usd": str(micros_to_decimal(self.total_capital_micros)),
            "deployable_usd": str(micros_to_decimal(self.deployable_micros)),
            "unallocated_usd": str(micros_to_decimal(self.unallocated_micros)),
            "reason": self.reason,
            "allocations": [a.to_dict() for a in self.allocations],
        }
