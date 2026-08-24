"""Capital allocation.

Strategies compete for a fixed, small pool of money. This module decides how much
each one gets. It does not enforce safety -- every intent it scales still goes
through a risk engine that may cut it further or refuse it. Allocation is an
optimisation problem and risk is a constraint problem, and merging them produces
a system where making the optimiser more aggressive quietly makes the constraints
weaker.

Not equal weight
----------------
Equal weighting treats a strategy with twelve trades exactly like one with four
hundred, which is the single most common way a research pipeline converts noise
into position size. Allocation here is proportional to a **shrunk, correlation-
penalised edge estimate**, in the spirit of fractional Kelly:

1. Start from realised edge per trade (expectancy relative to risk), measured
   **out-of-sample or later only**. A backtest buys nothing.
2. **Shrink toward zero by sample size.** A Sharpe of 3 over twelve trades is
   mostly luck; the shrinkage factor ``n / (n + k)`` reflects that directly, and
   with ``k`` at the configured minimum-trades threshold a strategy reaches half
   its raw score exactly when it reaches that threshold.
3. **Penalise correlation.** Capital in a strategy that moves with the rest of
   the book is worth less at the margin, so the score is multiplied by
   ``(1 - correlation)``.
4. **Apply a Kelly fraction well below one.** Full Kelly is the growth-optimal
   bet only if the edge estimate is exactly right, and it never is.
5. **Stage multiplier.** A strategy on probation gets a configured fraction of
   what it would otherwise receive.
6. **Suppress degraded strategies entirely.**

The small-account problem
-------------------------
With a $200 account and five eligible strategies, "diversify" is the wrong
answer: a fifth of $200 is $40, and after the minimum order size and the round
trip fee there is no trade left worth taking. Below a configured floor the
allocator therefore **concentrates** -- it funds strategies in score order until
the money runs out, rather than spreading it into slices too small to deploy.
This is stated loudly because it is the opposite of what a large-account
allocator should do, and the switch between the two behaviours is a threshold
that must be visible rather than emergent.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from kat.common.enums import MarketType
from kat.common.fixedpoint import micros_to_decimal
from kat.observability.logging import get_logger
from kat.portfolio.types import AllocationDecision, StrategyAllocation, StrategyScore

log = get_logger(__name__)

__all__ = ["AllocationPolicy", "CapitalAllocator"]

#: Below this much deployable capital the allocator concentrates rather than
#: divides. $50 is roughly the point at which a 10% position is $5, which is
#: still several contracts on a mid-priced market after fees.
DEFAULT_CONCENTRATION_FLOOR_MICROS = 50_000_000

#: Smallest allocation worth making. An allocation below this cannot fund a
#: single sensible order once the round trip fee is paid, so granting it would
#: just immobilise the money.
DEFAULT_MIN_ALLOCATION_MICROS = 10_000_000  # $10


class AllocationPolicy:
    """Tunable knobs for allocation. None of these affect a risk limit."""

    def __init__(
        self,
        *,
        kelly_fraction: Decimal = Decimal("0.25"),
        shrinkage_trades: int = 100,
        probation_fraction: Decimal = Decimal("0.10"),
        max_single_strategy_pct: Decimal = Decimal("0.50"),
        reserve_pct: Decimal = Decimal("0.10"),
        concentration_floor_micros: int = DEFAULT_CONCENTRATION_FLOOR_MICROS,
        min_allocation_micros: int = DEFAULT_MIN_ALLOCATION_MICROS,
    ) -> None:
        #: Fraction of the Kelly-optimal bet actually taken. Well below one,
        #: because full Kelly is optimal only for a perfectly known edge.
        self.kelly_fraction = kelly_fraction
        #: ``k`` in the ``n / (n + k)`` shrinkage. A strategy reaches half its
        #: raw score at exactly this many trades.
        self.shrinkage_trades = max(1, shrinkage_trades)
        self.probation_fraction = probation_fraction
        #: No strategy may hold more than this share, however good it looks.
        #: Concentration risk is not diversifiable by conviction.
        self.max_single_strategy_pct = max_single_strategy_pct
        #: Held back so a new opportunity does not require liquidating something.
        self.reserve_pct = reserve_pct
        self.concentration_floor_micros = concentration_floor_micros
        self.min_allocation_micros = min_allocation_micros


class CapitalAllocator:
    """Allocates capital within one domain.

    Event contracts and perpetuals are allocated **separately**, from separate
    pools, because they are independent domains with independent capital. There
    is no cross-domain optimisation and no shared budget.
    """

    def __init__(self, policy: AllocationPolicy | None = None) -> None:
        self.policy = policy or AllocationPolicy()

    def allocate(
        self,
        *,
        market_type: MarketType,
        total_capital_micros: int,
        scores: list[StrategyScore],
    ) -> AllocationDecision:
        """Divide ``total_capital_micros`` across ``scores``."""
        policy = self.policy
        deployable = int(Decimal(total_capital_micros) * (Decimal(1) - policy.reserve_pct))
        decision = AllocationDecision(
            market_type=market_type,
            total_capital_micros=total_capital_micros,
            deployable_micros=max(0, deployable),
        )

        if deployable <= 0:
            decision.reason = "no deployable capital"
            decision.unallocated_micros = max(0, total_capital_micros)
            return decision

        eligible = [s for s in scores if s.market_type is market_type]
        weighted = [(s, self._score(s)) for s in eligible]
        positive = [(s, w) for s, w in weighted if w > 0]

        for strategy_score, weight in weighted:
            if weight <= 0:
                decision.allocations.append(
                    StrategyAllocation(
                        strategy_id=strategy_score.strategy_id,
                        market_type=market_type,
                        allocated_micros=0,
                        weight=Decimal(0),
                        reason=self._zero_reason(strategy_score),
                        score=strategy_score,
                    )
                )

        if not positive:
            decision.unallocated_micros = deployable
            decision.reason = "no strategy has a positive risk-adjusted score"
            log.info(
                "allocation.none_eligible",
                domain=market_type.value, candidates=len(eligible),
            )
            return decision

        positive.sort(key=lambda pair: pair[1], reverse=True)

        if deployable < policy.concentration_floor_micros:
            allocations = self._concentrate(positive, deployable, market_type)
            decision.reason = (
                f"deployable capital {micros_to_decimal(deployable)} is below the "
                f"concentration floor; funding strategies in score order rather "
                f"than dividing into slices too small to trade"
            )
        else:
            allocations = self._proportional(positive, deployable, market_type)
            decision.reason = "proportional to shrunk, correlation-penalised edge"

        decision.allocations.extend(allocations)
        granted = sum(a.allocated_micros for a in allocations)
        decision.unallocated_micros = max(0, total_capital_micros - granted)

        log.info(
            "allocation.complete",
            domain=market_type.value,
            deployable_usd=str(micros_to_decimal(deployable)),
            granted_usd=str(micros_to_decimal(granted)),
            funded=sum(1 for a in allocations if a.allocated_micros > 0),
            candidates=len(eligible),
            mode="concentrated" if deployable < policy.concentration_floor_micros else "proportional",
        )
        return decision

    # -- scoring -------------------------------------------------------------

    def _score(self, score: StrategyScore) -> Decimal:
        """Risk-adjusted, shrunk, correlation-penalised weight for one strategy.

        Returns zero -- meaning "no capital" -- rather than a small number
        whenever the strategy is degraded, has no realised edge, or has too
        little evidence to distinguish its edge from noise.
        """
        policy = self.policy

        if score.is_degraded:
            return Decimal(0)
        if score.trade_count <= 0 or score.expectancy_micros <= 0:
            return Decimal(0)
        # A strategy whose out-of-sample Sharpe is not positive has not shown an
        # edge, whatever its expectancy says about a handful of large wins.
        if score.sharpe <= 0:
            return Decimal(0)

        raw = Decimal(str(score.sharpe))

        # Shrink toward zero by sample size: n / (n + k).
        n = Decimal(score.trade_count)
        k = Decimal(policy.shrinkage_trades)
        shrunk = raw * (n / (n + k))

        # Correlated capital is worth less at the margin.
        correlation = Decimal(str(max(0.0, min(1.0, score.correlation))))
        penalised = shrunk * (Decimal(1) - correlation)

        # Confidence, where the scorer supplied one, scales it further.
        if score.confidence > 0:
            penalised *= Decimal(str(min(1.0, score.confidence)))

        # Fractional Kelly.
        weighted = penalised * policy.kelly_fraction

        # Probation trades real money at a deliberately small size.
        if score.stage == "probation":
            weighted *= policy.probation_fraction

        return max(Decimal(0), weighted)

    @staticmethod
    def _zero_reason(score: StrategyScore) -> str:
        if score.is_degraded:
            return f"degraded: {', '.join(score.degradation_flags)}"
        if score.trade_count <= 0:
            return "no realised trades out of sample"
        if score.expectancy_micros <= 0:
            return "expectancy is not positive"
        if score.sharpe <= 0:
            return "out-of-sample Sharpe is not positive"
        return "score is zero"

    # -- distribution --------------------------------------------------------

    def _proportional(
        self,
        ranked: list[tuple[StrategyScore, Decimal]],
        deployable: int,
        market_type: MarketType,
    ) -> list[StrategyAllocation]:
        policy = self.policy
        total_weight = sum(w for _, w in ranked)
        cap = int(Decimal(deployable) * policy.max_single_strategy_pct)

        allocations: list[StrategyAllocation] = []
        remaining = deployable
        for score, weight in ranked:
            share = Decimal(weight) / total_weight
            amount = min(int(Decimal(deployable) * share), cap, remaining)
            if amount < policy.min_allocation_micros:
                # Too small to deploy; leave it unallocated rather than
                # immobilising money in a position that cannot be traded.
                allocations.append(
                    StrategyAllocation(
                        strategy_id=score.strategy_id,
                        market_type=market_type,
                        allocated_micros=0,
                        weight=share,
                        reason=(
                            f"share {micros_to_decimal(amount)} is below the minimum "
                            f"tradeable allocation"
                        ),
                        score=score,
                    )
                )
                continue
            remaining -= amount
            allocations.append(
                StrategyAllocation(
                    strategy_id=score.strategy_id,
                    market_type=market_type,
                    allocated_micros=amount,
                    weight=share,
                    reason="proportional share of shrunk risk-adjusted score",
                    score=score,
                )
            )
        return allocations

    def _concentrate(
        self,
        ranked: list[tuple[StrategyScore, Decimal]],
        deployable: int,
        market_type: MarketType,
    ) -> list[StrategyAllocation]:
        """Fund strategies in score order until the money runs out.

        The small-account behaviour. Dividing $200 five ways produces five
        allocations that cannot pay their own round trip; funding the best one
        or two properly produces trades that can actually make money.
        """
        policy = self.policy
        allocations: list[StrategyAllocation] = []
        remaining = deployable
        cap = int(Decimal(deployable) * policy.max_single_strategy_pct)
        # With very little capital the cap would prevent funding anything at all,
        # so a single strategy may take the whole pool when it is the only one
        # that can be funded meaningfully.
        if cap < policy.min_allocation_micros:
            cap = deployable

        for score, _weight in ranked:
            if remaining < policy.min_allocation_micros:
                allocations.append(
                    StrategyAllocation(
                        strategy_id=score.strategy_id,
                        market_type=market_type,
                        allocated_micros=0,
                        weight=Decimal(0),
                        reason="capital exhausted by higher-scoring strategies",
                        score=score,
                    )
                )
                continue
            amount = min(cap, remaining)
            remaining -= amount
            allocations.append(
                StrategyAllocation(
                    strategy_id=score.strategy_id,
                    market_type=market_type,
                    allocated_micros=amount,
                    weight=Decimal(1) if deployable == 0 else Decimal(amount) / Decimal(deployable),
                    reason="concentrated allocation on a small account",
                    score=score,
                )
            )
        return allocations

    # -- application ---------------------------------------------------------

    @staticmethod
    def apply(decision: AllocationDecision, strategies: dict[str, Any]) -> int:
        """Push allocations onto the live strategy objects.

        Returns how many were updated. A strategy that has no allocation is set
        to zero explicitly rather than left holding a stale figure from the
        previous cycle.
        """
        updated = 0
        for allocation in decision.allocations:
            strategy = strategies.get(allocation.strategy_id)
            if strategy is None:
                continue
            strategy.allocated_capital_micros = allocation.allocated_micros
            updated += 1
        return updated
