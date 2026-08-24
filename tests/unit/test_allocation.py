"""Capital allocation.

The property under test throughout: **evidence buys capital, and a good-looking
number without evidence does not.** A twelve-trade Sharpe of 3.0 must not
outrank a four-hundred-trade Sharpe of 1.2, because the first is mostly luck and
the second is mostly signal.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from kat.common.enums import MarketType
from kat.portfolio.allocator import AllocationPolicy, CapitalAllocator
from kat.portfolio.types import StrategyScore


def score(
    strategy_id: str,
    *,
    sharpe: float = 1.0,
    trades: int = 300,
    expectancy: int = 50_000,
    correlation: float = 0.0,
    confidence: float = 0.8,
    stage: str = "live",
    flags: tuple[str, ...] = (),
    market_type: MarketType = MarketType.EVENT,
) -> StrategyScore:
    return StrategyScore(
        strategy_id=strategy_id,
        market_type=market_type,
        sharpe=sharpe,
        expectancy_micros=expectancy,
        trade_count=trades,
        correlation=correlation,
        confidence=confidence,
        stage=stage,
        degradation_flags=flags,
    )


@pytest.fixture
def allocator() -> CapitalAllocator:
    return CapitalAllocator()


def granted(decision) -> dict[str, int]:
    return {a.strategy_id: a.allocated_micros for a in decision.allocations}


class TestNeverOverAllocates:
    @pytest.mark.parametrize("capital", [0, 1_000_000, 200_000_000, 5_000_000_000])
    def test_total_never_exceeds_deployable(self, allocator: CapitalAllocator, capital: int):
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=capital,
            scores=[score(f"s{i}", sharpe=1.0 + i * 0.3) for i in range(6)],
        )
        total = sum(a.allocated_micros for a in decision.allocations)
        assert total <= decision.deployable_micros
        assert total <= capital

    def test_a_reserve_is_always_held_back(self, allocator: CapitalAllocator):
        """So a new opportunity does not require liquidating something first."""
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=1_000_000_000,
            scores=[score("only")],
        )
        assert decision.deployable_micros < 1_000_000_000

    def test_zero_capital_allocates_nothing(self, allocator: CapitalAllocator):
        decision = allocator.allocate(
            market_type=MarketType.EVENT, total_capital_micros=0, scores=[score("s")]
        )
        assert sum(a.allocated_micros for a in decision.allocations) == 0
        assert "no deployable capital" in decision.reason


class TestEvidenceBuysCapital:
    def test_small_sample_is_shrunk_below_a_proven_strategy(self, allocator: CapitalAllocator):
        """A Sharpe of 3.0 over 12 trades must lose to 1.2 over 400."""
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=5_000_000_000,
            scores=[
                score("proven", sharpe=1.2, trades=400),
                score("lucky", sharpe=3.0, trades=12),
            ],
        )
        amounts = granted(decision)
        assert amounts["lucky"] < amounts["proven"]

    def test_shrinkage_reaches_half_at_the_configured_threshold(self):
        """n / (n + k) is 0.5 exactly when n == k, which makes the knob legible."""
        allocator = CapitalAllocator(AllocationPolicy(shrinkage_trades=100))
        at_threshold = allocator._score(score("s", sharpe=2.0, trades=100))
        unshrunk = allocator._score(score("s", sharpe=2.0, trades=10_000_000))
        assert at_threshold == pytest.approx(unshrunk / 2, rel=Decimal("0.01"))

    def test_correlated_capital_is_penalised(self, allocator: CapitalAllocator):
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=5_000_000_000,
            scores=[
                score("independent", sharpe=1.2, trades=400, correlation=0.0),
                score("correlated", sharpe=1.2, trades=400, correlation=0.9),
            ],
        )
        amounts = granted(decision)
        assert amounts["correlated"] < amounts["independent"]

    def test_probation_gets_a_fraction(self, allocator: CapitalAllocator):
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=5_000_000_000,
            scores=[
                score("live_one", sharpe=1.5, trades=300, stage="live"),
                score("probation_one", sharpe=1.5, trades=300, stage="probation"),
            ],
        )
        amounts = granted(decision)
        assert amounts["probation_one"] < amounts["live_one"]

    def test_no_strategy_may_take_the_whole_book(self, allocator: CapitalAllocator):
        """Conviction does not make concentration risk go away."""
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=5_000_000_000,
            scores=[score("dominant", sharpe=5.0, trades=5000), score("other", sharpe=0.6)],
        )
        amounts = granted(decision)
        assert amounts["dominant"] <= decision.deployable_micros * 0.51


class TestSuppression:
    @pytest.mark.parametrize(
        ("label", "candidate"),
        [
            ("degraded", score("s", flags=("calibration_drift",))),
            ("negative sharpe", score("s", sharpe=-0.4)),
            ("zero sharpe", score("s", sharpe=0.0)),
            ("no trades", score("s", trades=0)),
            ("negative expectancy", score("s", expectancy=-1000)),
        ],
    )
    def test_gets_nothing(self, allocator: CapitalAllocator, label: str, candidate: StrategyScore):
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=5_000_000_000,
            scores=[candidate],
        )
        assert granted(decision)["s"] == 0, f"{label} should receive no capital"

    def test_the_reason_is_recorded(self, allocator: CapitalAllocator):
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=5_000_000_000,
            scores=[score("s", flags=("slippage_up", "edge_decay"))],
        )
        allocation = decision.allocations[0]
        assert "degraded" in allocation.reason
        assert "slippage_up" in allocation.reason


class TestSmallAccount:
    """The platform must work with an account under $250."""

    def test_a_tiny_account_concentrates_rather_than_dividing(
        self, allocator: CapitalAllocator
    ):
        """Splitting $40 five ways funds nothing that can pay its own round trip."""
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=40_000_000,
            scores=[score(f"s{i}", sharpe=1.5 - i * 0.2) for i in range(5)],
        )
        funded = [a for a in decision.allocations if a.allocated_micros > 0]
        assert len(funded) <= 2, "a $40 account should not be spread across five strategies"
        assert "concentration floor" in decision.reason

    def test_concentration_funds_the_best_first(self, allocator: CapitalAllocator):
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=40_000_000,
            scores=[score("weak", sharpe=0.6), score("strong", sharpe=2.0)],
        )
        amounts = granted(decision)
        assert amounts["strong"] >= amounts["weak"]

    def test_a_larger_account_diversifies(self, allocator: CapitalAllocator):
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=5_000_000_000,
            scores=[score(f"s{i}", sharpe=1.5 - i * 0.2) for i in range(5)],
        )
        funded = [a for a in decision.allocations if a.allocated_micros > 0]
        assert len(funded) >= 3
        assert "concentration floor" not in decision.reason

    def test_allocations_below_the_tradeable_minimum_are_not_made(
        self, allocator: CapitalAllocator
    ):
        """Money in a slice too small to trade is money immobilised for nothing."""
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=5_000_000_000,
            scores=[score("big", sharpe=5.0, trades=5000)]
            + [score(f"tiny{i}", sharpe=0.01, trades=200) for i in range(30)],
        )
        for allocation in decision.allocations:
            assert allocation.allocated_micros == 0 or allocation.allocated_micros >= 10_000_000


class TestDomainSeparation:
    def test_perp_scores_are_ignored_when_allocating_event_capital(
        self, allocator: CapitalAllocator
    ):
        """The two domains have independent capital and never share a budget."""
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=5_000_000_000,
            scores=[
                score("event_one", market_type=MarketType.EVENT),
                score("perp_one", market_type=MarketType.PERP, sharpe=9.0, trades=9999),
            ],
        )
        assert "perp_one" not in granted(decision)
        assert "event_one" in granted(decision)


class TestApplication:
    def test_allocations_are_pushed_onto_strategies(self, allocator: CapitalAllocator):
        class FakeStrategy:
            def __init__(self) -> None:
                self.allocated_capital_micros = 999

        strategies = {"a": FakeStrategy(), "b": FakeStrategy()}
        decision = allocator.allocate(
            market_type=MarketType.EVENT,
            total_capital_micros=5_000_000_000,
            scores=[score("a", sharpe=2.0), score("b", sharpe=-1.0)],
        )
        updated = CapitalAllocator.apply(decision, strategies)
        assert updated == 2
        assert strategies["a"].allocated_capital_micros > 0
        # A strategy that earned nothing is zeroed explicitly, not left stale.
        assert strategies["b"].allocated_capital_micros == 0
