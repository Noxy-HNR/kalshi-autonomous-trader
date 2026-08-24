"""Strategies: they must fire when the edge is real and decline when it is not.

A strategy that never trades and a strategy that is broken look identical from
the outside, so these tests do both halves: each strategy is put in the
situation it targets and must produce an intent, and is then put in a situation
that only *looks* like it and must produce nothing.

The Kalshi fee curve drives much of this. The quadratic fee is proportional to
``price * (1 - price)``, so a round trip costs about 3.5 cents per contract near
$0.50 and about 1.3 cents near the extremes. An edge of three cents is a losing
trade in the middle of the range and a good one at the tails, and the strategies
are expected to know the difference.
"""

from __future__ import annotations

import inspect

import pytest

from kat.common.enums import MarketType
from kat.kalshi.fees import FeeSchedule, estimate_fee_micros
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketSnapshot, funding_rate_bps_per_hour
from kat.strategies import common as C
from kat.strategies.base import Strategy, StrategyContext

NOW = 1_700_000_000_000
DEEP = 3_000_000  # 30k contracts a side: liquidity is never the binding filter


def strategy_class(module_path: str) -> type[Strategy]:
    module = __import__(module_path, fromlist=["x"])
    return next(
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and issubclass(value, Strategy)
        and value.__module__ == module.__name__
    )


def event_book(bid_cc: int, ask_cc: int) -> OrderBook:
    return OrderBook.from_event_contract_levels(
        "EVT",
        yes_levels=[(bid_cc, DEEP), (bid_cc - 50, DEEP)],
        no_levels=[(10_000 - ask_cc, DEEP), (10_000 - ask_cc - 50, DEEP)],
    )


def event_snapshot(bid_cc: int = 4000, ask_cc: int = 4100, **kw) -> MarketSnapshot:
    return MarketSnapshot(
        ticker="EVT",
        market_type=MarketType.EVENT,
        book=event_book(bid_cc, ask_cc),
        updated_ms=NOW,
        event_ticker="E",
        volume_qc=DEEP,
        status="active",
        **kw,
    )


def perp_snapshot(**kw) -> MarketSnapshot:
    defaults = {
        "ticker": "PERP",
        "market_type": MarketType.PERP,
        "book": OrderBook.from_two_sided_levels("PERP", [(4990, DEEP)], [(5010, DEEP)]),
        "updated_ms": NOW,
        "mark_price_cc": 5000,
        "volume_qc": DEEP,
        "status": "active",
    }
    defaults.update(kw)
    return MarketSnapshot(**defaults)


def make(module_path: str) -> Strategy:
    instance = strategy_class(module_path)()
    # The portfolio layer normally sets this; without it every size is minimal.
    instance.allocated_capital_micros = 150_000_000
    return instance


# ---------------------------------------------------------------------------
# The fee curve the strategies are reasoning about
# ---------------------------------------------------------------------------


class TestFeeCurve:
    @pytest.mark.parametrize(
        ("price_cc", "max_round_trip_usd"),
        [(1000, 0.015), (5000, 0.036), (9000, 0.015)],
    )
    def test_round_trip_cost_peaks_mid_range(self, price_cc: int, max_round_trip_usd: float):
        """Fees are largest at $0.50 and smallest at the tails.

        This is why a three-cent edge is tradeable near the extremes and not in
        the middle, and why the strategies below decline mid-range trades that
        look superficially attractive.
        """
        schedule = FeeSchedule.from_series("quadratic", 0.07)
        one_way = estimate_fee_micros(price_cc=price_cc, qty_qc=100, schedule=schedule)
        round_trip_usd = 2 * one_way.fee_micros / 1_000_000
        assert round_trip_usd <= max_round_trip_usd

    def test_mid_range_costs_more_than_the_tails(self):
        schedule = FeeSchedule.from_series("quadratic", 0.07)
        mid = estimate_fee_micros(price_cc=5000, qty_qc=100, schedule=schedule).fee_micros
        tail = estimate_fee_micros(price_cc=1000, qty_qc=100, schedule=schedule).fee_micros
        assert mid > tail * 2


# ---------------------------------------------------------------------------
# Funding units -- the regression
# ---------------------------------------------------------------------------


class TestFundingUnits:
    """``funding_rate`` is a raw per-interval fraction, never bps per hour.

    The regression: the carry strategy compared the raw fraction directly
    against a threshold expressed in basis points per hour, so a 5% funding rate
    read as 0.05 bps and the strategy declined every trade at every rate. It
    never fired once, and nothing failed loudly -- it simply never traded.
    """

    def test_conversion_is_rate_times_ten_thousand_over_interval(self):
        assert funding_rate_bps_per_hour(0.0001, 1.0) == pytest.approx(1.0)
        assert funding_rate_bps_per_hour(0.01, 1.0) == pytest.approx(100.0)
        # An 8-hour interval spreads the same charge over eight hours.
        assert funding_rate_bps_per_hour(0.08, 8.0) == pytest.approx(100.0)

    def test_unknown_rate_is_none_not_zero(self):
        """A missing rate must not read as "funding is free"."""
        assert funding_rate_bps_per_hour(None) is None

    def test_non_positive_interval_is_rejected(self):
        assert funding_rate_bps_per_hour(0.01, 0.0) is None

    def test_carry_fires_on_a_large_rate(self):
        strategy = make("kat.strategies.perp.funding_carry")
        snapshot = perp_snapshot(funding_rate=0.05, next_funding_ms=NOW + 900_000)
        intents = strategy.generate(StrategyContext(now_ms=NOW, market=snapshot))
        assert intents, f"expected an intent, got skip={strategy.health()['last_skip']}"
        # A positive rate means longs pay shorts, so the receiving side is short.
        assert intents[0].book_side is not None
        assert intents[0].book_side.value == "ask"

    def test_carry_declines_a_negligible_rate(self):
        strategy = make("kat.strategies.perp.funding_carry")
        snapshot = perp_snapshot(funding_rate=0.00001, next_funding_ms=NOW + 900_000)
        assert strategy.generate(StrategyContext(now_ms=NOW, market=snapshot)) == []

    def test_carry_declines_when_the_rate_is_unknown(self):
        strategy = make("kat.strategies.perp.funding_carry")
        snapshot = perp_snapshot(funding_rate=None)
        assert strategy.generate(StrategyContext(now_ms=NOW, market=snapshot)) == []
        assert strategy.health()["last_skip"] == "funding_rate_unknown"


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------


def _reversion_context(bid_cc: int, ask_cc: int, **overrides) -> StrategyContext:
    features = {
        C.F_MID_CHANGE_5M_CC: -1200.0,   # a large move down
        C.F_MID_CHANGE_1M_CC: -100.0,    # already decelerating
        C.F_REALIZED_VOL_5M_CC: 150.0,
        C.F_VOLUME_ZSCORE_5M: 0.2,       # unconfirmed by volume
        C.F_VOLUME_5M_QC: float(DEEP),
    }
    features.update(overrides)
    return StrategyContext(now_ms=NOW, market=event_snapshot(bid_cc, ask_cc), features=features)


class TestMeanReversion:
    def test_fires_where_fees_are_small(self):
        """At $0.10 a four-cent reversion comfortably clears a 1.3 cent round trip."""
        strategy = make("kat.strategies.event.mean_reversion")
        intents = strategy.generate(_reversion_context(900, 1000))
        assert intents, f"expected an intent, got skip={strategy.health()['last_skip']}"
        intent = intents[0]
        assert intent.outcome_side is not None
        assert intent.outcome_side.value == "yes"  # the move was down, so YES is cheap
        assert intent.fair_value_cc > intent.limit_price_cc
        assert (intent.expected_value_micros or 0) > 0

    def test_declines_the_same_setup_mid_range_where_fees_eat_it(self):
        """The identical signal at $0.41 is not worth doing after costs."""
        strategy = make("kat.strategies.event.mean_reversion")
        assert strategy.generate(_reversion_context(4000, 4100)) == []
        assert strategy.health()["last_skip"] == "expected_value_below_floor"

    def test_declines_a_move_confirmed_by_volume(self):
        """High volume means the market is repricing, not overreacting."""
        strategy = make("kat.strategies.event.mean_reversion")
        assert strategy.generate(_reversion_context(900, 1000, **{C.F_VOLUME_ZSCORE_5M: 4.0})) == []
        assert strategy.health()["last_skip"] == "move_confirmed_by_volume"

    def test_declines_a_move_still_extending(self):
        """A move whose final minute is still large has not finished."""
        strategy = make("kat.strategies.event.mean_reversion")
        assert (
            strategy.generate(_reversion_context(900, 1000, **{C.F_MID_CHANGE_1M_CC: -1100.0}))
            == []
        )
        assert strategy.health()["last_skip"] == "move_still_extending"

    def test_declines_when_volatility_is_unknown(self):
        """Zero volatility is a missing estimate, not a calm market.

        Dividing by it would make every move infinitely significant.
        """
        strategy = make("kat.strategies.event.mean_reversion")
        assert (
            strategy.generate(_reversion_context(900, 1000, **{C.F_REALIZED_VOL_5M_CC: 0.0}))
            == []
        )
        assert strategy.health()["last_skip"] == "realized_vol_unavailable"

    def test_declines_on_stale_data(self):
        strategy = make("kat.strategies.event.mean_reversion")
        ctx = _reversion_context(900, 1000)
        ctx.now_ms = NOW + 10_000_000  # snapshot is now ancient
        assert strategy.generate(ctx) == []


# ---------------------------------------------------------------------------
# Every strategy: structural obligations
# ---------------------------------------------------------------------------

ALL_STRATEGIES = [
    "kat.strategies.event.mispricing",
    "kat.strategies.event.mean_reversion",
    "kat.strategies.event.momentum",
    "kat.strategies.event.expiry_drift",
    "kat.strategies.event.news_reaction",
    "kat.strategies.perp.funding_carry",
    "kat.strategies.perp.trend",
    "kat.strategies.perp.basis",
]


@pytest.mark.parametrize("module_path", ALL_STRATEGIES)
class TestStrategyContract:
    def test_declares_a_bounded_parameter_space(self, module_path: str):
        """An unbounded search space is how an optimiser finds a silly position."""
        strategy = make(module_path)
        space = strategy.metadata.param_space
        assert space, f"{module_path} declares no tunable parameters"
        for param in space:
            numeric = isinstance(param.default, int | float) and not isinstance(
                param.default, bool
            )
            if numeric:
                assert param.low is not None and param.high is not None, (
                    f"{module_path}:{param.name} is numeric but unbounded"
                )
                assert param.low <= param.high

    def test_declares_its_identity(self, module_path: str):
        strategy = make(module_path)
        meta = strategy.metadata
        assert meta.strategy_id and meta.version and meta.name and meta.family
        assert meta.market_type in (MarketType.EVENT, MarketType.PERP)

    def test_produces_nothing_from_an_empty_market(self, module_path: str):
        """No book, no data, no opinion -- and no exception."""
        strategy = make(module_path)
        empty = MarketSnapshot(
            ticker="EMPTY",
            market_type=strategy.market_type,
            book=OrderBook("EMPTY"),
            updated_ms=NOW,
        )
        assert strategy.generate(StrategyContext(now_ms=NOW, market=empty)) == []

    def test_never_reads_the_wall_clock(self, module_path: str):
        """A strategy that reads real time sees the future during a backtest."""
        import ast
        import pathlib

        module = __import__(module_path, fromlist=["x"])
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        banned = {"time", "datetime", "utcnow", "now"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"time", "utcnow", "now_ms"}:
                # ctx.now_ms is the simulated clock and is the correct source.
                if isinstance(node.value, ast.Name) and node.value.id == "ctx":
                    continue
                if node.attr in banned:
                    raise AssertionError(
                        f"{module_path} reads {node.attr} directly; use ctx.now_ms"
                    )

    def test_reports_a_skip_reason_when_it_declines(self, module_path: str):
        """"It did nothing" must always come with "because"."""
        strategy = make(module_path)
        empty = MarketSnapshot(
            ticker="EMPTY",
            market_type=strategy.market_type,
            book=OrderBook("EMPTY"),
            updated_ms=NOW,
        )
        strategy.generate(StrategyContext(now_ms=NOW, market=empty))
        assert strategy.health()["last_skip"], f"{module_path} declined without a reason"
