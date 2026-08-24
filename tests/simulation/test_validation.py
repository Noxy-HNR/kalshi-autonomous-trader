"""Tests for the simulation stack: scenarios, features, validation, gates.

The load-bearing ones here are the negative results. A validation harness that
cannot be made to *fail* is not measuring anything, so several of these check
that a strategy given too few trades, or a fitted parameter, or a walk-forward
that never turned a profit, is correctly refused rather than promoted.
"""

from __future__ import annotations

import itertools

import pytest

from kat.common.enums import DataQuality, MarketType, StrategyState
from kat.common.errors import DataQualityError
from kat.config.settings import ResearchSettings
from kat.discovery.lifecycle import evaluate_promotion
from kat.simulation.engine import BacktestConfig, EventFeed, EventKind, book_event
from kat.simulation.features import SimFeatureProvider, build_histories
from kat.simulation.scenarios import (
    MAX_PRICE_CC,
    MIN_PRICE_CC,
    ScenarioSpec,
    generate_market_series,
    generate_scenario,
    market_definitions,
    realised_volatility_cc,
)
from kat.simulation.validation import (
    monte_carlo_sharpe,
    run_once,
    split_chronologically,
    validate,
    walk_forward,
)
from kat.strategies.event.mean_reversion import MeanReversionStrategy

HOUR_MS = 3_600_000


def spec(**kw) -> ScenarioSpec:
    base = {
        "kind": "overreaction",
        "seed": 11,
        "duration_ms": 4 * HOUR_MS,
        "start_price_cc": 1_200,
        "close_offset_ms": 5 * HOUR_MS,
        "shock_probability": 0.008,
        "shock_cc": 1_200.0,
        "reversion": 0.05,
        "base_half_spread_cc": 25,
    }
    base.update(kw)
    return ScenarioSpec(**base)


def factory():
    strategy = MeanReversionStrategy()
    strategy.state = StrategyState.BACKTEST
    return [strategy]


CONFIG = BacktestConfig(market_type=MarketType.EVENT, seed=1, decision_interval_ms=30_000)


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


class TestScenarios:
    @pytest.mark.parametrize(
        "kind",
        ["mean_reverting", "trending", "drifting", "random_walk", "settling", "overreaction"],
    )
    def test_every_generator_produces_a_usable_feed(self, kind):
        events = generate_scenario(spec(kind=kind))
        feed = EventFeed(events)
        assert len(feed) > 100
        assert feed.end_ms > feed.start_ms

    def test_generated_history_is_marked_synthetic(self):
        """The safety property the whole module rests on.

        A number measured on generated data must never be presentable as a
        number measured on the exchange.
        """
        assert EventFeed(generate_scenario(spec())).is_synthetic

    def test_synthetic_and_recorded_history_cannot_be_mixed(self):
        """Blending the two produces a result that is neither."""
        synthetic = generate_scenario(spec())[:50]
        recorded = book_event(
            "REAL-1", info_ts_ms=0, bids=[(1_000, 500)], asks=[(1_100, 500)],
            quality=DataQuality.OK,
        )
        with pytest.raises(DataQualityError, match="synthetic"):
            EventFeed([*synthetic, recorded])

    def test_prices_stay_inside_the_tradeable_band(self):
        """A contract at zero or a dollar has resolved and is not a market."""
        events = generate_scenario(spec(kind="trending", drift_cc=40.0, duration_ms=8 * HOUR_MS))
        for event in events:
            for level in (event.payload.get("bids") or []) + (event.payload.get("asks") or []):
                assert MIN_PRICE_CC <= level[0] <= MAX_PRICE_CC

    def test_requested_volatility_is_what_comes_out(self):
        """A generator asked for 40 that produces 4 is broken."""
        events = generate_scenario(spec(kind="random_walk", volatility_cc=40.0))
        assert 30.0 < realised_volatility_cc(events) < 55.0

    def test_nothing_is_knowable_before_it_happens(self):
        for event in generate_scenario(spec()):
            assert event.available_ms >= event.info_ts_ms
            assert event.lag_ms > 0

    def test_an_unknown_scenario_kind_is_refused(self):
        with pytest.raises(ValueError, match="unknown scenario kind"):
            generate_scenario(spec(kind="wishful_thinking"))

    def test_a_market_series_settles_every_market(self):
        """Event-contract positions close by resolving, not by a later sale.

        A run whose markets never resolve leaves open positions and produces no
        completed trades, and a backtest with no completed trades has measured
        nothing however long it ran.
        """
        events, closes = generate_market_series(spec(), count=6)
        settlements = [e for e in events if e.kind is EventKind.SETTLEMENT]
        assert len(closes) == 6
        assert len(settlements) == 6
        assert len({e.ticker for e in settlements}) == 6

    def test_series_markets_overlap_in_time(self):
        """Real trading days have several contracts live at once."""
        events, _ = generate_market_series(spec(), count=4)
        by_ticker: dict[str, tuple[int, int]] = {}
        for event in events:
            lo, hi = by_ticker.get(event.ticker, (event.available_ms, event.available_ms))
            by_ticker[event.ticker] = (min(lo, event.available_ms), max(hi, event.available_ms))
        spans = sorted(by_ticker.values())
        assert spans[1][0] < spans[0][1], "the second market should open before the first closes"


# ---------------------------------------------------------------------------
# features inside the simulator
# ---------------------------------------------------------------------------


class TestSimFeatures:
    def test_features_compute_for_a_generated_market(self):
        """The defect: with no provider, every strategy skipped for the whole run."""
        s = spec()
        events = generate_scenario(s)
        provider = SimFeatureProvider(events, MarketType.EVENT, close_ts_ms=market_definitions(s))
        # Well past the longest feature window.
        vector = provider(s.ticker, 3 * HOUR_MS)
        assert vector is not None
        assert provider.computed == 1
        assert provider.failures == 0

    def test_an_event_contract_needs_its_close_time(self):
        """Term features are undefined without it, so the vector must not exist.

        Returning a partial vector would be worse: the strategy would act on a
        market whose time-to-expiry it does not know.
        """
        s = spec(close_offset_ms=None)
        events = generate_scenario(s)
        provider = SimFeatureProvider(events, MarketType.EVENT, close_ts_ms=market_definitions(s))
        assert provider(s.ticker, 3 * HOUR_MS) is None
        assert provider.failures == 1

    def test_nothing_is_computed_before_there_is_history(self):
        s = spec()
        provider = SimFeatureProvider(
            generate_scenario(s), MarketType.EVENT, close_ts_ms=market_definitions(s)
        )
        assert provider(s.ticker, 0) is None

    def test_an_unknown_market_returns_nothing(self):
        s = spec()
        provider = SimFeatureProvider(generate_scenario(s), MarketType.EVENT)
        assert provider("NOT-A-MARKET", HOUR_MS) is None
        assert provider.no_history == 1

    def test_bars_are_stamped_with_their_close_not_their_open(self):
        """Stamping the open understates availability by a whole period."""
        s = spec()
        histories = build_histories(generate_scenario(s), MarketType.EVENT)
        bars = histories[s.ticker].bars
        assert bars is not None and len(bars) > 3
        # A one-minute bar closing at T is available from T + LAG_CANDLE_MS.
        first = bars.latest(as_of_ms=10 * 60_000, lag_ms=0)
        assert first is not None
        assert (first.ts_ms + 1) % 60_000 == 0


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------


class TestSplitting:
    def test_the_cut_is_on_time_not_on_count(self):
        """Equal counts of events are not equal periods.

        A quiet overnight span and a busy hour hold very different numbers of
        messages, and annualising a Sharpe from the wrong one is nonsense.
        """
        events = [
            book_event("M", info_ts_ms=t, lag_ms=1, bids=[(1_000, 100)], asks=[(1_100, 100)],
                       quality=DataQuality.SYNTHETIC)
            # Dense for the first tenth of the span, sparse afterwards.
            for t in [*range(0, 1_000, 10), *range(1_000, 10_000, 1_000)]
        ]
        first, second = split_chronologically(events, (0.5, 0.5))
        boundary = max(e.available_ms for e in first)
        assert boundary <= 5_100
        assert min(e.available_ms for e in second) > boundary
        assert len(first) > len(second), "the dense half holds more events, as it should"

    def test_the_two_halves_are_disjoint_and_complete(self):
        events = generate_scenario(spec())
        first, second = split_chronologically(events, (0.7, 0.3))
        assert len(first) + len(second) == len(events)
        assert max(e.available_ms for e in first) <= min(e.available_ms for e in second)

    def test_degenerate_inputs(self):
        assert split_chronologically([], (0.5, 0.5)) == [[], []]
        with pytest.raises(ValueError, match="positive"):
            split_chronologically(generate_scenario(spec())[:10], (0.0, 0.0))


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------


class TestWalkForward:
    def test_the_first_window_is_not_tested_on_an_untrained_strategy(self):
        """The defect: window 0 had no history in front of it.

        Its train span was empty, so it scored a strategy that had seen nothing.
        For a fitted strategy that is not an out-of-sample result at all.
        """
        events = generate_scenario(spec(duration_ms=8 * HOUR_MS))
        _, windows = walk_forward(events, factory, CONFIG, windows=3, train_fraction=0.5)
        start = min(e.available_ms for e in events)
        for window in windows:
            assert window.train_end_ms > window.train_start_ms
            assert window.test_start_ms > start

    def test_test_spans_tile_the_history_without_overlapping(self):
        """No period may be scored twice."""
        events = generate_scenario(spec(duration_ms=8 * HOUR_MS))
        _, windows = walk_forward(events, factory, CONFIG, windows=4)
        for previous, following in itertools.pairwise(windows):
            assert following.test_start_ms >= previous.test_end_ms

    def test_a_window_with_no_trades_is_not_counted_as_a_loss(self):
        """Standing aside is a decision the strategy is allowed to make.

        Counting an untraded window as unprofitable would fail a strategy for
        declining bad conditions, which is the behaviour it is supposed to have.
        """
        events = generate_scenario(spec(kind="mean_reverting", shock_probability=0.0))
        block, windows = walk_forward(events, factory, CONFIG, windows=3)
        assert block["positive_windows"] == 0
        if all(w.trades == 0 for w in windows):
            assert block["windows"] == 0
            assert block["empty_windows"] == len(windows)

    def test_too_few_windows_is_refused_outright(self):
        with pytest.raises(ValueError, match="fewer than two windows"):
            walk_forward(generate_scenario(spec()), factory, CONFIG, windows=1)

    def test_a_nonsensical_train_fraction_is_refused(self):
        for fraction in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="strictly between"):
                walk_forward(
                    generate_scenario(spec()), factory, CONFIG,
                    windows=3, train_fraction=fraction,
                )


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    def test_a_handful_of_trades_is_refused_rather_than_resampled(self):
        """A bootstrap over four trades measures the four trades."""
        events, closes = generate_market_series(spec(), count=2)
        provider = SimFeatureProvider(events, MarketType.EVENT, close_ts_ms=closes)
        result = run_once(events, factory, CONFIG, feature_provider=provider)
        assert result is not None
        if len(result.trades) < 5:
            block = monte_carlo_sharpe(result, runs=100)
            assert block["monte_carlo_p5_sharpe"] is None
            assert "too few" in block["detail"]


# ---------------------------------------------------------------------------
# the whole protocol, and the gates
# ---------------------------------------------------------------------------


class TestValidationProtocol:
    @pytest.fixture(scope="class")
    def report(self):
        s = spec(duration_ms=3 * HOUR_MS)
        events, closes = generate_market_series(s, count=6)
        provider = SimFeatureProvider(events, MarketType.EVENT, close_ts_ms=closes)
        return validate(
            events, factory, CONFIG,
            strategy_id="event.mean_reversion",
            walk_forward_windows=3, monte_carlo_runs=100,
            feature_provider=provider,
        )

    def test_every_block_the_gates_read_is_present(self, report):
        blocks = report.to_dict()
        assert blocks["backtest"] is not None
        assert blocks["walk_forward"] is not None
        assert blocks["robustness"] is not None
        for key in ("trades", "sharpe", "max_drawdown_pct", "expectancy_micros"):
            assert key in blocks["backtest"], key
        for key in ("windows", "positive_windows", "sharpe"):
            assert key in blocks["walk_forward"], key

    def test_a_generated_history_is_labelled_as_one(self, report):
        """A synthetic run must never be mistaken for evidence of an edge."""
        assert report.synthetic is True
        assert any("generated" in note for note in report.notes)

    def test_expectancy_reaches_the_gate_as_a_number(self, report):
        """BacktestMetrics emits expectancy as a dollar string for display.

        The gate reads ``expectancy_micros``. Without the adapter it saw nothing
        and reported "unknown" rather than failing on the value.
        """
        value = report.backtest["expectancy_micros"]
        assert isinstance(value, int)

    def test_the_gates_refuse_a_strategy_with_no_evidence(self, report):
        """The whole point: promotion is refused, and the refusal is explicable."""
        decision = evaluate_promotion(
            strategy_id="event.mean_reversion",
            current_state=StrategyState.BACKTEST,
            settings=ResearchSettings(),
            **report.gate_kwargs(),
        )
        assert decision.promoted is False
        failed = [g for g in decision.gates if not g.passed]
        assert failed, "a refusal must name at least one failing gate"
        assert all(g.requirement for g in failed)
        assert all(g.observed for g in failed)

    def test_a_missing_perturbation_test_refuses_promotion(self, report):
        """No perturbed factory means parameter sensitivity was never measured.

        Unmeasured must fail, not pass by default.
        """
        assert report.robustness["perturbation_sharpe_ratio"] is None
        decision = evaluate_promotion(
            strategy_id="event.mean_reversion",
            current_state=StrategyState.ROBUSTNESS,
            settings=ResearchSettings(),
            **report.gate_kwargs(),
        )
        assert decision.promoted is False
        assert any(g.name == "parameter_perturbation" and not g.passed for g in decision.gates)

    def test_no_leakage_anywhere_in_the_protocol(self):
        """Strict leakage detection is on, and nothing tripped it."""
        s = spec(duration_ms=2 * HOUR_MS)
        events, closes = generate_market_series(s, count=3)
        provider = SimFeatureProvider(events, MarketType.EVENT, close_ts_ms=closes)
        result = run_once(events, factory, CONFIG, feature_provider=provider)
        assert result is not None
        assert result.leakage["clean"] is True
        assert result.leakage["violations"] == 0
        assert result.leakage["checks"] > 0
