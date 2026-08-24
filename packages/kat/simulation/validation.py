"""The validation protocol, and the metrics the promotion gates consume.

:mod:`kat.discovery.lifecycle` decides whether a strategy may advance. It asks
for specific numbers and refuses to promote without them. This module is what
produces those numbers, and it is deliberately the only thing that does: a gate
reading a metric block assembled somewhere else has no way to know how the split
was made or whether the same period was used twice.

The contract
------------
Four blocks, keyed exactly as the gates read them:

``backtest`` / ``oos`` / ``paper`` / ``live``
    ``trades``, ``sharpe``, ``max_drawdown_pct``, ``expectancy_micros``
``walk_forward``
    ``windows``, ``positive_windows``, ``sharpe``
``robustness``
    ``monte_carlo_p5_sharpe``, ``perturbation_sharpe_ratio``

Note ``trades`` and ``expectancy_micros``: :class:`~kat.simulation.metrics.
BacktestMetrics` names them ``trade_count`` and emits expectancy as a dollar
*string* for display. :func:`gate_metrics` is the adapter, and it exists so the
gates read integers rather than parsing formatted money.

Why the splits are chronological and never shuffled
---------------------------------------------------
Every split here cuts on time. Shuffling trades across a boundary lets the
"out-of-sample" period contain trades taken from the middle of the training
period, whose outcomes were already used to fit the strategy. The result looks
like out-of-sample evidence and is not, which is the exact failure the whole
lifecycle exists to prevent. There is no option to shuffle.

Why each strategy is rebuilt per run
------------------------------------
A strategy accumulates state -- price history, position memory, fitted
parameters. Reusing one instance across the in-sample and out-of-sample runs
would carry the training period's memory into the test period, so every entry
point here takes a *factory* and builds fresh instances for every single run.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from kat.common.enums import MarketType
from kat.observability.logging import get_logger
from kat.simulation.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    EventFeed,
    SimEvent,
)
from kat.strategies.base import Strategy

log = get_logger(__name__)

__all__ = [
    "ValidationReport",
    "WalkForwardWindow",
    "gate_metrics",
    "monte_carlo_sharpe",
    "perturbation_ratio",
    "run_once",
    "split_chronologically",
    "validate",
    "walk_forward",
]

#: A factory, because an engine runs once and a strategy remembers.
StrategyFactory = Callable[[], Sequence[Strategy]]

#: Given a multiplier, build strategies with their parameters scaled by it.
#: Used only by the perturbation test.
PerturbedFactory = Callable[[float], Sequence[Strategy]]


# ---------------------------------------------------------------------------
# The gate-facing view of a result
# ---------------------------------------------------------------------------


def gate_metrics(result: BacktestResult) -> dict[str, Any]:
    """The metric block :mod:`kat.discovery.lifecycle` reads.

    Kept minimal and explicitly named. The full :meth:`BacktestMetrics.to_dict`
    is richer and is what gets stored for a human to read; this is the subset the
    gates actually consume, in the units they consume it in.
    """
    m = result.metrics
    return {
        "trades": m.trade_count,
        "sharpe": m.sharpe,
        "max_drawdown_pct": m.max_drawdown_pct,
        "expectancy_micros": m.expectancy_micros,
        # Carried for explanation rather than gating. A refusal that can say
        # "fees were 140% of gross profit" is far more useful than one that
        # says "expectancy was negative".
        "win_rate": m.win_rate,
        "total_fees_micros": m.total_fees_micros,
        "fee_drag": m.fee_drag,
        "duration_days": m.duration_days,
        "final_equity_micros": m.final_equity_micros,
    }


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_chronologically(
    events: Sequence[SimEvent], fractions: Sequence[float]
) -> list[list[SimEvent]]:
    """Cut an event list into consecutive spans by time, never by index.

    Cutting by index would put equal *counts* of events in each span, which is
    not the same thing: a quiet overnight period and a busy hour would be
    treated as equal-length periods, and a Sharpe annualised from the second
    would be nonsense. The cut is on ``available_ms``, so each span is a real
    interval of simulated time.
    """
    if not events:
        return [[] for _ in fractions]
    total = sum(fractions)
    if total <= 0:
        raise ValueError("split fractions must sum to something positive")

    ordered = sorted(events, key=lambda e: (e.available_ms, e.seq))
    start = ordered[0].available_ms
    span = ordered[-1].available_ms - start
    if span <= 0:
        # Everything arrived at the same instant; there is no time axis to cut.
        return [list(ordered)] + [[] for _ in fractions[1:]]

    bounds: list[int] = []
    running = 0.0
    for fraction in fractions[:-1]:
        running += fraction / total
        bounds.append(start + int(span * running))

    spans: list[list[SimEvent]] = [[] for _ in fractions]
    for event in ordered:
        index = 0
        while index < len(bounds) and event.available_ms > bounds[index]:
            index += 1
        spans[index].append(event)
    return spans


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_once(
    events: Sequence[SimEvent],
    factory: StrategyFactory,
    config: BacktestConfig,
    *,
    label: str = "",
    **engine_kwargs: Any,
) -> BacktestResult | None:
    """One backtest over one span, with freshly built strategies.

    Returns ``None`` when the span is too short to contain a result, rather than
    raising: a walk-forward sweep over a real history will hit empty windows
    (weekends, halted markets), and one empty window is not a failed validation.
    """
    if len(events) < 2:
        log.debug("validation.span_too_short", label=label, events=len(events))
        return None

    feed = EventFeed(events, name=label or config.label)
    strategies = list(factory())
    if not strategies:
        raise ValueError("the strategy factory produced no strategies")

    run_config = _relabel(config, label or config.label)
    engine = BacktestEngine(
        feed=feed, strategies=strategies, config=run_config, **engine_kwargs
    )
    return engine.run()


def _relabel(config: BacktestConfig, label: str) -> BacktestConfig:
    """A copy carrying a different label, so each run is identifiable."""
    from dataclasses import replace

    return replace(config, label=label)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WalkForwardWindow:
    """One rolling window's out-of-sample result."""

    index: int
    train_start_ms: int
    train_end_ms: int
    test_start_ms: int
    test_end_ms: int
    trades: int
    sharpe: float
    total_return: float
    expectancy_micros: int

    @property
    def profitable(self) -> bool:
        return self.expectancy_micros > 0 and self.total_return > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_start_ms": self.train_start_ms,
            "train_end_ms": self.train_end_ms,
            "test_start_ms": self.test_start_ms,
            "test_end_ms": self.test_end_ms,
            "trades": self.trades,
            "sharpe": round(self.sharpe, 4),
            "total_return": round(self.total_return, 6),
            "expectancy_micros": self.expectancy_micros,
            "profitable": self.profitable,
        }


def walk_forward(
    events: Sequence[SimEvent],
    factory: StrategyFactory,
    config: BacktestConfig,
    *,
    windows: int = 5,
    train_fraction: float = 0.6,
    **engine_kwargs: Any,
) -> tuple[dict[str, Any], list[WalkForwardWindow]]:
    """Roll a train/test pair forward through the history.

    Each window trains on its own leading span and is *measured* only on the
    span that follows it, so every reported number comes from data the strategy
    had not seen when it was built. The windows advance rather than expanding,
    which is the stricter choice: an expanding window lets the earliest data
    influence every later result, and a strategy that only worked in the first
    month would keep being propped up by it.

    Returns ``(metric_block, per_window_detail)``. The block is what the gates
    read; the detail is what explains a refusal.
    """
    if windows < 2:
        raise ValueError("a walk-forward with fewer than two windows measures nothing")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must sit strictly between 0 and 1")

    ordered = sorted(events, key=lambda e: (e.available_ms, e.seq))
    if len(ordered) < 2:
        return _empty_walk_forward(), []

    start = ordered[0].available_ms
    end = ordered[-1].available_ms
    span = end - start
    if span <= 0:
        return _empty_walk_forward(), []

    # The leading `train_fraction` of the history is training only, never
    # tested. Without this reservation the first window has no history in front
    # of it and is scored on a strategy that has seen nothing -- which for a
    # fitted strategy is not an out-of-sample result, it is an untrained one.
    warmup_span = int(span * train_fraction)
    test_region_start = start + warmup_span
    test_span = end - test_region_start
    if test_span <= 0:
        return _empty_walk_forward(), []

    # Windows overlap in training but never in testing: the test spans tile the
    # region after the reservation exactly once, so no period is scored twice.
    step = test_span / windows
    train_span = int(step * train_fraction / (1.0 - train_fraction))
    results: list[WalkForwardWindow] = []

    for index in range(windows):
        test_start = test_region_start + int(step * index)
        test_end = test_region_start + int(step * (index + 1))
        train_start = max(start, test_start - train_span)

        train_events = [e for e in ordered if train_start <= e.available_ms < test_start]
        test_events = [e for e in ordered if test_start <= e.available_ms <= test_end]

        # The training span is run so the strategy sees it, but its result is
        # deliberately discarded: it is in-sample by construction.
        if len(train_events) >= 2:
            run_once(
                train_events, factory, config,
                label=f"wf{index}_train", **engine_kwargs,
            )

        result = run_once(
            test_events, factory, config, label=f"wf{index}_test", **engine_kwargs
        )
        if result is None:
            log.debug("validation.walk_forward_window_empty", index=index)
            continue

        m = result.metrics
        results.append(
            WalkForwardWindow(
                index=index,
                train_start_ms=train_start,
                train_end_ms=test_start,
                test_start_ms=test_start,
                test_end_ms=test_end,
                trades=m.trade_count,
                sharpe=m.sharpe,
                total_return=m.total_return,
                expectancy_micros=m.expectancy_micros,
            )
        )

    if not results:
        return _empty_walk_forward(), []

    # A window in which the strategy never traded is not evidence of anything.
    # Counting it as an unprofitable window would penalise standing aside, which
    # is a decision the strategy is supposed to be able to make; counting it as
    # a window at all would let a strategy clear the "enough windows" bar
    # without producing enough periods that it actually traded.
    traded = [w for w in results if w.trades > 0]
    empty = len(results) - len(traded)
    if not traded:
        block = _empty_walk_forward()
        block["empty_windows"] = empty
        block["detail"] = (
            f"the strategy did not trade in any of the {len(results)} windows; "
            "there is nothing to measure"
        )
        return block, results

    sharpes = [w.sharpe for w in traded]
    block = {
        "windows": len(traded),
        "empty_windows": empty,
        "positive_windows": sum(1 for w in traded if w.profitable),
        # The gate reads a single Sharpe for the whole walk-forward. The mean
        # across windows is the right one: it weights each period equally, so a
        # single extraordinary window cannot carry the number the way it would
        # if the windows were concatenated and measured as one series.
        "sharpe": statistics.fmean(sharpes),
        "sharpe_stdev": statistics.pstdev(sharpes) if len(sharpes) > 1 else 0.0,
        "worst_window_sharpe": min(sharpes),
        "best_window_sharpe": max(sharpes),
        "trades": sum(w.trades for w in traded),
    }
    return block, results


def _empty_walk_forward() -> dict[str, Any]:
    """A block that will fail the gates, which is the correct outcome.

    Reported as zero windows rather than omitted, so the refusal reads "0
    windows, needed 4" instead of "unknown".
    """
    return {"windows": 0, "positive_windows": 0, "sharpe": 0.0, "trades": 0}


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def monte_carlo_sharpe(
    result: BacktestResult,
    *,
    runs: int = 500,
    percentile: float = 5.0,
    seed: int = 0,
    periods_per_year: float | None = None,
) -> dict[str, Any]:
    """Resample the trade sequence and ask what the bad orderings look like.

    The question this answers: *if the same trades had arrived in a different
    order, would the strategy still have made money?* A strategy whose result
    depends on the order -- one big early win funding a long losing streak --
    has a result that will not repeat, because the order was arbitrary.

    Resampling is with replacement over the per-trade returns, which is the
    standard bootstrap. It deliberately does **not** resample the equity curve:
    equity points are serially dependent, and shuffling them would destroy the
    drawdown structure that makes the question meaningful.

    The gate reads the 5th percentile and requires it above zero, so a strategy
    passes only if it makes money in 95% of plausible reorderings.
    """
    trades = result.trades
    if len(trades) < 5:
        return {
            "monte_carlo_p5_sharpe": None,
            "runs": 0,
            "detail": (
                f"{len(trades)} trades is too few to resample; a bootstrap over "
                "a handful of trades measures the handful, not the strategy"
            ),
        }

    import numpy as np

    returns = np.array([t.return_on_cost() for t in trades], dtype=float)
    rng = np.random.default_rng(seed)
    n = len(returns)

    # Sharpe on a per-trade series annualises by trades-per-year, not by the
    # equity sampling period: the series being resampled is trades.
    if periods_per_year is None:
        days = max(result.metrics.duration_days, 1e-9)
        periods_per_year = n / days * 365.25

    sharpes = np.empty(runs, dtype=float)
    for i in range(runs):
        sample = returns[rng.integers(0, n, size=n)]
        std = float(sample.std(ddof=1)) if n > 1 else 0.0
        sharpes[i] = 0.0 if std <= 0 else float(sample.mean()) / std * math.sqrt(
            periods_per_year
        )

    return {
        "monte_carlo_p5_sharpe": float(np.percentile(sharpes, percentile)),
        "monte_carlo_median_sharpe": float(np.percentile(sharpes, 50)),
        "monte_carlo_p95_sharpe": float(np.percentile(sharpes, 95)),
        "monte_carlo_loss_probability": float((sharpes <= 0).mean()),
        "runs": runs,
        "trades_resampled": n,
    }


def perturbation_ratio(
    events: Sequence[SimEvent],
    perturbed_factory: PerturbedFactory,
    config: BacktestConfig,
    *,
    shift: float = 0.10,
    baseline_sharpe: float | None = None,
    **engine_kwargs: Any,
) -> dict[str, Any]:
    """How much of the edge survives a small parameter change.

    A strategy whose Sharpe collapses when a threshold moves by ten percent has
    parameters fitted to the sample, not chosen for a reason. The test runs the
    same history at ``1 - shift`` and ``1 + shift`` and reports the *worse* of
    the two as a fraction of the unperturbed Sharpe.

    Taking the worse side rather than the average is the point: a strategy that
    improves when a parameter moves one way and dies when it moves the other has
    still demonstrated fragility, and averaging would hide it.
    """
    if baseline_sharpe is None:
        baseline = run_once(
            events, lambda: perturbed_factory(1.0), config,
            label="perturb_base", **engine_kwargs,
        )
        if baseline is None:
            return {"perturbation_sharpe_ratio": None, "detail": "baseline run produced nothing"}
        baseline_sharpe = baseline.metrics.sharpe

    if baseline_sharpe <= 0:
        # There is no edge to be fragile. The Sharpe gate already refuses this;
        # reporting a ratio here would be dividing by a non-edge.
        return {
            "perturbation_sharpe_ratio": None,
            "baseline_sharpe": baseline_sharpe,
            "detail": "baseline Sharpe is not positive, so retention is not meaningful",
        }

    observed: dict[str, float] = {}
    for name, multiplier in (("down", 1.0 - shift), ("up", 1.0 + shift)):
        result = run_once(
            events, lambda m=multiplier: perturbed_factory(m), config,
            label=f"perturb_{name}", **engine_kwargs,
        )
        observed[name] = 0.0 if result is None else result.metrics.sharpe

    worst = min(observed.values())
    return {
        "perturbation_sharpe_ratio": worst / baseline_sharpe,
        "baseline_sharpe": baseline_sharpe,
        "perturbed_down_sharpe": observed["down"],
        "perturbed_up_sharpe": observed["up"],
        "shift": shift,
    }


# ---------------------------------------------------------------------------
# The whole protocol
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ValidationReport:
    """Everything one validation produced, in the shape the gates read."""

    strategy_id: str
    market_type: MarketType
    backtest: dict[str, Any] | None = None
    oos: dict[str, Any] | None = None
    walk_forward: dict[str, Any] | None = None
    robustness: dict[str, Any] | None = None
    windows: list[WalkForwardWindow] = field(default_factory=list)
    #: True when the history was generated rather than recorded. Carried all the
    #: way through so a caller cannot mistake one for the other.
    synthetic: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "market_type": self.market_type.value,
            "backtest": self.backtest,
            "oos": self.oos,
            "walk_forward": self.walk_forward,
            "robustness": self.robustness,
            "windows": [w.to_dict() for w in self.windows],
            "synthetic": self.synthetic,
            "notes": list(self.notes),
        }

    def gate_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :func:`kat.discovery.lifecycle.evaluate_promotion`."""
        return {
            "backtest_metrics": self.backtest,
            "oos_metrics": self.oos,
            "walk_forward_metrics": self.walk_forward,
            "robustness_metrics": self.robustness,
        }


def validate(
    events: Sequence[SimEvent],
    factory: StrategyFactory,
    config: BacktestConfig,
    *,
    strategy_id: str,
    perturbed_factory: PerturbedFactory | None = None,
    train_fraction: float = 0.7,
    walk_forward_windows: int = 5,
    monte_carlo_runs: int = 500,
    seed: int = 0,
    **engine_kwargs: Any,
) -> ValidationReport:
    """Run the full protocol and return every block the gates need.

    The in-sample and out-of-sample spans are cut once, chronologically, and the
    same cut is used for both the backtest block and the degradation gate that
    compares them. Doing the cut here rather than at each call site is what
    guarantees the two numbers being compared came from disjoint periods.
    """
    ordered = sorted(events, key=lambda e: (e.available_ms, e.seq))
    report = ValidationReport(
        strategy_id=strategy_id,
        market_type=config.market_type,
        synthetic=bool(ordered) and EventFeed(ordered).is_synthetic,
    )
    if report.synthetic:
        report.notes.append(
            "history is generated, not recorded: these numbers show the pipeline "
            "ran, not that an edge exists"
        )

    in_sample, out_of_sample = split_chronologically(
        ordered, (train_fraction, 1.0 - train_fraction)
    )

    backtest = run_once(in_sample, factory, config, label="in_sample", **engine_kwargs)
    if backtest is None:
        report.notes.append("in-sample span was too short to run")
        return report
    report.backtest = gate_metrics(backtest)

    oos = run_once(out_of_sample, factory, config, label="out_of_sample", **engine_kwargs)
    if oos is None:
        report.notes.append("out-of-sample span was too short to run")
    else:
        report.oos = gate_metrics(oos)

    block, windows = walk_forward(
        ordered, factory, config,
        windows=walk_forward_windows, **engine_kwargs,
    )
    report.walk_forward = block
    report.windows = windows

    # Robustness is measured on the out-of-sample trades where they exist. Using
    # the in-sample trades would ask "is the fitted result stable", which is the
    # wrong question -- a thoroughly overfitted strategy is perfectly stable on
    # the data it was fitted to.
    robustness_source = oos if oos is not None and oos.trades else backtest
    robustness = monte_carlo_sharpe(robustness_source, runs=monte_carlo_runs, seed=seed)

    if perturbed_factory is not None:
        robustness.update(
            perturbation_ratio(
                in_sample, perturbed_factory, config,
                baseline_sharpe=backtest.metrics.sharpe, **engine_kwargs,
            )
        )
    else:
        robustness["perturbation_sharpe_ratio"] = None
        robustness["perturbation_detail"] = (
            "no perturbed factory was supplied, so parameter sensitivity was not "
            "measured; the robustness gate will refuse promotion"
        )
    report.robustness = robustness

    log.info(
        "validation.complete",
        strategy_id=strategy_id,
        synthetic=report.synthetic,
        backtest_trades=report.backtest.get("trades"),
        oos_trades=(report.oos or {}).get("trades"),
        walk_forward_windows=block.get("windows"),
        positive_windows=block.get("positive_windows"),
    )
    return report
