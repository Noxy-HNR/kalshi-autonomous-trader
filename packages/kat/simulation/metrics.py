"""Performance measurement, with the uncertainty attached.

A backtest that reports "Sharpe 1.8" from forty trades has reported a number
whose sampling error is larger than the number. Every headline statistic here
therefore comes with a bootstrap confidence interval, and the interval is part
of the metric rather than an optional extra: a promotion gate that reads only
the point estimate is a gate that promotes noise.

What the bootstrap here does and does not do
--------------------------------------------
:func:`bootstrap_ci` is the percentile bootstrap. It resamples the observed
series and takes quantiles of the resulting statistic. Two limitations are worth
stating rather than hiding:

* It is not bias-corrected (no BCa). For a skewed statistic on a short series
  the interval is approximately, not exactly, at the nominal level.
* Financial returns are serially correlated, and an i.i.d. resample of them
  produces an interval that is too narrow. The Sharpe interval therefore uses a
  **moving-block** bootstrap whose block length defaults to ``sqrt(n)``, which
  preserves short-range dependence. Trade profits are resampled i.i.d., which is
  the right model for "was the average edge real" and the wrong one for "did the
  order of trades matter" -- the second question belongs to
  :mod:`kat.simulation.validation`, which answers it by permutation.

Calibration, because these are probability markets
--------------------------------------------------
On Kalshi the price *is* a probability, so a model can be profitable while being
badly calibrated, and it can be well calibrated while losing money to costs.
They are separate properties and both are measured: the Brier score and its
decomposition into a reliability curve say whether a stated 70% happened about
70% of the time, which no profit-and-loss statistic can tell you.

Calibration inputs from a backtest are *decision-time* predictions and are
therefore serially correlated -- a market predicted every five seconds for an
hour contributes hundreds of near-identical rows. The Brier score is a
diagnostic under those conditions, not an independent-sample statistic, and the
bin counts are reported so a reader can see how few genuinely independent
observations there are.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from kat.common.fixedpoint import micros_to_decimal

__all__ = [
    "MS_PER_YEAR",
    "BacktestMetrics",
    "Calibration",
    "ConfidenceInterval",
    "ReliabilityBin",
    "TradeLike",
    "bootstrap_ci",
    "brier_score",
    "calibration",
    "compute_metrics",
    "max_drawdown",
    "returns_from_equity",
    "sharpe_ratio",
    "sortino_ratio",
]

#: Julian year in milliseconds; the annualisation base for every rate here.
MS_PER_YEAR: float = 365.25 * 24 * 60 * 60 * 1000


class TradeLike(Protocol):
    """The parts of a closed trade this module measures.

    A protocol rather than an import so that the metrics module has no
    dependency on the engine, which lets a caller score trades reconstructed
    from the database exactly as it scores trades from a live run.
    """

    net_pnl_micros: int
    gross_pnl_micros: int
    fees_micros: int
    cost_micros: int

    @property
    def holding_ms(self) -> int: ...

    @property
    def is_win(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A point estimate and the range the data can actually support."""

    point: float
    low: float
    high: float
    alpha: float
    resamples: int
    block_size: int = 1

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def excludes_zero(self) -> bool:
        """True when the whole interval is on one side of zero.

        The only form in which a Sharpe ratio or an expectancy is evidence of
        anything.
        """
        return self.low > 0.0 or self.high < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": round(self.point, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
            "confidence": round(1 - self.alpha, 4),
            "resamples": self.resamples,
            "block_size": self.block_size,
            "excludes_zero": self.excludes_zero,
        }


def _as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def returns_from_equity(equity: Sequence[int]) -> np.ndarray:
    """Simple period returns from an equity series in micro-dollars.

    A period in which equity was zero or negative yields a zero return rather
    than a division by zero: an account that is wiped out has no further return
    to measure, and propagating a NaN would poison every statistic downstream.
    """
    arr = _as_array(equity)
    if arr.size < 2:
        return np.zeros(0, dtype=float)
    prev = arr[:-1]
    curr = arr[1:]
    out = np.zeros(prev.size, dtype=float)
    usable = prev > 0
    out[usable] = (curr[usable] - prev[usable]) / prev[usable]
    return out


def sharpe_ratio(returns: Sequence[float] | np.ndarray, *, periods_per_year: float) -> float:
    """Annualised Sharpe. Zero when the series has no dispersion to speak of."""
    arr = _as_array(np.asarray(returns))
    if arr.size < 2:
        return 0.0
    sd = float(arr.std(ddof=1))
    if sd <= 0.0 or not math.isfinite(sd):
        return 0.0
    return float(arr.mean() / sd * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: Sequence[float] | np.ndarray, *, periods_per_year: float, mar: float = 0.0
) -> float:
    """Annualised Sortino: dispersion measured only below the target return.

    Zero downside deviation returns 0.0 rather than infinity. A strategy that
    has never had a losing period has not demonstrated an infinite Sortino, it
    has demonstrated a short sample.
    """
    arr = _as_array(np.asarray(returns))
    if arr.size < 2:
        return 0.0
    downside = np.minimum(arr - mar, 0.0)
    dd = float(math.sqrt(float((downside**2).mean())))
    if dd <= 0.0:
        return 0.0
    return float((arr.mean() - mar) / dd * math.sqrt(periods_per_year))


def max_drawdown(equity: Sequence[int]) -> tuple[float, int]:
    """Largest peak-to-trough decline as ``(fraction, micros)``."""
    arr = _as_array(equity)
    if arr.size == 0:
        return 0.0, 0
    peaks = np.maximum.accumulate(arr)
    drops = peaks - arr
    idx = int(np.argmax(drops))
    worst_micros = int(drops[idx])
    peak = float(peaks[idx])
    fraction = worst_micros / peak if peak > 0 else 0.0
    return fraction, worst_micros


def _resample_indices(
    rng: np.random.Generator, n: int, block_size: int
) -> np.ndarray:
    if block_size <= 1:
        return rng.integers(0, n, size=n)
    starts = rng.integers(0, n, size=math.ceil(n / block_size))
    idx = (starts[:, None] + np.arange(block_size)[None, :]) % n
    return idx.reshape(-1)[:n]


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic,
    *,
    resamples: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
    block_size: int = 1,
) -> ConfidenceInterval | None:
    """Percentile bootstrap interval for ``statistic`` over ``values``.

    ``block_size`` > 1 selects a moving-block bootstrap, which is what serially
    correlated series need. Returns ``None`` for a sample too small to resample
    meaningfully; a two-point confidence interval is not a confidence interval.
    """
    arr = _as_array(np.asarray(values))
    if arr.size < 3:
        return None
    rng = np.random.default_rng(seed)
    block = max(1, min(int(block_size), arr.size))
    draws = np.empty(resamples, dtype=float)
    for i in range(resamples):
        sample = arr[_resample_indices(rng, arr.size, block)]
        draws[i] = statistic(sample)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return None
    low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    return ConfidenceInterval(
        point=float(statistic(arr)),
        low=float(low),
        high=float(high),
        alpha=alpha,
        resamples=resamples,
        block_size=block,
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    low: float
    high: float
    count: int
    mean_predicted: float
    observed_frequency: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "count": self.count,
            "mean_predicted": round(self.mean_predicted, 4),
            "observed_frequency": round(self.observed_frequency, 4),
            "gap": round(self.observed_frequency - self.mean_predicted, 4),
        }


@dataclass(frozen=True, slots=True)
class Calibration:
    """How well stated probabilities matched what happened."""

    count: int
    brier: float
    base_rate: float
    #: 1 - brier / brier_of_always_predicting_the_base_rate. Negative means the
    #: predictions are worse than knowing nothing but the base rate.
    skill_score: float
    expected_calibration_error: float
    bins: tuple[ReliabilityBin, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "brier": round(self.brier, 6),
            "base_rate": round(self.base_rate, 4),
            "skill_score": round(self.skill_score, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "reliability": [b.to_dict() for b in self.bins],
        }


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error of probabilistic predictions. Lower is better."""
    p = _as_array(probabilities)
    y = _as_array(outcomes)
    if p.size == 0 or p.size != y.size:
        return float("nan")
    return float(((p - y) ** 2).mean())


def calibration(
    probabilities: Sequence[float], outcomes: Sequence[int], *, bins: int = 10
) -> Calibration | None:
    """Brier score plus a reliability curve, or ``None`` without observations.

    ``None`` rather than a zero: no settled market means no evidence about
    calibration, and a zero would read as perfect.
    """
    p = _as_array(probabilities)
    y = _as_array(outcomes)
    if p.size == 0 or p.size != y.size:
        return None

    brier = float(((p - y) ** 2).mean())
    base = float(y.mean())
    base_brier = base * (1 - base)
    skill = 0.0 if base_brier <= 0 else 1.0 - brier / base_brier

    edges = np.linspace(0.0, 1.0, bins + 1)
    # ``right=True`` on the last bin so a prediction of exactly 1.0 lands in it.
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    out: list[ReliabilityBin] = []
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(p[mask].mean())
        observed = float(y[mask].mean())
        ece += (count / p.size) * abs(observed - mean_pred)
        out.append(
            ReliabilityBin(
                low=float(edges[b]),
                high=float(edges[b + 1]),
                count=count,
                mean_predicted=mean_pred,
                observed_frequency=observed,
            )
        )
    return Calibration(
        count=int(p.size),
        brier=brier,
        base_rate=base,
        skill_score=skill,
        expected_calibration_error=ece,
        bins=tuple(out),
    )


# ---------------------------------------------------------------------------
# The headline metric set
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Everything one run measured, with intervals where they are meaningful."""

    # -- sample --------------------------------------------------------------
    periods: int
    period_ms: int
    duration_days: float
    trade_count: int
    fill_count: int

    # -- return --------------------------------------------------------------
    initial_equity_micros: int
    final_equity_micros: int
    total_return: float
    annualised_return: float

    # -- risk-adjusted -------------------------------------------------------
    sharpe: float
    sharpe_ci: ConfidenceInterval | None
    sortino: float
    max_drawdown_pct: float
    max_drawdown_micros: int
    calmar: float

    # -- trades --------------------------------------------------------------
    win_rate: float
    expectancy_micros: int
    expectancy_ci: ConfidenceInterval | None
    profit_factor: float | None
    avg_win_micros: int
    avg_loss_micros: int
    avg_holding_ms: int
    largest_loss_micros: int

    # -- costs ---------------------------------------------------------------
    total_fees_micros: int
    avg_slippage_cc: float
    maker_fill_pct: float
    #: Fees as a fraction of gross profit. Above 1.0 means the strategy is a
    #: fee-generation programme that happens to also trade.
    fee_drag: float | None

    # -- deployment ----------------------------------------------------------
    avg_exposure_pct: float
    max_exposure_pct: float

    # -- probability quality -------------------------------------------------
    calibration: Calibration | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "periods": self.periods,
            "period_ms": self.period_ms,
            "duration_days": round(self.duration_days, 4),
            "trade_count": self.trade_count,
            "fill_count": self.fill_count,
            "initial_equity_usd": str(micros_to_decimal(self.initial_equity_micros)),
            "final_equity_usd": str(micros_to_decimal(self.final_equity_micros)),
            "total_return": round(self.total_return, 6),
            "annualised_return": round(self.annualised_return, 6),
            "sharpe": round(self.sharpe, 4),
            "sharpe_ci": self.sharpe_ci.to_dict() if self.sharpe_ci else None,
            "sortino": round(self.sortino, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 6),
            "max_drawdown_usd": str(micros_to_decimal(self.max_drawdown_micros)),
            "calmar": round(self.calmar, 4),
            "win_rate": round(self.win_rate, 4),
            "expectancy_usd": str(micros_to_decimal(self.expectancy_micros)),
            "expectancy_ci_micros": self.expectancy_ci.to_dict() if self.expectancy_ci else None,
            "profit_factor": None if self.profit_factor is None else round(self.profit_factor, 4),
            "avg_win_usd": str(micros_to_decimal(self.avg_win_micros)),
            "avg_loss_usd": str(micros_to_decimal(self.avg_loss_micros)),
            "largest_loss_usd": str(micros_to_decimal(self.largest_loss_micros)),
            "avg_holding_ms": self.avg_holding_ms,
            "total_fees_usd": str(micros_to_decimal(self.total_fees_micros)),
            "avg_slippage_cc": round(self.avg_slippage_cc, 4),
            "maker_fill_pct": round(self.maker_fill_pct, 4),
            "fee_drag": None if self.fee_drag is None else round(self.fee_drag, 4),
            "avg_exposure_pct": round(self.avg_exposure_pct, 4),
            "max_exposure_pct": round(self.max_exposure_pct, 4),
            "calibration": self.calibration.to_dict() if self.calibration else None,
        }


def _time_weighted_exposure(
    exposure: Sequence[tuple[int, int, int]]
) -> tuple[float, float]:
    """``(average, maximum)`` exposure as a fraction of equity, weighted by time."""
    if len(exposure) < 2:
        if not exposure:
            return 0.0, 0.0
        _, exp, eq = exposure[0]
        ratio = exp / eq if eq > 0 else 0.0
        return ratio, ratio
    total_ms = 0
    weighted = 0.0
    peak = 0.0
    for (t0, exp0, eq0), (t1, _, _) in itertools.pairwise(exposure):
        dt = max(0, t1 - t0)
        ratio = exp0 / eq0 if eq0 > 0 else 0.0
        peak = max(peak, ratio)
        weighted += ratio * dt
        total_ms += dt
    _, last_exp, last_eq = exposure[-1]
    last_ratio = last_exp / last_eq if last_eq > 0 else 0.0
    peak = max(peak, last_ratio)
    return (weighted / total_ms if total_ms > 0 else last_ratio), peak


def compute_metrics(
    *,
    equity_curve: Sequence[tuple[int, int]],
    trades: Sequence[TradeLike],
    period_ms: int,
    exposure: Sequence[tuple[int, int, int]] = (),
    avg_slippage_cc: float = 0.0,
    fill_count: int = 0,
    maker_fill_pct: float = 0.0,
    total_fees_micros: int = 0,
    probabilities: Sequence[float] = (),
    outcomes: Sequence[int] = (),
    resamples: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BacktestMetrics:
    """Score one run. Pure function of the run's own output.

    ``equity_curve`` is ``(ts_ms, equity_micros)`` sampled on a fixed period;
    ``exposure`` is ``(ts_ms, exposure_micros, equity_micros)`` on the same
    samples. Both come from the engine, but nothing here requires that -- the
    same function scores a paper-trading or live equity curve, which is what
    makes backtest and live numbers comparable.
    """
    equity = [e for _, e in equity_curve]
    timestamps = [t for t, _ in equity_curve]
    initial = equity[0] if equity else 0
    final = equity[-1] if equity else 0

    duration_ms = (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0
    duration_days = duration_ms / 86_400_000
    periods_per_year = MS_PER_YEAR / period_ms if period_ms > 0 else 0.0

    returns = returns_from_equity(equity)
    sharpe = sharpe_ratio(returns, periods_per_year=periods_per_year)
    # Serially correlated series need a block bootstrap; sqrt(n) is the usual
    # rule of thumb for the block length and is honest about being one.
    block = max(1, int(math.sqrt(returns.size))) if returns.size else 1
    sharpe_ci = bootstrap_ci(
        returns,
        lambda sample: sharpe_ratio(sample, periods_per_year=periods_per_year),
        resamples=resamples,
        alpha=alpha,
        seed=seed,
        block_size=block,
    )

    total_return = (final - initial) / initial if initial > 0 else 0.0
    if initial > 0 and duration_ms > 0 and final > 0:
        years = duration_ms / MS_PER_YEAR
        annualised = (final / initial) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    else:
        annualised = 0.0

    dd_pct, dd_micros = max_drawdown(equity)
    calmar = annualised / dd_pct if dd_pct > 0 else 0.0

    pnls = [t.net_pnl_micros for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    expectancy = int(sum(pnls) / len(pnls)) if pnls else 0
    expectancy_ci = bootstrap_ci(
        pnls, lambda sample: float(sample.mean()), resamples=resamples, alpha=alpha, seed=seed + 1
    )
    loss_total = abs(sum(losses))
    profit_factor = (sum(wins) / loss_total) if loss_total > 0 else None
    gross_profit = sum(t.gross_pnl_micros for t in trades)
    fee_drag = (total_fees_micros / gross_profit) if gross_profit > 0 else None

    avg_exposure, max_exposure = _time_weighted_exposure(exposure)

    return BacktestMetrics(
        periods=len(equity),
        period_ms=period_ms,
        duration_days=duration_days,
        trade_count=len(pnls),
        fill_count=fill_count,
        initial_equity_micros=initial,
        final_equity_micros=final,
        total_return=total_return,
        annualised_return=annualised,
        sharpe=sharpe,
        sharpe_ci=sharpe_ci,
        sortino=sortino_ratio(returns, periods_per_year=periods_per_year),
        max_drawdown_pct=dd_pct,
        max_drawdown_micros=dd_micros,
        calmar=calmar,
        win_rate=win_rate,
        expectancy_micros=expectancy,
        expectancy_ci=expectancy_ci,
        profit_factor=profit_factor,
        avg_win_micros=int(sum(wins) / len(wins)) if wins else 0,
        avg_loss_micros=int(sum(losses) / len(losses)) if losses else 0,
        avg_holding_ms=int(sum(t.holding_ms for t in trades) / len(trades)) if trades else 0,
        largest_loss_micros=min(pnls) if pnls else 0,
        total_fees_micros=total_fees_micros,
        avg_slippage_cc=avg_slippage_cc,
        maker_fill_pct=maker_fill_pct,
        fee_drag=fee_drag,
        avg_exposure_pct=avg_exposure,
        max_exposure_pct=max_exposure,
        calibration=calibration(probabilities, outcomes),
    )
