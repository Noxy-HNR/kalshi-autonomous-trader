"""Strategy lifecycle gates.

Mode D -- the recommended default -- means a strategy may be discovered,
optimised and validated entirely autonomously, but **cannot reach live capital
without clearing every gate**. This module is where that promise is kept.

The design principle throughout: a gate answers "is there enough evidence?", not
"does this look good?". Those are different questions, and conflating them is how
a research pipeline promotes noise. Concretely:

* A gate that cannot be evaluated **fails**. Missing evidence is not favourable
  evidence.
* Sample size is checked before performance, because a Sharpe of 3 over twelve
  trades is not a Sharpe of 3.
* The in-sample-to-out-of-sample degradation gate catches the specific failure
  that every other gate misses: a strategy that genuinely worked in the period it
  was fitted to and nowhere else.
* Promotion moves one rung. There is no path that skips from ``BACKTEST`` to
  ``LIVE`` however good the backtest is.

Demotion is always permitted and never gated. Pausing a strategy that is
misbehaving should never require evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from kat.common.enums import StrategyState
from kat.config.settings import ResearchSettings
from kat.observability.logging import get_logger
from kat.strategies.base import next_promotion_state

log = get_logger(__name__)

__all__ = ["GateResult", "PromotionDecision", "evaluate_promotion", "may_demote"]


@dataclass(frozen=True, slots=True)
class GateResult:
    """One gate and whether the evidence clears it."""

    name: str
    passed: bool
    requirement: str
    observed: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "requirement": self.requirement,
            "observed": self.observed,
            "detail": self.detail,
        }


@dataclass(slots=True)
class PromotionDecision:
    """Whether a strategy may move up one rung, and exactly why not."""

    strategy_id: str
    current_state: StrategyState
    target_state: StrategyState | None
    promoted: bool
    gates: list[GateResult] = field(default_factory=list)
    reason: str = ""

    @property
    def failures(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "from": self.current_state.value,
            "to": self.target_state.value if self.target_state else None,
            "promoted": self.promoted,
            "reason": self.reason,
            "gates": [g.to_dict() for g in self.gates],
        }


def _metric(block: dict[str, Any] | None, key: str) -> float | None:
    """Read a metric, treating absent and non-finite as absent."""
    if not block:
        return None
    value = block.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _present(name: str, block: dict[str, Any] | None, stage: str) -> GateResult:
    """The stage ran at all."""
    return GateResult(
        name=name,
        passed=bool(block),
        requirement=f"{stage} results must exist",
        observed="present" if block else "absent",
        detail="" if block else f"{stage} has not been run",
    )


def _sample_size(block: dict[str, Any] | None, minimum: int, stage: str) -> GateResult:
    trades = _metric(block, "trades")
    return GateResult(
        name=f"{stage}_sample_size",
        passed=trades is not None and trades >= minimum,
        requirement=f">= {minimum} trades",
        observed=f"{int(trades)}" if trades is not None else "unknown",
        detail=(
            ""
            if trades is not None and trades >= minimum
            else "below this, none of the performance numbers mean anything"
        ),
    )


def _sharpe(block: dict[str, Any] | None, minimum: Decimal, stage: str) -> GateResult:
    sharpe = _metric(block, "sharpe")
    return GateResult(
        name=f"{stage}_sharpe",
        passed=sharpe is not None and sharpe >= float(minimum),
        requirement=f">= {minimum}",
        observed=f"{sharpe:.4f}" if sharpe is not None else "unknown",
    )


def _drawdown(block: dict[str, Any] | None, maximum: Decimal, stage: str) -> GateResult:
    drawdown = _metric(block, "max_drawdown_pct")
    return GateResult(
        name=f"{stage}_drawdown",
        passed=drawdown is not None and drawdown <= float(maximum),
        requirement=f"<= {maximum}",
        observed=f"{drawdown:.4f}" if drawdown is not None else "unknown",
    )


def _degradation(
    in_sample: dict[str, Any] | None,
    out_of_sample: dict[str, Any] | None,
    maximum: Decimal,
) -> GateResult:
    """How much of the in-sample edge survived out of sample.

    This is the gate that catches the failure the others miss: a strategy whose
    backtest and out-of-sample numbers are each individually acceptable, but
    where the second is a small fraction of the first. That pattern means the
    edge was mostly specific to the fitting period, and it will keep decaying.
    """
    train = _metric(in_sample, "sharpe")
    test = _metric(out_of_sample, "sharpe")
    if train is None or test is None:
        return GateResult(
            name="degradation",
            passed=False,
            requirement=f"out-of-sample Sharpe within {maximum:.0%} decay of in-sample",
            observed="unknown",
            detail="both in-sample and out-of-sample Sharpe are needed to measure decay",
        )
    if train <= 0:
        # Nothing to decay from; the out-of-sample gate carries this case.
        return GateResult(
            name="degradation",
            passed=True,
            requirement="in-sample Sharpe positive to measure decay",
            observed=f"in-sample {train:.4f}",
            detail="in-sample edge was not positive, so decay is not meaningful",
        )
    decay = (train - test) / train
    return GateResult(
        name="degradation",
        passed=decay <= float(maximum),
        requirement=f"Sharpe decay <= {float(maximum):.0%}",
        observed=f"{decay:.1%} (in-sample {train:.3f} -> out-of-sample {test:.3f})",
        detail=(
            ""
            if decay <= float(maximum)
            else "most of the edge did not survive out of sample; it was fitted to the period"
        ),
    )


def _windows(block: dict[str, Any] | None, minimum: int) -> GateResult:
    windows = _metric(block, "windows")
    return GateResult(
        name="walk_forward_windows",
        passed=windows is not None and windows >= minimum,
        requirement=f">= {minimum} rolling windows",
        observed=f"{int(windows)}" if windows is not None else "unknown",
    )


def _consistency(block: dict[str, Any] | None) -> GateResult:
    """Most walk-forward windows should be profitable, not just the average.

    A strategy whose average is carried by one extraordinary window has not
    demonstrated a repeatable edge; it has demonstrated one good period.
    """
    positive = _metric(block, "positive_windows")
    total = _metric(block, "windows")
    if positive is None or total is None or total <= 0:
        return GateResult(
            name="walk_forward_consistency",
            passed=False,
            requirement="majority of windows profitable",
            observed="unknown",
        )
    ratio = positive / total
    return GateResult(
        name="walk_forward_consistency",
        passed=ratio > 0.5,
        requirement="> 50% of windows profitable",
        observed=f"{int(positive)}/{int(total)} ({ratio:.0%})",
        detail=(
            ""
            if ratio > 0.5
            else "the average is carried by a minority of windows, not a repeatable edge"
        ),
    )


def _robustness(block: dict[str, Any] | None) -> list[GateResult]:
    """Monte Carlo and parameter perturbation."""
    mc = _metric(block, "monte_carlo_p5_sharpe")
    perturbation = _metric(block, "perturbation_sharpe_ratio")
    return [
        GateResult(
            name="monte_carlo",
            passed=mc is not None and mc > 0,
            requirement="5th-percentile resampled Sharpe > 0",
            observed=f"{mc:.4f}" if mc is not None else "unknown",
            detail=(
                ""
                if mc is not None and mc > 0
                else "a plausible reordering of the same trades loses money"
            ),
        ),
        GateResult(
            name="parameter_perturbation",
            passed=perturbation is not None and perturbation >= 0.5,
            requirement="Sharpe retains >= 50% under a 10% parameter shift",
            observed=f"{perturbation:.2f}" if perturbation is not None else "unknown",
            detail=(
                ""
                if perturbation is not None and perturbation >= 0.5
                else "performance collapses under a small parameter change; the "
                "parameters were fitted to the sample"
            ),
        ),
    ]


def evaluate_promotion(
    *,
    strategy_id: str,
    current_state: StrategyState,
    settings: ResearchSettings,
    backtest_metrics: dict[str, Any] | None = None,
    oos_metrics: dict[str, Any] | None = None,
    walk_forward_metrics: dict[str, Any] | None = None,
    robustness_metrics: dict[str, Any] | None = None,
    paper_metrics: dict[str, Any] | None = None,
    live_metrics: dict[str, Any] | None = None,
) -> PromotionDecision:
    """Decide whether a strategy may move up exactly one rung.

    Returns a decision carrying every gate that was evaluated, passing or
    failing, so a refusal is explainable without re-running anything.
    """
    target = next_promotion_state(current_state)
    if target is None:
        return PromotionDecision(
            strategy_id=strategy_id,
            current_state=current_state,
            target_state=None,
            promoted=False,
            reason=(
                f"{current_state.value} has no next rung; promotion out of this "
                "state is an explicit transition, not an automatic one"
            ),
        )

    gates: list[GateResult] = []

    # Each target rung demands the evidence produced by the rung below it.
    if target is StrategyState.OUT_OF_SAMPLE:
        gates.append(_present("backtest_present", backtest_metrics, "backtest"))
        gates.append(_sample_size(backtest_metrics, settings.min_trades_for_promotion, "backtest"))
        expectancy = _metric(backtest_metrics, "expectancy_micros")
        gates.append(
            GateResult(
                name="backtest_expectancy",
                passed=expectancy is not None and expectancy > 0,
                requirement="positive expectancy after realistic costs",
                observed=f"{expectancy:.0f} micros" if expectancy is not None else "unknown",
            )
        )

    elif target is StrategyState.WALK_FORWARD:
        gates.append(_present("oos_present", oos_metrics, "out-of-sample"))
        gates.append(_sample_size(oos_metrics, settings.min_trades_for_promotion, "oos"))
        gates.append(_sharpe(oos_metrics, settings.min_oos_sharpe, "oos"))
        gates.append(_drawdown(oos_metrics, settings.max_oos_drawdown_pct, "oos"))
        gates.append(
            _degradation(backtest_metrics, oos_metrics, settings.max_is_oos_sharpe_degradation)
        )

    elif target is StrategyState.ROBUSTNESS:
        gates.append(_present("walk_forward_present", walk_forward_metrics, "walk-forward"))
        gates.append(_windows(walk_forward_metrics, settings.min_walk_forward_windows))
        gates.append(_consistency(walk_forward_metrics))
        gates.append(_sharpe(walk_forward_metrics, settings.min_oos_sharpe, "walk_forward"))

    elif target is StrategyState.PAPER:
        gates.append(_present("robustness_present", robustness_metrics, "robustness"))
        gates.extend(_robustness(robustness_metrics))

    elif target is StrategyState.PROBATION:
        gates.append(_present("paper_present", paper_metrics, "paper trading"))
        gates.append(_sample_size(paper_metrics, settings.min_paper_trades_for_live, "paper"))
        gates.append(_sharpe(paper_metrics, settings.min_oos_sharpe, "paper"))
        gates.append(
            _degradation(oos_metrics, paper_metrics, settings.max_is_oos_sharpe_degradation)
        )

    elif target is StrategyState.LIVE:
        # Probation already trades real money at a reduced size, so the evidence
        # for full allocation is its own live record, not the paper record.
        gates.append(_present("probation_present", live_metrics, "probation"))
        gates.append(_sample_size(live_metrics, settings.min_trades_for_promotion, "probation"))
        gates.append(_sharpe(live_metrics, settings.min_oos_sharpe, "probation"))
        gates.append(_drawdown(live_metrics, settings.max_oos_drawdown_pct, "probation"))

    else:
        # Early rungs (DEVELOPMENT, BACKTEST) are reached by work being done
        # rather than by evidence clearing a bar.
        gates.append(
            GateResult(
                name="stage_work_complete",
                passed=True,
                requirement=f"work for {target.value} has been performed",
                observed="assumed complete by the caller",
            )
        )

    promoted = bool(gates) and all(g.passed for g in gates)
    failures = [g for g in gates if not g.passed]
    decision = PromotionDecision(
        strategy_id=strategy_id,
        current_state=current_state,
        target_state=target,
        promoted=promoted,
        gates=gates,
        reason=(
            f"all {len(gates)} gates cleared for {target.value}"
            if promoted
            else "; ".join(f"{g.name}: {g.observed} (needs {g.requirement})" for g in failures)
        ),
    )

    log_fn = log.info if promoted else log.warning
    log_fn(
        "lifecycle.promotion_evaluated",
        strategy_id=strategy_id,
        **{"from": current_state.value},
        to=target.value,
        promoted=promoted,
        failed_gates=[g.name for g in failures],
    )
    return decision


def may_demote(current_state: StrategyState, target: StrategyState) -> bool:
    """Whether a demotion is allowed. It always is.

    Pausing or retiring a misbehaving strategy must never require evidence: the
    cost of stopping something that was fine is an opportunity, and the cost of
    not stopping something that is not is money.
    """
    return target in (StrategyState.PAUSED, StrategyState.RETIRED, StrategyState.REVIEW)
