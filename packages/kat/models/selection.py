"""Model selection.

One rule, and it is the entire module: **selection is on out-of-sample
performance only.** Training performance is never a selection criterion. It is
recorded, reported and used to *disqualify* -- a large train-to-test gap is the
overfitting signal -- but a model can never win by fitting its training data
better.

Why a composite rather than a single metric
-------------------------------------------
AUC alone rewards a model that ranks well but reports nonsense probabilities,
which is useless when the number is going to be compared against a market price.
Brier alone conflates discrimination with calibration. The score used here is a
weighted combination of both, plus an explicit calibration term, and any
candidate that fails a hard gate is excluded outright rather than being allowed
to compensate elsewhere.

The gates
---------
* **Discrimination.** Test AUC must beat a coin flip by a stated margin. A model
  that cannot rank cannot be fixed by calibration.
* **Calibration.** Test ECE must be tight enough that its error is smaller than
  the round-trip fee, or the "edge" it reports is an artefact of its own bias.
* **Overfitting.** The train-to-test AUC gap must be below a threshold. A model
  that scores 0.95 in training and 0.55 in test has memorised, and its test
  score is a lucky draw rather than an estimate.
* **Sample size.** Below a floor, none of the above numbers mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kat.models.trainers import TrainedModel
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["SelectionCriteria", "SelectionResult", "select_best"]


@dataclass(slots=True)
class SelectionCriteria:
    """Hard gates and score weights. Gates are pass/fail, not tradeable."""

    #: Test AUC must exceed this. 0.5 is a coin flip; the margin is the point.
    min_test_auc: float = 0.53
    #: **Excess** calibration error on test -- the part of the gap not explained
    #: by binomial sampling noise. Gating on raw ECE would be a test of sample
    #: size rather than of the model: a perfectly calibrated model shows a raw
    #: ECE near 0.029 at 1200 samples and 0.074 at 200. A systematic two-point
    #: error, by contrast, manufactures a two-cent edge on a $0.50 market and
    #: biases every trade the same way.
    max_test_excess_ece: float = 0.02
    #: Train AUC minus test AUC. Above this the model has memorised.
    max_overfitting_gap: float = 0.15
    #: Below this many test rows nothing here is measurable.
    min_test_samples: int = 200
    #: Brier must at least beat always predicting the base rate.
    require_beats_base_rate: bool = True

    # Score weights, applied only to candidates that already passed every gate.
    auc_weight: float = 0.5
    brier_weight: float = 0.3
    calibration_weight: float = 0.2


@dataclass(slots=True)
class Candidate:
    model: TrainedModel
    passed: bool
    score: float
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.model.name,
            "passed": self.passed,
            "score": round(self.score, 6),
            "failures": list(self.failures),
            # Always reported separately, never blended.
            "train_metrics": self.model.train_metrics,
            "validation_metrics": self.model.validation_metrics,
            "test_metrics": self.model.test_metrics,
            "overfitting_gap": self.model.overfitting_gap,
            "calibration": self.model.calibration,
        }


@dataclass(slots=True)
class SelectionResult:
    """Which model won, which did not, and exactly why."""

    best: TrainedModel | None
    candidates: list[Candidate] = field(default_factory=list)
    reason: str = ""

    @property
    def deployable(self) -> bool:
        return self.best is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.best.name if self.best else None,
            "deployable": self.deployable,
            "reason": self.reason,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def _base_rate_brier(metrics: dict[str, float]) -> float:
    """Brier score of always predicting the base rate.

    This is the number a model has to beat to have contributed anything at all:
    predicting the unconditional frequency requires no model.
    """
    base = metrics.get("base_rate")
    if base is None or not np.isfinite(base):
        return float("nan")
    return float(base * (1 - base))


def _evaluate(model: TrainedModel, criteria: SelectionCriteria) -> Candidate:
    failures: list[str] = []

    if model.failed is not None:
        return Candidate(model, False, 0.0, [f"training failed: {model.failed}"])

    test = model.test_metrics
    n = int(test.get("n", 0))
    if n < criteria.min_test_samples:
        failures.append(f"only {n} test rows, need {criteria.min_test_samples}")

    auc = test.get("auc")
    if auc is None:
        failures.append("test AUC is undefined (single-class test split)")
    elif auc < criteria.min_test_auc:
        failures.append(f"test AUC {auc:.4f} below {criteria.min_test_auc}")

    ece = test.get("ece")
    null_ece = test.get("null_ece")
    excess_ece = test.get("excess_ece")
    if excess_ece is None and ece is not None and null_ece is not None:
        excess_ece = max(0.0, float(ece) - float(null_ece))
    if excess_ece is None or not np.isfinite(excess_ece):
        failures.append("test calibration could not be measured")
    elif excess_ece > criteria.max_test_excess_ece:
        failures.append(
            f"excess calibration error {excess_ece:.4f} above "
            f"{criteria.max_test_excess_ece} (raw ECE {ece:.4f}, noise floor "
            f"{null_ece:.4f} at n={n})"
        )

    gap = model.overfitting_gap
    if gap is not None and gap > criteria.max_overfitting_gap:
        failures.append(
            f"train-to-test AUC gap {gap:.4f} above {criteria.max_overfitting_gap}; "
            "the model has memorised its training data"
        )

    brier = test.get("brier")
    if criteria.require_beats_base_rate:
        baseline = _base_rate_brier(test)
        if brier is not None and np.isfinite(baseline) and brier >= baseline:
            failures.append(
                f"test Brier {brier:.4f} does not beat always predicting the "
                f"base rate ({baseline:.4f})"
            )

    if failures:
        return Candidate(model, False, 0.0, failures)

    # Composite score, all terms in [0, 1] and higher is better.
    auc_term = (float(auc) - 0.5) * 2 if auc is not None else 0.0
    brier_term = max(0.0, 1.0 - float(brier) / 0.25) if brier is not None else 0.0
    calib_term = (
        max(0.0, 1.0 - float(excess_ece) / criteria.max_test_excess_ece)
        if excess_ece is not None
        else 0.0
    )
    score = (
        criteria.auc_weight * auc_term
        + criteria.brier_weight * brier_term
        + criteria.calibration_weight * calib_term
    )
    return Candidate(model, True, float(score), [])


def select_best(
    models: list[TrainedModel], criteria: SelectionCriteria | None = None
) -> SelectionResult:
    """Pick the model to deploy, or refuse to pick one.

    Returning no model is a legitimate and common outcome. Deploying the least
    bad of several failing models is how a platform ends up trading on noise.
    """
    criteria = criteria or SelectionCriteria()
    candidates = [_evaluate(m, criteria) for m in models]
    passing = [c for c in candidates if c.passed]

    candidates.sort(key=lambda c: (-c.score, c.model.name))

    if not passing:
        result = SelectionResult(
            best=None,
            candidates=candidates,
            reason="no candidate cleared every out-of-sample gate",
        )
        log.warning(
            "model.selection_none",
            attempted=len(models),
            failures={c.model.name: c.failures for c in candidates},
        )
        return result

    winner = max(passing, key=lambda c: c.score)
    result = SelectionResult(
        best=winner.model,
        candidates=candidates,
        reason=(
            f"highest composite out-of-sample score ({winner.score:.4f}) among "
            f"{len(passing)} of {len(models)} candidates that cleared every gate"
        ),
    )
    log.info(
        "model.selected",
        family=winner.model.name,
        score=round(winner.score, 6),
        test_auc=winner.model.test_metrics.get("auc"),
        test_brier=winner.model.test_metrics.get("brier"),
        test_ece=winner.model.test_metrics.get("ece"),
        train_auc=winner.model.train_metrics.get("auc"),
        overfitting_gap=winner.model.overfitting_gap,
        rejected=[c.model.name for c in candidates if not c.passed],
    )
    return result
