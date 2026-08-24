"""Probability calibration.

These are probability markets. A model that is *accurate* but badly *calibrated*
is useless here, and the distinction is the whole reason this module exists.

Accuracy asks "was the model right?". Calibration asks "when the model said 70%,
did it happen 70% of the time?". A model with 65% accuracy that is perfectly
calibrated can be traded profitably, because you can compare its number directly
with the market price and size on the difference. A model with 80% accuracy that
systematically says 90% when it means 70% will lose money on every trade it is
most confident about, which is precisely where it will size largest.

Gradient-boosted trees are usually badly calibrated out of the box -- they push
probabilities toward the extremes -- so calibrating them is not optional here.

What is measured
----------------
``brier``
    Mean squared error of the probability against the outcome. Proper scoring
    rule: it is minimised only by reporting your true belief.
``ece``
    Expected calibration error -- the average gap between predicted probability
    and observed frequency, weighted by bin population. This is the number that
    says "your 70% bucket actually came in at 55%".
``mce``
    Maximum calibration error, the worst single bin. A small ECE can hide a
    badly broken tail, and the tails are where prediction markets pay.
``reliability``
    The full curve, so the shape of the miscalibration is visible rather than
    collapsed to one number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "CalibrationReport",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "brier_score",
    "expected_calibration_error",
    "null_calibration_error",
    "reliability_curve",
]


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and outcome.

    Lower is better; 0.25 is what you get by always predicting 0.5, so anything
    above that is worse than useless.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, *, bins: int = 10
) -> list[dict[str, float]]:
    """Observed frequency against predicted probability, per bin.

    Empty bins are omitted rather than reported as zero, because "no
    observations" and "never happened" are different facts and conflating them
    invents miscalibration that is not there.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return []

    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[dict[str, float]] = []
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        # Include the right edge only in the final bin so 1.0 is not dropped.
        mask = (y_prob >= low) & (y_prob < high) if i < bins - 1 else (y_prob >= low)
        count = int(mask.sum())
        if count == 0:
            continue
        out.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "count": count,
                "mean_predicted": float(np.mean(y_prob[mask])),
                "observed_frequency": float(np.mean(y_true[mask])),
            }
        )
    return out


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, bins: int = 10
) -> tuple[float, float]:
    """Return ``(ece, mce)``: the average and worst bin-level calibration gap."""
    curve = reliability_curve(y_true, y_prob, bins=bins)
    if not curve:
        return float("nan"), float("nan")
    total = sum(b["count"] for b in curve)
    gaps = [abs(b["mean_predicted"] - b["observed_frequency"]) for b in curve]
    weighted = sum(g * b["count"] for g, b in zip(gaps, curve, strict=True)) / total
    return float(weighted), float(max(gaps))


#: Mean absolute deviation of a standard normal, ``sqrt(2 / pi)``. Converts a
#: standard error into an expected absolute gap.
_MAD_OF_NORMAL = float(np.sqrt(2.0 / np.pi))


def null_calibration_error(y_prob: np.ndarray, *, bins: int = 10) -> float:
    """ECE a **perfectly calibrated** model would show at this sample size.

    This is the number a fixed ECE threshold forgets, and forgetting it makes the
    threshold meaningless. Observed frequency in a bin of ``n`` samples is a
    binomial estimate with standard error ``sqrt(p(1-p)/n)``, so even a model
    whose probabilities are exactly right shows a non-zero gap. Empirically that
    noise floor is about 0.074 at 200 samples, 0.029 at 1200 and 0.015 at 5000 --
    so a flat 0.05 gate rejects perfect calibration on any realistic validation
    set and accepts sloppy calibration on a huge one.

    Comparing measured ECE against *this* is what makes the calibration gate a
    test of the model rather than a test of the sample size.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0
    weighted = 0.0
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high) if i < bins - 1 else (y_prob >= low)
        count = int(mask.sum())
        if count == 0:
            continue
        p = float(np.mean(y_prob[mask]))
        standard_error = float(np.sqrt(max(p * (1.0 - p), 1e-12) / count))
        weighted += standard_error * _MAD_OF_NORMAL * count
        total += count
    if total == 0:
        return float("nan")
    return float(weighted / total)


@dataclass(slots=True)
class CalibrationReport:
    """Everything measured about one model's probability quality."""

    brier: float
    ece: float
    mce: float
    #: ECE a perfectly calibrated model would show at this sample size.
    null_ece: float = float("nan")
    reliability: list[dict[str, float]] = field(default_factory=list)
    n_samples: int = 0
    method: str = "none"

    @property
    def excess_ece(self) -> float:
        """Miscalibration beyond what sampling noise explains.

        This, not raw ECE, is the number with economic meaning: it is the part
        of the gap that would still be there with infinite data.
        """
        if not np.isfinite(self.ece) or not np.isfinite(self.null_ece):
            return float("nan")
        return max(0.0, self.ece - self.null_ece)

    @property
    def is_usable(self) -> bool:
        """Whether these probabilities can be compared to a market price.

        Gated on *excess* ECE. A systematic two-point error on a market priced at
        $0.50 manufactures a two-cent edge that does not exist, which is most of
        a round-trip fee -- so the model would trade constantly and lose
        constantly. Noise-driven error does not have that property: it averages
        out across trades rather than biasing every one of them the same way.
        """
        if not np.isfinite(self.brier) or not np.isfinite(self.excess_ece):
            return False
        return self.brier < 0.25 and self.excess_ece < 0.02 and self.n_samples >= 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "brier": round(self.brier, 6) if np.isfinite(self.brier) else None,
            "ece": round(self.ece, 6) if np.isfinite(self.ece) else None,
            "null_ece": round(self.null_ece, 6) if np.isfinite(self.null_ece) else None,
            "excess_ece": round(self.excess_ece, 6) if np.isfinite(self.excess_ece) else None,
            "mce": round(self.mce, 6) if np.isfinite(self.mce) else None,
            "n_samples": self.n_samples,
            "method": self.method,
            "usable": self.is_usable,
            "reliability": self.reliability,
        }


def assess(
    y_true: np.ndarray, y_prob: np.ndarray, *, method: str = "none", bins: int = 10
) -> CalibrationReport:
    """Measure calibration quality on a held-out set."""
    ece, mce = expected_calibration_error(y_true, y_prob, bins=bins)
    return CalibrationReport(
        brier=brier_score(y_true, y_prob),
        ece=ece,
        mce=mce,
        null_ece=null_calibration_error(y_prob, bins=bins),
        reliability=reliability_curve(y_true, y_prob, bins=bins),
        n_samples=int(np.asarray(y_true).size),
        method=method,
    )


class PlattCalibrator:
    """Logistic (Platt) scaling.

    Fits a one-dimensional logistic regression from the raw score to the
    outcome. Cheap, needs little data, and can only apply a monotone sigmoid
    correction -- which is the right tool when the miscalibration is a smooth
    over- or under-confidence rather than a strange shape.
    """

    def __init__(self) -> None:
        self._model: Any = None

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> PlattCalibrator:
        from sklearn.linear_model import LogisticRegression

        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        y_true = np.asarray(y_true, dtype=int)
        if len(np.unique(y_true)) < 2:
            # One-class data cannot calibrate anything; leaving the model unset
            # makes transform a no-op rather than producing a constant.
            log.warning("calibration.single_class", note="Platt scaling skipped")
            return self
        self._model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self._model.fit(scores, y_true)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        if self._model is None:
            return np.clip(scores.ravel(), 0.0, 1.0)
        return self._model.predict_proba(scores)[:, 1]


class IsotonicCalibrator:
    """Isotonic regression.

    Fits an arbitrary monotone mapping, so it can correct shapes Platt cannot.
    The cost is that it needs considerably more data and will happily overfit a
    small validation set into a step function -- which is why
    :func:`choose_calibrator` refuses it below a sample threshold rather than
    letting it look better than it is.
    """

    def __init__(self) -> None:
        self._model: Any = None

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        from sklearn.isotonic import IsotonicRegression

        scores = np.asarray(scores, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        if len(np.unique(y_true)) < 2:
            log.warning("calibration.single_class", note="isotonic regression skipped")
            return self
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._model.fit(scores, y_true)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self._model is None:
            return np.clip(scores, 0.0, 1.0)
        return np.clip(self._model.predict(scores), 0.0, 1.0)


#: Below this many validation samples, isotonic regression fits noise.
ISOTONIC_MIN_SAMPLES = 1_000


def choose_calibrator(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    bins: int = 10,
) -> tuple[Any, CalibrationReport, str]:
    """Fit both calibrators and keep whichever genuinely calibrates better.

    Selection is on ECE rather than Brier: Brier mixes discrimination and
    calibration together, and a calibrator cannot change discrimination at all,
    so scoring it on Brier partly scores it on something it does not affect.

    Isotonic is only considered when there is enough data for it, and the
    uncalibrated model is included as a candidate so that a model which was
    already well calibrated is not made worse.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    candidates: list[tuple[str, Any, np.ndarray]] = [("none", None, np.clip(scores, 0.0, 1.0))]

    platt = PlattCalibrator().fit(scores, y_true)
    candidates.append(("platt", platt, platt.transform(scores)))

    if scores.size >= ISOTONIC_MIN_SAMPLES:
        isotonic = IsotonicCalibrator().fit(scores, y_true)
        candidates.append(("isotonic", isotonic, isotonic.transform(scores)))
    else:
        log.info(
            "calibration.isotonic_skipped",
            n_samples=int(scores.size),
            required=ISOTONIC_MIN_SAMPLES,
            note="isotonic regression overfits a small validation set",
        )

    best_name, best_model, best_report = "none", None, None
    best_ece = float("inf")
    for name, model, calibrated in candidates:
        report = assess(y_true, calibrated, method=name, bins=bins)
        # Excess rather than raw ECE: comparing raw values across candidates is
        # fine here (same sample), but excess is the quantity that means
        # something and keeps the comparison consistent with the gate.
        score = report.excess_ece
        if np.isfinite(score) and score < best_ece:
            best_name, best_model, best_report, best_ece = name, model, report, score

    if best_report is None:
        best_report = assess(y_true, np.clip(scores, 0.0, 1.0), method="none", bins=bins)

    log.info(
        "calibration.selected",
        method=best_name,
        ece=round(best_report.ece, 6) if np.isfinite(best_report.ece) else None,
        excess_ece=(
            round(best_report.excess_ece, 6) if np.isfinite(best_report.excess_ece) else None
        ),
        brier=round(best_report.brier, 6) if np.isfinite(best_report.brier) else None,
        usable=best_report.is_usable,
    )
    return best_model, best_report, best_name
