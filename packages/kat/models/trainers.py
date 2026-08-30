"""Competing model families.

No family is assumed best. Several are trained on identical data, scored on an
identical held-out split, and the winner is chosen on **out-of-sample**
performance by :mod:`kat.models.selection`. That is not diplomacy: which family
wins genuinely varies by market and by feature set, and a platform that hard-codes
"gradient boosting is best" will be wrong on the markets where it is not.

Splitting is always temporal
----------------------------
:func:`temporal_split` never shuffles. A random split on time-series data lets
the model learn from Tuesday to predict Monday, which produces a spectacular
validation score and a strategy that loses money. Every split here is by time,
and the train set always ends before the validation set begins.

Every family here is free and open-source: scikit-learn's logistic regression and
random forest, XGBoost, and LightGBM, plus a simple averaging ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "TRAINERS",
    "Fold",
    "SplitResult",
    "TrainedModel",
    "temporal_split",
    "train_all",
    "walk_forward",
    "walk_forward_folds",
]


class SupportsPredictProba(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(slots=True)
class SplitResult:
    """A strictly temporal train/validation/test split."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_end_index: int
    val_end_index: int

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.y_train.size),
            "validation": int(self.y_val.size),
            "test": int(self.y_test.size),
        }


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    timestamps: np.ndarray | None = None,
) -> SplitResult:
    """Split by time, never at random.

    ``timestamps``, when supplied, is used to sort the rows first, so the caller
    does not have to guarantee the data arrives in order. If it is omitted the
    existing row order is assumed to be chronological -- which it must be, and
    which the feature pipeline guarantees.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be positive and sum to less than 1")

    if timestamps is not None:
        order = np.argsort(np.asarray(timestamps), kind="stable")
        X, y = X[order], y[order]

    n = X.shape[0]
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    if train_end == 0 or val_end == train_end or val_end >= n:
        raise ValueError(
            f"{n} rows cannot be split into three non-empty temporal blocks at "
            f"train_frac={train_frac}, val_frac={val_frac}"
        )

    return SplitResult(
        X_train=X[:train_end],
        y_train=y[:train_end],
        X_val=X[train_end:val_end],
        y_val=y[train_end:val_end],
        X_test=X[val_end:],
        y_test=y[val_end:],
        train_end_index=train_end,
        val_end_index=val_end,
    )


@dataclass(slots=True)
class TrainedModel:
    """One trained estimator, with metrics kept separate per split.

    ``train_metrics`` exists to be *compared against* the others, not to be
    reported on its own. A large gap between train and test is the overfitting
    signal, and hiding the training number would hide it.
    """

    name: str
    family: str
    estimator: Any
    feature_names: list[str]
    train_metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, Any] | None = None
    calibrator: Any = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    n_train_rows: int = 0
    failed: str | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probability of the positive class."""
        raw = self.estimator.predict_proba(np.asarray(X))[:, 1]
        if self.calibrator is not None:
            return np.clip(self.calibrator.transform(raw), 0.0, 1.0)
        return np.clip(raw, 0.0, 1.0)

    @property
    def overfitting_gap(self) -> float | None:
        """Train AUC minus test AUC. Large and positive means memorised."""
        train_auc = self.train_metrics.get("auc")
        test_auc = self.test_metrics.get("auc")
        if train_auc is None or test_auc is None:
            return None
        return float(train_auc - test_auc)


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Discrimination and probability quality on one split."""
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    from kat.models.calibration import (
        brier_score,
        expected_calibration_error,
        null_calibration_error,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    out: dict[str, float] = {
        "brier": brier_score(y_true, y_prob),
        "accuracy": float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
        "n": int(y_true.size),
        "base_rate": float(np.mean(y_true)) if y_true.size else float("nan"),
    }
    # AUC and log loss are undefined on a single-class split; report them as
    # absent rather than as a number that means nothing.
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["log_loss"] = float(log_loss(y_true, y_prob))
    ece, mce = expected_calibration_error(y_true, y_prob)
    null_ece = null_calibration_error(y_prob)
    out["ece"] = float(ece)
    out["mce"] = float(mce)
    # The noise floor travels with the metric, so any consumer can tell
    # miscalibration from sample size without recomputing it.
    out["null_ece"] = float(null_ece)
    out["excess_ece"] = (
        float(max(0.0, ece - null_ece))
        if np.isfinite(ece) and np.isfinite(null_ece)
        else float("nan")
    )
    return out


# --- family builders --------------------------------------------------------


def _logistic(**kw: Any) -> tuple[Any, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    params = {"C": 1.0, "max_iter": 2000, "solver": "lbfgs", **kw}
    # Scaling matters for a penalised linear model: without it the penalty falls
    # on whichever feature happens to have the largest units.
    return make_pipeline(StandardScaler(), LogisticRegression(**params)), params


def _random_forest(**kw: Any) -> tuple[Any, dict[str, Any]]:
    from sklearn.ensemble import RandomForestClassifier

    params = {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 20,
        "n_jobs": -1,
        "random_state": 7,
        **kw,
    }
    return RandomForestClassifier(**params), params


def _xgboost(**kw: Any) -> tuple[Any, dict[str, Any]]:
    from xgboost import XGBClassifier

    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        # Regularisation defaults are deliberately firm: these datasets are
        # small and noisy, and an unregularised booster memorises them.
        "reg_lambda": 2.0,
        "min_child_weight": 10,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 7,
        **kw,
    }
    return XGBClassifier(**params), params


def _lightgbm(**kw: Any) -> tuple[Any, dict[str, Any]]:
    from lightgbm import LGBMClassifier

    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_lambda": 2.0,
        "random_state": 7,
        "verbosity": -1,
        **kw,
    }
    return LGBMClassifier(**params), params


#: Every family that competes. Adding one here is all it takes to enter it.
TRAINERS: dict[str, Any] = {
    "logistic": _logistic,
    "random_forest": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
}


def _importance(estimator: Any, feature_names: list[str]) -> dict[str, float]:
    """Best available feature importance, normalised to sum to one."""
    values: np.ndarray | None = None
    target = estimator
    # Unwrap a pipeline to reach the estimator that actually has coefficients.
    if hasattr(estimator, "steps"):
        target = estimator.steps[-1][1]
    if hasattr(target, "feature_importances_"):
        values = np.asarray(target.feature_importances_, dtype=float)
    elif hasattr(target, "coef_"):
        values = np.abs(np.asarray(target.coef_, dtype=float)).ravel()
    if values is None or values.size != len(feature_names):
        return {}
    total = float(values.sum())
    if total <= 0:
        return {}
    return {
        name: round(float(v / total), 6)
        for name, v in sorted(
            zip(feature_names, values, strict=True), key=lambda kv: -kv[1]
        )
    }


def train_one(
    name: str,
    split: SplitResult,
    feature_names: list[str],
    *,
    calibrate: bool = True,
    **hyperparameters: Any,
) -> TrainedModel:
    """Train one family and score it on all three splits.

    Calibration is fitted on the **validation** split and evaluated on the
    **test** split. Fitting and evaluating it on the same data would make every
    model look perfectly calibrated.
    """
    builder = TRAINERS.get(name)
    if builder is None:
        raise ValueError(f"unknown model family {name!r}; known: {sorted(TRAINERS)}")

    estimator, params = builder(**hyperparameters)
    model = TrainedModel(
        name=name,
        family=name,
        estimator=estimator,
        feature_names=list(feature_names),
        hyperparameters=params,
        n_train_rows=int(split.y_train.size),
    )

    try:
        estimator.fit(split.X_train, split.y_train)
    except Exception as exc:
        model.failed = f"{type(exc).__name__}: {exc}"
        log.error("model.train_failed", family=name, error=str(exc))
        return model

    train_prob = estimator.predict_proba(split.X_train)[:, 1]
    val_prob = estimator.predict_proba(split.X_val)[:, 1]

    if calibrate:
        from kat.models.calibration import choose_calibrator

        # The chosen method name is already inside report.to_dict().
        calibrator, report, _ = choose_calibrator(val_prob, split.y_val)
        model.calibrator = calibrator
        model.calibration = report.to_dict()
        model.calibration["fitted_on"] = "validation"
    else:
        model.calibration = None

    test_prob = model.predict_proba(split.X_test)

    model.train_metrics = _metrics(split.y_train, train_prob)
    model.validation_metrics = _metrics(split.y_val, val_prob)
    model.test_metrics = _metrics(split.y_test, test_prob)
    model.feature_importance = _importance(estimator, list(feature_names))

    log.info(
        "model.trained",
        family=name,
        train_auc=model.train_metrics.get("auc"),
        val_auc=model.validation_metrics.get("auc"),
        test_auc=model.test_metrics.get("auc"),
        test_brier=model.test_metrics.get("brier"),
        overfitting_gap=model.overfitting_gap,
        calibration=(model.calibration or {}).get("method"),
    )
    return model


def train_all(
    split: SplitResult,
    feature_names: list[str],
    *,
    families: list[str] | None = None,
    calibrate: bool = True,
) -> list[TrainedModel]:
    """Train every family on identical data. Failures are returned, not raised."""
    chosen = families or list(TRAINERS)
    models: list[TrainedModel] = []
    for name in chosen:
        models.append(train_one(name, split, feature_names, calibrate=calibrate))
    succeeded = [m for m in models if m.failed is None]
    log.info(
        "model.train_all_complete",
        attempted=len(models),
        succeeded=len(succeeded),
        failed=[m.name for m in models if m.failed],
    )
    return models


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Fold:
    """One rolling window's out-of-sample result for one family."""

    index: int
    train_rows: int
    test_rows: int
    train_base_rate: float
    test_base_rate: float
    train_auc: float | None
    test_auc: float | None
    test_brier: float
    test_excess_ece: float
    failed: str | None = None

    @property
    def gap(self) -> float | None:
        """Train AUC minus test AUC for this window alone."""
        if self.train_auc is None or self.test_auc is None:
            return None
        return self.train_auc - self.test_auc

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "train_base_rate": round(self.train_base_rate, 4),
            "test_base_rate": round(self.test_base_rate, 4),
            "train_auc": None if self.train_auc is None else round(self.train_auc, 4),
            "test_auc": None if self.test_auc is None else round(self.test_auc, 4),
            "test_brier": round(self.test_brier, 4),
            "test_excess_ece": round(self.test_excess_ece, 4),
            "gap": None if self.gap is None else round(self.gap, 4),
            "failed": self.failed,
        }


def walk_forward_folds(
    n_rows: int, *, folds: int = 5, train_fraction: float = 0.6
) -> list[tuple[slice, slice]]:
    """Rolling train/test index pairs over time-ordered rows.

    Rolling, not expanding. An expanding window lets the earliest data influence
    every later result, so a model that only worked in the first period keeps
    being propped up by it -- the same reasoning the backtest walk-forward gives
    for advancing its windows rather than growing them.

    The leading ``train_fraction`` of the rows is training only and never
    tested: without that reservation the first window would score a model that
    had seen almost nothing, which is not an out-of-sample result but an
    untrained one. Test spans then tile the remainder exactly once, so no row is
    scored twice.
    """
    if folds < 2:
        raise ValueError("a walk-forward with fewer than two folds measures nothing")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must sit strictly between 0 and 1")

    reserved = int(n_rows * train_fraction)
    test_region = n_rows - reserved
    if test_region < folds:
        return []

    step = test_region / folds
    train_span = int(step * train_fraction / (1.0 - train_fraction))
    pairs: list[tuple[slice, slice]] = []
    for index in range(folds):
        test_start = reserved + int(step * index)
        test_end = reserved + int(step * (index + 1)) if index < folds - 1 else n_rows
        train_start = max(0, test_start - train_span)
        if test_end - test_start < 1 or test_start - train_start < 1:
            continue
        pairs.append((slice(train_start, test_start), slice(test_start, test_end)))
    return pairs


def walk_forward(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    timestamps: np.ndarray | None = None,
    folds: int = 5,
    train_fraction: float = 0.6,
    **hyperparameters: Any,
) -> tuple[dict[str, Any], list[Fold]]:
    """Score one family across rolling windows instead of one split.

    A single 60/20/20 temporal split bets everything on one test period. On
    market data that is a coin toss about regime: the platform's first attempt
    produced base rates of 7.6%, 10.9% and 35.9% across its three splits, so
    every model was judged on conditions it had never trained on, and the
    train-to-test AUC gap could not distinguish memorisation from a market that
    had simply changed.

    Rolling windows separate those two. A gap that is large in every window is
    the model memorising; a gap that swings between windows is the market
    moving, and ``gap_spread`` is what tells them apart.

    Returns ``(summary, per_fold_detail)``.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if timestamps is not None:
        order = np.argsort(np.asarray(timestamps), kind="stable")
        X, y = X[order], y[order]

    pairs = walk_forward_folds(X.shape[0], folds=folds, train_fraction=train_fraction)
    if not pairs:
        return {"family": name, "folds": 0, "reason": "not enough rows to roll a window"}, []

    builder = TRAINERS.get(name)
    if builder is None:
        raise ValueError(f"unknown model family {name!r}; known: {sorted(TRAINERS)}")

    results: list[Fold] = []
    for index, (train_idx, test_idx) in enumerate(pairs):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        fold = Fold(
            index=index,
            train_rows=int(y_tr.size),
            test_rows=int(y_te.size),
            train_base_rate=float(np.mean(y_tr)) if y_tr.size else float("nan"),
            test_base_rate=float(np.mean(y_te)) if y_te.size else float("nan"),
            train_auc=None,
            test_auc=None,
            test_brier=float("nan"),
            test_excess_ece=float("nan"),
        )
        # A fold whose training half is one class cannot be fitted, and one whose
        # test half is one class cannot be ranked. Both are recorded rather than
        # skipped: a walk-forward that quietly drops its hard windows is
        # reporting on the easy ones.
        if len(np.unique(y_tr)) < 2:
            fold.failed = "training window is one class"
            results.append(fold)
            continue
        try:
            estimator, _ = builder(**hyperparameters)
            estimator.fit(X_tr, y_tr)
            train_metrics = _metrics(y_tr, estimator.predict_proba(X_tr)[:, 1])
            test_metrics = _metrics(y_te, estimator.predict_proba(X_te)[:, 1])
        except Exception as exc:
            fold.failed = f"{type(exc).__name__}: {exc}"
            results.append(fold)
            continue

        fold.train_auc = train_metrics.get("auc")
        fold.test_auc = test_metrics.get("auc")
        fold.test_brier = test_metrics.get("brier", float("nan"))
        fold.test_excess_ece = test_metrics.get("excess_ece", float("nan"))
        results.append(fold)

    scored = [f for f in results if f.test_auc is not None]
    gaps = [f.gap for f in results if f.gap is not None]
    aucs = [f.test_auc for f in scored]
    rates = [f.test_base_rate for f in results if f.test_base_rate == f.test_base_rate]

    summary: dict[str, Any] = {
        "family": name,
        "folds": len(results),
        "scored": len(scored),
        "failed": sum(1 for f in results if f.failed),
        "mean_test_auc": float(np.mean(aucs)) if aucs else None,
        "worst_test_auc": float(np.min(aucs)) if aucs else None,
        "mean_gap": float(np.mean(gaps)) if gaps else None,
        # The number this whole function exists to produce. A consistent gap is
        # a model that memorises; a gap that swings is a market that moved.
        "gap_spread": float(np.max(gaps) - np.min(gaps)) if len(gaps) > 1 else None,
        "base_rate_spread": float(np.max(rates) - np.min(rates)) if len(rates) > 1 else None,
        "mean_excess_ece": (
            float(np.mean([f.test_excess_ece for f in scored])) if scored else None
        ),
    }
    log.info("model.walk_forward", **{k: v for k, v in summary.items() if v is not None})
    return summary, results
