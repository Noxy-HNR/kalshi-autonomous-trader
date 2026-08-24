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
    "SplitResult",
    "TrainedModel",
    "temporal_split",
    "train_all",
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
