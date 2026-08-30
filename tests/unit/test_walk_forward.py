"""Rolling windows, so a gate failure can be explained rather than just reported.

A single 60/20/20 temporal split bets everything on one test period, and on
market data that is a coin toss about regime -- the platform's first attempt
produced base rates of 7.6%, 10.9% and 35.9% across its three splits. Every
model was then judged on conditions it had never trained on, and the
train-to-test AUC gap could not distinguish two very different problems:

* the model memorised its training data, which more data will not fix;
* the market changed underneath it, which is exactly what more data does fix.

``gap_spread`` separates them. These tests pin the geometry that makes the
measurement honest, and the semantics that make it mean anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from kat.models.trainers import Fold, walk_forward, walk_forward_folds


# -- geometry ----------------------------------------------------------------


def test_windows_never_test_on_data_they_trained_on() -> None:
    """The property everything else rests on."""
    for train, test in walk_forward_folds(1_000, folds=5):
        assert train.stop <= test.start, "a training window ran into its own test window"


def test_test_spans_tile_the_region_exactly_once() -> None:
    """Scoring a period twice would weight it twice in the average."""
    covered = [i for _, test in walk_forward_folds(1_000, folds=5) for i in range(test.start, test.stop)]
    assert len(covered) == len(set(covered))


def test_the_leading_fraction_is_never_tested() -> None:
    """Without the reservation the first window scores a model that has seen
    almost nothing, which is not an out-of-sample result but an untrained one."""
    folds = walk_forward_folds(1_000, folds=5, train_fraction=0.6)
    assert min(test.start for _, test in folds) >= 600


def test_windows_roll_rather_than_expand() -> None:
    """An expanding window lets the earliest data influence every later result,
    so a model that only worked in the first period keeps being propped up."""
    folds = walk_forward_folds(1_000, folds=5)
    starts = [train.start for train, _ in folds]
    assert starts == sorted(starts)
    assert starts[0] < starts[-1], "training windows must advance, not accumulate"
    widths = {train.stop - train.start for train, _ in folds}
    assert len(widths) == 1, "a rolling window keeps a constant width"


def test_the_last_window_reaches_the_end_of_the_data() -> None:
    """Rounding must not silently drop the most recent rows, which are the ones
    most like the period the model would actually run in."""
    folds = walk_forward_folds(1_003, folds=5)
    assert folds[-1][1].stop == 1_003


@pytest.mark.parametrize("folds", [0, 1])
def test_too_few_folds_is_refused(folds: int) -> None:
    with pytest.raises(ValueError, match="fewer than two folds"):
        walk_forward_folds(1_000, folds=folds)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_an_impossible_train_fraction_is_refused(fraction: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        walk_forward_folds(1_000, train_fraction=fraction)


def test_too_little_data_yields_no_folds_rather_than_a_bad_one() -> None:
    assert walk_forward_folds(6, folds=5) == []


# -- semantics ---------------------------------------------------------------


def _separable(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A signal a model can actually learn, stationary across the whole span."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    logit = 1.6 * X[:, 0] - 1.1 * X[:, 1]
    y = (logit + rng.normal(scale=0.5, size=n) > 0).astype(float)
    return X, y, np.arange(n, dtype=float)


def test_a_stationary_signal_scores_above_chance_in_every_window() -> None:
    X, y, ts = _separable(3_000)
    summary, folds = walk_forward("logistic", X, y, ["a", "b", "c", "d"], timestamps=ts, folds=4)
    assert summary["scored"] == summary["folds"] == 4
    assert summary["mean_test_auc"] > 0.75
    assert summary["worst_test_auc"] > 0.6, "a stationary signal should hold up in every window"


def test_a_stationary_signal_has_a_small_gap_spread() -> None:
    """The control for the diagnostic: when nothing changes between windows, the
    gap should not swing between them either."""
    X, y, ts = _separable(3_000)
    summary, _ = walk_forward("logistic", X, y, ["a", "b", "c", "d"], timestamps=ts, folds=4)
    assert summary["gap_spread"] is not None
    assert summary["gap_spread"] < 0.15


def test_rows_are_sorted_by_timestamp_before_folding() -> None:
    """Callers must not have to hand the data over in order; folding unsorted
    rows would put later data in an earlier window."""
    X, y, ts = _separable(2_000)
    shuffle = np.random.default_rng(3).permutation(len(y))
    ordered, _ = walk_forward("logistic", X, y, ["a", "b", "c", "d"], timestamps=ts, folds=3)
    jumbled, _ = walk_forward(
        "logistic", X[shuffle], y[shuffle], ["a", "b", "c", "d"],
        timestamps=ts[shuffle], folds=3,
    )
    assert ordered["mean_test_auc"] == pytest.approx(jumbled["mean_test_auc"], abs=1e-9)


def test_a_one_class_training_window_is_recorded_not_skipped() -> None:
    """A walk-forward that quietly drops its hard windows reports on the easy ones."""
    n = 2_000
    X = np.random.default_rng(1).normal(size=(n, 3))
    y = np.zeros(n)
    y[int(n * 0.75) :] = 1.0  # every early window is single-class
    summary, folds = walk_forward(
        "logistic", X, y, ["a", "b", "c"], timestamps=np.arange(n, dtype=float), folds=4
    )
    assert summary["folds"] == 4
    assert summary["failed"] >= 1
    assert any(f.failed for f in folds)
    assert summary["scored"] < summary["folds"]


def test_an_unknown_family_is_refused() -> None:
    X, y, ts = _separable(500)
    with pytest.raises(ValueError, match="unknown model family"):
        walk_forward("not_a_family", X, y, ["a", "b", "c", "d"], timestamps=ts, folds=2)


def test_too_little_data_reports_a_reason_rather_than_raising() -> None:
    X, y, ts = _separable(8)
    summary, folds = walk_forward("logistic", X, y, ["a", "b", "c", "d"], timestamps=ts, folds=5)
    assert summary["folds"] == 0
    assert "reason" in summary
    assert folds == []


def test_fold_gap_is_absent_when_either_auc_is() -> None:
    """None, not zero: an unmeasurable gap is not a zero one."""
    fold = Fold(
        index=0, train_rows=10, test_rows=10, train_base_rate=0.5, test_base_rate=0.5,
        train_auc=None, test_auc=0.7, test_brier=0.2, test_excess_ece=0.01,
    )
    assert fold.gap is None
    assert fold.to_dict()["gap"] is None
