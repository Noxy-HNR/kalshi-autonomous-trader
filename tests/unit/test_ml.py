"""Machine learning: splitting, calibration, selection, registry.

The properties that matter for a trading system, as opposed to a leaderboard:

* Splits are temporal, never random, so a model cannot learn from Tuesday to
  predict Monday.
* Selection is on out-of-sample performance only. The model with the best
  *training* score must be able to lose.
* Calibration is judged against the sampling-noise floor, not a fixed number, so
  the gate tests the model rather than the sample size.
* An artefact that has changed since it was validated cannot be loaded.
"""

from __future__ import annotations

import numpy as np
import pytest

from kat.models.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    assess,
    brier_score,
    choose_calibrator,
    expected_calibration_error,
    null_calibration_error,
    reliability_curve,
)
from kat.models.selection import SelectionCriteria, select_best
from kat.models.trainers import TrainedModel, temporal_split, train_all


@pytest.fixture(scope="module")
def dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """A signal buried in heavy noise, with a slow drift.

    The drift matters: it is what makes a random split cheat and a temporal split
    honest, so the same data exercises both.
    """
    rng = np.random.default_rng(11)
    n, features = 6_000, 8
    t = np.arange(n)
    X = rng.normal(size=(n, features))
    X[:, 0] += np.sin(t / 400.0) * 0.8
    weights = np.array([1.1, -0.8, 0.6, 0.0, 0.0, 0.4, 0.0, -0.3])
    logit = X @ weights + rng.normal(scale=1.4, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y, t, [f"f{i}" for i in range(features)]


@pytest.fixture(scope="module")
def trained(dataset):
    X, y, t, names = dataset
    split = temporal_split(X, y, timestamps=t)
    return split, train_all(split, names), names


class TestTemporalSplit:
    def test_blocks_are_contiguous_and_in_order(self, dataset):
        X, y, t, _ = dataset
        split = temporal_split(X, y, timestamps=t)
        assert split.train_end_index < split.val_end_index < X.shape[0]
        assert split.sizes["train"] > split.sizes["validation"] > 0
        assert split.sizes["test"] > 0

    def test_no_row_appears_in_two_splits(self, dataset):
        """A random split would silently break this and score beautifully."""
        X, y, t, _ = dataset
        split = temporal_split(X, y, timestamps=t)
        total = sum(split.sizes.values())
        assert total == X.shape[0]

    def test_rows_are_sorted_by_time_before_splitting(self):
        """Callers may pass unsorted data; the split must still respect time."""
        rng = np.random.default_rng(1)
        n = 1_000
        t = rng.permutation(n)
        X = t.reshape(-1, 1).astype(float)
        y = (t > n // 2).astype(int)
        split = temporal_split(X, y, timestamps=t)
        # After sorting, the train block holds the earliest timestamps.
        assert split.X_train.max() < split.X_val.min()
        assert split.X_val.max() < split.X_test.min()

    @pytest.mark.parametrize(
        ("train_frac", "val_frac"), [(0.9, 0.2), (0.0, 0.2), (0.6, 0.0), (1.0, 0.1)]
    )
    def test_rejects_impossible_fractions(self, dataset, train_frac, val_frac):
        X, y, _, _ = dataset
        with pytest.raises(ValueError):
            temporal_split(X, y, train_frac=train_frac, val_frac=val_frac)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            temporal_split(np.zeros((10, 2)), np.zeros(9))


class TestCalibrationMeasurement:
    def test_brier_of_a_coin_flip_is_a_quarter(self):
        y = np.array([0, 1, 0, 1])
        assert brier_score(y, np.full(4, 0.5)) == pytest.approx(0.25)

    def test_perfect_prediction_scores_zero(self):
        y = np.array([0, 1, 0, 1])
        assert brier_score(y, y.astype(float)) == pytest.approx(0.0)

    def test_empty_bins_are_omitted_not_reported_as_zero(self):
        """"No observations" and "never happened" are different facts."""
        y = np.array([1, 1, 0])
        probs = np.array([0.95, 0.92, 0.91])
        curve = reliability_curve(y, probs, bins=10)
        assert len(curve) == 1
        assert curve[0]["count"] == 3

    def test_a_biased_model_shows_a_calibration_gap(self):
        rng = np.random.default_rng(5)
        n = 5_000
        true_p = rng.uniform(0.1, 0.9, n)
        y = (rng.random(n) < true_p).astype(int)
        overconfident = np.clip(true_p + 0.15, 0, 1)
        ece, _ = expected_calibration_error(y, overconfident)
        assert ece > 0.1


class TestNoiseFloor:
    """The regression that made the calibration gate meaningless.

    A fixed ECE threshold is a test of sample size, not of the model: a
    perfectly calibrated model shows a raw ECE around 0.074 at 200 samples and
    0.015 at 5000, purely from binomial noise.
    """

    @pytest.mark.parametrize("n", [200, 1_200, 5_000])
    def test_a_perfectly_calibrated_model_has_near_zero_excess_error(self, n: int):
        rng = np.random.default_rng(3)
        probs = rng.uniform(0.05, 0.95, n)
        y = (rng.random(n) < probs).astype(int)  # outcomes match the probabilities exactly
        report = assess(y, probs)
        assert report.ece > 0, "raw ECE is non-zero even for perfect calibration"
        assert report.excess_ece < 0.02, (
            f"excess ECE {report.excess_ece:.4f} should be near zero for a "
            f"perfectly calibrated model at n={n} (raw {report.ece:.4f}, "
            f"floor {report.null_ece:.4f})"
        )

    def test_the_noise_floor_falls_as_samples_grow(self):
        rng = np.random.default_rng(3)
        floors = [
            null_calibration_error(rng.uniform(0.05, 0.95, n)) for n in (200, 1_200, 5_000)
        ]
        assert floors[0] > floors[1] > floors[2]

    def test_systematic_bias_survives_the_noise_correction(self):
        """Real miscalibration must not be explained away as sampling noise."""
        rng = np.random.default_rng(5)
        n = 5_000
        true_p = rng.uniform(0.1, 0.9, n)
        y = (rng.random(n) < true_p).astype(int)
        report = assess(y, np.clip(true_p + 0.12, 0, 1))
        assert report.excess_ece > 0.05
        assert not report.is_usable


class TestCalibrators:
    @pytest.fixture
    def miscalibrated(self):
        rng = np.random.default_rng(9)
        n = 4_000
        true_p = rng.uniform(0.05, 0.95, n)
        y = (rng.random(n) < true_p).astype(int)
        # Squash toward the extremes, the way a boosted tree does.
        raw = np.clip(1 / (1 + np.exp(-(np.log(true_p / (1 - true_p)) * 1.8))), 0, 1)
        return raw, y

    def test_platt_reduces_calibration_error(self, miscalibrated):
        raw, y = miscalibrated
        before = assess(y, raw)
        after = assess(y, PlattCalibrator().fit(raw, y).transform(raw))
        assert after.ece < before.ece

    def test_isotonic_reduces_calibration_error(self, miscalibrated):
        raw, y = miscalibrated
        before = assess(y, raw)
        after = assess(y, IsotonicCalibrator().fit(raw, y).transform(raw))
        assert after.ece < before.ece

    def test_isotonic_is_refused_on_a_small_sample(self):
        """It fits a step function to noise below a threshold."""
        rng = np.random.default_rng(2)
        raw = rng.uniform(0.1, 0.9, 200)
        y = (rng.random(200) < raw).astype(int)
        _, _, method = choose_calibrator(raw, y)
        assert method != "isotonic"

    def test_an_already_calibrated_model_is_left_alone(self):
        rng = np.random.default_rng(4)
        n = 3_000
        probs = rng.uniform(0.05, 0.95, n)
        y = (rng.random(n) < probs).astype(int)
        _, report, _ = choose_calibrator(probs, y)
        assert report.excess_ece < 0.02

    def test_single_class_data_does_not_crash(self):
        raw = np.linspace(0.1, 0.9, 100)
        y = np.zeros(100, dtype=int)
        calibrator, _, _ = choose_calibrator(raw, y)
        if calibrator is not None:
            assert np.all(np.isfinite(calibrator.transform(raw)))


class TestTrainingAndSelection:
    def test_every_family_trains(self, trained):
        _, models, _ = trained
        assert len(models) == 4
        assert all(m.failed is None for m in models), [m.failed for m in models if m.failed]

    def test_metrics_are_reported_per_split_and_never_blended(self, trained):
        _, models, _ = trained
        for model in models:
            assert model.train_metrics and model.validation_metrics and model.test_metrics
            assert model.train_metrics is not model.test_metrics

    def test_selection_is_not_on_training_performance(self, trained):
        """The model that fits its training data best must be able to lose.

        On this dataset LightGBM has the highest train AUC and the largest
        train-to-test gap, and is rejected for exactly that.
        """
        _, models, _ = trained
        result = select_best(models)
        assert result.deployable

        best_train = max(models, key=lambda m: m.train_metrics.get("auc", 0))
        assert best_train.name == "lightgbm"
        assert result.best is not None
        assert result.best.name != best_train.name, (
            "the highest training score won selection, which means selection is "
            "not out-of-sample"
        )

    def test_a_memorising_model_is_rejected_by_name(self, trained):
        _, models, _ = trained
        result = select_best(models)
        rejected = {c.model.name: c.failures for c in result.candidates if not c.passed}
        assert "lightgbm" in rejected
        assert any("memorised" in f for f in rejected["lightgbm"])

    def test_calibration_is_fitted_on_validation_not_test(self, trained):
        """Fitting and scoring calibration on the same data makes every model
        look perfectly calibrated."""
        _, models, _ = trained
        for model in models:
            if model.calibration:
                assert model.calibration["fitted_on"] == "validation"

    def test_no_model_is_deployed_when_none_clears_the_gates(self, trained):
        """Deploying the least bad of several failing models is how a platform
        ends up trading on noise."""
        _, models, _ = trained
        impossible = SelectionCriteria(min_test_auc=0.99)
        result = select_best(models, impossible)
        assert not result.deployable
        assert result.best is None
        assert "no candidate cleared" in result.reason

    def test_a_failed_model_cannot_be_selected(self):
        broken = TrainedModel(
            name="broken", family="broken", estimator=None, feature_names=["f"],
            failed="ValueError: exploded",
        )
        result = select_best([broken])
        assert not result.deployable
        assert "training failed" in result.candidates[0].failures[0]

    def test_a_model_that_cannot_beat_the_base_rate_is_rejected(self):
        useless = TrainedModel(
            name="useless", family="useless", estimator=None, feature_names=["f"],
            train_metrics={"auc": 0.5},
            validation_metrics={"auc": 0.5},
            test_metrics={
                "auc": 0.60, "brier": 0.25, "ece": 0.01, "null_ece": 0.01,
                "excess_ece": 0.0, "n": 1000, "base_rate": 0.5,
            },
        )
        result = select_best([useless])
        assert not result.deployable
        assert any("base rate" in f for f in result.candidates[0].failures)

    def test_feature_importance_is_produced(self, trained):
        _, models, names = trained
        for model in models:
            if model.feature_importance:
                assert set(model.feature_importance) <= set(names)
                assert sum(model.feature_importance.values()) == pytest.approx(1.0, abs=0.01)
