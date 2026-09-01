"""Model registry against a real database and real artefacts on disk.

The property this exists to enforce: a model that is loaded for trading is
*byte-for-byte* the model whose out-of-sample metrics justified deploying it.
Anything else means trading with something that was never validated.
"""

from __future__ import annotations

import numpy as np
import pytest

from kat.common.enums import MarketType
from kat.db.engine import Database
from kat.models.registry import DeploymentState, ModelRegistry
from kat.models.selection import select_best
from kat.models.trainers import temporal_split, train_all


@pytest.fixture
def trained_models():
    rng = np.random.default_rng(11)
    n, features = 3_000, 6
    t = np.arange(n)
    X = rng.normal(size=(n, features))
    weights = np.array([1.0, -0.7, 0.5, 0.0, 0.3, 0.0])
    logit = X @ weights + rng.normal(scale=1.3, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    split = temporal_split(X, y, timestamps=t)
    return train_all(split, [f"f{i}" for i in range(features)], families=["logistic", "random_forest"])


@pytest.fixture
async def registry(sqlite_db: Database, tmp_path):
    return ModelRegistry(sqlite_db, artifact_dir=tmp_path / "artifacts")


def usable(models):
    """Models that actually trained.

    A plain ``next(...)`` generator would raise ``StopIteration`` here, and
    inside a coroutine that surfaces as an opaque ``RuntimeError`` instead of a
    readable failure.
    """
    out = [m for m in models if m.failed is None]
    assert out, f"no model trained successfully: {[m.failed for m in models]}"
    return out


class TestRegistration:
    async def test_registers_with_full_provenance(self, registry, trained_models):
        model = usable(trained_models)[0]
        registered = await registry.register(
            model, market_type=MarketType.EVENT, target="resolves_yes"
        )
        assert registered.artifact_path.exists()
        assert len(registered.artifact_sha256) == 64

        listed = await registry.list_models(market_type=MarketType.EVENT)
        assert len(listed) == 1
        row = listed[0]
        assert row["deployment_state"] == DeploymentState.CANDIDATE
        assert row["test_metrics"]["auc"] is not None
        # The commit is what makes a result reproducible from recorded artefacts.
        assert row["git_commit"] is not None

    async def test_refuses_to_register_a_model_that_failed_to_train(self, registry):
        from kat.models.trainers import TrainedModel

        broken = TrainedModel(
            name="broken", family="broken", estimator=None,
            feature_names=[], failed="ValueError: exploded",
        )
        with pytest.raises(ValueError, match="failed to train"):
            await registry.register(broken, market_type=MarketType.EVENT, target="t")

    async def test_the_calibrator_round_trips_with_the_estimator(
        self, registry, trained_models
    ):
        """A model deployed without its calibrator reports different
        probabilities from the ones it was selected on.

        The calibrator may legitimately be ``None`` -- an already well-calibrated
        model needs no correction -- so what is asserted is that whatever was
        fitted survives the round trip, and that the reloaded model produces the
        same probabilities as the one that was selected.
        """
        model = usable(trained_models)[0]
        registered = await registry.register(
            model, market_type=MarketType.EVENT, target="resolves_yes"
        )
        loaded = await registry.load(registered.model_id, registered.version)

        assert (loaded.calibrator is None) == (model.calibrator is None)

        X = np.random.default_rng(0).normal(size=(50, len(model.feature_names)))
        expected = model.predict_proba(X)
        raw = loaded.estimator.predict_proba(X)[:, 1]
        actual = (
            loaded.calibrator.transform(raw) if loaded.calibrator is not None else raw
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


class TestIntegrity:
    async def test_a_tampered_artefact_is_refused(self, registry, trained_models):
        model = usable(trained_models)[0]
        registered = await registry.register(
            model, market_type=MarketType.EVENT, target="resolves_yes"
        )
        with registered.artifact_path.open("ab") as handle:
            handle.write(b"tampered")

        with pytest.raises(ValueError, match="does not match its registered"):
            await registry.load(registered.model_id, registered.version)

    async def test_a_missing_artefact_is_reported_clearly(self, registry, trained_models):
        model = usable(trained_models)[0]
        registered = await registry.register(
            model, market_type=MarketType.EVENT, target="resolves_yes"
        )
        registered.artifact_path.unlink()
        with pytest.raises(FileNotFoundError):
            await registry.load(registered.model_id, registered.version)

    async def test_an_unknown_model_raises(self, registry):
        with pytest.raises(KeyError):
            await registry.load("nope", "v1")


class TestDeployment:
    async def test_deploying_archives_the_incumbent(self, registry, trained_models):
        models = usable(trained_models)
        first = await registry.register(
            models[0], market_type=MarketType.EVENT, target="resolves_yes"
        )
        second = await registry.register(
            models[1], market_type=MarketType.EVENT, target="resolves_yes"
        )

        await registry.deploy(first.model_id, first.version)
        assert (await registry.deployed(MarketType.EVENT, "resolves_yes")).version == first.version

        await registry.deploy(second.model_id, second.version)
        deployed = await registry.deployed(MarketType.EVENT, "resolves_yes")
        assert deployed.version == second.version

        states = {r["version"]: r["deployment_state"] for r in await registry.list_models()}
        assert states[first.version] == DeploymentState.ARCHIVED
        assert states[second.version] == DeploymentState.DEPLOYED

    async def test_rollback_restores_the_previous_model(self, registry, trained_models):
        models = usable(trained_models)
        first = await registry.register(
            models[0], market_type=MarketType.EVENT, target="resolves_yes"
        )
        second = await registry.register(
            models[1], market_type=MarketType.EVENT, target="resolves_yes"
        )
        await registry.deploy(first.model_id, first.version)
        await registry.deploy(second.model_id, second.version)

        restored = await registry.rollback(MarketType.EVENT, "resolves_yes")
        assert restored is not None
        assert restored.version == first.version

        states = {r["version"]: r["deployment_state"] for r in await registry.list_models()}
        assert states[first.version] == DeploymentState.DEPLOYED
        # The rolled-back model is rejected, not archived, so it is not the next
        # thing a further rollback restores.
        assert states[second.version] == DeploymentState.REJECTED

    async def test_rollback_with_nothing_to_fall_back_to_returns_none(
        self, registry, trained_models
    ):
        model = usable(trained_models)[0]
        registered = await registry.register(
            model, market_type=MarketType.EVENT, target="resolves_yes"
        )
        await registry.deploy(registered.model_id, registered.version)
        assert await registry.rollback(MarketType.EVENT, "resolves_yes") is None

    async def test_domains_have_independent_deployment_slots(self, registry, trained_models):
        models = usable(trained_models)
        event = await registry.register(
            models[0], market_type=MarketType.EVENT, target="resolves_yes"
        )
        perp = await registry.register(
            models[1], market_type=MarketType.PERP, target="direction"
        )
        await registry.deploy(event.model_id, event.version)
        await registry.deploy(perp.model_id, perp.version)

        # Deploying a perp model must not archive the event model.
        assert (await registry.deployed(MarketType.EVENT, "resolves_yes")) is not None
        assert (await registry.deployed(MarketType.PERP, "direction")) is not None


class TestEndToEnd:
    async def test_train_select_register_deploy_predict(self, registry, trained_models):
        """The whole chain, ending in a prediction from a hash-verified artefact."""
        result = select_best(trained_models)
        assert result.deployable, result.reason

        registered = await registry.register(
            result.best, market_type=MarketType.EVENT, target="resolves_yes"
        )
        await registry.deploy(registered.model_id, registered.version)

        loaded = await registry.deployed(MarketType.EVENT, "resolves_yes")
        assert loaded is not None

        X = np.random.default_rng(0).normal(size=(10, len(loaded.feature_names)))
        raw = loaded.estimator.predict_proba(X)[:, 1]
        probabilities = (
            loaded.calibrator.transform(raw) if loaded.calibrator is not None else raw
        )
        assert probabilities.shape == (10,)
        assert np.all((probabilities >= 0) & (probabilities <= 1))


class TestServingFindsWhatWasDeployed:
    """The trainer and the serving path must not have to agree on a string.

    ``InferenceEngine`` defaulted to target "settlement" while the trainer builds
    its target from the label horizon and deploys "forward_move_30m". They never
    matched, so ``deployed()`` always returned None and the platform served no
    predictions at all, reporting nothing wrong.
    """

    async def test_a_deployed_model_is_found_without_naming_its_target(
        self, registry, trained_models
    ):
        registered = await registry.register(
            usable(trained_models)[0], market_type=MarketType.EVENT, target="forward_move_30m"
        )
        await registry.deploy(registered.model_id, registered.version)

        found = await registry.deployed(MarketType.EVENT)

        assert found is not None
        assert found.model_id == registered.model_id
        assert found.target == "forward_move_30m"

    async def test_naming_the_wrong_target_still_finds_nothing(
        self, registry, trained_models
    ):
        registered = await registry.register(
            usable(trained_models)[0], market_type=MarketType.EVENT, target="forward_move_30m"
        )
        await registry.deploy(registered.model_id, registered.version)

        assert await registry.deployed(MarketType.EVENT, "settlement") is None

    async def test_nothing_deployed_is_none_not_an_error(self, registry):
        assert await registry.deployed(MarketType.EVENT) is None

    async def test_the_newest_deployment_wins_when_targets_differ(
        self, registry, trained_models
    ):
        """A domain with two deployed targets serves the latest rather than failing."""
        first = await registry.register(
            usable(trained_models)[0], market_type=MarketType.EVENT, target="forward_move_30m"
        )
        await registry.deploy(first.model_id, first.version)
        second = await registry.register(
            usable(trained_models)[0], market_type=MarketType.EVENT, target="forward_move_60m"
        )
        await registry.deploy(second.model_id, second.version)

        found = await registry.deployed(MarketType.EVENT)
        assert found is not None
        assert found.target == "forward_move_60m"
