"""Model registry.

Every live prediction must be traceable back to the exact thing that produced
it::

    model -> model version -> feature set -> feature timestamp -> inputs -> prediction

This module owns the first half of that chain. It records, for every model, the
dataset it was trained on, the feature set it consumes, the precise train,
validation and test windows, the metrics from each split kept separate, its
calibration, the git commit of the code that trained it, and a SHA-256 of the
artefact on disk.

The hash is not ceremony. An artefact whose hash does not match its registry row
has been replaced by something, and "something" is not a model you should trade
with: :meth:`ModelRegistry.load` refuses it.

Deployment and rollback
-----------------------
Exactly one model per (market type, target) is ``deployed`` at a time. Deploying
a new one demotes the incumbent to ``archived`` rather than deleting it, which is
what makes rollback a single state change instead of a retraining job.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from kat.common.enums import MarketType
from kat.common.ids import new_id
from kat.common.timeutil import utcnow_aware
from kat.db.engine import Database
from kat.db.models import ModelRegistry as ModelRow
from kat.engine.decision import git_commit
from kat.models.trainers import TrainedModel
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["DeploymentState", "ModelRegistry", "RegisteredModel"]


class DeploymentState:
    CANDIDATE = "candidate"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    REJECTED = "rejected"


@dataclass(slots=True)
class RegisteredModel:
    """A model as the registry knows it."""

    model_id: str
    version: str
    name: str
    model_type: str
    market_type: MarketType
    target: str
    artifact_path: Path
    artifact_sha256: str
    feature_names: list[str] = field(default_factory=list)
    deployment_state: str = DeploymentState.CANDIDATE
    estimator: Any = None
    calibrator: Any = None

    @property
    def key(self) -> str:
        return f"{self.model_id}:{self.version}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRegistry:
    """Persists models to disk and their provenance to the database."""

    def __init__(self, database: Database, *, artifact_dir: Path | str = "data/models") -> None:
        self.db = database
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    # -- registration --------------------------------------------------------

    async def register(
        self,
        model: TrainedModel,
        *,
        market_type: MarketType,
        target: str,
        dataset_id: str | None = None,
        feature_set_id: str | None = None,
        train_window: tuple[datetime | None, datetime | None] = (None, None),
        validation_window: tuple[datetime | None, datetime | None] = (None, None),
        test_window: tuple[datetime | None, datetime | None] = (None, None),
        model_id: str | None = None,
        version: str | None = None,
        deployment_state: str = DeploymentState.CANDIDATE,
    ) -> RegisteredModel:
        """Write the artefact and its provenance. Does not deploy it."""
        if model.failed is not None:
            raise ValueError(f"refusing to register a model that failed to train: {model.failed}")

        model_id = model_id or f"{market_type.value}-{target}-{model.name}"
        version = version or new_id("v").split("_", 1)[1]
        path = self.artifact_dir / f"{model_id}-{version}.pkl"

        # The calibrator travels with the estimator: a model deployed without
        # the calibrator that was fitted for it reports different probabilities
        # from the ones it was selected on.
        payload = {
            "estimator": model.estimator,
            "calibrator": model.calibrator,
            "feature_names": model.feature_names,
            "family": model.name,
            "hyperparameters": model.hyperparameters,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        digest = _sha256(path)

        async with self.db.session() as session:
            session.add(
                ModelRow(
                    model_id=model_id,
                    version=version,
                    name=f"{model.name} ({target})",
                    model_type=model.name,
                    market_type=market_type.value,
                    target=target,
                    dataset_id=dataset_id,
                    feature_set_id=feature_set_id,
                    feature_names=list(model.feature_names),
                    hyperparameters=_jsonable(model.hyperparameters),
                    train_start=train_window[0],
                    train_end=train_window[1],
                    validation_start=validation_window[0],
                    validation_end=validation_window[1],
                    test_start=test_window[0],
                    test_end=test_window[1],
                    # Kept separate on purpose; nothing merges these.
                    train_metrics=_jsonable(model.train_metrics),
                    validation_metrics=_jsonable(model.validation_metrics),
                    test_metrics=_jsonable(model.test_metrics),
                    calibration=_jsonable(model.calibration),
                    feature_importance=_jsonable(model.feature_importance),
                    deployment_state=deployment_state,
                    artifact_path=str(path),
                    artifact_sha256=digest,
                    n_train_rows=model.n_train_rows,
                    git_commit=git_commit(),
                )
            )

        log.info(
            "model.registered",
            model_id=model_id,
            version=version,
            family=model.name,
            test_auc=model.test_metrics.get("auc"),
            sha256=digest[:12],
        )
        return RegisteredModel(
            model_id=model_id,
            version=version,
            name=model.name,
            model_type=model.name,
            market_type=market_type,
            target=target,
            artifact_path=path,
            artifact_sha256=digest,
            feature_names=list(model.feature_names),
            deployment_state=deployment_state,
            estimator=model.estimator,
            calibrator=model.calibrator,
        )

    # -- loading -------------------------------------------------------------

    async def load(self, model_id: str, version: str) -> RegisteredModel:
        """Load an artefact, verifying it is the one that was registered.

        A hash mismatch raises. An artefact that has changed since registration
        is not the model whose out-of-sample metrics justified deploying it, and
        trading with it would mean trading with something unvalidated.
        """
        async with self.db.session() as session:
            row = (
                await session.execute(
                    select(ModelRow).where(
                        ModelRow.model_id == model_id, ModelRow.version == version
                    )
                )
            ).scalar_one_or_none()

        if row is None:
            raise KeyError(f"no model {model_id}:{version} in the registry")
        if not row.artifact_path:
            raise FileNotFoundError(f"{model_id}:{version} has no artefact path recorded")

        path = Path(row.artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"artefact for {model_id}:{version} is missing at {path}")

        digest = _sha256(path)
        if row.artifact_sha256 and digest != row.artifact_sha256:
            raise ValueError(
                f"artefact for {model_id}:{version} does not match its registered "
                f"hash (expected {row.artifact_sha256[:12]}, found {digest[:12]}); "
                "refusing to load a model that has been altered since validation"
            )

        with path.open("rb") as handle:
            payload = pickle.load(handle)

        return RegisteredModel(
            model_id=row.model_id,
            version=row.version,
            name=row.name,
            model_type=row.model_type,
            market_type=MarketType(row.market_type),
            target=row.target or "",
            artifact_path=path,
            artifact_sha256=digest,
            feature_names=list(row.feature_names or []),
            deployment_state=row.deployment_state,
            estimator=payload["estimator"],
            calibrator=payload.get("calibrator"),
        )

    # -- deployment ----------------------------------------------------------

    async def deploy(self, model_id: str, version: str) -> RegisteredModel:
        """Deploy a model, archiving whichever was deployed for the same slot.

        The incumbent is archived rather than deleted so that
        :meth:`rollback` is a state change and not a retraining job.
        """
        loaded = await self.load(model_id, version)

        async with self.db.session() as session:
            incumbents = (
                await session.execute(
                    select(ModelRow).where(
                        ModelRow.market_type == loaded.market_type.value,
                        ModelRow.target == loaded.target,
                        ModelRow.deployment_state == DeploymentState.DEPLOYED,
                    )
                )
            ).scalars().all()
            for row in incumbents:
                row.deployment_state = DeploymentState.ARCHIVED
                log.info(
                    "model.archived",
                    model_id=row.model_id, version=row.version,
                    reason=f"superseded by {model_id}:{version}",
                )

            new_row = (
                await session.execute(
                    select(ModelRow).where(
                        ModelRow.model_id == model_id, ModelRow.version == version
                    )
                )
            ).scalar_one()
            new_row.deployment_state = DeploymentState.DEPLOYED

        loaded.deployment_state = DeploymentState.DEPLOYED
        log.info(
            "model.deployed",
            model_id=model_id, version=version,
            market_type=loaded.market_type.value, target=loaded.target,
            archived=[r.version for r in incumbents],
        )
        return loaded

    async def rollback(
        self, market_type: MarketType, target: str
    ) -> RegisteredModel | None:
        """Return to the most recently archived model for this slot.

        Rollback is deliberately dumb: the previous model was validated by the
        same gates the current one was, so restoring it needs no new evidence.
        """
        async with self.db.session() as session:
            current = (
                await session.execute(
                    select(ModelRow).where(
                        ModelRow.market_type == market_type.value,
                        ModelRow.target == target,
                        ModelRow.deployment_state == DeploymentState.DEPLOYED,
                    )
                )
            ).scalars().all()
            previous = (
                await session.execute(
                    select(ModelRow)
                    .where(
                        ModelRow.market_type == market_type.value,
                        ModelRow.target == target,
                        ModelRow.deployment_state == DeploymentState.ARCHIVED,
                    )
                    .order_by(ModelRow.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

            if previous is None:
                log.error(
                    "model.rollback_impossible",
                    market_type=market_type.value, target=target,
                    note="no archived model to fall back to",
                )
                return None

            for row in current:
                row.deployment_state = DeploymentState.REJECTED
            previous.deployment_state = DeploymentState.DEPLOYED
            model_id, version = previous.model_id, previous.version

        log.warning(
            "model.rolled_back",
            market_type=market_type.value, target=target,
            restored=f"{model_id}:{version}",
            rejected=[f"{r.model_id}:{r.version}" for r in current],
        )
        return await self.load(model_id, version)

    async def deployed(self, market_type: MarketType, target: str) -> RegisteredModel | None:
        async with self.db.session() as session:
            row = (
                await session.execute(
                    select(ModelRow).where(
                        ModelRow.market_type == market_type.value,
                        ModelRow.target == target,
                        ModelRow.deployment_state == DeploymentState.DEPLOYED,
                    )
                )
            ).scalar_one_or_none()
        if row is None:
            return None
        return await self.load(row.model_id, row.version)

    async def list_models(self, *, market_type: MarketType | None = None) -> list[dict[str, Any]]:
        async with self.db.session() as session:
            query = select(ModelRow).order_by(ModelRow.created_at.desc())
            if market_type is not None:
                query = query.where(ModelRow.market_type == market_type.value)
            rows = (await session.execute(query)).scalars().all()
        return [
            {
                "model_id": r.model_id,
                "version": r.version,
                "model_type": r.model_type,
                "market_type": r.market_type,
                "target": r.target,
                "deployment_state": r.deployment_state,
                "test_metrics": r.test_metrics,
                "calibration": r.calibration,
                "git_commit": r.git_commit,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars and datetimes into something JSON can hold."""
    import numpy as np

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        # NaN and infinity are not representable in JSON and become null rather
        # than silently corrupting the row.
        return None
    return value


def _now() -> datetime:
    return utcnow_aware()
