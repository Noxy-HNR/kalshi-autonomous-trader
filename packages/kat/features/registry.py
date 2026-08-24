"""Versioned feature sets, and why a feature set needs an identity at all.

A model is a function of a *meaning*, not of a list of column names. When
``book_imbalance`` is redefined from three levels to five, every model trained
against the old meaning becomes wrong in a way nothing detects: the column is
still there, still populated, still plausible, and the model's live predictions
quietly degrade while its recorded test metrics stay excellent.

:class:`~kat.features.spec.FeatureSetSpec` closes that by hashing the actual
definitions into its ``feature_set_id``. This module's job is to make that
identity durable and discoverable: registered in process, persisted to the
``feature_sets`` table, and loadable back into the exact specs a model was
trained against.

Two consequences follow, and both are deliberate:

* **Editing a definition without bumping the version is allowed, and visible.**
  It produces a new id, so old models keep pointing at the old definitions.
  Registering it logs a warning naming both ids, because two live definitions of
  ``event_core.v1`` is nearly always an accident.
* **The registry never rewrites a stored set.** Persistence updates the mutable
  presentation columns of a row and leaves the definitions alone, since the id
  is derived from them -- a differing body under the same id would mean the hash
  was computed over something other than what was stored.

Offline by default
------------------
Every read is served from memory. The database is written to, and read from at
startup, but no inference or backtest waits on it: a research process with no
database still resolves feature sets, which is what makes the pipeline runnable
on a laptop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from sqlalchemy import select

from kat.common.enums import MarketType
from kat.common.errors import ConfigError, DataQualityError
from kat.config.settings import REPO_ROOT
from kat.db.engine import Database, get_database
from kat.db.models import FeatureSet
from kat.features.compute import EVENT_FEATURE_SET, PERP_FEATURE_SET
from kat.features.spec import FeatureSetSpec, FeatureSpec
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "FeatureSetRegistry",
    "get_feature_set_registry",
    "reset_feature_set_registry",
    "resolve_git_commit",
    "spec_from_orm",
    "spec_to_orm",
]


# ---------------------------------------------------------------------------
# Git provenance
# ---------------------------------------------------------------------------


def resolve_git_commit(repo_root: Path | None = None) -> str | None:
    """The commit this process is running, or ``None`` if it cannot be determined.

    Read from ``.git`` directly rather than by shelling out to ``git``. A
    subprocess on a trading host is a stall waiting to happen -- it inherits the
    environment, can block on a network filesystem, and would run on every model
    save. Reading two small files cannot.

    Lives here, rather than in the model registry, because the feature set is the
    first artefact in the pipeline that needs provenance, and the model registry
    imports it from here to avoid a second copy that could drift.
    """
    override = os.environ.get("KAT_GIT_COMMIT")
    if override:
        return override.strip()[:40]

    root = repo_root or REPO_ROOT
    head_file = root / ".git" / "HEAD"
    try:
        head = head_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if not head.startswith("ref:"):
        return head[:40] or None

    ref = head.partition("ref:")[2].strip()
    try:
        return (root / ".git" / ref).read_text(encoding="utf-8").strip()[:40] or None
    except OSError:
        pass

    # A repository that has been packed keeps its refs in one file instead.
    try:
        for line in (root / ".git" / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or " " not in line:
                continue
            sha, _, name = line.partition(" ")
            if name.strip() == ref:
                return sha[:40]
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# ORM mapping
# ---------------------------------------------------------------------------


def spec_to_orm(spec: FeatureSetSpec) -> FeatureSet:
    """The ``feature_sets`` row for a spec.

    ``availability_lag_ms`` is stored as its own column rather than left implicit
    in the feature bodies, because the simulator enforces it per feature and
    should not have to re-derive it from a source enum whose mapping may have
    changed since the row was written.
    """
    return FeatureSet(
        feature_set_id=spec.feature_set_id,
        version=spec.version,
        market_type=spec.market_type.value,
        features=[s.to_dict() for s in spec.specs],
        availability_lag_ms=spec.availability_lag_ms,
        git_commit=spec.git_commit,
    )


def spec_from_orm(row: FeatureSet) -> FeatureSetSpec:
    """Rebuild a spec from a stored row, verifying it still hashes to its own id.

    A mismatch means the row was edited in the database, or that
    :meth:`FeatureSpec.to_dict` changed shape since it was written. Either way
    the stored definitions no longer mean what the id says they mean, and a model
    pinned to that id would be loaded against the wrong contract.
    """
    payload = list(row.features or [])
    specs = tuple(FeatureSpec.from_dict(f) for f in payload)
    name, _, _rest = row.feature_set_id.partition(".")
    spec = FeatureSetSpec(
        name=name,
        version=row.version,
        market_type=MarketType(row.market_type),
        specs=specs,
        git_commit=row.git_commit,
    )
    if spec.feature_set_id != row.feature_set_id:
        raise DataQualityError(
            f"stored feature set {row.feature_set_id} no longer hashes to its own id "
            f"(recomputed {spec.feature_set_id}); the definitions in the row have been altered"
        )
    return spec


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class FeatureSetRegistry:
    """In-memory catalogue of feature sets, mirrored to the database."""

    __slots__ = ("_by_id", "_by_name", "_database")

    def __init__(self, database: Database | None = None) -> None:
        self._by_id: dict[str, FeatureSetSpec] = {}
        # name -> version -> spec. Two versions of a name coexist normally; two
        # *bodies* under one version do not, and the warning below says so.
        self._by_name: dict[str, dict[str, FeatureSetSpec]] = {}
        self._database = database

    # -- in-memory ----------------------------------------------------------

    def register(self, spec: FeatureSetSpec) -> FeatureSetSpec:
        existing = self._by_id.get(spec.feature_set_id)
        if existing is not None:
            return existing

        versions = self._by_name.setdefault(spec.name, {})
        clash = versions.get(spec.version)
        if clash is not None:
            log.warning(
                "features.version_body_changed",
                name=spec.name,
                version=spec.version,
                previous_id=clash.feature_set_id,
                new_id=spec.feature_set_id,
                detail="two different definitions share one version string; bump the version",
            )
        versions[spec.version] = spec
        self._by_id[spec.feature_set_id] = spec
        log.info(
            "features.registered",
            feature_set_id=spec.feature_set_id,
            market_type=spec.market_type.value,
            features=len(spec.specs),
            max_window_ms=spec.max_window_ms,
        )
        return spec

    def get(self, feature_set_id: str) -> FeatureSetSpec:
        spec = self._by_id.get(feature_set_id)
        if spec is None:
            raise ConfigError(
                f"feature set {feature_set_id!r} is not registered; "
                f"known: {', '.join(sorted(self._by_id)) or '<none>'}"
            )
        return spec

    def find(self, feature_set_id: str) -> FeatureSetSpec | None:
        return self._by_id.get(feature_set_id)

    def resolve(self, name: str, version: str | None = None) -> FeatureSetSpec:
        """Resolve by name, defaulting to the highest version string registered.

        Version ordering is lexicographic over the version string, which is why
        versions are ``v1``/``v2`` rather than ``1``/``2``: it keeps the ordering
        obvious and stops ``v10`` sorting before ``v9`` for as long as anyone is
        likely to care.
        """
        versions = self._by_name.get(name)
        if not versions:
            raise ConfigError(f"no feature set named {name!r} is registered")
        if version is not None:
            spec = versions.get(version)
            if spec is None:
                raise ConfigError(
                    f"feature set {name!r} has no version {version!r}; "
                    f"have {', '.join(sorted(versions))}"
                )
            return spec
        return versions[max(versions)]

    def for_market(self, market_type: MarketType) -> tuple[FeatureSetSpec, ...]:
        return tuple(s for s in self._by_id.values() if s.market_type is market_type)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_id))

    def __len__(self) -> int:
        return len(self._by_id)

    # -- persistence --------------------------------------------------------

    def _db(self) -> Database:
        return self._database or get_database()

    async def persist(self, spec: FeatureSetSpec) -> bool:
        """Write the spec to ``feature_sets``. Returns ``True`` if it was new.

        Existing rows are left alone. The id is a hash of the body, so a row that
        already exists under this id already holds these definitions; rewriting
        it could only replace them with something that does not hash to the id it
        is stored under.
        """
        db = self._db()
        async with db.session() as session:
            row = await session.get(FeatureSet, spec.feature_set_id)
            if row is not None:
                return False
            session.add(spec_to_orm(spec))
        log.info("features.persisted", feature_set_id=spec.feature_set_id)
        return True

    async def persist_all(self) -> int:
        """Persist every registered set. Returns the number newly written."""
        written = 0
        for spec in list(self._by_id.values()):
            if await self.persist(spec):
                written += 1
        return written

    async def load(self, feature_set_id: str) -> FeatureSetSpec | None:
        """Load one stored set and register it. ``None`` if it is not in the table."""
        db = self._db()
        async with db.session() as session:
            row = await session.get(FeatureSet, feature_set_id)
            if row is None:
                return None
            spec = spec_from_orm(row)
        return self.register(spec)

    async def load_all(self, market_type: MarketType | None = None) -> list[FeatureSetSpec]:
        """Load every stored set, registering each. Called once at startup.

        A row that fails its own hash check is skipped with an error rather than
        aborting the load: one corrupted definition should not stop the platform
        from resolving the others, and the model that pins the bad id will fail
        loudly when it is loaded.
        """
        db = self._db()
        stmt = select(FeatureSet)
        if market_type is not None:
            stmt = stmt.where(FeatureSet.market_type == market_type.value)
        async with db.session() as session:
            rows = list((await session.execute(stmt)).scalars())

        loaded: list[FeatureSetSpec] = []
        for row in rows:
            try:
                loaded.append(self.register(spec_from_orm(row)))
            except DataQualityError as exc:
                log.error("features.load_failed", feature_set_id=row.feature_set_id, error=str(exc))
        return loaded


# ---------------------------------------------------------------------------
# Process singleton
# ---------------------------------------------------------------------------

_registry: FeatureSetRegistry | None = None

#: Sets registered on first use. These are the definitions the platform ships
#: with; anything else has to be registered by whatever created it.
_BUILT_IN: Final[tuple[FeatureSetSpec, ...]] = (EVENT_FEATURE_SET, PERP_FEATURE_SET)


def get_feature_set_registry() -> FeatureSetRegistry:
    """The process-wide registry, pre-loaded with the built-in sets.

    The built-ins are stamped with the running commit at first use rather than at
    import, so that a process started from a different checkout records the
    commit it is actually running.
    """
    global _registry
    if _registry is None:
        registry = FeatureSetRegistry()
        commit = resolve_git_commit()
        for spec in _BUILT_IN:
            stamped = (
                spec
                if commit is None
                else FeatureSetSpec(
                    name=spec.name,
                    version=spec.version,
                    market_type=spec.market_type,
                    specs=spec.specs,
                    description=spec.description,
                    git_commit=commit,
                )
            )
            registry.register(stamped)
        _registry = registry
    return _registry


def reset_feature_set_registry() -> None:
    """Drop the singleton. Tests only."""
    global _registry
    _registry = None
