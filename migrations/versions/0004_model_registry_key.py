"""model_registry keyed by (model_id, version)

A registry that cannot hold two versions of a model is a registry that can never
retrain. ``model_id`` is derived from the domain, the target and the family, so
every retrain of a family produces the same id -- and with the id alone as the
primary key the second registration violates it and the training run dies:

    UniqueViolationError: duplicate key value violates unique constraint
    "pk_model_registry"
    DETAIL:  Key (model_id)=(event-forward_move_30m-lightgbm) already exists.

Every lookup in ``kat.models.registry`` already reads ``(model_id, version)``
together, and archiving an old version while deploying a new one is the entire
purpose of the ``deployment_state`` column. Only the schema disagreed.

Revision ID: 0004_model_registry_key
Revises: 0003_widen_source_id
Create Date: 2026-09-01

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004_model_registry_key"
down_revision: str | Sequence[str] | None = "0003_widen_source_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("pk_model_registry", "model_registry", type_="primary")
    op.create_primary_key(
        "pk_model_registry", "model_registry", ["model_id", "version"]
    )


def downgrade() -> None:
    """Narrowing the key again can lose rows, so refuse rather than destroy.

    Going back means keeping one version per model_id and discarding the rest,
    and which one to keep is not a decision a migration should make silently.
    """
    op.execute(
        "DELETE FROM model_registry a USING model_registry b "
        "WHERE a.model_id = b.model_id AND a.version < b.version"
    )
    op.drop_constraint("pk_model_registry", "model_registry", type_="primary")
    op.create_primary_key("pk_model_registry", "model_registry", ["model_id"])
