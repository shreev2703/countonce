"""corrections.key + consent: idempotent corrections from the web UI (additive)

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("corrections") as batch:
        batch.add_column(sa.Column("key", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column("consent", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch.create_unique_constraint("corrections_job_key", ["job_id", "key"])


def downgrade() -> None:
    with op.batch_alter_table("corrections") as batch:
        batch.drop_constraint("corrections_job_key", type_="unique")
        batch.drop_column("consent")
        batch.drop_column("key")
