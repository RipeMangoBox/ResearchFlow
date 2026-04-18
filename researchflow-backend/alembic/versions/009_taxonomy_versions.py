"""Add taxonomy_versions table for ontology change tracking.

Revision ID: 009
Revises: 008
Create Date: 2026-04-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "taxonomy_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("entity_id", UUID(as_uuid=True), nullable=False),
        sa.Column("action", sa.String(30), nullable=False),
        sa.Column("version_label", sa.String(100)),
        sa.Column("changed_by", sa.String(50)),
        sa.Column("change_summary", sa.Text),
        sa.Column("before_snapshot", JSONB),
        sa.Column("after_snapshot", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_taxonomy_versions_entity", "taxonomy_versions", ["entity_type", "entity_id"])
    op.create_index("idx_taxonomy_versions_action", "taxonomy_versions", ["action"])


def downgrade() -> None:
    op.drop_table("taxonomy_versions")
