"""Add search_branches and render_artifacts tables.

Revision ID: 011
Revises: 010
Create Date: 2026-04-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "search_branches",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("search_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("branch_name", sa.String(200), nullable=False),
        sa.Column("hypothesis", sa.Text),
        sa.Column("rejected_patterns", ARRAY(sa.Text)),
        sa.Column("result_paper_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("status", sa.String(20), server_default="active"),
        sa.Column("parent_branch_id", UUID(as_uuid=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_search_branches_session", "search_branches", ["session_id"])

    op.create_table(
        "render_artifacts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("artifact_type", sa.String(30), nullable=False),
        sa.Column("title", sa.Text),
        sa.Column("content_md", sa.Text),
        sa.Column("object_key", sa.Text),
        sa.Column("paper_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("parameters", JSONB),
        sa.Column("generated_by", sa.String(50)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_render_artifacts_type", "render_artifacts", ["artifact_type"])


def downgrade() -> None:
    op.drop_table("render_artifacts")
    op.drop_table("search_branches")
