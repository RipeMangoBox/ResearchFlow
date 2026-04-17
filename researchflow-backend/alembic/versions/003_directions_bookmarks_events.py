"""Add direction_cards, user_bookmarks, user_events tables.

Revision ID: 003
Revises: 002
Create Date: 2026-04-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "direction_cards",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("bottleneck_id", UUID(as_uuid=True)),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("rationale", sa.Text),
        sa.Column("is_structural", sa.Boolean),
        sa.Column("required_assets", JSONB),
        sa.Column("estimated_cost", sa.Text),
        sa.Column("max_risk", sa.Text),
        sa.Column("confidence", sa.Float),
        sa.Column("related_paper_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("feasibility_plan_md", sa.Text),
        sa.Column("source_topic", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "user_bookmarks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("target_type", sa.String(30), nullable=False),
        sa.Column("target_id", UUID(as_uuid=True), nullable=False),
        sa.Column("note", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_bookmarks_target", "user_bookmarks", ["target_type", "target_id"])

    op.create_table(
        "user_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("target_type", sa.String(30)),
        sa.Column("target_id", UUID(as_uuid=True)),
        sa.Column("payload", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_events_type", "user_events", ["event_type"])


def downgrade() -> None:
    op.drop_table("user_events")
    op.drop_table("user_bookmarks")
    op.drop_table("direction_cards")
