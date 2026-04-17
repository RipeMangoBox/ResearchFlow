"""Add baseline evolution tracking to delta_cards and paradigm_templates.

Enables method lineage DAG: tracking which methods build on which,
how improvements chain together, and when a method becomes established
enough to be considered a new baseline.

Revision ID: 007
Revises: 006
Create Date: 2026-04-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, UUID

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # delta_cards: baseline evolution fields
    op.add_column("delta_cards", sa.Column("parent_delta_card_ids", ARRAY(UUID(as_uuid=True))))
    op.add_column("delta_cards", sa.Column("baseline_paper_ids", ARRAY(UUID(as_uuid=True))))
    op.add_column("delta_cards", sa.Column("lineage_depth", sa.SmallInteger, server_default="0"))
    op.add_column("delta_cards", sa.Column("is_established_baseline", sa.Boolean, server_default="false"))
    op.add_column("delta_cards", sa.Column("downstream_count", sa.SmallInteger, server_default="0"))

    op.create_index("idx_delta_cards_lineage", "delta_cards", ["lineage_depth"])
    op.create_index("idx_delta_cards_baseline", "delta_cards", ["is_established_baseline"])

    # paradigm_templates: evolution tracking
    op.add_column("paradigm_templates", sa.Column("superseded_by", UUID(as_uuid=True)))
    op.add_column("paradigm_templates", sa.Column("anchor_paper_id", UUID(as_uuid=True)))
    op.add_column("paradigm_templates", sa.Column("adoption_count", sa.SmallInteger, server_default="0"))


def downgrade() -> None:
    op.drop_column("paradigm_templates", "adoption_count")
    op.drop_column("paradigm_templates", "anchor_paper_id")
    op.drop_column("paradigm_templates", "superseded_by")

    op.drop_index("idx_delta_cards_baseline", "delta_cards")
    op.drop_index("idx_delta_cards_lineage", "delta_cards")
    op.drop_column("delta_cards", "downstream_count")
    op.drop_column("delta_cards", "is_established_baseline")
    op.drop_column("delta_cards", "lineage_depth")
    op.drop_column("delta_cards", "baseline_paper_ids")
    op.drop_column("delta_cards", "parent_delta_card_ids")
