"""Add key_equations, key_figures, same_family_paper_ids to delta_cards.

Revision ID: 012
Revises: 011
Create Date: 2026-04-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("delta_cards", sa.Column("key_equations", JSONB))
    op.add_column("delta_cards", sa.Column("key_figures", JSONB))
    op.add_column("delta_cards", sa.Column("same_family_paper_ids", ARRAY(UUID(as_uuid=True))))


def downgrade() -> None:
    op.drop_column("delta_cards", "same_family_paper_ids")
    op.drop_column("delta_cards", "key_figures")
    op.drop_column("delta_cards", "key_equations")
