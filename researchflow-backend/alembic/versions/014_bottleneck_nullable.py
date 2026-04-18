"""Make PaperBottleneckClaim.bottleneck_id nullable + add raw_title.

Revision ID: 014
Revises: 013
Create Date: 2026-04-18
"""

from alembic import op
import sqlalchemy as sa

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("paper_bottleneck_claims", "bottleneck_id", nullable=True)
    op.add_column("paper_bottleneck_claims", sa.Column("raw_title", sa.Text))


def downgrade() -> None:
    op.drop_column("paper_bottleneck_claims", "raw_title")
    op.alter_column("paper_bottleneck_claims", "bottleneck_id", nullable=False)
