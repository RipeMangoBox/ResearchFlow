"""Add extracted_figure_images to paper_analyses.

Revision ID: 013
Revises: 012
Create Date: 2026-04-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("paper_analyses", sa.Column("extracted_figure_images", JSONB))


def downgrade() -> None:
    op.drop_column("paper_analyses", "extracted_figure_images")
