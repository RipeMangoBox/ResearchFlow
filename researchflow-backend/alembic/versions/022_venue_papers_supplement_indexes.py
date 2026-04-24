"""Add partial indexes for venue_papers supplement queries.

Revision ID: 022
Revises: 021
"""

revision = "022"
down_revision = "021"

from alembic import op


def upgrade() -> None:
    # Partial index: find rows needing supplement (pdf_url, arxiv_id, or doi empty)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_vp_supplement_pending
        ON venue_papers (conf_year)
        WHERE (pdf_url = '' OR arxiv_id = '' OR doi = '')
    """)
    # Partial index: openreview_forum_id lookup (not in original migration)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_vp_orf_id
        ON venue_papers (openreview_forum_id)
        WHERE openreview_forum_id != ''
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_vp_supplement_pending")
    op.execute("DROP INDEX IF EXISTS ix_vp_orf_id")
