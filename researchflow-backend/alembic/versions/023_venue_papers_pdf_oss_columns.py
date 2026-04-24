"""Add PDF/OSS tracking columns to venue_papers.

Supports bulk PDF download → OSS pipeline:
  - pdf_object_key: OSS key (e.g. venue-papers/pdf/CVPR_2025/2504.12345.pdf)
  - pdf_size_bytes: validated PDF file size
  - pdf_checksum: SHA-256 hex digest
  - pdf_downloaded_at: when PDF was successfully stored

Also adds partial index for rows pending PDF download.

Revision ID: 023
Revises: 022
"""

revision = "023"
down_revision = "022"

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    op.add_column("venue_papers", sa.Column("pdf_object_key", sa.String(500), server_default=""))
    op.add_column("venue_papers", sa.Column("pdf_size_bytes", sa.Integer, nullable=True))
    op.add_column("venue_papers", sa.Column("pdf_checksum", sa.String(64), server_default=""))
    op.add_column("venue_papers", sa.Column("pdf_downloaded_at", sa.DateTime, nullable=True))

    # Partial index: rows that have a pdf_url but haven't been downloaded to OSS yet
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_vp_pdf_pending
        ON venue_papers (id)
        WHERE pdf_url != '' AND (pdf_object_key IS NULL OR pdf_object_key = '')
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_vp_pdf_pending")
    op.drop_column("venue_papers", "pdf_downloaded_at")
    op.drop_column("venue_papers", "pdf_checksum")
    op.drop_column("venue_papers", "pdf_size_bytes")
    op.drop_column("venue_papers", "pdf_object_key")
