"""Add ephemeral input states, per-claim confidence, and evidence basis.

- Extend paper_state enum with: ephemeral_received, canonicalized, enriched,
  archived_or_expired
- Add is_ephemeral, expires_at, retention_days to papers
- Add confidence_notes JSONB to paper_analyses
- Add confidence, basis, source_page to evidence_units
- Create evidence_basis enum

Revision ID: 002
Revises: 001
Create Date: 2026-04-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Extend paper_state enum ─────────────────────────────────
    # PostgreSQL requires ALTER TYPE ... ADD VALUE for enum extension
    op.execute("ALTER TYPE paper_state ADD VALUE IF NOT EXISTS 'ephemeral_received' BEFORE 'wait'")
    op.execute("ALTER TYPE paper_state ADD VALUE IF NOT EXISTS 'canonicalized' AFTER 'ephemeral_received'")
    op.execute("ALTER TYPE paper_state ADD VALUE IF NOT EXISTS 'enriched' AFTER 'canonicalized'")
    op.execute("ALTER TYPE paper_state ADD VALUE IF NOT EXISTS 'archived_or_expired' AFTER 'analysis_mismatch'")

    # ── Create evidence_basis enum ──────────────────────────────
    evidence_basis = sa.Enum(
        "code_verified", "experiment_backed", "text_stated",
        "inferred", "speculative",
        name="evidence_basis",
    )
    evidence_basis.create(op.get_bind(), checkfirst=True)

    # ── Add ephemeral columns to papers ─────────────────────────
    op.add_column("papers", sa.Column("is_ephemeral", sa.Boolean, server_default="false", nullable=False))
    op.add_column("papers", sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("papers", sa.Column("retention_days", sa.SmallInteger, server_default="30", nullable=True))

    # Index for cleanup job: find expired ephemeral papers
    op.create_index(
        "idx_papers_ephemeral_expires",
        "papers",
        ["expires_at"],
        postgresql_where="is_ephemeral AND expires_at IS NOT NULL",
    )

    # ── Add confidence_notes to paper_analyses ──────────────────
    op.add_column("paper_analyses", sa.Column("confidence_notes", JSONB, nullable=True))

    # ── Add confidence, basis, source_page to evidence_units ────
    op.add_column("evidence_units", sa.Column("confidence", sa.Float, nullable=True))
    op.add_column("evidence_units", sa.Column(
        "basis", evidence_basis, nullable=True,
    ))
    op.add_column("evidence_units", sa.Column("source_page", sa.SmallInteger, nullable=True))


def downgrade() -> None:
    # ── Remove evidence_units columns ───────────────────────────
    op.drop_column("evidence_units", "source_page")
    op.drop_column("evidence_units", "basis")
    op.drop_column("evidence_units", "confidence")

    # ── Remove paper_analyses column ────────────────────────────
    op.drop_column("paper_analyses", "confidence_notes")

    # ── Remove papers columns ───────────────────────────────────
    op.drop_index("idx_papers_ephemeral_expires", "papers")
    op.drop_column("papers", "retention_days")
    op.drop_column("papers", "expires_at")
    op.drop_column("papers", "is_ephemeral")

    # ── Drop evidence_basis enum ────────────────────────────────
    sa.Enum(name="evidence_basis").drop(op.get_bind(), checkfirst=True)

    # Note: PostgreSQL does not support removing values from an existing enum.
    # The 4 new paper_state values will remain but are harmless.
