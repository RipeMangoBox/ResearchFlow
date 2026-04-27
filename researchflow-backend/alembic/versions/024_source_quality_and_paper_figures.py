"""Add papers.source_quality + paper_figures table (additive only).

Goals:
  - Tag every paper with a quality bucket so test/low-quality rows can be
    filtered or soft-deleted without scanning text.
  - Give figures a first-class row instead of being buried inside
    PaperAnalysis.extracted_figure_images JSONB. New ingestions should write
    here; the JSONB column is kept for backward compatibility (no drop).

Nothing existing is dropped or modified. A separate, optional backfill
script (scripts/backfill_paper_figures.py) can copy historical rows from
the JSONB column into the new table.

Revision ID: 024
Revises: 023
"""

revision = "024"
down_revision = "023"

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


def upgrade() -> None:
    # ── papers.source_quality ─────────────────────────────────────
    # Use a CHECK constraint instead of a real ENUM type to keep
    # downgrades cheap and avoid pg_enum bloat.
    op.add_column(
        "papers",
        sa.Column(
            "source_quality",
            sa.String(16),
            nullable=False,
            server_default="normal",
        ),
    )
    op.execute("""
        ALTER TABLE papers
        ADD CONSTRAINT ck_papers_source_quality
        CHECK (source_quality IN ('test', 'low', 'normal', 'published'))
    """)
    op.create_index(
        "ix_papers_source_quality",
        "papers",
        ["source_quality"],
    )

    # ── paper_figures table ───────────────────────────────────────
    op.create_table(
        "paper_figures",
        sa.Column("id", postgresql.UUID(as_uuid=True),
                  server_default=sa.text("gen_random_uuid()"),
                  primary_key=True),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("papers.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("label", sa.String(64), nullable=False),
        sa.Column("type", sa.String(16), nullable=False, server_default="figure"),
        sa.Column("semantic_role", sa.String(32), server_default="other"),
        sa.Column("page_num", sa.SmallInteger, nullable=True),
        sa.Column("bbox", postgresql.JSONB, nullable=True),
        sa.Column("object_key", sa.String(500), nullable=False),
        sa.Column("public_url", sa.Text, nullable=True),
        sa.Column("caption", sa.Text, server_default=""),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("width", sa.Integer, nullable=True),
        sa.Column("height", sa.Integer, nullable=True),
        sa.Column("size_bytes", sa.Integer, nullable=True),
        sa.Column("extraction_method", sa.String(32),
                  server_default="vlm_precise"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(),
                  nullable=False),
        sa.UniqueConstraint("paper_id", "label",
                            name="uq_paper_figures_paper_label"),
    )
    op.create_index(
        "ix_paper_figures_paper_id",
        "paper_figures",
        ["paper_id"],
    )
    op.create_index(
        "ix_paper_figures_role",
        "paper_figures",
        ["semantic_role"],
    )


def downgrade() -> None:
    op.drop_index("ix_paper_figures_role", table_name="paper_figures")
    op.drop_index("ix_paper_figures_paper_id", table_name="paper_figures")
    op.drop_table("paper_figures")

    op.drop_index("ix_papers_source_quality", table_name="papers")
    op.execute("ALTER TABLE papers DROP CONSTRAINT IF EXISTS ck_papers_source_quality")
    op.drop_column("papers", "source_quality")
