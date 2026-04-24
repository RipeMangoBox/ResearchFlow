"""venue_papers — pre-crawled conference/journal paper metadata index.

Stores ALL accepted papers from 21 venues, used as local lookup cache
to avoid per-paper API calls during enrich.

Design principles:
  - Dedup chain: arxiv_id > doi > openreview_forum_id > title_normalized+author
  - authors stored as JSONB (preserves institution, not just names)
  - Heavy data (PDF, figures) stays on OSS — this table is metadata only
  - extra_data JSONB holds venue-specific fields (session, poster_position, etc.)
  - Covering index on (venue, year) for fast conf_year scans

Revision ID: 017
Revises: 016
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "017"
down_revision = "016"


def upgrade():
    op.create_table(
        "venue_papers",
        # ── Identity ──
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),

        # ── Dedup keys (all indexed) ──
        sa.Column("arxiv_id", sa.String(30), server_default="", index=True),
        sa.Column("doi", sa.String(200), server_default="", index=True),
        sa.Column("openreview_forum_id", sa.String(50), server_default=""),
        sa.Column("title_normalized", sa.String(500), nullable=False, index=True),

        # ── Core metadata ──
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("authors", JSONB, server_default="'[]'"),  # [{fullname, institution}]
        sa.Column("abstract", sa.Text, server_default=""),
        sa.Column("venue", sa.String(50), nullable=False),
        sa.Column("year", sa.SmallInteger, nullable=False),
        sa.Column("conf_year", sa.String(50), nullable=False),  # e.g. "CVPR_2026"

        # ── Acceptance ──
        sa.Column("decision", sa.String(100), server_default=""),     # raw: "Accept (Oral)"
        sa.Column("acceptance_type", sa.String(50), server_default=""),  # normalized: "oral"

        # ── Links ──
        sa.Column("paper_link", sa.String(500), server_default=""),   # canonical landing page
        sa.Column("pdf_url", sa.String(500), server_default=""),      # direct PDF link
        sa.Column("code_url", sa.String(500), server_default=""),
        sa.Column("project_url", sa.String(500), server_default=""),

        # ── Enrichment fields ──
        sa.Column("keywords", sa.Text, server_default=""),
        sa.Column("topic", sa.String(200), server_default=""),

        # ── Source tracking ──
        sa.Column("source_type", sa.String(50), server_default=""),   # e.g. "vc_json", "cvf_html"
        sa.Column("source_url", sa.String(500), server_default=""),

        # ── Venue-specific overflow ──
        sa.Column("extra_data", JSONB, server_default="'{}'"),
        # Examples of what goes in extra_data:
        #   VC JSON: session, room_name, poster_position, starttime, eventtype, virtualsite_url, sourceid
        #   ACL: paper_id (anthology ID for acceptance_type inference)
        #   SIGGRAPH: keywords_raw (SIG/TOG)
        #   Journals: citation_count
        #   HF Daily: hf_upvotes

        # ── Timestamps ──
        sa.Column("crawled_at", sa.DateTime, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime, server_default=sa.text("now()")),
    )

    # Composite index for venue+year scans (most common query pattern)
    op.create_index("ix_venue_papers_venue_year", "venue_papers", ["venue", "year"])


def downgrade():
    op.drop_index("ix_venue_papers_venue_year")
    op.drop_table("venue_papers")
