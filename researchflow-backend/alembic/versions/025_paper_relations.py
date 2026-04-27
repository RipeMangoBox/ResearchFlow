"""Add paper_relations table — paper-to-paper baseline / cite DAG.

Materialized from `agent_blackboard_items` rows where item_type='reference_role_map'.
A separate service (paper_relation_service.materialize_relations) walks the
classifications, fuzzy-matches each ref_title against the papers table, and
inserts a row here when a confident match is found.

The vault exporter renders a "## 直接 baseline" / "## 引用本文" section using
this table; Obsidian's graph view picks up the wiki-link edges naturally.

Additive only — no existing tables modified.

Revision ID: 025
Revises: 024
"""

revision = "025"
down_revision = "024"

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


def upgrade() -> None:
    op.create_table(
        "paper_relations",
        sa.Column("id", postgresql.UUID(as_uuid=True),
                  server_default=sa.text("gen_random_uuid()"),
                  primary_key=True),
        sa.Column("source_paper_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("papers.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("target_paper_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("papers.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("relation_type", sa.String(32), nullable=False),
        # direct_baseline / comparison_baseline / method_source / formula_source
        # / dataset_source / benchmark_source / same_task_prior_work / extends
        # / cites / followup_of (reverse edge)
        sa.Column("evidence", sa.Text, server_default=""),
        sa.Column("confidence", sa.Numeric(3, 2), nullable=True),
        sa.Column("ref_index", sa.String(20), nullable=True),  # e.g. "[12]"
        sa.Column("ref_title_raw", sa.Text, nullable=True),    # what the source called it
        sa.Column("match_method", sa.String(20), server_default="title_fuzzy"),
        # title_fuzzy / arxiv_id / doi / dblp_key
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("source_paper_id", "target_paper_id", "relation_type",
                            name="uq_paper_relations_triple"),
    )
    op.create_index("ix_paper_relations_source",
                    "paper_relations", ["source_paper_id"])
    op.create_index("ix_paper_relations_target",
                    "paper_relations", ["target_paper_id"])
    op.create_index("ix_paper_relations_type",
                    "paper_relations", ["relation_type"])


def downgrade() -> None:
    op.drop_index("ix_paper_relations_type", table_name="paper_relations")
    op.drop_index("ix_paper_relations_target", table_name="paper_relations")
    op.drop_index("ix_paper_relations_source", table_name="paper_relations")
    op.drop_table("paper_relations")
