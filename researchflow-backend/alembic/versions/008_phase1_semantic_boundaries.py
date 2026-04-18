"""Phase 1: Semantic boundary restructuring.

New tables:
  - paper_bottleneck_claims: paper-level fact (what a paper claims to solve)
  - project_focus_bottlenecks: project-level decision (what the user cares about)
  - canonical_ideas: cross-paper normalized concept layer
  - contribution_to_canonical_idea: maps paper contributions to canonical ideas
  - delta_card_lineage: independent method lineage DAG edges
  - paradigm_candidates: candidate paradigms pending review
  - slot_candidates: candidate slots pending review
  - mechanism_candidates: candidate mechanisms pending review

Column additions:
  - papers.current_delta_card_id: pointer to current published DeltaCard (append-only)
  - delta_cards.analysis_run_id, source_asset_hash, model_run_id: immutable snapshot provenance

Revision ID: 008
Revises: 007
Create Date: 2026-04-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── papers: current_delta_card_id ──
    op.add_column("papers", sa.Column("current_delta_card_id", UUID(as_uuid=True)))

    # ── delta_cards: append-only provenance fields ──
    op.add_column("delta_cards", sa.Column("analysis_run_id", UUID(as_uuid=True)))
    op.add_column("delta_cards", sa.Column("source_asset_hash", sa.String(64)))
    op.add_column("delta_cards", sa.Column("model_run_id", sa.String(100)))

    # ── paper_bottleneck_claims ──
    op.create_table(
        "paper_bottleneck_claims",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("bottleneck_id", UUID(as_uuid=True), sa.ForeignKey("project_bottlenecks.id"), nullable=False),
        sa.Column("claim_text", sa.Text, nullable=False),
        sa.Column("is_primary", sa.Boolean, server_default="true"),
        sa.Column("is_fundamental", sa.Boolean),
        sa.Column("confidence", sa.Float),
        sa.Column("source", sa.String(30), server_default="system_inferred"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_pbc_paper", "paper_bottleneck_claims", ["paper_id"])
    op.create_index("idx_pbc_bottleneck", "paper_bottleneck_claims", ["bottleneck_id"])

    # ── project_focus_bottlenecks ──
    op.create_table(
        "project_focus_bottlenecks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("bottleneck_id", UUID(as_uuid=True), sa.ForeignKey("project_bottlenecks.id"), nullable=False),
        sa.Column("project_name", sa.String(200)),
        sa.Column("user_description", sa.Text),
        sa.Column("priority", sa.SmallInteger, server_default="3"),
        sa.Column("status", sa.String(20), server_default="active"),
        sa.Column("constraints", JSONB),
        sa.Column("negative_constraints", ARRAY(sa.Text)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_pfb_bottleneck", "project_focus_bottlenecks", ["bottleneck_id"])
    op.create_index("idx_pfb_status", "project_focus_bottlenecks", ["status"])

    # ── canonical_ideas ──
    op.create_table(
        "canonical_ideas",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("domain", sa.String(100)),
        sa.Column("mechanism_family_id", UUID(as_uuid=True), sa.ForeignKey("mechanism_families.id")),
        sa.Column("contribution_count", sa.SmallInteger, server_default="0"),
        sa.Column("status", sa.String(20), server_default="candidate"),
        sa.Column("merged_into_id", UUID(as_uuid=True)),
        sa.Column("aliases", ARRAY(sa.Text)),
        sa.Column("tags", ARRAY(sa.Text)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_canonical_ideas_domain", "canonical_ideas", ["domain"])
    op.create_index("idx_canonical_ideas_status", "canonical_ideas", ["status"])
    op.create_index("idx_canonical_ideas_mechanism", "canonical_ideas", ["mechanism_family_id"])

    # ── contribution_to_canonical_idea ──
    op.create_table(
        "contribution_to_canonical_idea",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("idea_delta_id", UUID(as_uuid=True), sa.ForeignKey("idea_deltas.id", ondelete="CASCADE"), nullable=False),
        sa.Column("canonical_idea_id", UUID(as_uuid=True), sa.ForeignKey("canonical_ideas.id", ondelete="CASCADE"), nullable=False),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("contribution_type", sa.String(30), server_default="instance"),
        sa.Column("confidence", sa.Float),
        sa.Column("source", sa.String(30), server_default="system_inferred"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_ctci_idea_delta", "contribution_to_canonical_idea", ["idea_delta_id"])
    op.create_index("idx_ctci_canonical", "contribution_to_canonical_idea", ["canonical_idea_id"])
    op.create_index("idx_ctci_paper", "contribution_to_canonical_idea", ["paper_id"])

    # ── delta_card_lineage ──
    op.create_table(
        "delta_card_lineage",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("child_delta_card_id", UUID(as_uuid=True), sa.ForeignKey("delta_cards.id", ondelete="CASCADE"), nullable=False),
        sa.Column("parent_delta_card_id", UUID(as_uuid=True), sa.ForeignKey("delta_cards.id", ondelete="CASCADE"), nullable=False),
        sa.Column("relation_type", sa.String(30), nullable=False, server_default="builds_on"),
        sa.Column("confidence", sa.Float),
        sa.Column("status", sa.String(20), server_default="candidate"),
        sa.Column("evidence_type", sa.String(50)),
        sa.Column("source", sa.String(30), server_default="system_inferred"),
        sa.Column("reviewed_by", sa.String(50)),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_lineage_child", "delta_card_lineage", ["child_delta_card_id"])
    op.create_index("idx_lineage_parent", "delta_card_lineage", ["parent_delta_card_id"])
    op.create_index("idx_lineage_status", "delta_card_lineage", ["status"])
    op.create_index("idx_lineage_type", "delta_card_lineage", ["relation_type"])

    # ── paradigm_candidates ──
    op.create_table(
        "paradigm_candidates",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("domain", sa.String(100)),
        sa.Column("description", sa.Text),
        sa.Column("slots_json", JSONB),
        sa.Column("trigger_count", sa.SmallInteger, server_default="1"),
        sa.Column("trigger_paper_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("max_similarity_to_existing", sa.Float),
        sa.Column("most_similar_paradigm_id", UUID(as_uuid=True)),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("promoted_paradigm_id", UUID(as_uuid=True)),
        sa.Column("reviewed_by", sa.String(50)),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
        sa.Column("review_notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_paradigm_cand_status", "paradigm_candidates", ["status"])
    op.create_index("idx_paradigm_cand_domain", "paradigm_candidates", ["domain"])

    # ── slot_candidates ──
    op.create_table(
        "slot_candidates",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("paradigm_candidate_id", UUID(as_uuid=True)),
        sa.Column("paradigm_id", UUID(as_uuid=True)),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("slot_type", sa.String(50)),
        sa.Column("trigger_count", sa.SmallInteger, server_default="1"),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("promoted_slot_id", UUID(as_uuid=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_slot_cand_status", "slot_candidates", ["status"])

    # ── mechanism_candidates ──
    op.create_table(
        "mechanism_candidates",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("domain", sa.String(100)),
        sa.Column("description", sa.Text),
        sa.Column("aliases", ARRAY(sa.Text)),
        sa.Column("trigger_count", sa.SmallInteger, server_default="1"),
        sa.Column("trigger_paper_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("max_similarity_to_existing", sa.Float),
        sa.Column("most_similar_mechanism_id", UUID(as_uuid=True)),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("promoted_mechanism_id", UUID(as_uuid=True)),
        sa.Column("reviewed_by", sa.String(50)),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_mechanism_cand_status", "mechanism_candidates", ["status"])
    op.create_index("idx_mechanism_cand_domain", "mechanism_candidates", ["domain"])


def downgrade() -> None:
    op.drop_table("mechanism_candidates")
    op.drop_table("slot_candidates")
    op.drop_table("paradigm_candidates")
    op.drop_table("delta_card_lineage")
    op.drop_table("contribution_to_canonical_idea")
    op.drop_table("canonical_ideas")
    op.drop_table("project_focus_bottlenecks")
    op.drop_table("paper_bottleneck_claims")
    op.drop_column("delta_cards", "model_run_id")
    op.drop_column("delta_cards", "source_asset_hash")
    op.drop_column("delta_cards", "analysis_run_id")
    op.drop_column("papers", "current_delta_card_id")
