"""Add knowledge graph schema: idea_deltas, slots, mechanism_families,
graph_edges, implementation_units. Extend evidence_units, papers,
project_bottlenecks with graph FK fields.

Revision ID: 004
Revises: 003
Create Date: 2026-04-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── slots ───────────────────────────────────────────────────
    op.create_table(
        "slots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("paradigm_id", UUID(as_uuid=True), sa.ForeignKey("paradigm_templates.id"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("slot_type", sa.String(50)),
        sa.Column("is_required", sa.Boolean, server_default="true"),
        sa.Column("sort_order", sa.SmallInteger, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── mechanism_families ──────────────────────────────────────
    op.create_table(
        "mechanism_families",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("domain", sa.String(100)),
        sa.Column("description", sa.Text),
        sa.Column("parent_id", UUID(as_uuid=True), sa.ForeignKey("mechanism_families.id")),
        sa.Column("aliases", ARRAY(sa.Text)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── idea_deltas (core graph object) ─────────────────────────
    op.create_table(
        "idea_deltas",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("analysis_id", UUID(as_uuid=True), sa.ForeignKey("paper_analyses.id")),
        sa.Column("primary_bottleneck_id", UUID(as_uuid=True), sa.ForeignKey("project_bottlenecks.id")),
        sa.Column("paradigm_id", UUID(as_uuid=True), sa.ForeignKey("paradigm_templates.id")),
        sa.Column("delta_statement", sa.Text, nullable=False),
        sa.Column("changed_slots", JSONB),
        sa.Column("mechanism_family_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("structurality_score", sa.Float),
        sa.Column("transferability_score", sa.Float),
        sa.Column("local_keyness_score", sa.Float),
        sa.Column("field_keyness_score", sa.Float),
        sa.Column("confidence", sa.Float),
        sa.Column("evidence_count", sa.Integer, server_default="0"),
        sa.Column("publish_status", sa.String(20), server_default="draft"),
        sa.Column("is_structural", sa.Boolean),
        sa.Column("primary_gain_source", sa.String(100)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_idea_deltas_paper", "idea_deltas", ["paper_id"])
    op.create_index("idx_idea_deltas_bottleneck", "idea_deltas", ["primary_bottleneck_id"])
    op.create_index("idx_idea_deltas_paradigm", "idea_deltas", ["paradigm_id"])
    op.create_index("idx_idea_deltas_publish", "idea_deltas", ["publish_status"])

    # ── graph_edges (unified edge table) ────────────────────────
    op.create_table(
        "graph_edges",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_type", sa.String(30), nullable=False),
        sa.Column("source_id", UUID(as_uuid=True), nullable=False),
        sa.Column("target_type", sa.String(30), nullable=False),
        sa.Column("target_id", UUID(as_uuid=True), nullable=False),
        sa.Column("edge_type", sa.String(50), nullable=False),
        sa.Column("assertion_source", sa.String(20), nullable=False, server_default="inferred_by_system"),
        sa.Column("confidence", sa.Float),
        sa.Column("evidence_id", UUID(as_uuid=True)),
        sa.Column("metadata", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_edges_source", "graph_edges", ["source_type", "source_id"])
    op.create_index("idx_edges_target", "graph_edges", ["target_type", "target_id"])
    op.create_index("idx_edges_type", "graph_edges", ["edge_type"])
    op.create_index("idx_edges_assertion", "graph_edges", ["assertion_source"])

    # ── implementation_units ────────────────────────────────────
    op.create_table(
        "implementation_units",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id")),
        sa.Column("idea_delta_id", UUID(as_uuid=True), sa.ForeignKey("idea_deltas.id")),
        sa.Column("repo_url", sa.Text),
        sa.Column("file_path", sa.Text),
        sa.Column("class_or_function", sa.Text),
        sa.Column("description", sa.Text),
        sa.Column("config_snippet", sa.Text),
        sa.Column("shape_trace", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Extend existing tables ──────────────────────────────────

    # evidence_units: add graph FK links
    op.add_column("evidence_units", sa.Column("idea_delta_id", UUID(as_uuid=True), nullable=True))
    op.add_column("evidence_units", sa.Column("slot_id", UUID(as_uuid=True), nullable=True))

    # project_bottlenecks: add domain and paradigm link
    op.add_column("project_bottlenecks", sa.Column("domain", sa.String(100), nullable=True))
    op.add_column("project_bottlenecks", sa.Column("paradigm_id", UUID(as_uuid=True), nullable=True))

    # papers: add scholarly backbone fields
    op.add_column("papers", sa.Column("openalex_id", sa.String(50), nullable=True))
    op.add_column("papers", sa.Column("cited_by_count", sa.SmallInteger, nullable=True))
    op.add_column("papers", sa.Column("role_in_kb", sa.String(30), nullable=True))


def downgrade() -> None:
    # Remove paper fields
    op.drop_column("papers", "role_in_kb")
    op.drop_column("papers", "cited_by_count")
    op.drop_column("papers", "openalex_id")

    # Remove bottleneck fields
    op.drop_column("project_bottlenecks", "paradigm_id")
    op.drop_column("project_bottlenecks", "domain")

    # Remove evidence_units fields
    op.drop_column("evidence_units", "slot_id")
    op.drop_column("evidence_units", "idea_delta_id")

    # Drop new tables
    op.drop_table("implementation_units")
    op.drop_table("graph_edges")
    op.drop_table("idea_deltas")
    op.drop_table("mechanism_families")
    op.drop_table("slots")
