"""v2: metadata observations, taxonomy DAG, method evolution, problems

Revision ID: 005_v2
Revises: 004_idea_graph_schema
Create Date: 2026-04-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

revision = "005_v2"
down_revision = "004_idea_graph_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Metadata Observation Ledger ─────────────────────────────
    op.create_table(
        "metadata_observations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("entity_id", UUID(as_uuid=True), nullable=False),
        sa.Column("field_name", sa.String(100), nullable=False),
        sa.Column("value_json", JSONB, nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("source_url", sa.Text),
        sa.Column("raw_payload_object_key", sa.Text),
        sa.Column("observed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("confidence", sa.Float, default=0.5),
        sa.Column("authority_rank", sa.SmallInteger, default=5),
        sa.Column("conflict_group_id", UUID(as_uuid=True)),
    )
    op.create_index("idx_obs_entity", "metadata_observations", ["entity_type", "entity_id"])
    op.create_index("idx_obs_field", "metadata_observations", ["entity_id", "field_name"])
    op.create_index("idx_obs_source", "metadata_observations", ["source"])

    # ── Canonical Paper Metadata ────────────────────────────────
    op.create_table(
        "canonical_paper_metadata",
        sa.Column("paper_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("canonical_title", sa.Text),
        sa.Column("canonical_authors", JSONB),
        sa.Column("canonical_affiliations", JSONB),
        sa.Column("canonical_venue", sa.String(200)),
        sa.Column("canonical_acceptance_status", sa.String(50)),
        sa.Column("canonical_year", sa.SmallInteger),
        sa.Column("canonical_citation_count", sa.SmallInteger),
        sa.Column("canonical_code_url", sa.Text),
        sa.Column("selected_observation_ids", ARRAY(sa.Text)),
        sa.Column("unresolved_conflicts", JSONB),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.Column("resolver_version", sa.String(20)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Taxonomy Nodes ──────────────────────────────────────────
    op.create_table(
        "taxonomy_nodes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("name_zh", sa.String(200)),
        sa.Column("dimension", sa.String(50), nullable=False),
        sa.Column("aliases", ARRAY(sa.Text)),
        sa.Column("description", sa.Text),
        sa.Column("version", sa.SmallInteger, default=1),
        sa.Column("status", sa.String(20), default="candidate"),
        sa.Column("sort_order", sa.SmallInteger, default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_taxnode_dimension", "taxonomy_nodes", ["dimension"])
    op.create_index("idx_taxnode_name", "taxonomy_nodes", ["name"])
    op.create_index("idx_taxnode_status", "taxonomy_nodes", ["status"])

    # ── Taxonomy Edges ──────────────────────────────────────────
    op.create_table(
        "taxonomy_edges",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("parent_id", UUID(as_uuid=True), nullable=False),
        sa.Column("child_id", UUID(as_uuid=True), nullable=False),
        sa.Column("relation_type", sa.String(30), nullable=False),
        sa.Column("confidence", sa.Float, default=1.0),
        sa.Column("evidence_ids", ARRAY(sa.Text)),
    )
    op.create_index("idx_taxedge_parent", "taxonomy_edges", ["parent_id"])
    op.create_index("idx_taxedge_child", "taxonomy_edges", ["child_id"])
    op.create_unique_constraint("uq_taxonomy_edge", "taxonomy_edges",
                                ["parent_id", "child_id", "relation_type"])

    # ── Paper Facets ────────────────────────────────────────────
    op.create_table(
        "paper_facets",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", UUID(as_uuid=True), nullable=False),
        sa.Column("node_id", UUID(as_uuid=True), nullable=False),
        sa.Column("facet_role", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float, default=1.0),
        sa.Column("source", sa.String(30)),
        sa.Column("evidence_refs", JSONB),
    )
    op.create_index("idx_facet_paper", "paper_facets", ["paper_id"])
    op.create_index("idx_facet_node", "paper_facets", ["node_id"])
    op.create_index("idx_facet_role", "paper_facets", ["facet_role"])
    op.create_unique_constraint("uq_paper_facet", "paper_facets",
                                ["paper_id", "node_id", "facet_role"])

    # ── Problem Nodes ───────────────────────────────────────────
    op.create_table(
        "problem_nodes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("name_zh", sa.Text),
        sa.Column("parent_task_id", UUID(as_uuid=True)),
        sa.Column("scope_facet_ids", ARRAY(sa.Text)),
        sa.Column("symptom", sa.Text),
        sa.Column("root_cause", sa.Text),
        sa.Column("why_common", sa.Text),
        sa.Column("solution_families", JSONB),
        sa.Column("evidence_paper_ids", ARRAY(sa.Text)),
        sa.Column("status", sa.String(20), default="candidate"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_problem_task", "problem_nodes", ["parent_task_id"])

    # ── Problem Claims ──────────────────────────────────────────
    op.create_table(
        "problem_claims",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", UUID(as_uuid=True), nullable=False),
        sa.Column("problem_id", UUID(as_uuid=True), nullable=False),
        sa.Column("claim_type", sa.String(30), nullable=False),
        sa.Column("evidence_refs", JSONB),
        sa.Column("confidence", sa.Float, default=0.5),
    )
    op.create_index("idx_claim_paper", "problem_claims", ["paper_id"])
    op.create_index("idx_claim_problem", "problem_claims", ["problem_id"])

    # ── Method Nodes ────────────────────────────────────────────
    op.create_table(
        "method_nodes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("name_zh", sa.String(200)),
        sa.Column("type", sa.String(50), nullable=False),
        sa.Column("canonical_paper_id", UUID(as_uuid=True)),
        sa.Column("version", sa.String(20)),
        sa.Column("maturity", sa.String(30), default="seed"),
        sa.Column("description", sa.Text),
        sa.Column("promotion_criteria", JSONB),
        sa.Column("downstream_count", sa.SmallInteger, default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_method_name", "method_nodes", ["name"])
    op.create_index("idx_method_maturity", "method_nodes", ["maturity"])

    # ── Method Slots ────────────────────────────────────────────
    op.create_table(
        "method_slots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("method_id", UUID(as_uuid=True), nullable=False),
        sa.Column("slot_name", sa.String(100), nullable=False),
        sa.Column("default_description", sa.Text),
        sa.Column("sort_order", sa.SmallInteger, default=0),
    )
    op.create_index("idx_mslot_method", "method_slots", ["method_id"])
    op.create_unique_constraint("uq_method_slot", "method_slots",
                                ["method_id", "slot_name"])

    # ── Method Edges ────────────────────────────────────────────
    op.create_table(
        "method_edges",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("parent_method_id", UUID(as_uuid=True), nullable=False),
        sa.Column("child_method_id", UUID(as_uuid=True), nullable=False),
        sa.Column("relation_type", sa.String(30), nullable=False),
        sa.Column("scope_facet_ids", ARRAY(sa.Text)),
        sa.Column("changed_slot_ids", ARRAY(sa.Text)),
        sa.Column("delta_description", sa.Text),
        sa.Column("evidence_refs", JSONB),
        sa.Column("confidence", sa.Float, default=0.5),
        sa.Column("status", sa.String(20), default="candidate"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_medge_parent", "method_edges", ["parent_method_id"])
    op.create_index("idx_medge_child", "method_edges", ["child_method_id"])
    op.create_index("idx_medge_type", "method_edges", ["relation_type"])

    # ── Method Applications ─────────────────────────────────────
    op.create_table(
        "method_applications",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", UUID(as_uuid=True), nullable=False),
        sa.Column("method_id", UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(30), nullable=False),
        sa.Column("task_id", UUID(as_uuid=True)),
        sa.Column("scenario_id", UUID(as_uuid=True)),
        sa.Column("dataset_ids", ARRAY(sa.Text)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_mapp_paper", "method_applications", ["paper_id"])
    op.create_index("idx_mapp_method", "method_applications", ["method_id"])
    op.create_unique_constraint("uq_method_application", "method_applications",
                                ["paper_id", "method_id", "role"])

    # ── Add acceptance_type to papers ───────────────────────────
    op.add_column("papers", sa.Column("acceptance_type", sa.String(50)))
    op.add_column("papers", sa.Column("review_scores", JSONB))
    op.add_column("papers", sa.Column("dblp_key", sa.String(200)))


def downgrade() -> None:
    op.drop_column("papers", "dblp_key")
    op.drop_column("papers", "review_scores")
    op.drop_column("papers", "acceptance_type")
    op.drop_table("method_applications")
    op.drop_table("method_edges")
    op.drop_table("method_slots")
    op.drop_table("method_nodes")
    op.drop_table("problem_claims")
    op.drop_table("problem_nodes")
    op.drop_table("paper_facets")
    op.drop_table("taxonomy_edges")
    op.drop_table("taxonomy_nodes")
    op.drop_table("canonical_paper_metadata")
    op.drop_table("metadata_observations")
