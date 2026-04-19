"""V6: candidate queue, agent infrastructure, KB profiles, structured reports.

16 new tables + DomainSpec scope extensions.

Revision ID: 016
Revises: 015
Create Date: 2026-04-19
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ═══════════════════════════════════════════════════
    # 1. DomainSpec: scope + budget extensions
    # ═══════════════════════════════════════════════════
    op.add_column("domain_specs", sa.Column("scope_modalities", ARRAY(sa.Text), server_default="{}"))
    op.add_column("domain_specs", sa.Column("scope_tasks", ARRAY(sa.Text), server_default="{}"))
    op.add_column("domain_specs", sa.Column("scope_paradigms", ARRAY(sa.Text), server_default="{}"))
    op.add_column("domain_specs", sa.Column("scope_seed_methods", ARRAY(sa.Text), server_default="{}"))
    op.add_column("domain_specs", sa.Column("scope_seed_models", ARRAY(sa.Text), server_default="{}"))
    op.add_column("domain_specs", sa.Column("scope_seed_datasets", ARRAY(sa.Text), server_default="{}"))
    op.add_column("domain_specs", sa.Column("negative_scope", ARRAY(sa.Text), server_default="{}"))
    op.add_column("domain_specs", sa.Column("budget_metadata_candidates", sa.Integer, server_default="500"))
    op.add_column("domain_specs", sa.Column("budget_shallow_ingest", sa.Integer, server_default="200"))
    op.add_column("domain_specs", sa.Column("budget_deep_ingest", sa.Integer, server_default="50"))
    op.add_column("domain_specs", sa.Column("budget_anchor_methods", sa.Integer, server_default="20"))

    # ═══════════════════════════════════════════════════
    # 2. paper_candidates — candidate pool
    # ═══════════════════════════════════════════════════
    op.create_table(
        "paper_candidates",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("normalized_title", sa.Text),
        sa.Column("arxiv_id", sa.String(30)),
        sa.Column("doi", sa.String(100)),
        sa.Column("s2_paper_id", sa.String(50)),
        sa.Column("openalex_id", sa.String(50)),
        sa.Column("openreview_id", sa.String(100)),
        sa.Column("dblp_id", sa.String(100)),
        sa.Column("paper_link", sa.Text),
        sa.Column("discovered_from_paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id")),
        sa.Column("discovered_from_domain_id", UUID(as_uuid=True), sa.ForeignKey("domain_specs.id")),
        sa.Column("discovery_source", sa.String(30), nullable=False),
        sa.Column("discovery_reason", sa.String(50)),
        sa.Column("relation_hint", sa.String(30)),
        sa.Column("authors_json", JSONB),
        sa.Column("venue", sa.String(100)),
        sa.Column("year", sa.SmallInteger),
        sa.Column("abstract", sa.Text),
        sa.Column("citation_count", sa.Integer),
        sa.Column("code_url", sa.Text),
        sa.Column("metadata_json", JSONB),
        sa.Column("status", sa.String(25), server_default="discovered"),
        sa.Column("absorption_level", sa.SmallInteger, server_default="0"),
        sa.Column("ingested_paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id")),
        sa.Column("reject_reason", sa.Text),
        sa.Column("duplicate_of_candidate_id", UUID(as_uuid=True), sa.ForeignKey("paper_candidates.id")),
        sa.Column("duplicate_of_paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_pc_arxiv_id", "paper_candidates", ["arxiv_id"])
    op.create_index("ix_pc_doi", "paper_candidates", ["doi"])
    op.create_index("ix_pc_status_absorption", "paper_candidates", ["status", "absorption_level"])
    op.create_index("ix_pc_discovered_from", "paper_candidates", ["discovered_from_paper_id"])
    op.create_index("ix_pc_domain", "paper_candidates", ["discovered_from_domain_id", "status"])

    # ═══════════════════════════════════════════════════
    # 3. candidate_scores — multi-stage scoring
    # ═══════════════════════════════════════════════════
    op.create_table(
        "candidate_scores",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("candidate_id", UUID(as_uuid=True), sa.ForeignKey("paper_candidates.id", ondelete="CASCADE"), nullable=False),
        sa.Column("discovery_score", sa.Float),
        sa.Column("deep_ingest_score", sa.Float),
        sa.Column("graph_promotion_score", sa.Float),
        sa.Column("anchor_score", sa.Float),
        sa.Column("discovery_breakdown", JSONB),
        sa.Column("deep_ingest_breakdown", JSONB),
        sa.Column("hard_caps_applied", JSONB),
        sa.Column("boosts_applied", JSONB),
        sa.Column("penalties_applied", JSONB),
        sa.Column("decision", sa.String(25)),
        sa.Column("decision_reason", sa.Text),
        sa.Column("score_version", sa.SmallInteger, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_cs_candidate", "candidate_scores", ["candidate_id"])

    # ═══════════════════════════════════════════════════
    # 4. score_signals — traceable scoring evidence
    # ═══════════════════════════════════════════════════
    op.create_table(
        "score_signals",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("entity_type", sa.String(30), nullable=False),
        sa.Column("entity_id", UUID(as_uuid=True), nullable=False),
        sa.Column("signal_name", sa.String(80), nullable=False),
        sa.Column("signal_value", JSONB, nullable=False),
        sa.Column("signal_strength", sa.Float),
        sa.Column("evidence_refs", JSONB),
        sa.Column("producer", sa.String(30), nullable=False),
        sa.Column("confidence", sa.Float),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_ss_entity", "score_signals", ["entity_type", "entity_id"])

    # ═══════════════════════════════════════════════════
    # 5. agent_runs — agent execution tracking
    # ═══════════════════════════════════════════════════
    op.create_table(
        "agent_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id")),
        sa.Column("candidate_id", UUID(as_uuid=True), sa.ForeignKey("paper_candidates.id")),
        sa.Column("domain_id", UUID(as_uuid=True), sa.ForeignKey("domain_specs.id")),
        sa.Column("agent_name", sa.String(50), nullable=False),
        sa.Column("phase", sa.String(20), nullable=False),
        sa.Column("status", sa.String(15), server_default="running"),
        sa.Column("model_name", sa.String(50)),
        sa.Column("prompt_version", sa.String(20)),
        sa.Column("input_token_count", sa.Integer),
        sa.Column("output_token_count", sa.Integer),
        sa.Column("cost_usd", sa.Float),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("error_message", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_ar_paper", "agent_runs", ["paper_id"])
    op.create_index("ix_ar_candidate", "agent_runs", ["candidate_id"])

    # ═══════════════════════════════════════════════════
    # 6. agent_blackboard_items — inter-agent shared data
    # ═══════════════════════════════════════════════════
    op.create_table(
        "agent_blackboard_items",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", UUID(as_uuid=True), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id")),
        sa.Column("candidate_id", UUID(as_uuid=True), sa.ForeignKey("paper_candidates.id")),
        sa.Column("item_type", sa.String(30), nullable=False),
        sa.Column("value_json", JSONB, nullable=False),
        sa.Column("confidence", sa.Float),
        sa.Column("evidence_refs", JSONB),
        sa.Column("producer_agent", sa.String(50), nullable=False),
        sa.Column("is_verified", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_abi_paper_type", "agent_blackboard_items", ["paper_id", "item_type"])
    op.create_index("ix_abi_candidate_type", "agent_blackboard_items", ["candidate_id", "item_type"])

    # ═══════════════════════════════════════════════════
    # 7. paper_extractions — typed extraction results
    # ═══════════════════════════════════════════════════
    op.create_table(
        "paper_extractions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("extraction_type", sa.String(30), nullable=False),
        sa.Column("value_json", JSONB, nullable=False),
        sa.Column("evidence_refs", JSONB),
        sa.Column("producer_run_id", UUID(as_uuid=True), sa.ForeignKey("agent_runs.id")),
        sa.Column("extraction_version", sa.SmallInteger, server_default="1"),
        sa.Column("review_status", sa.String(20), server_default="auto"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("paper_id", "extraction_type", "extraction_version", name="uq_extraction_paper_type_ver"),
    )
    op.create_index("ix_pe_paper_type", "paper_extractions", ["paper_id", "extraction_type"])

    # ═══════════════════════════════════════════════════
    # 8. reference_role_maps — citation role classification
    # ═══════════════════════════════════════════════════
    op.create_table(
        "reference_role_maps",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ref_index", sa.String(10)),
        sa.Column("ref_title", sa.Text),
        sa.Column("ref_arxiv_id", sa.String(30)),
        sa.Column("ref_candidate_id", UUID(as_uuid=True), sa.ForeignKey("paper_candidates.id")),
        sa.Column("ref_paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="SET NULL")),
        sa.Column("role", sa.String(30), nullable=False),
        sa.Column("role_confidence", sa.Float),
        sa.Column("where_mentioned", ARRAY(sa.Text), server_default="{}"),
        sa.Column("mention_count", sa.SmallInteger, server_default="1"),
        sa.Column("recommended_ingest_level", sa.String(20)),
        sa.Column("recommendation_reason", sa.Text),
        sa.Column("evidence_refs", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_rrm_paper", "reference_role_maps", ["paper_id"])

    # ═══════════════════════════════════════════════════
    # 9. evidence_items — fine-grained evidence index
    # ═══════════════════════════════════════════════════
    op.create_table(
        "evidence_items",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_type", sa.String(30), nullable=False),
        sa.Column("source_id", sa.String(100)),
        sa.Column("section_name", sa.String(50)),
        sa.Column("page", sa.SmallInteger),
        sa.Column("bbox", JSONB),
        sa.Column("text", sa.Text),
        sa.Column("image_object_key", sa.Text),
        sa.Column("table_html", sa.Text),
        sa.Column("formula_latex", sa.Text),
        sa.Column("token_count", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_ei_paper", "evidence_items", ["paper_id"])
    op.create_index("ix_ei_paper_type", "evidence_items", ["paper_id", "source_type"])
    # Note: pgvector embedding column added later when needed

    # ═══════════════════════════════════════════════════
    # 10. graph_node_candidates
    # ═══════════════════════════════════════════════════
    op.create_table(
        "graph_node_candidates",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id")),
        sa.Column("candidate_id", UUID(as_uuid=True), sa.ForeignKey("paper_candidates.id")),
        sa.Column("node_type", sa.String(20), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("name_zh", sa.String(200)),
        sa.Column("one_liner", sa.Text),
        sa.Column("promotion_score", sa.Float),
        sa.Column("promotion_breakdown", JSONB),
        sa.Column("status", sa.String(20), server_default="candidate"),
        sa.Column("promoted_entity_type", sa.String(30)),
        sa.Column("promoted_entity_id", UUID(as_uuid=True)),
        sa.Column("evidence_refs", JSONB),
        sa.Column("confidence", sa.Float),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ═══════════════════════════════════════════════════
    # 11. graph_edge_candidates
    # ═══════════════════════════════════════════════════
    op.create_table(
        "graph_edge_candidates",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id")),
        sa.Column("source_entity_type", sa.String(30), nullable=False),
        sa.Column("source_entity_id", UUID(as_uuid=True)),
        sa.Column("source_candidate_id", UUID(as_uuid=True)),
        sa.Column("target_entity_type", sa.String(30), nullable=False),
        sa.Column("target_entity_id", UUID(as_uuid=True)),
        sa.Column("target_candidate_id", UUID(as_uuid=True)),
        sa.Column("relation_type", sa.String(50), nullable=False),
        sa.Column("slot_name", sa.String(100)),
        sa.Column("confidence_score", sa.Float),
        sa.Column("confidence_breakdown", JSONB),
        sa.Column("one_liner", sa.Text),
        sa.Column("status", sa.String(20), server_default="candidate"),
        sa.Column("promoted_edge_id", UUID(as_uuid=True)),
        sa.Column("evidence_refs", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ═══════════════════════════════════════════════════
    # 12. kb_node_profiles — entity profile pages
    # ═══════════════════════════════════════════════════
    op.create_table(
        "kb_node_profiles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("entity_type", sa.String(30), nullable=False),
        sa.Column("entity_id", UUID(as_uuid=True), nullable=False),
        sa.Column("profile_kind", sa.String(20), server_default="page"),
        sa.Column("lang", sa.String(5), server_default="zh"),
        sa.Column("one_liner", sa.Text),
        sa.Column("short_intro_md", sa.Text),
        sa.Column("detailed_md", sa.Text),
        sa.Column("structured_json", JSONB),
        sa.Column("evidence_refs", JSONB),
        sa.Column("generated_by_run_id", UUID(as_uuid=True), sa.ForeignKey("agent_runs.id")),
        sa.Column("model_name", sa.String(50)),
        sa.Column("prompt_version", sa.String(20)),
        sa.Column("profile_version", sa.SmallInteger, server_default="1"),
        sa.Column("review_status", sa.String(20), server_default="auto"),
        sa.Column("staleness_trigger_count", sa.SmallInteger, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("entity_type", "entity_id", "profile_kind", "lang", name="uq_node_profile_entity"),
    )
    op.create_index("ix_knp_entity", "kb_node_profiles", ["entity_type", "entity_id"])

    # ═══════════════════════════════════════════════════
    # 13. kb_edge_profiles — relationship descriptions
    # ═══════════════════════════════════════════════════
    op.create_table(
        "kb_edge_profiles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("source_entity_type", sa.String(30), nullable=False),
        sa.Column("source_entity_id", UUID(as_uuid=True), nullable=False),
        sa.Column("target_entity_type", sa.String(30), nullable=False),
        sa.Column("target_entity_id", UUID(as_uuid=True), nullable=False),
        sa.Column("relation_type", sa.String(50), nullable=False),
        sa.Column("edge_table", sa.String(30)),
        sa.Column("edge_id", UUID(as_uuid=True)),
        sa.Column("lang", sa.String(5), server_default="zh"),
        sa.Column("one_liner", sa.Text),
        sa.Column("relation_summary", sa.Text),
        sa.Column("source_context", sa.Text),
        sa.Column("target_context", sa.Text),
        sa.Column("display_priority", sa.SmallInteger, server_default="5"),
        sa.Column("evidence_refs", JSONB),
        sa.Column("generated_by_run_id", UUID(as_uuid=True), sa.ForeignKey("agent_runs.id")),
        sa.Column("review_status", sa.String(20), server_default="auto"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_kep_source", "kb_edge_profiles", ["source_entity_type", "source_entity_id"])
    op.create_index("ix_kep_target", "kb_edge_profiles", ["target_entity_type", "target_entity_id"])

    # ═══════════════════════════════════════════════════
    # 14. paper_reports — structured reports
    # ═══════════════════════════════════════════════════
    op.create_table(
        "paper_reports",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("report_version", sa.SmallInteger, server_default="1"),
        sa.Column("title_zh", sa.Text),
        sa.Column("title_en", sa.Text),
        sa.Column("generated_by_run_id", UUID(as_uuid=True), sa.ForeignKey("agent_runs.id")),
        sa.Column("model_name", sa.String(50)),
        sa.Column("prompt_version", sa.String(20)),
        sa.Column("review_status", sa.String(20), server_default="auto"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ═══════════════════════════════════════════════════
    # 15. paper_report_sections
    # ═══════════════════════════════════════════════════
    op.create_table(
        "paper_report_sections",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("report_id", UUID(as_uuid=True), sa.ForeignKey("paper_reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_type", sa.String(30), nullable=False),
        sa.Column("title", sa.Text),
        sa.Column("body_md", sa.Text),
        sa.Column("evidence_refs", JSONB),
        sa.Column("order_index", sa.SmallInteger, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ═══════════════════════════════════════════════════
    # 16. review_queue_items — unified review queue
    # ═══════════════════════════════════════════════════
    op.create_table(
        "review_queue_items",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("item_type", sa.String(30), nullable=False),
        sa.Column("entity_type", sa.String(30), nullable=False),
        sa.Column("entity_id", UUID(as_uuid=True), nullable=False),
        sa.Column("priority_score", sa.Float),
        sa.Column("reason", sa.Text),
        sa.Column("suggested_decision", sa.String(20)),
        sa.Column("evidence_refs", JSONB),
        sa.Column("status", sa.String(25), server_default="pending"),
        sa.Column("reviewed_by", sa.String(50)),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
        sa.Column("review_notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_rqi_status_priority", "review_queue_items", ["status", sa.text("priority_score DESC NULLS LAST")])


def downgrade() -> None:
    op.drop_table("review_queue_items")
    op.drop_table("paper_report_sections")
    op.drop_table("paper_reports")
    op.drop_table("kb_edge_profiles")
    op.drop_table("kb_node_profiles")
    op.drop_table("graph_edge_candidates")
    op.drop_table("graph_node_candidates")
    op.drop_table("evidence_items")
    op.drop_table("reference_role_maps")
    op.drop_table("paper_extractions")
    op.drop_table("agent_blackboard_items")
    op.drop_table("agent_runs")
    op.drop_table("score_signals")
    op.drop_table("candidate_scores")
    op.drop_table("paper_candidates")

    op.drop_column("domain_specs", "budget_anchor_methods")
    op.drop_column("domain_specs", "budget_deep_ingest")
    op.drop_column("domain_specs", "budget_shallow_ingest")
    op.drop_column("domain_specs", "budget_metadata_candidates")
    op.drop_column("domain_specs", "negative_scope")
    op.drop_column("domain_specs", "scope_seed_datasets")
    op.drop_column("domain_specs", "scope_seed_models")
    op.drop_column("domain_specs", "scope_seed_methods")
    op.drop_column("domain_specs", "scope_paradigms")
    op.drop_column("domain_specs", "scope_tasks")
    op.drop_column("domain_specs", "scope_modalities")
