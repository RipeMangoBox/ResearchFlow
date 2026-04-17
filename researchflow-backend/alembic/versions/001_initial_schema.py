"""Initial schema: papers, paper_assets, paper_versions, paper_analyses,
method_deltas, paradigm_templates, evidence_units, transfer_atoms,
project_bottlenecks, search_sessions, reading_plans, digests,
execution_memories, jobs, model_runs, user_feedback.

Revision ID: 001
Revises:
Create Date: 2026-04-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── Enum types ──────────────────────────────────────────────
    paper_state = sa.Enum(
        "wait", "downloaded", "l1_metadata", "l2_parsed",
        "l3_skimmed", "l4_deep", "checked",
        "skip", "missing", "too_large", "analysis_mismatch",
        name="paper_state",
    )
    importance = sa.Enum("S", "A", "B", "C", "D", name="importance")
    analysis_level = sa.Enum(
        "l1_metadata", "l2_parse", "l3_skim", "l4_deep",
        name="analysis_level",
    )
    asset_type = sa.Enum(
        "raw_pdf", "raw_html", "extracted_text", "figure",
        "code_snapshot", "skim_report", "deep_report", "exported_md",
        name="asset_type",
    )
    period_type = sa.Enum("day", "week", "month", name="period_type")
    job_status = sa.Enum(
        "pending", "running", "completed", "failed", "cancelled",
        name="job_status",
    )
    feedback_type = sa.Enum(
        "correction", "confirmation", "rejection", "tag_edit",
        name="feedback_type",
    )
    tier = sa.Enum(
        "A_open_data", "B_open_code", "C_accepted_no_code", "D_preprint",
        name="tier",
    )

    paper_state.create(op.get_bind(), checkfirst=True)
    importance.create(op.get_bind(), checkfirst=True)
    analysis_level.create(op.get_bind(), checkfirst=True)
    asset_type.create(op.get_bind(), checkfirst=True)
    period_type.create(op.get_bind(), checkfirst=True)
    job_status.create(op.get_bind(), checkfirst=True)
    feedback_type.create(op.get_bind(), checkfirst=True)
    tier.create(op.get_bind(), checkfirst=True)

    # ── papers ──────────────────────────────────────────────────
    op.create_table(
        "papers",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("title_sanitized", sa.Text, nullable=False),
        sa.Column("venue", sa.String(100)),
        sa.Column("year", sa.SmallInteger),
        sa.Column("category", sa.String(100), nullable=False),
        sa.Column("state", paper_state, nullable=False, server_default="wait"),
        sa.Column("importance", importance, server_default="C"),
        sa.Column("tier", tier),
        sa.Column("paper_link", sa.Text),
        sa.Column("project_link", sa.Text),
        sa.Column("arxiv_id", sa.String(20)),
        sa.Column("doi", sa.String(100)),
        sa.Column("keep_score", sa.Float),
        sa.Column("analysis_priority", sa.Float),
        sa.Column("structurality_score", sa.Float),
        sa.Column("extensionability_score", sa.Float),
        sa.Column("authors", JSONB),
        sa.Column("abstract", sa.Text),
        sa.Column("keywords", ARRAY(sa.Text)),
        sa.Column("license", sa.String(50)),
        sa.Column("funding", sa.Text),
        sa.Column("open_data", sa.Boolean, server_default="false"),
        sa.Column("open_code", sa.Boolean, server_default="false"),
        sa.Column("code_url", sa.Text),
        sa.Column("data_url", sa.Text),
        sa.Column("tags", ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column("mechanism_family", sa.String(100)),
        sa.Column("supervision_type", sa.String(50)),
        sa.Column("inference_pattern", sa.String(100)),
        sa.Column("core_operator", sa.Text),
        sa.Column("primary_logic", sa.Text),
        sa.Column("claims", ARRAY(sa.Text)),
        sa.Column("pdf_path_local", sa.Text),
        sa.Column("pdf_object_key", sa.Text),
        sa.Column("collected_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("downloaded_at", sa.DateTime(timezone=True)),
        sa.Column("analyzed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("source", sa.String(50)),
        sa.Column("source_ref", sa.Text),
    )
    # Add vector column separately (pgvector syntax)
    op.execute("ALTER TABLE papers ADD COLUMN embedding vector(1536)")

    op.create_index("idx_papers_state", "papers", ["state"])
    op.create_index("idx_papers_category", "papers", ["category"])
    op.create_index("idx_papers_venue_year", "papers", ["venue", "year"])
    op.execute("CREATE INDEX idx_papers_tags ON papers USING GIN(tags)")
    op.execute("CREATE INDEX idx_papers_title_fts ON papers USING GIN(to_tsvector('english', title))")

    # ── paper_assets ────────────────────────────────────────────
    op.create_table(
        "paper_assets",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("asset_type", asset_type, nullable=False),
        sa.Column("object_key", sa.Text, nullable=False),
        sa.Column("mime_type", sa.String(100)),
        sa.Column("size_bytes", sa.BigInteger),
        sa.Column("checksum", sa.String(64)),
        sa.Column("metadata", JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_paper_assets_paper", "paper_assets", ["paper_id"])

    # ── paper_versions ──────────────────────────────────────────
    op.create_table(
        "paper_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.SmallInteger, nullable=False, server_default="1"),
        sa.Column("diff_summary", sa.Text),
        sa.Column("arxiv_version", sa.String(10)),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("paper_id", "version", name="uq_paper_version"),
    )

    # ── paper_analyses ──────────────────────────────────────────
    op.create_table(
        "paper_analyses",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("level", analysis_level, nullable=False),
        sa.Column("model_provider", sa.String(50)),
        sa.Column("model_name", sa.String(100)),
        sa.Column("prompt_version", sa.String(20), nullable=False),
        sa.Column("schema_version", sa.String(20), nullable=False),
        sa.Column("confidence", sa.Float),
        sa.Column("extracted_sections", JSONB),
        sa.Column("extracted_formulas", ARRAY(sa.Text)),
        sa.Column("extracted_tables", JSONB),
        sa.Column("figure_captions", JSONB),
        sa.Column("problem_summary", sa.Text),
        sa.Column("method_summary", sa.Text),
        sa.Column("evidence_summary", sa.Text),
        sa.Column("core_intuition", sa.Text),
        sa.Column("changed_slots", ARRAY(sa.Text)),
        sa.Column("is_plugin_patch", sa.Boolean),
        sa.Column("worth_deep_read", sa.Boolean),
        sa.Column("full_report_md", sa.Text),
        sa.Column("full_report_object_key", sa.Text),
        sa.Column("evidence_spans", JSONB),
        sa.Column("superseded_by", UUID(as_uuid=True)),
        sa.Column("is_current", sa.Boolean, server_default="true"),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_analyses_paper", "paper_analyses", ["paper_id"])
    op.create_index("idx_analyses_level", "paper_analyses", ["level"])
    op.execute(
        "CREATE INDEX idx_analyses_current ON paper_analyses(paper_id, is_current) WHERE is_current"
    )

    # ── method_deltas ───────────────────────────────────────────
    op.create_table(
        "method_deltas",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("analysis_id", UUID(as_uuid=True)),
        sa.Column("paradigm_name", sa.String(100), nullable=False),
        sa.Column("paradigm_version", sa.String(20), server_default="v1"),
        sa.Column("slots", JSONB, nullable=False),
        sa.Column("is_structural", sa.Boolean),
        sa.Column("primary_gain_source", sa.String(100)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_deltas_paper", "method_deltas", ["paper_id"])
    op.create_index("idx_deltas_paradigm", "method_deltas", ["paradigm_name"])

    # ── paradigm_templates ──────────────────────────────────────
    op.create_table(
        "paradigm_templates",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("version", sa.String(20), server_default="v1"),
        sa.Column("domain", sa.String(100)),
        sa.Column("slots", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── evidence_units ──────────────────────────────────────────
    op.create_table(
        "evidence_units",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("analysis_id", UUID(as_uuid=True)),
        sa.Column("atom_type", sa.String(30), nullable=False),
        sa.Column("claim", sa.Text, nullable=False),
        sa.Column("evidence_type", sa.String(30)),
        sa.Column("causal_strength", sa.Float),
        sa.Column("source_section", sa.String(50)),
        sa.Column("source_quote", sa.Text),
        sa.Column("conditions", sa.Text),
        sa.Column("failure_modes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.execute("ALTER TABLE evidence_units ADD COLUMN embedding vector(1536)")
    op.create_index("idx_evidence_paper", "evidence_units", ["paper_id"])
    op.create_index("idx_evidence_type", "evidence_units", ["atom_type"])

    # ── transfer_atoms ──────────────────────────────────────────
    op.create_table(
        "transfer_atoms",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_domain", sa.String(100), nullable=False),
        sa.Column("target_domain", sa.String(100), nullable=False),
        sa.Column("mechanism", sa.Text, nullable=False),
        sa.Column("preconditions", sa.Text),
        sa.Column("failure_risks", sa.Text),
        sa.Column("confidence", sa.Float),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── project_bottlenecks ─────────────────────────────────────
    op.create_table(
        "project_bottlenecks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("symptom_query", sa.Text),
        sa.Column("latent_need", sa.Text),
        sa.Column("constraints", JSONB),
        sa.Column("status", sa.String(20), server_default="active"),
        sa.Column("priority", sa.SmallInteger, server_default="3"),
        sa.Column("related_paper_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("rejected_patterns", ARRAY(sa.Text)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.execute("ALTER TABLE project_bottlenecks ADD COLUMN embedding vector(1536)")

    # ── search_sessions ─────────────────────────────────────────
    op.create_table(
        "search_sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("symptom_query", sa.Text, nullable=False),
        sa.Column("latent_need", sa.Text),
        sa.Column("candidate_bottleneck_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("rejected_solution_patterns", ARRAY(sa.Text)),
        sa.Column("search_branches", JSONB),
        sa.Column("result_paper_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("rewrite_history", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── reading_plans ───────────────────────────────────────────
    op.create_table(
        "reading_plans",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("title", sa.Text),
        sa.Column("bottleneck_id", UUID(as_uuid=True), sa.ForeignKey("project_bottlenecks.id")),
        sa.Column("canonical_baselines", ARRAY(UUID(as_uuid=True)), nullable=False, server_default="{}"),
        sa.Column("structural_improvements", ARRAY(UUID(as_uuid=True)), nullable=False, server_default="{}"),
        sa.Column("strong_team_followups", ARRAY(UUID(as_uuid=True)), nullable=False, server_default="{}"),
        sa.Column("patches_and_negatives", ARRAY(UUID(as_uuid=True)), nullable=False, server_default="{}"),
        sa.Column("rationale", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── digests ─────────────────────────────────────────────────
    op.create_table(
        "digests",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("period_type", period_type, nullable=False),
        sa.Column("period_start", sa.Date, nullable=False),
        sa.Column("period_end", sa.Date, nullable=False),
        sa.Column("source_paper_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("source_bottleneck_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("source_search_session_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("rendered_text", sa.Text, nullable=False),
        sa.Column("render_version", sa.String(20), server_default="v1"),
        sa.Column("metadata", JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_digests_period", "digests", ["period_type", "period_start"])

    # ── execution_memories ──────────────────────────────────────
    op.create_table(
        "execution_memories",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("repo_url", sa.Text),
        sa.Column("env_fingerprint", JSONB, nullable=False),
        sa.Column("failed_command", sa.Text, nullable=False),
        sa.Column("error_message", sa.Text),
        sa.Column("fix_action", sa.Text, nullable=False),
        sa.Column("verified", sa.Boolean, server_default="false"),
        sa.Column("applicable_conditions", sa.Text),
        sa.Column("invalidation_conditions", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_exec_mem_repo", "execution_memories", ["repo_url"])

    # ── jobs ────────────────────────────────────────────────────
    op.create_table(
        "jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_type", sa.String(50), nullable=False),
        sa.Column("status", job_status, nullable=False, server_default="pending"),
        sa.Column("payload", JSONB, nullable=False, server_default="{}"),
        sa.Column("result", JSONB),
        sa.Column("error", sa.Text),
        sa.Column("priority", sa.SmallInteger, server_default="5"),
        sa.Column("retries", sa.SmallInteger, server_default="0"),
        sa.Column("max_retries", sa.SmallInteger, server_default="3"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_jobs_status", "jobs", ["status", "priority"])
    op.create_index("idx_jobs_type", "jobs", ["job_type"])

    # ── model_runs ──────────────────────────────────────────────
    op.create_table(
        "model_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_id", UUID(as_uuid=True), sa.ForeignKey("jobs.id")),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id")),
        sa.Column("model_provider", sa.String(50), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("prompt_version", sa.String(20)),
        sa.Column("input_tokens", sa.Integer),
        sa.Column("output_tokens", sa.Integer),
        sa.Column("cost_usd", sa.Float),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── user_feedback ───────────────────────────────────────────
    op.create_table(
        "user_feedback",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("target_type", sa.String(30), nullable=False),
        sa.Column("target_id", UUID(as_uuid=True), nullable=False),
        sa.Column("feedback_type", feedback_type, nullable=False),
        sa.Column("old_value", JSONB),
        sa.Column("new_value", JSONB),
        sa.Column("comment", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_feedback_target", "user_feedback", ["target_type", "target_id"])


def downgrade() -> None:
    tables = [
        "user_feedback", "model_runs", "jobs", "execution_memories",
        "digests", "reading_plans", "search_sessions", "project_bottlenecks",
        "transfer_atoms", "evidence_units", "paradigm_templates", "method_deltas",
        "paper_analyses", "paper_versions", "paper_assets", "papers",
    ]
    for t in tables:
        op.drop_table(t)

    enums = [
        "tier", "feedback_type", "job_status", "period_type",
        "asset_type", "analysis_level", "importance", "paper_state",
    ]
    for e in enums:
        sa.Enum(name=e).drop(op.get_bind(), checkfirst=True)
