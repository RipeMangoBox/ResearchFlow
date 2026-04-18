"""CQRS-lite projection views for read-optimized queries.

Creates 4 materialized views:
  - paper_search_docs: denormalized paper + analysis + scores for search
  - idea_search_docs: denormalized idea + paper + evidence for idea search
  - lineage_view: flattened lineage DAG with paper titles
  - review_queue_view: pending reviews with target summaries

Revision ID: 010
Revises: 009
Create Date: 2026-04-18
"""

from alembic import op
import sqlalchemy as sa

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # paper_search_docs: denormalized view for fast paper search
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS paper_search_docs AS
        SELECT
            p.id AS paper_id,
            p.title,
            p.venue,
            p.year,
            p.category,
            p.state,
            p.importance,
            p.tags,
            p.core_operator,
            p.primary_logic,
            p.abstract,
            p.keep_score,
            p.structurality_score,
            p.extensionability_score,
            p.analysis_priority,
            p.open_code,
            p.open_data,
            p.code_url,
            p.arxiv_id,
            p.doi,
            p.mechanism_family,
            p.current_delta_card_id,
            p.cited_by_count,
            p.role_in_kb,
            dc.delta_statement,
            dc.structurality_score AS dc_structurality,
            dc.extensionability_score AS dc_extensionability,
            dc.transferability_score AS dc_transferability,
            dc.status AS dc_status,
            dc.key_ideas_ranked,
            (SELECT count(*) FROM evidence_units eu WHERE eu.paper_id = p.id) AS evidence_count,
            (SELECT count(*) FROM idea_deltas id2 WHERE id2.paper_id = p.id) AS idea_count,
            p.created_at,
            p.updated_at
        FROM papers p
        LEFT JOIN delta_cards dc ON dc.id = p.current_delta_card_id
        WHERE p.state NOT IN ('archived_or_expired', 'skip')
    """)
    op.execute("CREATE UNIQUE INDEX idx_psd_paper_id ON paper_search_docs (paper_id)")
    op.execute("CREATE INDEX idx_psd_category ON paper_search_docs (category)")
    op.execute("CREATE INDEX idx_psd_structurality ON paper_search_docs (structurality_score)")

    # idea_search_docs: denormalized idea + paper for idea-centric search
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS idea_search_docs AS
        SELECT
            id2.id AS idea_delta_id,
            id2.paper_id,
            p.title AS paper_title,
            p.venue,
            p.year,
            p.category,
            id2.delta_statement,
            id2.changed_slots,
            id2.structurality_score,
            id2.transferability_score,
            id2.confidence,
            id2.publish_status,
            id2.evidence_count,
            id2.is_structural,
            id2.delta_card_id,
            dc.key_ideas_ranked,
            dc.assumptions,
            dc.failure_modes,
            dc.baseline_paradigm,
            p.open_code,
            p.code_url,
            p.mechanism_family,
            p.tags,
            id2.created_at
        FROM idea_deltas id2
        JOIN papers p ON p.id = id2.paper_id
        LEFT JOIN delta_cards dc ON dc.id = id2.delta_card_id
    """)
    op.execute("CREATE UNIQUE INDEX idx_isd_idea_id ON idea_search_docs (idea_delta_id)")
    op.execute("CREATE INDEX idx_isd_paper ON idea_search_docs (paper_id)")
    op.execute("CREATE INDEX idx_isd_structurality ON idea_search_docs (structurality_score)")
    op.execute("CREATE INDEX idx_isd_publish ON idea_search_docs (publish_status)")

    # lineage_view: flattened lineage with paper titles
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS lineage_view AS
        SELECT
            dcl.id AS lineage_id,
            dcl.child_delta_card_id,
            dcl.parent_delta_card_id,
            dcl.relation_type,
            dcl.confidence,
            dcl.status,
            dcl.evidence_type,
            dcl.source,
            child_dc.paper_id AS child_paper_id,
            child_p.title AS child_paper_title,
            child_dc.delta_statement AS child_delta_statement,
            child_dc.structurality_score AS child_structurality,
            parent_dc.paper_id AS parent_paper_id,
            parent_p.title AS parent_paper_title,
            parent_dc.delta_statement AS parent_delta_statement,
            parent_dc.is_established_baseline AS parent_is_baseline,
            dcl.created_at
        FROM delta_card_lineage dcl
        JOIN delta_cards child_dc ON child_dc.id = dcl.child_delta_card_id
        JOIN papers child_p ON child_p.id = child_dc.paper_id
        JOIN delta_cards parent_dc ON parent_dc.id = dcl.parent_delta_card_id
        JOIN papers parent_p ON parent_p.id = parent_dc.paper_id
    """)
    op.execute("CREATE UNIQUE INDEX idx_lv_lineage_id ON lineage_view (lineage_id)")
    op.execute("CREATE INDEX idx_lv_child ON lineage_view (child_delta_card_id)")
    op.execute("CREATE INDEX idx_lv_parent ON lineage_view (parent_delta_card_id)")
    op.execute("CREATE INDEX idx_lv_status ON lineage_view (status)")

    # review_queue_view: pending reviews with target summaries
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS review_queue_view AS
        SELECT
            rt.id AS task_id,
            rt.target_type,
            rt.target_id,
            rt.task_type,
            rt.status,
            rt.priority,
            rt.assigned_to,
            rt.notes,
            rt.created_at,
            CASE
                WHEN rt.target_type = 'delta_card' THEN (
                    SELECT p.title FROM delta_cards dc JOIN papers p ON p.id = dc.paper_id
                    WHERE dc.id = rt.target_id LIMIT 1
                )
                WHEN rt.target_type = 'assertion' THEN (
                    SELECT ga.edge_type FROM graph_assertions ga WHERE ga.id = rt.target_id LIMIT 1
                )
                WHEN rt.target_type = 'lineage' THEN (
                    SELECT child_p.title FROM delta_card_lineage dcl
                    JOIN delta_cards child_dc ON child_dc.id = dcl.child_delta_card_id
                    JOIN papers child_p ON child_p.id = child_dc.paper_id
                    WHERE dcl.id = rt.target_id LIMIT 1
                )
                WHEN rt.target_type = 'paradigm_candidate' THEN (
                    SELECT pc.name FROM paradigm_candidates pc WHERE pc.id = rt.target_id LIMIT 1
                )
                ELSE NULL
            END AS target_summary
        FROM review_tasks rt
        WHERE rt.status IN ('pending', 'in_progress')
    """)
    op.execute("CREATE UNIQUE INDEX idx_rqv_task_id ON review_queue_view (task_id)")
    op.execute("CREATE INDEX idx_rqv_status ON review_queue_view (status)")
    op.execute("CREATE INDEX idx_rqv_priority ON review_queue_view (priority)")


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS review_queue_view")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS lineage_view")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS idea_search_docs")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS paper_search_docs")
