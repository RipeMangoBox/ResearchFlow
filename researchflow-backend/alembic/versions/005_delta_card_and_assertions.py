"""Add DeltaCard truth layer, assertion-based graph model, review/override/alias tables.

New tables:
  - delta_cards (intermediate truth layer)
  - graph_nodes (unified node registry)
  - graph_assertions (replaces graph_edges with lifecycle)
  - graph_assertion_evidence (assertion-evidence linking)
  - review_tasks (audit queue)
  - human_overrides (manual correction tracking)
  - aliases (entity name normalization)

Modified tables:
  - idea_deltas: add delta_card_id FK
  - evidence_units: add delta_card_id FK

Data migration:
  - method_deltas → delta_cards
  - graph_edges → graph_assertions (via graph_nodes)

Revision ID: 005
Revises: 004
Create Date: 2026-04-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── delta_cards ────────────────────────────────────────────────
    op.create_table(
        "delta_cards",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("paper_id", UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("analysis_id", UUID(as_uuid=True), sa.ForeignKey("paper_analyses.id")),
        sa.Column("frame_id", UUID(as_uuid=True), sa.ForeignKey("paradigm_templates.id")),
        sa.Column("baseline_paradigm", sa.Text),
        sa.Column("primary_bottleneck_id", UUID(as_uuid=True), sa.ForeignKey("project_bottlenecks.id")),
        sa.Column("changed_slot_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("unchanged_slot_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("mechanism_family_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("delta_statement", sa.Text, nullable=False),
        sa.Column("key_ideas_ranked", JSONB),
        sa.Column("structurality_score", sa.Float),
        sa.Column("extensionability_score", sa.Float),
        sa.Column("transferability_score", sa.Float),
        sa.Column("assumptions", ARRAY(sa.Text)),
        sa.Column("failure_modes", ARRAY(sa.Text)),
        sa.Column("evaluation_context", sa.Text),
        sa.Column("evidence_refs", ARRAY(UUID(as_uuid=True))),
        sa.Column("extraction_confidence", sa.Float),
        sa.Column("linkage_confidence", sa.Float),
        sa.Column("evidence_confidence", sa.Float),
        sa.Column("status", sa.String(20), server_default="draft"),
        sa.Column("model_provider", sa.String(50)),
        sa.Column("model_name", sa.String(100)),
        sa.Column("prompt_version", sa.String(20)),
        sa.Column("schema_version", sa.String(20)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_delta_cards_paper", "delta_cards", ["paper_id"])
    op.create_index("idx_delta_cards_status", "delta_cards", ["status"])
    op.create_index("idx_delta_cards_frame", "delta_cards", ["frame_id"])
    op.create_index("idx_delta_cards_bottleneck", "delta_cards", ["primary_bottleneck_id"])

    # ── graph_nodes ────────────────────────────────────────────────
    op.create_table(
        "graph_nodes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("node_type", sa.String(30), nullable=False),
        sa.Column("ref_table", sa.String(50), nullable=False),
        sa.Column("ref_id", UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(20), server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_graph_nodes_type", "graph_nodes", ["node_type"])
    op.create_index("idx_graph_nodes_ref", "graph_nodes", ["ref_table", "ref_id"], unique=True)
    op.create_index("idx_graph_nodes_status", "graph_nodes", ["status"])

    # ── graph_assertions ───────────────────────────────────────────
    op.create_table(
        "graph_assertions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("from_node_id", UUID(as_uuid=True), sa.ForeignKey("graph_nodes.id"), nullable=False),
        sa.Column("to_node_id", UUID(as_uuid=True), sa.ForeignKey("graph_nodes.id"), nullable=False),
        sa.Column("edge_type", sa.String(50), nullable=False),
        sa.Column("assertion_source", sa.String(30), nullable=False, server_default="system_inferred"),
        sa.Column("confidence", sa.Float),
        sa.Column("status", sa.String(20), server_default="candidate"),
        sa.Column("reviewed_by", sa.String(50)),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
        sa.Column("metadata", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_assertions_from", "graph_assertions", ["from_node_id"])
    op.create_index("idx_assertions_to", "graph_assertions", ["to_node_id"])
    op.create_index("idx_assertions_type", "graph_assertions", ["edge_type"])
    op.create_index("idx_assertions_status", "graph_assertions", ["status"])
    op.create_index("idx_assertions_source", "graph_assertions", ["assertion_source"])

    # ── graph_assertion_evidence ───────────────────────────────────
    op.create_table(
        "graph_assertion_evidence",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("assertion_id", UUID(as_uuid=True), sa.ForeignKey("graph_assertions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("evidence_unit_id", UUID(as_uuid=True), sa.ForeignKey("evidence_units.id", ondelete="CASCADE"), nullable=False),
        sa.Column("weight", sa.Float),
        sa.Column("role", sa.String(30), nullable=False, server_default="supports"),
    )
    op.create_index("idx_assertion_evidence_assertion", "graph_assertion_evidence", ["assertion_id"])
    op.create_index("idx_assertion_evidence_unit", "graph_assertion_evidence", ["evidence_unit_id"])

    # ── review_tasks ───────────────────────────────────────────────
    op.create_table(
        "review_tasks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("target_type", sa.String(30), nullable=False),
        sa.Column("target_id", UUID(as_uuid=True), nullable=False),
        sa.Column("task_type", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("priority", sa.SmallInteger, server_default="3"),
        sa.Column("assigned_to", sa.String(50)),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_review_tasks_status", "review_tasks", ["status"])
    op.create_index("idx_review_tasks_target", "review_tasks", ["target_type", "target_id"])
    op.create_index("idx_review_tasks_priority", "review_tasks", ["priority"])

    # ── human_overrides ────────────────────────────────────────────
    op.create_table(
        "human_overrides",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("target_type", sa.String(30), nullable=False),
        sa.Column("target_id", UUID(as_uuid=True), nullable=False),
        sa.Column("field_name", sa.String(100), nullable=False),
        sa.Column("old_value", JSONB),
        sa.Column("new_value", JSONB),
        sa.Column("reason", sa.Text),
        sa.Column("overridden_by", sa.String(50)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_overrides_target", "human_overrides", ["target_type", "target_id"])

    # ── aliases ────────────────────────────────────────────────────
    op.create_table(
        "aliases",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("entity_type", sa.String(30), nullable=False),
        sa.Column("entity_id", UUID(as_uuid=True), nullable=False),
        sa.Column("alias", sa.Text, nullable=False),
        sa.Column("source", sa.String(30), server_default="auto_detected"),
        sa.Column("confidence", sa.Float),
    )
    op.create_index("idx_aliases_entity", "aliases", ["entity_type", "entity_id"])
    op.create_index("idx_aliases_alias", "aliases", ["alias"])

    # ── Modify existing tables ─────────────────────────────────────

    # idea_deltas: add delta_card_id FK
    op.add_column("idea_deltas", sa.Column(
        "delta_card_id", UUID(as_uuid=True),
        sa.ForeignKey("delta_cards.id"),
        nullable=True,
    ))
    op.create_index("idx_idea_deltas_delta_card", "idea_deltas", ["delta_card_id"])

    # evidence_units: add delta_card_id FK
    op.add_column("evidence_units", sa.Column(
        "delta_card_id", UUID(as_uuid=True),
        sa.ForeignKey("delta_cards.id"),
        nullable=True,
    ))
    op.create_index("idx_evidence_units_delta_card", "evidence_units", ["delta_card_id"])

    # ── Data migration: method_deltas → delta_cards ────────────────
    op.execute("""
        INSERT INTO delta_cards (id, paper_id, analysis_id, delta_statement, status,
                                 structurality_score, schema_version, created_at, updated_at)
        SELECT id, paper_id, analysis_id,
               COALESCE(primary_gain_source, paradigm_name || ' delta'),
               'published',
               CASE WHEN is_structural THEN 0.8 ELSE 0.3 END,
               'migrated_from_method_deltas',
               created_at, updated_at
        FROM method_deltas
    """)

    # ── Data migration: graph_edges → graph_nodes + graph_assertions ──
    # Step 1: Create graph_nodes for all unique source/target references in graph_edges
    op.execute("""
        INSERT INTO graph_nodes (node_type, ref_table, ref_id)
        SELECT DISTINCT source_type,
               CASE source_type
                   WHEN 'idea_delta' THEN 'idea_deltas'
                   WHEN 'evidence_unit' THEN 'evidence_units'
                   WHEN 'slot' THEN 'slots'
                   WHEN 'mechanism_family' THEN 'mechanism_families'
                   WHEN 'bottleneck' THEN 'project_bottlenecks'
                   WHEN 'paper' THEN 'papers'
                   ELSE source_type
               END,
               source_id
        FROM graph_edges
        ON CONFLICT (ref_table, ref_id) DO NOTHING
    """)
    op.execute("""
        INSERT INTO graph_nodes (node_type, ref_table, ref_id)
        SELECT DISTINCT target_type,
               CASE target_type
                   WHEN 'idea_delta' THEN 'idea_deltas'
                   WHEN 'evidence_unit' THEN 'evidence_units'
                   WHEN 'slot' THEN 'slots'
                   WHEN 'mechanism_family' THEN 'mechanism_families'
                   WHEN 'bottleneck' THEN 'project_bottlenecks'
                   WHEN 'paper' THEN 'papers'
                   ELSE target_type
               END,
               target_id
        FROM graph_edges
        ON CONFLICT (ref_table, ref_id) DO NOTHING
    """)

    # Step 2: Create graph_assertions from graph_edges
    op.execute("""
        INSERT INTO graph_assertions (from_node_id, to_node_id, edge_type,
                                      assertion_source, confidence, status, metadata, created_at)
        SELECT gn_from.id, gn_to.id, ge.edge_type,
               CASE ge.assertion_source
                   WHEN 'asserted_by_paper' THEN 'paper_asserted'
                   WHEN 'inferred_by_system' THEN 'system_inferred'
                   WHEN 'verified_by_human' THEN 'human_verified'
                   ELSE 'system_inferred'
               END,
               ge.confidence,
               'published',
               ge.metadata,
               ge.created_at
        FROM graph_edges ge
        JOIN graph_nodes gn_from ON gn_from.ref_id = ge.source_id
            AND gn_from.ref_table = CASE ge.source_type
                WHEN 'idea_delta' THEN 'idea_deltas'
                WHEN 'evidence_unit' THEN 'evidence_units'
                WHEN 'slot' THEN 'slots'
                WHEN 'mechanism_family' THEN 'mechanism_families'
                WHEN 'bottleneck' THEN 'project_bottlenecks'
                WHEN 'paper' THEN 'papers'
                ELSE ge.source_type
            END
        JOIN graph_nodes gn_to ON gn_to.ref_id = ge.target_id
            AND gn_to.ref_table = CASE ge.target_type
                WHEN 'idea_delta' THEN 'idea_deltas'
                WHEN 'evidence_unit' THEN 'evidence_units'
                WHEN 'slot' THEN 'slots'
                WHEN 'mechanism_family' THEN 'mechanism_families'
                WHEN 'bottleneck' THEN 'project_bottlenecks'
                WHEN 'paper' THEN 'papers'
                ELSE ge.target_type
            END
    """)

    # Step 3: Migrate evidence links from graph_edges to graph_assertion_evidence
    op.execute("""
        INSERT INTO graph_assertion_evidence (assertion_id, evidence_unit_id, role)
        SELECT ga.id, ge.evidence_id, 'supports'
        FROM graph_edges ge
        JOIN graph_nodes gn_from ON gn_from.ref_id = ge.source_id
            AND gn_from.ref_table = CASE ge.source_type
                WHEN 'idea_delta' THEN 'idea_deltas'
                WHEN 'evidence_unit' THEN 'evidence_units'
                WHEN 'slot' THEN 'slots'
                WHEN 'mechanism_family' THEN 'mechanism_families'
                WHEN 'bottleneck' THEN 'project_bottlenecks'
                WHEN 'paper' THEN 'papers'
                ELSE ge.source_type
            END
        JOIN graph_nodes gn_to ON gn_to.ref_id = ge.target_id
            AND gn_to.ref_table = CASE ge.target_type
                WHEN 'idea_delta' THEN 'idea_deltas'
                WHEN 'evidence_unit' THEN 'evidence_units'
                WHEN 'slot' THEN 'slots'
                WHEN 'mechanism_family' THEN 'mechanism_families'
                WHEN 'bottleneck' THEN 'project_bottlenecks'
                WHEN 'paper' THEN 'papers'
                ELSE ge.target_type
            END
        JOIN graph_assertions ga ON ga.from_node_id = gn_from.id
            AND ga.to_node_id = gn_to.id
            AND ga.edge_type = ge.edge_type
        WHERE ge.evidence_id IS NOT NULL
    """)


def downgrade() -> None:
    # Drop new FK columns
    op.drop_index("idx_evidence_units_delta_card", "evidence_units")
    op.drop_column("evidence_units", "delta_card_id")
    op.drop_index("idx_idea_deltas_delta_card", "idea_deltas")
    op.drop_column("idea_deltas", "delta_card_id")

    # Drop new tables (reverse order of creation)
    op.drop_table("aliases")
    op.drop_table("human_overrides")
    op.drop_table("review_tasks")
    op.drop_table("graph_assertion_evidence")
    op.drop_table("graph_assertions")
    op.drop_table("graph_nodes")
    op.drop_table("delta_cards")
