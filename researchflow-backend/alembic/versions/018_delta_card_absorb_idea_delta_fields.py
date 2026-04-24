"""Absorb IdeaDelta fields into DeltaCard + drop redundant columns/tables.

Phase 1 of Agent Pipeline Refactor:
  1A. Add IdeaDelta fields to delta_cards (dual-write transition)
  1B. (evidence_spans kept — used by L2 parse for GROBID data; L4 write removed in code only)
  1C. Drop delta_cards.parent_delta_card_ids (redundant with delta_card_lineage table)
  1D. Drop method_deltas table (legacy, superseded by delta_cards)

Revision ID: 018
Revises: 017
"""
from alembic import op
import sqlalchemy as sa

revision = "018"
down_revision = "017"


def upgrade():
    # ── 1A: Absorb IdeaDelta fields into delta_cards ──
    op.add_column("delta_cards", sa.Column("publish_status", sa.String(20), server_default="draft"))
    op.add_column("delta_cards", sa.Column("evidence_count", sa.Integer, server_default="0"))
    op.add_column("delta_cards", sa.Column("local_keyness_score", sa.Float, nullable=True))
    op.add_column("delta_cards", sa.Column("field_keyness_score", sa.Float, nullable=True))
    op.add_column("delta_cards", sa.Column("changed_slots_json", sa.dialects.postgresql.JSONB, nullable=True))
    op.add_column("delta_cards", sa.Column("is_structural", sa.Boolean, nullable=True))
    op.create_index("idx_delta_cards_publish", "delta_cards", ["publish_status"])

    # Backfill from idea_deltas
    op.execute("""
        UPDATE delta_cards dc SET
            publish_status = COALESCE(id.publish_status, 'draft'),
            evidence_count = COALESCE(id.evidence_count, 0),
            local_keyness_score = id.local_keyness_score,
            field_keyness_score = id.field_keyness_score,
            changed_slots_json = id.changed_slots,
            is_structural = id.is_structural
        FROM idea_deltas id
        WHERE id.delta_card_id = dc.id
    """)

    # ── 1B: evidence_spans kept (used by L2 parse for GROBID data) ──
    # L4 redundant write removed in code only (analysis_service.py).

    # ── 1C: Backfill lineage table, then drop parent_delta_card_ids ──
    # Ensure all parent links exist in delta_card_lineage before dropping the array
    op.execute("""
        INSERT INTO delta_card_lineage (id, child_delta_card_id, parent_delta_card_id,
                                        relation_type, confidence, status, source)
        SELECT gen_random_uuid(), dc.id, unnest(dc.parent_delta_card_ids),
               'builds_on', 0.7, 'candidate', 'migrated_from_array'
        FROM delta_cards dc
        WHERE dc.parent_delta_card_ids IS NOT NULL
          AND array_length(dc.parent_delta_card_ids, 1) > 0
        ON CONFLICT DO NOTHING
    """)
    op.drop_column("delta_cards", "parent_delta_card_ids")

    # ── 1D: Drop method_deltas table ──
    op.drop_table("method_deltas")


def downgrade():
    # 1D: Recreate method_deltas
    op.create_table(
        "method_deltas",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paradigm_name", sa.Text),
        sa.Column("slots", sa.dialects.postgresql.JSONB),
        sa.Column("is_structural", sa.Boolean),
        sa.Column("primary_gain_source", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # 1C: Re-add parent_delta_card_ids
    op.add_column("delta_cards", sa.Column(
        "parent_delta_card_ids",
        sa.dialects.postgresql.ARRAY(sa.dialects.postgresql.UUID(as_uuid=True)),
        nullable=True,
    ))

    # 1A: Drop new delta_cards columns
    op.drop_index("idx_delta_cards_publish", table_name="delta_cards")
    op.drop_column("delta_cards", "is_structural")
    op.drop_column("delta_cards", "changed_slots_json")
    op.drop_column("delta_cards", "field_keyness_score")
    op.drop_column("delta_cards", "local_keyness_score")
    op.drop_column("delta_cards", "evidence_count")
    op.drop_column("delta_cards", "publish_status")
