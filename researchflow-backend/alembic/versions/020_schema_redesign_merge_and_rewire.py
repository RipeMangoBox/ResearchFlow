"""Schema redesign: merge IdeaDelta→DeltaCard, merge ReviewTask→review_queue, rewire FKs.

Phase 1 of the 4-layer architecture alignment:
  - Add delta_card_id to contribution_to_canonical_idea (prep for IdeaDelta deletion)
  - Backfill FK data from idea_deltas → delta_cards
  - Update graph_nodes ref_table from 'idea_deltas' to 'delta_cards'
  - Merge review_tasks into review_queue_items, rename to review_queue
  - Drop idea_delta_id columns after backfill
  - Migrate sparse Paper columns to metadata_observations

Revision ID: 020
Revises: 019
"""
from alembic import op
import sqlalchemy as sa

revision = "020"
down_revision = "019"


def upgrade():
    # ══════════════════════════════════════════════════════════════
    # STEP 1: Prep columns
    # ══════════════════════════════════════════════════════════════

    # contribution_to_canonical_idea: add delta_card_id column
    op.add_column("contribution_to_canonical_idea", sa.Column(
        "delta_card_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True,
    ))

    # review_queue_items: absorb fields from review_tasks
    op.add_column("review_queue_items", sa.Column("task_type", sa.String(30), nullable=True))
    op.add_column("review_queue_items", sa.Column("assigned_to", sa.String(50), nullable=True))

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Backfill data
    # ══════════════════════════════════════════════════════════════

    # Backfill contribution_to_canonical_idea.delta_card_id from idea_deltas
    op.execute("""
        UPDATE contribution_to_canonical_idea c
        SET delta_card_id = id.delta_card_id
        FROM idea_deltas id
        WHERE c.idea_delta_id = id.id
          AND id.delta_card_id IS NOT NULL
    """)

    # Update graph_nodes: change ref_table from 'idea_deltas' to 'delta_cards'
    op.execute("""
        UPDATE graph_nodes gn
        SET ref_table = 'delta_cards',
            node_type = CASE WHEN node_type = 'idea_delta' THEN 'delta_card' ELSE node_type END,
            ref_id = COALESCE(
                (SELECT delta_card_id FROM idea_deltas WHERE id = gn.ref_id),
                gn.ref_id
            )
        WHERE gn.ref_table = 'idea_deltas'
    """)

    # Migrate review_tasks → review_queue_items
    op.execute("""
        INSERT INTO review_queue_items (id, item_type, entity_type, entity_id,
            task_type, priority_score, reason, status, assigned_to, created_at)
        SELECT gen_random_uuid(), task_type, target_type, target_id,
            task_type, priority, notes, status, assigned_to, created_at
        FROM review_tasks
        ON CONFLICT DO NOTHING
    """)

    # Migrate sparse Paper columns to metadata_observations
    op.execute("""
        INSERT INTO metadata_observations (id, entity_type, entity_id, field_name,
            value_json, source, authority_rank, observed_at)
        SELECT gen_random_uuid(), 'paper', id, 'supervision_type',
            to_jsonb(supervision_type), 'system_migration', 10, now()
        FROM papers WHERE supervision_type IS NOT NULL
        ON CONFLICT DO NOTHING
    """)
    for col in ['inference_pattern', 'license', 'funding', 'data_url']:
        op.execute(f"""
            INSERT INTO metadata_observations (id, entity_type, entity_id, field_name,
                value_json, source, authority_rank, observed_at)
            SELECT gen_random_uuid(), 'paper', id, '{col}',
                to_jsonb({col}), 'system_migration', 10, now()
            FROM papers WHERE {col} IS NOT NULL
            ON CONFLICT DO NOTHING
        """)

    # ══════════════════════════════════════════════════════════════
    # STEP 3: FK rewire + column drops
    # ══════════════════════════════════════════════════════════════

    # contribution_to_canonical_idea: add FK, drop old column
    op.create_foreign_key(
        "fk_contrib_delta_card", "contribution_to_canonical_idea", "delta_cards",
        ["delta_card_id"], ["id"],
    )
    op.execute("ALTER TABLE contribution_to_canonical_idea DROP CONSTRAINT IF EXISTS contribution_to_canonical_idea_idea_delta_id_fkey")
    op.drop_column("contribution_to_canonical_idea", "idea_delta_id")

    # evidence_units: drop idea_delta_id (delta_card_id already exists and populated)
    op.drop_column("evidence_units", "idea_delta_id")

    # Rename review_queue_items → review_queue
    op.rename_table("review_queue_items", "review_queue")

    # ══════════════════════════════════════════════════════════════
    # STEP 4: Drop Paper sparse columns
    # ══════════════════════════════════════════════════════════════
    for col in ['supervision_type', 'inference_pattern', 'open_data', 'open_code',
                'license', 'funding', 'data_url', 'core_operator', 'primary_logic',
                'claims', 'structurality_score', 'extensionability_score']:
        op.execute(f"ALTER TABLE papers DROP COLUMN IF EXISTS {col}")

    # Drop embedding separately (pgvector type)
    op.execute("ALTER TABLE papers DROP COLUMN IF EXISTS embedding")


def downgrade():
    # Re-add Paper columns
    op.add_column("papers", sa.Column("embedding", sa.LargeBinary, nullable=True))
    for col in ['extensionability_score', 'structurality_score', 'claims',
                'primary_logic', 'core_operator', 'data_url', 'funding',
                'license', 'open_code', 'open_data', 'inference_pattern',
                'supervision_type']:
        op.add_column("papers", sa.Column(col, sa.Text, nullable=True))

    op.rename_table("review_queue", "review_queue_items")
    op.add_column("evidence_units", sa.Column("idea_delta_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("contribution_to_canonical_idea", sa.Column("idea_delta_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True))
    op.drop_constraint("fk_contrib_delta_card", "contribution_to_canonical_idea", type_="foreignkey")
    op.drop_column("contribution_to_canonical_idea", "delta_card_id")
    op.drop_column("review_queue_items", "assigned_to")
    op.drop_column("review_queue_items", "task_type")
