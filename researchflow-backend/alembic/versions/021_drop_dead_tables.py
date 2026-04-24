"""Drop 16 unused/orphaned tables + review_tasks (merged into review_queue).

Tables dropped:
  - idea_deltas (absorbed into delta_cards)
  - graph_edges (replaced by graph_assertions)
  - slots (data in paradigm_templates.slots JSONB)
  - method_slots (never written)
  - implementation_units (not implemented)
  - transfer_atoms (not implemented)
  - search_sessions, search_branches (not implemented)
  - reading_plans (replaced by paper_reports)
  - direction_cards (not implemented)
  - render_artifacts (replaced by paper_reports)
  - user_bookmarks, user_events (not integrated)
  - execution_memories (not used)
  - user_feedback (replaced by human_overrides)
  - project_focus_bottlenecks (use domain_specs.constraints)
  - review_tasks (merged into review_queue)

Revision ID: 021
Revises: 020
"""
from alembic import op

revision = "021"
down_revision = "020"


def upgrade():
    # Drop in FK-dependency order (children first)
    op.drop_table("search_branches")          # FK → search_sessions
    op.drop_table("search_sessions")
    op.drop_table("implementation_units")     # FK → idea_deltas (being deleted)
    op.drop_table("idea_deltas")              # FK → papers, delta_cards
    op.drop_table("graph_edges")
    op.drop_table("slots")                    # FK → paradigm_templates
    op.drop_table("method_slots")
    op.drop_table("transfer_atoms")
    op.drop_table("reading_plans")
    op.drop_table("direction_cards")
    op.drop_table("render_artifacts")
    op.drop_table("user_bookmarks")
    op.drop_table("user_events")
    op.drop_table("execution_memories")
    op.drop_table("user_feedback")
    op.drop_table("project_focus_bottlenecks")  # FK → project_bottlenecks
    op.drop_table("review_tasks")             # merged into review_queue


def downgrade():
    # Recreating all 17 tables in reverse order would be needed for rollback.
    # In practice, restore from pg_dump backup taken before migration.
    raise NotImplementedError(
        "Migration 021 is destructive (17 table drops). "
        "Restore from pg_dump backup to rollback."
    )
