"""Add performance indexes for full-text search and common query patterns.

New indexes:
  - GIN tsvector on delta_cards.delta_statement (full-text search)
  - GIN tsvector on idea_deltas.delta_statement (full-text search)
  - Composite on graph_assertions(status, edge_type)
  - Composite on graph_assertions(status, from_node_id) (published edge queries)

Revision ID: 006
Revises: 005
Create Date: 2026-04-18
"""

from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Full-text search GIN indexes ──────────────────────────────
    op.execute(
        "CREATE INDEX idx_delta_cards_statement_tsvector "
        "ON delta_cards USING gin(to_tsvector('english', delta_statement))"
    )
    op.execute(
        "CREATE INDEX idx_idea_deltas_statement_tsvector "
        "ON idea_deltas USING gin(to_tsvector('english', delta_statement))"
    )

    # ── Composite indexes for common query patterns ───────────────
    op.create_index(
        "idx_assertions_status_edge_type",
        "graph_assertions",
        ["status", "edge_type"],
    )
    op.create_index(
        "idx_assertions_status_from_node",
        "graph_assertions",
        ["status", "from_node_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_assertions_status_from_node", "graph_assertions")
    op.drop_index("idx_assertions_status_edge_type", "graph_assertions")
    op.execute("DROP INDEX idx_idea_deltas_statement_tsvector")
    op.execute("DROP INDEX idx_delta_cards_statement_tsvector")
