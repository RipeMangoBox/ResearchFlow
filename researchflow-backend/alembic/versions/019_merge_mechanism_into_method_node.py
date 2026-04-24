"""Merge mechanism_families into method_nodes.

MechanismFamily and MethodNode represent related concepts at different
abstraction levels. This migration unifies them into a single method_nodes
table, migrates all data, renames all FK/field references, and drops the
old mechanism_families table.

Revision ID: 019
Revises: 018
"""
from alembic import op
import sqlalchemy as sa

revision = "019"
down_revision = "018"


def upgrade():
    # ── 1. Add MechanismFamily-specific fields to method_nodes ──
    op.add_column("method_nodes", sa.Column(
        "parent_method_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True,
    ))
    op.create_foreign_key(
        "fk_method_nodes_parent", "method_nodes", "method_nodes",
        ["parent_method_id"], ["id"],
    )
    op.add_column("method_nodes", sa.Column(
        "aliases", sa.dialects.postgresql.ARRAY(sa.Text), nullable=True,
    ))
    # domain may already exist on method_nodes; add only if missing
    op.execute("""
        DO $$ BEGIN
            ALTER TABLE method_nodes ADD COLUMN domain VARCHAR(100);
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$;
    """)

    # ── 2. Migrate mechanism_families → method_nodes ──
    op.execute("""
        INSERT INTO method_nodes (id, name, type, domain, description, parent_method_id, aliases, maturity, created_at)
        SELECT id, name, 'mechanism_family', domain, description, parent_id, aliases, 'seed', created_at
        FROM mechanism_families
        ON CONFLICT (id) DO UPDATE SET
            type = COALESCE(NULLIF(method_nodes.type, ''), EXCLUDED.type),
            domain = COALESCE(method_nodes.domain, EXCLUDED.domain),
            aliases = COALESCE(method_nodes.aliases, EXCLUDED.aliases),
            parent_method_id = COALESCE(method_nodes.parent_method_id, EXCLUDED.parent_method_id)
    """)

    # ── 3. Rename columns on delta_cards and idea_deltas ──
    op.alter_column("delta_cards", "mechanism_family_ids", new_column_name="method_node_ids")
    op.alter_column("idea_deltas", "mechanism_family_ids", new_column_name="method_node_ids")

    # ── 4. Rename column on canonical_ideas ──
    # Drop old FK first
    op.execute("ALTER TABLE canonical_ideas DROP CONSTRAINT IF EXISTS canonical_ideas_mechanism_family_id_fkey")
    op.alter_column("canonical_ideas", "mechanism_family_id", new_column_name="method_node_id")
    op.create_foreign_key(
        "canonical_ideas_method_node_id_fkey", "canonical_ideas", "method_nodes",
        ["method_node_id"], ["id"],
    )
    # Rename index
    op.execute("DROP INDEX IF EXISTS idx_canonical_ideas_mechanism")
    op.create_index("idx_canonical_ideas_method_node", "canonical_ideas", ["method_node_id"])

    # ── 5. Rename column on papers ──
    op.alter_column("papers", "mechanism_family", new_column_name="method_family")

    # ── 6. Update graph_assertions edge_type ──
    op.execute("UPDATE graph_assertions SET edge_type = 'instance_of_method' WHERE edge_type = 'instance_of_mechanism'")

    # ── 7. Update graph_nodes ref_table ──
    op.execute("UPDATE graph_nodes SET ref_table = 'method_nodes', node_type = 'method' WHERE ref_table = 'mechanism_families'")

    # ── 8. Rename mechanism_candidates → method_candidates ──
    op.rename_table("mechanism_candidates", "method_candidates")

    # ── 9. Drop mechanism_families (data already migrated) ──
    op.drop_table("mechanism_families")


def downgrade():
    # Recreate mechanism_families
    op.create_table(
        "mechanism_families",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("domain", sa.String(100)),
        sa.Column("description", sa.Text),
        sa.Column("parent_id", sa.dialects.postgresql.UUID(as_uuid=True)),
        sa.Column("aliases", sa.dialects.postgresql.ARRAY(sa.Text)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Restore data from method_nodes where type='mechanism_family'
    op.execute("""
        INSERT INTO mechanism_families (id, name, domain, description, parent_id, aliases, created_at)
        SELECT id, name, domain, description, parent_method_id, aliases, created_at
        FROM method_nodes WHERE type = 'mechanism_family'
    """)

    op.rename_table("method_candidates", "mechanism_candidates")
    op.execute("UPDATE graph_nodes SET ref_table = 'mechanism_families', node_type = 'mechanism' WHERE ref_table = 'method_nodes' AND node_type = 'method'")
    op.execute("UPDATE graph_assertions SET edge_type = 'instance_of_mechanism' WHERE edge_type = 'instance_of_method'")
    op.alter_column("papers", "method_family", new_column_name="mechanism_family")
    op.execute("DROP INDEX IF EXISTS idx_canonical_ideas_method_node")
    op.execute("ALTER TABLE canonical_ideas DROP CONSTRAINT IF EXISTS canonical_ideas_method_node_id_fkey")
    op.alter_column("canonical_ideas", "method_node_id", new_column_name="mechanism_family_id")
    op.create_foreign_key(
        "canonical_ideas_mechanism_family_id_fkey", "canonical_ideas", "mechanism_families",
        ["mechanism_family_id"], ["id"],
    )
    op.alter_column("idea_deltas", "method_node_ids", new_column_name="mechanism_family_ids")
    op.alter_column("delta_cards", "method_node_ids", new_column_name="mechanism_family_ids")
    op.drop_constraint("fk_method_nodes_parent", "method_nodes", type_="foreignkey")
    op.drop_column("method_nodes", "parent_method_id")
    op.drop_column("method_nodes", "aliases")
