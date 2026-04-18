"""DomainSpec + DomainSourceRegistry + IncrementalCheckpoint + paper ring column.

Revision ID: 015
Revises: 014
Create Date: 2026-04-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # papers: ring + domain_id
    op.add_column("papers", sa.Column("ring", sa.String(20)))
    op.add_column("papers", sa.Column("domain_id", UUID(as_uuid=True)))

    # domain_specs
    op.create_table(
        "domain_specs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("seed_paper_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("seed_repo_urls", ARRAY(sa.Text)),
        sa.Column("openalex_topic_ids", ARRAY(sa.Text)),
        sa.Column("openalex_source_ids", ARRAY(sa.Text)),
        sa.Column("paradigm_template_id", UUID(as_uuid=True), sa.ForeignKey("paradigm_templates.id")),
        sa.Column("constraints", JSONB),
        sa.Column("negative_constraints", ARRAY(sa.Text)),
        sa.Column("paper_count", sa.SmallInteger, server_default="0"),
        sa.Column("status", sa.String(20), server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_domain_specs_name", "domain_specs", ["name"])
    op.create_index("idx_domain_specs_status", "domain_specs", ["status"])

    # domain_source_registry
    op.create_table(
        "domain_source_registry",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("domain_id", UUID(as_uuid=True), sa.ForeignKey("domain_specs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_type", sa.String(30), nullable=False),
        sa.Column("source_ref", sa.Text, nullable=False),
        sa.Column("sync_frequency", sa.String(20), server_default="weekly"),
        sa.Column("last_synced_at", sa.DateTime(timezone=True)),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_dsr_domain", "domain_source_registry", ["domain_id"])
    op.create_index("idx_dsr_type", "domain_source_registry", ["source_type"])

    # incremental_checkpoints
    op.create_table(
        "incremental_checkpoints",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("source_registry_id", UUID(as_uuid=True), sa.ForeignKey("domain_source_registry.id", ondelete="CASCADE"), nullable=False),
        sa.Column("checkpoint_value", sa.Text, nullable=False),
        sa.Column("papers_found", sa.SmallInteger, server_default="0"),
        sa.Column("papers_new", sa.SmallInteger, server_default="0"),
        sa.Column("sync_mode", sa.String(20)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_checkpoint_source", "incremental_checkpoints", ["source_registry_id"])


def downgrade() -> None:
    op.drop_table("incremental_checkpoints")
    op.drop_table("domain_source_registry")
    op.drop_table("domain_specs")
    op.drop_column("papers", "domain_id")
    op.drop_column("papers", "ring")
