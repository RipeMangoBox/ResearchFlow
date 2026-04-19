"""Domain models — DomainSpec, DomainSourceRegistry, IncrementalCheckpoint.

DomainSpec: defines a research domain with seed papers, sources, constraints.
DomainSourceRegistry: tracks data sources for each domain.
IncrementalCheckpoint: sync progress per source for incremental updates.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class DomainSpec(Base):
    """A research domain specification — the blueprint for KB construction.

    Captures: what to study, where to look, what to include/exclude.
    """
    __tablename__ = "domain_specs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    # Seed inputs
    seed_paper_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    seed_repo_urls: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    # External source IDs
    openalex_topic_ids: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    openalex_source_ids: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    # Linked paradigm
    paradigm_template_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paradigm_templates.id")
    )

    # Constraints
    constraints: Mapped[dict | None] = mapped_column(JSONB)
    # e.g. {"min_year": 2023, "must_have_abstract": true, "min_cited_by": 5}
    negative_constraints: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    # e.g. ["no_plugin_only", "must_have_evidence", "must_have_open_code"]

    # ── V6: Scope Definition (Domain Manifest) ──
    scope_modalities: Mapped[list[str] | None] = mapped_column(ARRAY(Text), default=[])
    scope_tasks: Mapped[list[str] | None] = mapped_column(ARRAY(Text), default=[])
    scope_paradigms: Mapped[list[str] | None] = mapped_column(ARRAY(Text), default=[])
    scope_seed_methods: Mapped[list[str] | None] = mapped_column(ARRAY(Text), default=[])
    scope_seed_models: Mapped[list[str] | None] = mapped_column(ARRAY(Text), default=[])
    scope_seed_datasets: Mapped[list[str] | None] = mapped_column(ARRAY(Text), default=[])
    negative_scope: Mapped[list[str] | None] = mapped_column(ARRAY(Text), default=[])

    # ── V6: Budget Limits ──
    budget_metadata_candidates: Mapped[int | None] = mapped_column(Integer, default=500)
    budget_shallow_ingest: Mapped[int | None] = mapped_column(Integer, default=200)
    budget_deep_ingest: Mapped[int | None] = mapped_column(Integer, default=50)
    budget_anchor_methods: Mapped[int | None] = mapped_column(Integer, default=20)

    # Stats
    paper_count: Mapped[int] = mapped_column(SmallInteger, default=0)
    status: Mapped[str] = mapped_column(String(20), default="active")
    # active / archived

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_domain_specs_name", "name"),
        Index("idx_domain_specs_status", "status"),
    )


class DomainSourceRegistry(Base):
    """A registered data source for a domain — defines where to look for papers."""
    __tablename__ = "domain_source_registry"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    domain_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("domain_specs.id", ondelete="CASCADE"), nullable=False
    )
    source_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # openalex_topic / openalex_source / awesome_repo / semantic_scholar / zotero
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    # e.g. "T12345" for OpenAlex topic, repo URL for awesome, library ID for Zotero
    sync_frequency: Mapped[str] = mapped_column(
        String(20), default="weekly"
    )  # daily / weekly / monthly / manual
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_dsr_domain", "domain_id"),
        Index("idx_dsr_type", "source_type"),
    )


class IncrementalCheckpoint(Base):
    """Sync progress for a domain source — tracks where we left off."""
    __tablename__ = "incremental_checkpoints"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_registry_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("domain_source_registry.id", ondelete="CASCADE"), nullable=False
    )
    checkpoint_value: Mapped[str] = mapped_column(Text, nullable=False)
    # e.g. "2026-04-18" for date-based, commit SHA for awesome repo
    papers_found: Mapped[int] = mapped_column(SmallInteger, default=0)
    papers_new: Mapped[int] = mapped_column(SmallInteger, default=0)
    sync_mode: Mapped[str | None] = mapped_column(String(20))
    # hot / weekly / monthly

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_checkpoint_source", "source_registry_id"),
    )
