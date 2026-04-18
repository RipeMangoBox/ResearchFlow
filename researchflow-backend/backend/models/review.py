"""Review, override, alias, and taxonomy versioning models.

- ReviewTask: audit queue for delta_cards, assertions, idea_deltas, lineage, candidates
- HumanOverride: tracks manual corrections with before/after values
- Alias: entity name normalization for mechanism families, bottlenecks, etc.
- TaxonomyVersion: tracks ontology changes (paradigm/slot/mechanism creates/updates)
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class ReviewTask(Base):
    """A review task in the audit queue.

    Targets can be delta_cards, assertions, or idea_deltas.
    Tasks are created automatically (e.g., high-value assertions default to candidate)
    or manually by researchers.
    """
    __tablename__ = "review_tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    target_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # delta_card / assertion / idea_delta
    target_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    task_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # auto_review / human_review / re_analysis
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending / in_progress / approved / rejected
    priority: Mapped[int] = mapped_column(SmallInteger, default=3)
    assigned_to: Mapped[str | None] = mapped_column(String(50))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_review_tasks_status", "status"),
        Index("idx_review_tasks_target", "target_type", "target_id"),
        Index("idx_review_tasks_priority", "priority"),
    )


class HumanOverride(Base):
    """Records a human correction to any entity field.

    Preserves old_value and new_value as JSONB for audit trail.
    """
    __tablename__ = "human_overrides"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    target_type: Mapped[str] = mapped_column(String(30), nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    field_name: Mapped[str] = mapped_column(String(100), nullable=False)
    old_value: Mapped[dict | None] = mapped_column(JSONB)
    new_value: Mapped[dict | None] = mapped_column(JSONB)
    reason: Mapped[str | None] = mapped_column(Text)
    overridden_by: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_overrides_target", "target_type", "target_id"),
    )


class Alias(Base):
    """Entity name alias for normalization and deduplication.

    Used during entity resolution to map variant names to canonical entities
    (e.g., "DDPM" → diffusion mechanism family, "flow matching" → flow_matching).
    """
    __tablename__ = "aliases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # mechanism_family / bottleneck / slot
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    alias: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(
        String(30), default="auto_detected"
    )  # manual / auto_detected
    confidence: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        Index("idx_aliases_entity", "entity_type", "entity_id"),
        Index("idx_aliases_alias", "alias"),
    )


class TaxonomyVersion(Base):
    """Tracks ontology changes over time — paradigm/slot/mechanism creates/updates/merges.

    Every change to the taxonomy gets a version record for audit trail and rollback.
    """
    __tablename__ = "taxonomy_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # paradigm_template / slot / mechanism_family / canonical_idea
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    action: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # created / updated / merged / deprecated / created_from_candidate
    version_label: Mapped[str | None] = mapped_column(String(100))
    changed_by: Mapped[str | None] = mapped_column(String(50))
    change_summary: Mapped[str | None] = mapped_column(Text)
    before_snapshot: Mapped[dict | None] = mapped_column(JSONB)
    after_snapshot: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_taxonomy_versions_entity", "entity_type", "entity_id"),
        Index("idx_taxonomy_versions_action", "action"),
    )
