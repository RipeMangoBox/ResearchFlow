"""Method evolution models — MethodNode, MethodSlot, MethodEdge, MethodApplication.

Methods are abstracted from papers. A paper may use, adapt, or propose a method.
Methods evolve via edges (applies_to_domain, modifies_slot, combines_with, etc.)
and can be promoted to established_baseline when downstream count is sufficient.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class MethodNode(Base):
    """A unified method/mechanism entity abstracted from papers.

    Covers both specific methods (GRPO, DPO, QwenVL) and mechanism families
    (diffusion, flow_matching, reinforcement_learning). Distinguished by `type`.

    Types: algorithm / recipe / model_family / system / mechanism_family
    Maturity: seed → emerging → established_baseline
    """
    __tablename__ = "method_nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    name_zh: Mapped[str | None] = mapped_column(String(200))
    type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # algorithm / recipe / model_family / system / mechanism_family

    domain: Mapped[str | None] = mapped_column(String(100))
    canonical_paper_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    version: Mapped[str | None] = mapped_column(String(20))
    maturity: Mapped[str] = mapped_column(
        String(30), default="seed"
    )  # seed / emerging / established_baseline

    description: Mapped[str | None] = mapped_column(Text)
    promotion_criteria: Mapped[dict | None] = mapped_column(JSONB)

    # Hierarchy (absorbed from MechanismFamily.parent_id)
    parent_method_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("method_nodes.id")
    )
    # Entity resolution aliases (absorbed from MechanismFamily.aliases)
    aliases: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    downstream_count: Mapped[int] = mapped_column(SmallInteger, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_method_name", "name"),
        Index("idx_method_maturity", "maturity"),
        Index("idx_method_type", "type"),
    )


class MethodSlot(Base):
    """A named component slot within a method.

    Examples for GRPO: reward_function, advantage_estimator, policy_update.
    """
    __tablename__ = "method_slots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    method_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    slot_name: Mapped[str] = mapped_column(String(100), nullable=False)
    default_description: Mapped[str | None] = mapped_column(Text)
    sort_order: Mapped[int] = mapped_column(SmallInteger, default=0)

    __table_args__ = (
        Index("idx_mslot_method", "method_id"),
        UniqueConstraint("method_id", "slot_name", name="uq_method_slot"),
    )


class MethodEdge(Base):
    """An evolution edge between two methods.

    Captures: applies_to_domain, modifies_slot, combines_with,
    replaces, distills_from, extends, new_baseline_from.
    """
    __tablename__ = "method_edges"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    parent_method_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    child_method_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )
    # applies_to_domain / modifies_slot / combines_with / replaces
    # distills_from / extends / new_baseline_from

    scope_facet_ids: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    changed_slot_ids: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    delta_description: Mapped[str | None] = mapped_column(Text)
    # e.g., "在 video 领域把 reward 从 rule-based 改成 learned"

    evidence_refs: Mapped[dict | None] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    status: Mapped[str] = mapped_column(
        String(20), default="candidate"
    )  # candidate / published / rejected

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_medge_parent", "parent_method_id"),
        Index("idx_medge_child", "child_method_id"),
        Index("idx_medge_type", "relation_type"),
    )


class MethodApplication(Base):
    """How a paper uses a method — as baseline, proposed method, component, etc."""
    __tablename__ = "method_applications"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    method_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    role: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # baseline / adapted_baseline / proposed_method / component / comparison_baseline

    task_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    scenario_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    dataset_ids: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("paper_id", "method_id", "role",
                         name="uq_method_application"),
        Index("idx_mapp_paper", "paper_id"),
        Index("idx_mapp_method", "method_id"),
        Index("idx_mapp_role", "role"),
    )
