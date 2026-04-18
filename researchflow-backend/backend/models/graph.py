"""Knowledge graph core models — IdeaDelta, Slot, MechanismFamily, GraphEdge, ImplementationUnit.

IdeaDelta is the PRIMARY object in the graph (not Paper).
Paper is a container, Evidence is an anchor, Graph is the retrieval accelerator.

Hierarchy: ParadigmFrame → Slot → MechanismFamily → IdeaDelta → EvidenceUnit → ImplementationUnit
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


# ── Slot (独立槽位模型) ─────────────────────────────────────────

class Slot(Base):
    """A named slot within a ParadigmFrame.

    Examples: denoiser, reward, credit_assignment, vision_encoder, projector
    """
    __tablename__ = "slots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paradigm_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paradigm_templates.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    slot_type: Mapped[str | None] = mapped_column(String(50))  # architecture/objective/data/inference
    is_required: Mapped[bool] = mapped_column(Boolean, default=True)
    sort_order: Mapped[int] = mapped_column(SmallInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ── MechanismFamily (机制族，层级结构) ──────────────────────────

class MechanismFamily(Base):
    """A family of related mechanisms, optionally hierarchical.

    Examples: diffusion, flow_matching, masked_modeling, reinforcement_learning
    """
    __tablename__ = "mechanism_families"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    domain: Mapped[str | None] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("mechanism_families.id")
    )
    aliases: Mapped[list[str] | None] = mapped_column(ARRAY(Text))  # for entity resolution
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ── IdeaDelta (核心主对象) ──────────────────────────────────────

class IdeaDelta(Base):
    """The PRIMARY object in the knowledge graph.

    Represents what a paper actually CHANGED relative to the domain's
    canonical paradigm. Paper is the container, IdeaDelta is the knowledge.

    Hard constraint: publish_status stays 'draft' until evidence_count >= 2.
    """
    __tablename__ = "idea_deltas"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    analysis_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paper_analyses.id")
    )
    primary_bottleneck_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("project_bottlenecks.id")
    )
    paradigm_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paradigm_templates.id")
    )
    delta_card_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("delta_cards.id")
    )

    # Core content
    delta_statement: Mapped[str] = mapped_column(Text, nullable=False)
    changed_slots: Mapped[dict | None] = mapped_column(JSONB)
    # Schema: [{slot_id, slot_name, from, to, change_type: "structural"|"plugin"|"tweak"}]

    mechanism_family_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))

    # Scores (the 4 key dimensions from design doc)
    structurality_score: Mapped[float | None] = mapped_column()
    transferability_score: Mapped[float | None] = mapped_column()
    local_keyness_score: Mapped[float | None] = mapped_column()
    field_keyness_score: Mapped[float | None] = mapped_column()
    confidence: Mapped[float | None] = mapped_column()

    # Evidence tracking
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)

    # Publishing (hard constraint: draft until evidence_count >= 2)
    publish_status: Mapped[str] = mapped_column(
        String(20), default="draft"
    )  # draft / auto_published / human_verified

    # Legacy compatibility (migrate from method_deltas)
    is_structural: Mapped[bool | None] = mapped_column(Boolean)
    primary_gain_source: Mapped[str | None] = mapped_column(String(100))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # ORM relationships
    paper = relationship("Paper", foreign_keys=[paper_id], lazy="selectin")
    delta_card = relationship("DeltaCard", foreign_keys=[delta_card_id], lazy="selectin")

    __table_args__ = (
        Index("idx_idea_deltas_paper", "paper_id"),
        Index("idx_idea_deltas_bottleneck", "primary_bottleneck_id"),
        Index("idx_idea_deltas_paradigm", "paradigm_id"),
        Index("idx_idea_deltas_publish", "publish_status"),
    )


# ── GraphEdge (统一边表) ────────────────────────────────────────

class GraphEdge(Base):
    """Unified edge table for the knowledge graph.

    All relationships flow through here. Every edge has:
    - source/target with type discrimination
    - edge_type (12 core types)
    - assertion_source (hard constraint: must distinguish paper vs system vs human)
    - confidence score

    Hard constraint: paper→paper edges only allow edge_type='cites'.
    """
    __tablename__ = "graph_edges"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_type: Mapped[str] = mapped_column(String(30), nullable=False)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    target_type: Mapped[str] = mapped_column(String(30), nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    edge_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Hard constraint #2: every edge must declare its source
    assertion_source: Mapped[str] = mapped_column(
        String(20), nullable=False, default="inferred_by_system"
    )  # asserted_by_paper / inferred_by_system / verified_by_human

    confidence: Mapped[float | None] = mapped_column()
    evidence_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_edges_source", "source_type", "source_id"),
        Index("idx_edges_target", "target_type", "target_id"),
        Index("idx_edges_type", "edge_type"),
        Index("idx_edges_assertion", "assertion_source"),
    )


# ── ImplementationUnit (代码实现锚点) ──────────────────────────

class ImplementationUnit(Base):
    """Code-level anchor for an IdeaDelta.

    Links research ideas to specific code implementations:
    file, class, function, config, tensor shapes.
    """
    __tablename__ = "implementation_units"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id")
    )
    idea_delta_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("idea_deltas.id")
    )
    repo_url: Mapped[str | None] = mapped_column(Text)
    file_path: Mapped[str | None] = mapped_column(Text)
    class_or_function: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    config_snippet: Mapped[str | None] = mapped_column(Text)
    shape_trace: Mapped[dict | None] = mapped_column(JSONB)
    # Schema: [{layer, input_shape, output_shape, notes}]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
