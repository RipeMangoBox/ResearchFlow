"""Knowledge-base models — graph candidates, profiles, reports, review queue.

GraphNodeCandidate / GraphEdgeCandidate: staging area before KB promotion.
KBNodeProfile / KBEdgeProfile: generated wiki-style pages for KB entities.
PaperReport / PaperReportSection: structured per-paper reports.
ReviewQueueItem: human review queue for auto-generated artifacts.
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
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class GraphNodeCandidate(Base):
    """A candidate node extracted from a paper, pending promotion to the KB graph.

    Nodes represent methods, datasets, tasks, metrics, etc. discovered during
    paper analysis. They are scored and optionally promoted to canonical entities.
    """
    __tablename__ = "graph_node_candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id")
    )
    candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paper_candidates.id")
    )

    node_type: Mapped[str] = mapped_column(String(20), nullable=False)
    # method / dataset / task / metric / concept / tool
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    name_zh: Mapped[str | None] = mapped_column(String(200))
    one_liner: Mapped[str | None] = mapped_column(Text)

    # Promotion scoring
    promotion_score: Mapped[float | None] = mapped_column(Float)
    promotion_breakdown: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(20), default="candidate")
    # candidate / promoted / merged / rejected

    # Resolution
    promoted_entity_type: Mapped[str | None] = mapped_column(String(30))
    promoted_entity_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    evidence_refs: Mapped[dict | None] = mapped_column(JSONB)
    confidence: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class GraphEdgeCandidate(Base):
    """A candidate edge (relation) extracted from a paper, pending promotion.

    Connects two entities (nodes or candidates) with a typed relation
    discovered during paper analysis.
    """
    __tablename__ = "graph_edge_candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id")
    )

    # Source endpoint
    source_entity_type: Mapped[str] = mapped_column(String(30), nullable=False)
    source_entity_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    source_candidate_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    # Target endpoint
    target_entity_type: Mapped[str] = mapped_column(String(30), nullable=False)
    target_entity_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    target_candidate_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    # Relation details
    relation_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # improves / replaces / uses / evaluates_on / extends / combines_with
    slot_name: Mapped[str | None] = mapped_column(String(100))
    confidence_score: Mapped[float | None] = mapped_column(Float)
    confidence_breakdown: Mapped[dict | None] = mapped_column(JSONB)
    one_liner: Mapped[str | None] = mapped_column(Text)

    # Lifecycle
    status: Mapped[str] = mapped_column(String(20), default="candidate")
    # candidate / promoted / merged / rejected
    promoted_edge_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    evidence_refs: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class KBNodeProfile(Base):
    """A generated wiki-style profile page for a KB entity.

    Supports multiple languages and profile kinds (page, card, tooltip).
    Versioned and review-tracked for quality control.
    """
    __tablename__ = "kb_node_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_type: Mapped[str] = mapped_column(String(30), nullable=False)
    # method / dataset / task / metric / concept / paper
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    profile_kind: Mapped[str] = mapped_column(String(20), default="page")
    # page / card / tooltip
    lang: Mapped[str] = mapped_column(String(5), default="zh")

    # Content
    one_liner: Mapped[str | None] = mapped_column(Text)
    short_intro_md: Mapped[str | None] = mapped_column(Text)
    detailed_md: Mapped[str | None] = mapped_column(Text)
    structured_json: Mapped[dict | None] = mapped_column(JSONB)
    evidence_refs: Mapped[dict | None] = mapped_column(JSONB)

    # Generation provenance
    generated_by_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_runs.id")
    )
    model_name: Mapped[str | None] = mapped_column(String(50))
    prompt_version: Mapped[str | None] = mapped_column(String(20))

    # Versioning and review
    profile_version: Mapped[int] = mapped_column(SmallInteger, default=1)
    review_status: Mapped[str] = mapped_column(String(20), default="auto")
    # auto / human_approved / human_edited / rejected
    staleness_trigger_count: Mapped[int] = mapped_column(SmallInteger, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("entity_type", "entity_id", "profile_kind", "lang",
                         name="uq_knp_entity_kind_lang"),
        Index("idx_knp_entity", "entity_type", "entity_id"),
    )


class KBEdgeProfile(Base):
    """A generated profile describing the relationship between two KB entities.

    Provides human-readable summaries of edges for wiki rendering
    and contextual display in the knowledge graph UI.
    """
    __tablename__ = "kb_edge_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Source endpoint
    source_entity_type: Mapped[str] = mapped_column(String(30), nullable=False)
    source_entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    # Target endpoint
    target_entity_type: Mapped[str] = mapped_column(String(30), nullable=False)
    target_entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    # Relation
    relation_type: Mapped[str] = mapped_column(String(50), nullable=False)
    edge_table: Mapped[str | None] = mapped_column(String(30))
    edge_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    lang: Mapped[str] = mapped_column(String(5), default="zh")

    # Content
    one_liner: Mapped[str | None] = mapped_column(Text)
    relation_summary: Mapped[str | None] = mapped_column(Text)
    source_context: Mapped[str | None] = mapped_column(Text)
    target_context: Mapped[str | None] = mapped_column(Text)
    display_priority: Mapped[int] = mapped_column(SmallInteger, default=5)
    evidence_refs: Mapped[dict | None] = mapped_column(JSONB)

    # Generation provenance
    generated_by_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_runs.id")
    )
    review_status: Mapped[str] = mapped_column(String(20), default="auto")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_kep_source", "source_entity_type", "source_entity_id"),
        Index("idx_kep_target", "target_entity_type", "target_entity_id"),
    )


class PaperReport(Base):
    """A structured report generated for a single paper.

    Contains metadata and provenance; actual content lives in
    PaperReportSection rows linked to this report.
    """
    __tablename__ = "paper_reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    report_version: Mapped[int] = mapped_column(SmallInteger, default=1)
    title_zh: Mapped[str | None] = mapped_column(Text)
    title_en: Mapped[str | None] = mapped_column(Text)

    # Generation provenance
    generated_by_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_runs.id")
    )
    model_name: Mapped[str | None] = mapped_column(String(50))
    prompt_version: Mapped[str | None] = mapped_column(String(20))
    review_status: Mapped[str] = mapped_column(String(20), default="auto")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PaperReportSection(Base):
    """A single section within a PaperReport.

    Sections are ordered by order_index and typed by section_type
    (e.g. overview, method, formula, experiment, limitation).
    """
    __tablename__ = "paper_report_sections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paper_reports.id", ondelete="CASCADE"), nullable=False
    )
    section_type: Mapped[str] = mapped_column(String(30), nullable=False)
    # overview / method / formula / experiment / ablation / limitation / related_work
    title: Mapped[str | None] = mapped_column(Text)
    body_md: Mapped[str | None] = mapped_column(Text)
    evidence_refs: Mapped[dict | None] = mapped_column(JSONB)
    order_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ReviewQueueItem(Base):
    """An item queued for human review.

    Covers any auto-generated artifact (profile, extraction, report section)
    that needs human verification or approval.
    """
    __tablename__ = "review_queue_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    item_type: Mapped[str] = mapped_column(String(30), nullable=False)
    # profile / extraction / report / candidate / edge
    entity_type: Mapped[str] = mapped_column(String(30), nullable=False)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    # Priority and recommendation
    priority_score: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str | None] = mapped_column(Text)
    suggested_decision: Mapped[str | None] = mapped_column(String(20))
    # approve / reject / edit / merge
    evidence_refs: Mapped[dict | None] = mapped_column(JSONB)

    # Review lifecycle
    status: Mapped[str] = mapped_column(String(25), default="pending")
    # pending / in_progress / approved / rejected / deferred
    reviewed_by: Mapped[str | None] = mapped_column(String(50))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_notes: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_rqi_status_priority", "status", "priority_score"),
    )
