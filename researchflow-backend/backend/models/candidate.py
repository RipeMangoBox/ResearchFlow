"""PaperCandidate, CandidateScore, ScoreSignal models.

PaperCandidate: a paper discovered but not yet ingested into the KB.
CandidateScore: multi-stage scoring record for a candidate.
ScoreSignal: generic signal attached to any entity for scoring decisions.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class PaperCandidate(Base):
    """A paper discovered via citation walk, search, or recommendation.

    Tracks the full lifecycle from discovery through scoring to either
    ingestion (promoted to Paper) or rejection.
    """
    __tablename__ = "paper_candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_title: Mapped[str | None] = mapped_column(Text)

    # External IDs
    arxiv_id: Mapped[str | None] = mapped_column(String(30))
    doi: Mapped[str | None] = mapped_column(String(100))
    s2_paper_id: Mapped[str | None] = mapped_column(String(50))
    openalex_id: Mapped[str | None] = mapped_column(String(50))
    openreview_id: Mapped[str | None] = mapped_column(String(100))
    dblp_id: Mapped[str | None] = mapped_column(String(100))
    paper_link: Mapped[str | None] = mapped_column(Text)

    # Discovery provenance
    discovered_from_paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id")
    )
    discovered_from_domain_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("domain_specs.id")
    )
    discovery_source: Mapped[str] = mapped_column(String(30), nullable=False)
    # citation_walk / search / recommendation / manual / reference_list
    discovery_reason: Mapped[str | None] = mapped_column(String(50))
    relation_hint: Mapped[str | None] = mapped_column(String(30))
    # baseline / extension / sibling / foundational / negative

    # Basic metadata
    authors_json: Mapped[dict | None] = mapped_column(JSONB)
    venue: Mapped[str | None] = mapped_column(String(100))
    year: Mapped[int | None] = mapped_column(SmallInteger)
    abstract: Mapped[str | None] = mapped_column(Text)
    citation_count: Mapped[int | None] = mapped_column(Integer)
    code_url: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB)

    # Lifecycle
    status: Mapped[str] = mapped_column(String(25), default="discovered")
    # discovered / scoring / accepted / ingesting / ingested / rejected / duplicate
    absorption_level: Mapped[int] = mapped_column(SmallInteger, default=0)
    # 0=metadata only, 1=abstract scored, 2=deep ingested, 3=graph promoted

    # Resolution
    ingested_paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id")
    )
    reject_reason: Mapped[str | None] = mapped_column(Text)
    duplicate_of_candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paper_candidates.id")
    )
    duplicate_of_paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id")
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_pc_arxiv_id", "arxiv_id"),
        Index("idx_pc_doi", "doi"),
        Index("idx_pc_status", "status"),
        Index("idx_pc_status_absorption", "status", "absorption_level"),
    )


class CandidateScore(Base):
    """Multi-stage scoring record for a PaperCandidate.

    Captures discovery-stage, deep-ingest, and graph-promotion scores
    along with detailed breakdowns and applied adjustments.
    """
    __tablename__ = "candidate_scores"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paper_candidates.id", ondelete="CASCADE"), nullable=False
    )

    # Scores per stage
    discovery_score: Mapped[float | None] = mapped_column(Float)
    deep_ingest_score: Mapped[float | None] = mapped_column(Float)
    graph_promotion_score: Mapped[float | None] = mapped_column(Float)
    anchor_score: Mapped[float | None] = mapped_column(Float)

    # Breakdowns
    discovery_breakdown: Mapped[dict | None] = mapped_column(JSONB)
    deep_ingest_breakdown: Mapped[dict | None] = mapped_column(JSONB)

    # Adjustments
    hard_caps_applied: Mapped[dict | None] = mapped_column(JSONB)
    boosts_applied: Mapped[dict | None] = mapped_column(JSONB)
    penalties_applied: Mapped[dict | None] = mapped_column(JSONB)

    # Decision
    decision: Mapped[str | None] = mapped_column(String(25))
    # accept / reject / defer / manual_review
    decision_reason: Mapped[str | None] = mapped_column(Text)
    score_version: Mapped[int] = mapped_column(SmallInteger, default=1)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ScoreSignal(Base):
    """A generic scoring signal attached to any entity.

    Provides a flexible key-value store for signals consumed by
    scoring functions (e.g. citation velocity, venue rank, method overlap).
    """
    __tablename__ = "score_signals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_type: Mapped[str] = mapped_column(String(30), nullable=False)
    # paper / candidate / domain / method_node
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    signal_name: Mapped[str] = mapped_column(String(80), nullable=False)
    signal_value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    signal_strength: Mapped[float | None] = mapped_column(Float)
    evidence_refs: Mapped[dict | None] = mapped_column(JSONB)
    producer: Mapped[str] = mapped_column(String(30), nullable=False)
    # agent / heuristic / user / external
    confidence: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_ss_entity", "entity_type", "entity_id"),
    )
