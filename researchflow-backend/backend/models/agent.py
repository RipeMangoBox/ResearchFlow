"""AgentRun, AgentBlackboardItem, PaperExtraction, ReferenceRoleMap models.

AgentRun: tracks a single agent invocation with cost/token accounting.
AgentBlackboardItem: structured output written by an agent to the shared blackboard.
PaperExtraction: versioned structured extraction from a paper (methods, claims, etc.).
ReferenceRoleMap: role classification for each reference cited in a paper.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class AgentRun(Base):
    """A single agent invocation — tracks what ran, how long, and cost.

    Supports linking to paper, candidate, or domain as context.
    """
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id")
    )
    candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paper_candidates.id")
    )
    domain_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("domain_specs.id")
    )

    agent_name: Mapped[str] = mapped_column(String(50), nullable=False)
    phase: Mapped[str] = mapped_column(String(20), nullable=False)
    # discovery / triage / deep_ingest / graph_build / profile_gen / report_gen
    status: Mapped[str] = mapped_column(String(15), default="running")
    # running / success / failed / timeout

    # LLM details
    model_name: Mapped[str | None] = mapped_column(String(50))
    prompt_version: Mapped[str | None] = mapped_column(String(20))
    input_token_count: Mapped[int | None] = mapped_column(Integer)
    output_token_count: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_ar_paper", "paper_id"),
        Index("ix_ar_candidate", "candidate_id"),
    )


class AgentBlackboardItem(Base):
    """A structured item written to the shared blackboard by an agent.

    Agents read/write blackboard items to coordinate multi-step analysis
    without direct coupling between agent implementations.
    """
    __tablename__ = "agent_blackboard_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False
    )
    paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id")
    )
    candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paper_candidates.id")
    )

    item_type: Mapped[str] = mapped_column(String(30), nullable=False)
    # method_extract / claim / evidence / taxonomy_tag / score_signal
    value_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    evidence_refs: Mapped[dict | None] = mapped_column(JSONB)
    producer_agent: Mapped[str] = mapped_column(String(50), nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_abi_paper_type", "paper_id", "item_type"),
        Index("ix_abi_candidate_type", "candidate_id", "item_type"),
    )


class PaperExtraction(Base):
    """A versioned structured extraction from a paper.

    Stores typed extractions (methods, claims, formulas, datasets, etc.)
    with provenance back to the agent run that produced them.
    """
    __tablename__ = "paper_extractions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    extraction_type: Mapped[str] = mapped_column(String(30), nullable=False)
    # method / claim / formula / dataset / ablation / limitation / contribution
    value_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    evidence_refs: Mapped[dict | None] = mapped_column(JSONB)
    producer_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_runs.id")
    )
    extraction_version: Mapped[int] = mapped_column(SmallInteger, default=1)
    review_status: Mapped[str] = mapped_column(String(20), default="auto")
    # auto / human_approved / human_edited / rejected

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("paper_id", "extraction_type", "extraction_version",
                         name="uq_extraction_paper_type_ver"),
        Index("ix_pe_paper_type", "paper_id", "extraction_type"),
    )


class ReferenceRoleMap(Base):
    """Role classification for a single reference cited in a paper.

    Maps each reference to its role (baseline, foundation, extension, etc.)
    with optional links to candidates or existing papers in the KB.
    """
    __tablename__ = "reference_role_maps"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )

    # Reference identification
    ref_index: Mapped[str | None] = mapped_column(String(10))
    ref_title: Mapped[str | None] = mapped_column(Text)
    ref_arxiv_id: Mapped[str | None] = mapped_column(String(30))

    # Resolution links
    ref_candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paper_candidates.id")
    )
    ref_paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="SET NULL")
    )

    # Role classification
    role: Mapped[str] = mapped_column(String(30), nullable=False)
    # baseline / foundation / extension / negative / dataset / tool / sibling
    role_confidence: Mapped[float | None] = mapped_column(Float)
    where_mentioned: Mapped[list[str] | None] = mapped_column(ARRAY(Text), default=list)
    mention_count: Mapped[int] = mapped_column(SmallInteger, default=1)

    # Ingest recommendation
    recommended_ingest_level: Mapped[str | None] = mapped_column(String(20))
    # skip / metadata / abstract / deep / full
    recommendation_reason: Mapped[str | None] = mapped_column(Text)
    evidence_refs: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_rrm_paper", "paper_id"),
    )
