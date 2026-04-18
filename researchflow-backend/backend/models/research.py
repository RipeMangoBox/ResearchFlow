"""ProjectBottleneck, PaperBottleneckClaim, ProjectFocusBottleneck, SearchSession, ReadingPlan models."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, Index, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class ProjectBottleneck(Base):
    """Global bottleneck ontology — shared across all domains.

    This is the canonical registry. Paper claims and project focus both reference this.
    """
    __tablename__ = "project_bottlenecks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    symptom_query: Mapped[str | None] = mapped_column(Text)
    latent_need: Mapped[str | None] = mapped_column(Text)
    constraints: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(20), default="active")
    priority: Mapped[int] = mapped_column(SmallInteger, default=3)
    related_paper_ids: Mapped[list[str] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    rejected_patterns: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    # Graph links
    domain: Mapped[str | None] = mapped_column(String(100))
    paradigm_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paradigm_templates.id")
    )

    embedding = mapped_column(Vector(1536), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PaperBottleneckClaim(Base):
    """What a paper claims it is solving — paper-level fact, not project judgment.

    Extracted from L4 analysis. Multiple papers can claim the same bottleneck.
    """
    __tablename__ = "paper_bottleneck_claims"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    bottleneck_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("project_bottlenecks.id"), nullable=False
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=True)
    is_fundamental: Mapped[bool | None] = mapped_column(Boolean)
    confidence: Mapped[float | None] = mapped_column()
    source: Mapped[str] = mapped_column(
        String(30), default="system_inferred"
    )  # system_inferred / human_verified
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_pbc_paper", "paper_id"),
        Index("idx_pbc_bottleneck", "bottleneck_id"),
    )


class ProjectFocusBottleneck(Base):
    """What a specific project/user currently cares about — decision-layer object.

    This is NOT derived from papers. This is what the researcher is actually stuck on.
    """
    __tablename__ = "project_focus_bottlenecks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    bottleneck_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("project_bottlenecks.id"), nullable=False
    )
    project_name: Mapped[str | None] = mapped_column(String(200))
    user_description: Mapped[str | None] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(SmallInteger, default=3)
    status: Mapped[str] = mapped_column(String(20), default="active")
    # active / resolved / parked
    constraints: Mapped[dict | None] = mapped_column(JSONB)
    negative_constraints: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    # e.g. ["no plugin-only", "must have open code"]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_pfb_bottleneck", "bottleneck_id"),
        Index("idx_pfb_status", "status"),
    )


class SearchSession(Base):
    __tablename__ = "search_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    symptom_query: Mapped[str] = mapped_column(Text, nullable=False)
    latent_need: Mapped[str | None] = mapped_column(Text)
    candidate_bottleneck_ids: Mapped[list[str] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    rejected_solution_patterns: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    search_branches: Mapped[dict | None] = mapped_column(JSONB)
    result_paper_ids: Mapped[list[str] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    rewrite_history: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ReadingPlan(Base):
    __tablename__ = "reading_plans"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str | None] = mapped_column(Text)
    bottleneck_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    # Tiered recommendations (paper_ids)
    canonical_baselines: Mapped[list[str]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    structural_improvements: Mapped[list[str]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    strong_team_followups: Mapped[list[str]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    patches_and_negatives: Mapped[list[str]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )

    rationale: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class SearchBranch(Base):
    """A named exploration branch within a search session.

    Tracks branching decisions: when user rejects a solution pattern
    and pivots to a new direction.
    """
    __tablename__ = "search_branches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("search_sessions.id", ondelete="CASCADE"), nullable=False
    )
    branch_name: Mapped[str] = mapped_column(String(200), nullable=False)
    hypothesis: Mapped[str | None] = mapped_column(Text)
    rejected_patterns: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    result_paper_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    status: Mapped[str] = mapped_column(String(20), default="active")
    # active / exhausted / merged
    parent_branch_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_search_branches_session", "session_id"),
    )


class RenderArtifact(Base):
    """A rendered output artifact from the system.

    Tracks generated reports, digests, reading plans, Obsidian exports,
    and any other derived output for audit and re-generation.
    """
    __tablename__ = "render_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    artifact_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # report / digest / reading_plan / obsidian_vault / csv_export
    title: Mapped[str | None] = mapped_column(Text)
    content_md: Mapped[str | None] = mapped_column(Text)
    object_key: Mapped[str | None] = mapped_column(Text)
    # Object storage key if stored externally
    paper_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    parameters: Mapped[dict | None] = mapped_column(JSONB)
    # Generation parameters for reproducibility
    generated_by: Mapped[str | None] = mapped_column(String(50))
    # model / user / cron
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_render_artifacts_type", "artifact_type"),
    )
