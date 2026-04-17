"""ProjectBottleneck, SearchSession, ReadingPlan models."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class ProjectBottleneck(Base):
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

    embedding = mapped_column(Vector(1536), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
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
