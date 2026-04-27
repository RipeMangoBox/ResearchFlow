"""ProjectBottleneck and PaperBottleneckClaim models."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, Index, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class ProjectBottleneck(Base):
    """Global bottleneck ontology — shared across all domains."""
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
    """What a paper claims it is solving — paper-level fact."""
    __tablename__ = "paper_bottleneck_claims"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    bottleneck_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("project_bottlenecks.id"), nullable=True
    )
    raw_title: Mapped[str | None] = mapped_column(Text)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=True)
    is_fundamental: Mapped[bool | None] = mapped_column(Boolean)
    confidence: Mapped[float | None] = mapped_column()
    source: Mapped[str] = mapped_column(String(30), default="system_inferred")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_pbc_paper", "paper_id"),
        Index("idx_pbc_bottleneck", "bottleneck_id"),
    )
