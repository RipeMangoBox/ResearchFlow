"""PaperAnalysis, MethodDelta, ParadigmTemplate models."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Index,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base
from backend.models.enums import AnalysisLevel


class PaperAnalysis(Base):
    __tablename__ = "paper_analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    level: Mapped[AnalysisLevel] = mapped_column(
        Enum(AnalysisLevel, name="analysis_level", create_type=False,
             values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )

    # Versioning
    model_provider: Mapped[str | None] = mapped_column(String(50))
    model_name: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float | None] = mapped_column()

    # L2: extracted sections
    extracted_sections: Mapped[dict | None] = mapped_column(JSONB)
    extracted_formulas: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    extracted_tables: Mapped[dict | None] = mapped_column(JSONB)
    figure_captions: Mapped[dict | None] = mapped_column(JSONB)

    # L3: skim card
    problem_summary: Mapped[str | None] = mapped_column(Text)
    method_summary: Mapped[str | None] = mapped_column(Text)
    evidence_summary: Mapped[str | None] = mapped_column(Text)
    core_intuition: Mapped[str | None] = mapped_column(Text)
    changed_slots: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    is_plugin_patch: Mapped[bool | None] = mapped_column(Boolean)
    worth_deep_read: Mapped[bool | None] = mapped_column(Boolean)

    # L4: deep report
    full_report_md: Mapped[str | None] = mapped_column(Text)
    full_report_object_key: Mapped[str | None] = mapped_column(Text)

    # Evidence spans — per-section source anchors
    # Schema: [{page, paragraph, quote, section_ref}]
    evidence_spans: Mapped[dict | None] = mapped_column(JSONB)

    # Per-claim confidence notes — distinguishes fact from inference
    # Schema: [{claim, confidence: 0-1, basis: "code_verified"|"experiment_backed"|
    #           "text_stated"|"inferred"|"speculative", source_anchor, reasoning}]
    confidence_notes: Mapped[dict | None] = mapped_column(JSONB)

    # Version chain
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_analyses_paper", "paper_id"),
        Index("idx_analyses_current", "paper_id", "is_current", postgresql_where="is_current"),
        Index("idx_analyses_level", "level"),
    )


class MethodDelta(Base):
    __tablename__ = "method_deltas"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    analysis_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    # Domain paradigm reference
    paradigm_name: Mapped[str] = mapped_column(String(100), nullable=False)
    paradigm_version: Mapped[str] = mapped_column(String(20), default="v1")

    # Slot-level delta
    slots: Mapped[dict] = mapped_column(JSONB, nullable=False)

    is_structural: Mapped[bool | None] = mapped_column(Boolean)
    primary_gain_source: Mapped[str | None] = mapped_column(String(100))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_deltas_paper", "paper_id"),
        Index("idx_deltas_paradigm", "paradigm_name"),
    )


class ParadigmTemplate(Base):
    __tablename__ = "paradigm_templates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    version: Mapped[str] = mapped_column(String(20), default="v1")
    domain: Mapped[str | None] = mapped_column(String(100))
    slots: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
