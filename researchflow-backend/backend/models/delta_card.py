"""DeltaCard — the intermediate truth layer.

DeltaCard sits between PaperAnalysis (raw LLM output) and IdeaDelta (reusable knowledge atom).
It captures what a paper changed relative to a canonical paradigm, with structured slot changes,
confidence scores, and evidence references. Built once from L4 analysis, rendered many times.

Hierarchy: Paper → PaperAnalysis → DeltaCard → IdeaDelta → GraphAssertion
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class DeltaCard(Base):
    """Intermediate truth layer — extracted once, projected many times.

    A DeltaCard captures the structured "what changed" from a single paper's L4 analysis,
    aligned to the ontology (paradigm frame, slots, mechanisms, bottlenecks).

    Status lifecycle: draft → published → deprecated
    """
    __tablename__ = "delta_cards"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    analysis_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paper_analyses.id")
    )
    frame_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paradigm_templates.id")
    )

    # Paradigm alignment
    baseline_paradigm: Mapped[str | None] = mapped_column(Text)
    primary_bottleneck_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("project_bottlenecks.id")
    )
    changed_slot_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    unchanged_slot_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    mechanism_family_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))

    # Baseline evolution — DAG inheritance (core of method lineage tracking)
    # Which existing methods/papers does this paper build on?
    parent_delta_card_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    # Which specific papers are used as baselines? (may differ from parent_delta_cards)
    baseline_paper_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    # Method lineage depth (0 = foundational paradigm, 1 = direct improvement, 2+ = chain)
    lineage_depth: Mapped[int | None] = mapped_column(SmallInteger, default=0)
    # Has this DeltaCard become a baseline for others? (auto-updated)
    is_established_baseline: Mapped[bool | None] = mapped_column(default=False)
    # How many downstream papers use this as baseline?
    downstream_count: Mapped[int | None] = mapped_column(SmallInteger, default=0)

    # Core content
    delta_statement: Mapped[str] = mapped_column(Text, nullable=False)
    key_ideas_ranked: Mapped[dict | None] = mapped_column(JSONB)
    # Schema: [{rank: int, statement: str, confidence: float}]

    # Scores (3 key dimensions)
    structurality_score: Mapped[float | None] = mapped_column(Float)
    extensionability_score: Mapped[float | None] = mapped_column(Float)
    transferability_score: Mapped[float | None] = mapped_column(Float)

    # Assumptions & failure modes
    assumptions: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    failure_modes: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    evaluation_context: Mapped[str | None] = mapped_column(Text)

    # Evidence references
    evidence_refs: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))

    # Confidence breakdown
    extraction_confidence: Mapped[float | None] = mapped_column(Float)
    linkage_confidence: Mapped[float | None] = mapped_column(Float)
    evidence_confidence: Mapped[float | None] = mapped_column(Float)

    # Status lifecycle
    status: Mapped[str] = mapped_column(
        String(20), default="draft"
    )  # draft / published / deprecated

    # Provenance (append-only: each card is an immutable snapshot)
    analysis_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    source_asset_hash: Mapped[str | None] = mapped_column(String(64))
    model_provider: Mapped[str | None] = mapped_column(String(50))
    model_name: Mapped[str | None] = mapped_column(String(100))
    model_run_id: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(20))
    schema_version: Mapped[str | None] = mapped_column(String(20))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_delta_cards_paper", "paper_id"),
        Index("idx_delta_cards_status", "status"),
        Index("idx_delta_cards_frame", "frame_id"),
        Index("idx_delta_cards_bottleneck", "primary_bottleneck_id"),
    )
