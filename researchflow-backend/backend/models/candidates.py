"""Candidate tables for paradigms, slots, and mechanisms.

Auto-discovery creates candidates, not live ontology entries.
Candidates must pass review gates before promotion:
  - Low similarity to existing entries
  - Used by multiple papers
  - Alias resolution confirms independence
  - Human review approved
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class ParadigmCandidate(Base):
    """A candidate paradigm discovered by LLM, pending review before promotion."""
    __tablename__ = "paradigm_candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    domain: Mapped[str | None] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)
    slots_json: Mapped[dict | None] = mapped_column(JSONB)
    # Schema: [{name, slot_type, description, is_required}]

    # How many papers triggered this candidate
    trigger_count: Mapped[int] = mapped_column(SmallInteger, default=1)
    trigger_paper_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))

    # Similarity to existing paradigms (lower = more novel)
    max_similarity_to_existing: Mapped[float | None] = mapped_column(Float)
    most_similar_paradigm_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    # Review status
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending / approved / rejected / merged
    promoted_paradigm_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    # FK → paradigm_templates.id if approved
    reviewed_by: Mapped[str | None] = mapped_column(String(50))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_notes: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_paradigm_cand_status", "status"),
        Index("idx_paradigm_cand_domain", "domain"),
    )


class SlotCandidate(Base):
    """A candidate slot discovered during analysis, pending review."""
    __tablename__ = "slot_candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paradigm_candidate_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    paradigm_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    # One of the above should be set

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    slot_type: Mapped[str | None] = mapped_column(String(50))
    trigger_count: Mapped[int] = mapped_column(SmallInteger, default=1)

    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending / approved / rejected
    promoted_slot_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_slot_cand_status", "status"),
    )


class MechanismCandidate(Base):
    """A candidate mechanism family discovered during analysis, pending review."""
    __tablename__ = "mechanism_candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    domain: Mapped[str | None] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)
    aliases: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    trigger_count: Mapped[int] = mapped_column(SmallInteger, default=1)
    trigger_paper_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))

    max_similarity_to_existing: Mapped[float | None] = mapped_column(Float)
    most_similar_mechanism_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending / approved / rejected / merged
    promoted_mechanism_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    reviewed_by: Mapped[str | None] = mapped_column(String(50))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_mechanism_cand_status", "status"),
        Index("idx_mechanism_cand_domain", "domain"),
    )
