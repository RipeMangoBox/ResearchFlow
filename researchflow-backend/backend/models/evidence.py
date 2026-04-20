"""EvidenceUnit, TransferAtom models."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, ForeignKey, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base
from backend.models.enums import EvidenceBasis


class EvidenceUnit(Base):
    __tablename__ = "evidence_units"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    analysis_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paper_analyses.id")
    )

    # Graph FK links (Layer 3→4 connection)
    idea_delta_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    delta_card_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    slot_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    atom_type: Mapped[str] = mapped_column(String(30), nullable=False)
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_type: Mapped[str | None] = mapped_column(String(30))
    causal_strength: Mapped[float | None] = mapped_column()

    # Per-claim confidence & basis (GAP 2 fix)
    confidence: Mapped[float | None] = mapped_column()   # 0.0–1.0
    basis: Mapped[EvidenceBasis | None] = mapped_column(
        Enum(EvidenceBasis, name="evidence_basis", create_type=False,
             values_callable=lambda e: [m.value for m in e]),
    )

    # Source anchoring — where in the paper this evidence comes from
    source_section: Mapped[str | None] = mapped_column(String(200))
    source_page: Mapped[int | None] = mapped_column(SmallInteger)
    source_quote: Mapped[str | None] = mapped_column(Text)

    conditions: Mapped[str | None] = mapped_column(Text)
    failure_modes: Mapped[str | None] = mapped_column(Text)

    embedding = mapped_column(Vector(1536), nullable=True)

    # ORM relationships
    paper = relationship("Paper", foreign_keys=[paper_id], lazy="selectin")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TransferAtom(Base):
    __tablename__ = "transfer_atoms"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source_domain: Mapped[str] = mapped_column(String(100), nullable=False)
    target_domain: Mapped[str] = mapped_column(String(100), nullable=False)
    mechanism: Mapped[str] = mapped_column(Text, nullable=False)
    preconditions: Mapped[str | None] = mapped_column(Text)
    failure_risks: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
