"""PaperRelation — paper-to-paper baseline / cite DAG.

Materialized from `agent_blackboard_items.value_json` rows of type
`reference_role_map`, by paper_relation_service.materialize_relations.

The vault exporter renders these as Obsidian wiki-link sections so the graph
view picks up real method-evolution edges, not just tag co-occurrence.
"""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class PaperRelation(Base):
    __tablename__ = "paper_relations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
    )
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    evidence: Mapped[str | None] = mapped_column(Text, default="")
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(3, 2))
    ref_index: Mapped[str | None] = mapped_column(String(20))
    ref_title_raw: Mapped[str | None] = mapped_column(Text)
    match_method: Mapped[str | None] = mapped_column(String(20), default="title_fuzzy")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "source_paper_id", "target_paper_id", "relation_type",
            name="uq_paper_relations_triple",
        ),
        Index("ix_paper_relations_source", "source_paper_id"),
        Index("ix_paper_relations_target", "target_paper_id"),
        Index("ix_paper_relations_type", "relation_type"),
    )
