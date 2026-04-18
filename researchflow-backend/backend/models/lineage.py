"""DeltaCardLineage — independent lineage table for method evolution DAG.

Replaces the embedded parent_delta_card_ids array in delta_cards.
Each row represents one directed edge: child builds_on/extends/replaces parent.

depth, downstream_count, is_established_baseline become projections computed
from this table, not primary write fields.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, SmallInteger, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class DeltaCardLineage(Base):
    """A directed edge in the method lineage DAG.

    child_delta_card_id --[relation_type]--> parent_delta_card_id
    """
    __tablename__ = "delta_card_lineage"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    child_delta_card_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("delta_cards.id", ondelete="CASCADE"), nullable=False
    )
    parent_delta_card_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("delta_cards.id", ondelete="CASCADE"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(
        String(30), nullable=False, default="builds_on"
    )  # builds_on / extends / replaces / inherits_baseline

    # Confidence and review status
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(
        String(20), default="candidate"
    )  # candidate / published / rejected

    # Evidence for this lineage claim
    evidence_type: Mapped[str | None] = mapped_column(String(50))
    # method_section_cited / experiment_baseline / repo_reuse / slot_inheritance

    # Provenance
    source: Mapped[str] = mapped_column(
        String(30), default="system_inferred"
    )  # system_inferred / human_verified
    reviewed_by: Mapped[str | None] = mapped_column(String(50))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_lineage_child", "child_delta_card_id"),
        Index("idx_lineage_parent", "parent_delta_card_id"),
        Index("idx_lineage_status", "status"),
        Index("idx_lineage_type", "relation_type"),
    )
