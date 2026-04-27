"""PaperFigure — first-class storage for extracted paper figures.

Replaces the JSONB blob `PaperAnalysis.extracted_figure_images`. The blob
column is kept for backward compatibility; new ingestions write here, and
`scripts/backfill_paper_figures.py` migrates historical rows on demand.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class PaperFigure(Base):
    __tablename__ = "paper_figures"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
    )
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    type: Mapped[str] = mapped_column(String(16), nullable=False, default="figure")
    semantic_role: Mapped[str | None] = mapped_column(String(32), default="other")
    page_num: Mapped[int | None] = mapped_column(SmallInteger)
    bbox: Mapped[dict | None] = mapped_column(JSONB)
    object_key: Mapped[str] = mapped_column(String(500), nullable=False)
    public_url: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text, default="")
    description: Mapped[str | None] = mapped_column(Text, default="")
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    extraction_method: Mapped[str | None] = mapped_column(
        String(32), default="vlm_precise"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("paper_id", "label", name="uq_paper_figures_paper_label"),
        Index("ix_paper_figures_paper_id", "paper_id"),
        Index("ix_paper_figures_role", "semantic_role"),
    )
