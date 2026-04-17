"""Digest model."""

import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Enum, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base
from backend.models.enums import PeriodType


class Digest(Base):
    __tablename__ = "digests"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    period_type: Mapped[PeriodType] = mapped_column(
        Enum(PeriodType, name="period_type", create_type=True), nullable=False
    )
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)

    source_paper_ids: Mapped[list[str] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    source_bottleneck_ids: Mapped[list[str] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    source_search_session_ids: Mapped[list[str] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))

    rendered_text: Mapped[str] = mapped_column(Text, nullable=False)
    render_version: Mapped[str] = mapped_column(String(20), default="v1")
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
