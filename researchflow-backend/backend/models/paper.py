"""Paper, PaperAsset, PaperVersion models."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base
from backend.models.enums import AssetType, Importance, PaperState, Tier


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    title_sanitized: Mapped[str] = mapped_column(Text, nullable=False)
    venue: Mapped[str | None] = mapped_column(String(100))
    year: Mapped[int | None] = mapped_column(SmallInteger)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    state: Mapped[PaperState] = mapped_column(
        Enum(PaperState, name="paper_state", create_type=True),
        nullable=False,
        default=PaperState.WAIT,
    )
    importance: Mapped[Importance | None] = mapped_column(
        Enum(Importance, name="importance", create_type=True),
        default=Importance.C,
    )
    tier: Mapped[Tier | None] = mapped_column(
        Enum(Tier, name="tier", create_type=True),
    )

    # Links
    paper_link: Mapped[str | None] = mapped_column(Text)
    project_link: Mapped[str | None] = mapped_column(Text)
    arxiv_id: Mapped[str | None] = mapped_column(String(20))
    doi: Mapped[str | None] = mapped_column(String(100))

    # Scores (computed by triage)
    keep_score: Mapped[float | None] = mapped_column()
    analysis_priority: Mapped[float | None] = mapped_column()
    structurality_score: Mapped[float | None] = mapped_column()
    extensionability_score: Mapped[float | None] = mapped_column()

    # Metadata from Zotero/Crossref
    authors: Mapped[dict | None] = mapped_column(JSONB)
    abstract: Mapped[str | None] = mapped_column(Text)
    keywords: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    license: Mapped[str | None] = mapped_column(String(50))
    funding: Mapped[str | None] = mapped_column(Text)
    open_data: Mapped[bool] = mapped_column(Boolean, default=False)
    open_code: Mapped[bool] = mapped_column(Boolean, default=False)
    code_url: Mapped[str | None] = mapped_column(Text)
    data_url: Mapped[str | None] = mapped_column(Text)

    # Classification
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list
    )
    mechanism_family: Mapped[str | None] = mapped_column(String(100))
    supervision_type: Mapped[str | None] = mapped_column(String(50))
    inference_pattern: Mapped[str | None] = mapped_column(String(100))

    # Frontmatter compatibility
    core_operator: Mapped[str | None] = mapped_column(Text)
    primary_logic: Mapped[str | None] = mapped_column(Text)
    claims: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    # PDF storage
    pdf_path_local: Mapped[str | None] = mapped_column(Text)
    pdf_object_key: Mapped[str | None] = mapped_column(Text)

    # Vector embedding
    embedding = mapped_column(Vector(1536), nullable=True)

    # Timestamps
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    downloaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Source tracking
    source: Mapped[str | None] = mapped_column(String(50))
    source_ref: Mapped[str | None] = mapped_column(Text)

    # Ephemeral / library-external input support
    is_ephemeral: Mapped[bool] = mapped_column(Boolean, default=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retention_days: Mapped[int | None] = mapped_column(SmallInteger, default=30)

    # Relationships
    assets: Mapped[list["PaperAsset"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    versions: Mapped[list["PaperVersion"]] = relationship(back_populates="paper", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_papers_state", "state"),
        Index("idx_papers_category", "category"),
        Index("idx_papers_venue_year", "venue", "year"),
        Index("idx_papers_tags", "tags", postgresql_using="gin"),
    )


class PaperAsset(Base):
    __tablename__ = "paper_assets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    asset_type: Mapped[AssetType] = mapped_column(
        Enum(AssetType, name="asset_type", create_type=True), nullable=False
    )
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(100))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    checksum: Mapped[str | None] = mapped_column(String(64))
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    paper: Mapped["Paper"] = relationship(back_populates="assets")

    __table_args__ = (
        Index("idx_paper_assets_paper", "paper_id"),
    )


class PaperVersion(Base):
    __tablename__ = "paper_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    version: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)
    diff_summary: Mapped[str | None] = mapped_column(Text)
    arxiv_version: Mapped[str | None] = mapped_column(String(10))
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    paper: Mapped["Paper"] = relationship(back_populates="versions")
