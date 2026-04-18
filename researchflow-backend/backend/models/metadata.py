"""Metadata observation ledger and canonical resolver models.

Multi-source metadata observations are recorded as-is, then resolved
into canonical values via authority ranking.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class MetadataObservation(Base):
    """A single observation of a metadata field from an external source.

    Multiple sources may report different values for the same field.
    The canonical resolver picks the best one based on authority_rank.
    """
    __tablename__ = "metadata_observations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # What entity this observation is about
    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # paper / author / venue / repo / dataset
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )

    # Which field and its value
    field_name: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # venue / status / authors / affiliation / citation_count / code_url / acceptance_type / review_scores
    value_json: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Source provenance
    source: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # arxiv / crossref / openalex / semantic_scholar / dblp / openreview / official_conf / google_scholar / github / pdf_grobid
    source_url: Mapped[str | None] = mapped_column(Text)
    raw_payload_object_key: Mapped[str | None] = mapped_column(Text)

    # Quality signals
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    authority_rank: Mapped[int] = mapped_column(
        SmallInteger, default=5
    )  # 1 = highest authority, 10 = lowest

    # Conflict tracking
    conflict_group_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    __table_args__ = (
        Index("idx_obs_entity", "entity_type", "entity_id"),
        Index("idx_obs_field", "entity_id", "field_name"),
        Index("idx_obs_source", "source"),
        Index("idx_obs_conflict", "conflict_group_id",
              postgresql_where="conflict_group_id IS NOT NULL"),
    )


class CanonicalPaperMetadata(Base):
    """Resolved canonical metadata for a paper, selected from observations.

    The resolver picks the best observation for each field based on
    authority_rank and confidence. Unresolved conflicts are flagged.
    """
    __tablename__ = "canonical_paper_metadata"

    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    # Resolved canonical fields
    canonical_title: Mapped[str | None] = mapped_column(Text)
    canonical_authors: Mapped[dict | None] = mapped_column(JSONB)
    # Schema: [{name, given_name, surname, affiliation, email, orcid}]
    canonical_affiliations: Mapped[dict | None] = mapped_column(JSONB)
    # Schema: [{name, country, ror_id}]
    canonical_venue: Mapped[str | None] = mapped_column(String(200))
    canonical_acceptance_status: Mapped[str | None] = mapped_column(String(50))
    # oral / poster / spotlight / workshop / rejected / under_review / unknown
    canonical_year: Mapped[int | None] = mapped_column(SmallInteger)
    canonical_citation_count: Mapped[int | None] = mapped_column(SmallInteger)
    canonical_code_url: Mapped[str | None] = mapped_column(Text)

    # Provenance
    selected_observation_ids: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    unresolved_conflicts: Mapped[dict | None] = mapped_column(JSONB)
    # Schema: [{field, sources: [{source, value}], reason}]

    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolver_version: Mapped[str | None] = mapped_column(String(20))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ── Authority rank constants ──────────────────────────────────────

AUTHORITY_RANKS = {
    # Venue / acceptance status
    "acceptance_status": {
        "official_conf": 1,
        "openreview": 2,
        "dblp": 3,
        "crossref": 4,
        "openalex": 4,
        "arxiv": 5,
        "semantic_scholar": 6,
        "google_scholar": 8,
    },
    # Citation count
    "citation_count": {
        "semantic_scholar": 1,
        "openalex": 2,
        "crossref": 3,
        "google_scholar": 5,
    },
    # Authors / affiliations
    "authors": {
        "pdf_grobid": 1,
        "openalex": 2,
        "crossref": 3,
        "semantic_scholar": 4,
        "arxiv": 5,
    },
    # Code URL
    "code_url": {
        "github": 1,
        "pdf_grobid": 2,
        "semantic_scholar": 3,
        "openalex": 4,
    },
}


def get_authority_rank(field_name: str, source: str) -> int:
    """Get the authority rank for a source on a given field.

    Lower number = higher authority. Returns 5 as default.
    """
    field_ranks = AUTHORITY_RANKS.get(field_name, {})
    return field_ranks.get(source, 5)
