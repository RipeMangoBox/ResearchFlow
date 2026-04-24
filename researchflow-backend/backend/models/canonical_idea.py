"""CanonicalIdea, ContributionToCanonicalIdea models.

CanonicalIdea: cross-paper normalized concept — the "real idea" that multiple
paper contributions map to.

ContributionToCanonicalIdea: links a paper's specific contribution (IdeaDelta)
to a canonical idea. One paper can have N contributions to M canonical ideas.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Index, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class CanonicalIdea(Base):
    """Cross-paper normalized idea — the reusable concept layer.

    Multiple IdeaDeltas (paper contributions) can map to one CanonicalIdea.
    This separates "what this paper did" from "what this idea IS" in the field.
    """
    __tablename__ = "canonical_ideas"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str | None] = mapped_column(String(100))
    method_node_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("method_nodes.id")
    )

    # How many papers contribute to this idea
    contribution_count: Mapped[int] = mapped_column(SmallInteger, default=0)

    # Maturity tracking
    status: Mapped[str] = mapped_column(
        String(20), default="candidate"
    )  # candidate / established / deprecated / merged

    # If merged into another canonical idea
    merged_into_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    # Aliases for entity resolution
    aliases: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    # Tags for retrieval
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    embedding = mapped_column(Vector(1536), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_canonical_ideas_domain", "domain"),
        Index("idx_canonical_ideas_status", "status"),
        Index("idx_canonical_ideas_method_node", "method_node_id"),
    )


class ContributionToCanonicalIdea(Base):
    """Maps a paper's contribution (IdeaDelta) to a CanonicalIdea.

    One paper can have multiple contributions to different canonical ideas.
    E.g., a paper might contribute a reward design AND a sampling trick.
    """
    __tablename__ = "contribution_to_canonical_idea"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    idea_delta_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("idea_deltas.id", ondelete="CASCADE"), nullable=False
    )
    canonical_idea_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("canonical_ideas.id", ondelete="CASCADE"), nullable=False
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )

    # What role does this contribution play?
    contribution_type: Mapped[str] = mapped_column(
        String(30), default="instance"
    )  # instance / origin / extension / refinement

    # Confidence that this mapping is correct
    confidence: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(
        String(30), default="system_inferred"
    )  # system_inferred / human_verified

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_ctci_idea_delta", "idea_delta_id"),
        Index("idx_ctci_canonical", "canonical_idea_id"),
        Index("idx_ctci_paper", "paper_id"),
    )
