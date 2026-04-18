"""Graph assertion models — GraphNode, GraphAssertion, GraphAssertionEvidence.

Replaces the flat graph_edges table with a proper assertion-based graph model:
- GraphNode: unified node registry (type + ref_table + ref_id)
- GraphAssertion: edges with lifecycle (candidate → published → rejected/deprecated/superseded)
- GraphAssertionEvidence: links assertions to evidence units with roles and weights

This enables:
- Proper node identity (no type+id string hacks)
- Assertion lifecycle (candidate → review → publish)
- Multi-evidence support per assertion
- Explicit contradicts/qualifies roles
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class GraphNode(Base):
    """Unified node registry for the knowledge graph.

    Every entity that participates in graph assertions gets a node.
    The ref_table + ref_id point back to the actual data row.
    """
    __tablename__ = "graph_nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    node_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # paper/delta_card/idea_delta/evidence/bottleneck/slot/mechanism
    ref_table: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # actual table name
    ref_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )  # actual table PK
    status: Mapped[str] = mapped_column(
        String(20), default="active"
    )  # active / deprecated
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_graph_nodes_type", "node_type"),
        Index("idx_graph_nodes_ref", "ref_table", "ref_id", unique=True),
        Index("idx_graph_nodes_status", "status"),
    )


class GraphAssertion(Base):
    """A directed assertion between two graph nodes.

    Replaces graph_edges with proper lifecycle management:
    - candidate: proposed by system, needs review for high-value edge types
    - published: verified and visible in queries
    - rejected: reviewed and rejected
    - deprecated: superseded by newer assertion
    - superseded: replaced by a newer version

    High-value edge types (contradicts, transferable_to, patch_of) default to
    candidate status and require review before publishing.
    """
    __tablename__ = "graph_assertions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    from_node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("graph_nodes.id"), nullable=False
    )
    to_node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("graph_nodes.id"), nullable=False
    )
    edge_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    assertion_source: Mapped[str] = mapped_column(
        String(30), nullable=False, default="system_inferred"
    )  # paper_asserted / system_inferred / human_verified
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(
        String(20), default="candidate"
    )  # candidate / published / rejected / deprecated / superseded
    reviewed_by: Mapped[str | None] = mapped_column(String(50))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ORM relationships
    from_node = relationship("GraphNode", foreign_keys=[from_node_id], lazy="selectin")
    to_node = relationship("GraphNode", foreign_keys=[to_node_id], lazy="selectin")

    __table_args__ = (
        Index("idx_assertions_from", "from_node_id"),
        Index("idx_assertions_to", "to_node_id"),
        Index("idx_assertions_type", "edge_type"),
        Index("idx_assertions_status", "status"),
        Index("idx_assertions_source", "assertion_source"),
    )


class GraphAssertionEvidence(Base):
    """Links an assertion to its supporting/contradicting evidence.

    Each assertion can have multiple evidence units with different roles:
    - supports: evidence that backs this assertion
    - contradicts: evidence that weakens this assertion
    - qualifies: evidence that adds conditions/caveats
    """
    __tablename__ = "graph_assertion_evidence"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    assertion_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("graph_assertions.id", ondelete="CASCADE"), nullable=False
    )
    evidence_unit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("evidence_units.id", ondelete="CASCADE"), nullable=False
    )
    weight: Mapped[float | None] = mapped_column(Float)
    role: Mapped[str] = mapped_column(
        String(30), nullable=False, default="supports"
    )  # supports / contradicts / qualifies

    __table_args__ = (
        Index("idx_assertion_evidence_assertion", "assertion_id"),
        Index("idx_assertion_evidence_unit", "evidence_unit_id"),
    )
