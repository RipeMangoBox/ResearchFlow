"""Faceted taxonomy DAG models.

Supports multi-dimensional paper classification:
domain, modality, task, subtask, learning_paradigm, scenario,
constraint, mechanism, method_baseline, model_family, dataset,
benchmark, metric, lab, venue.
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
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class TaxonomyNode(Base):
    """A node in the faceted taxonomy DAG.

    Each node has a dimension (domain, task, mechanism, etc.)
    and can have parent-child relationships via TaxonomyEdge.
    """
    __tablename__ = "taxonomy_nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    name_zh: Mapped[str | None] = mapped_column(String(200))
    dimension: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    # domain / modality / task / subtask / learning_paradigm / scenario
    # constraint / mechanism / method_baseline / model_family
    # dataset / benchmark / metric / lab / venue

    aliases: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    description: Mapped[str | None] = mapped_column(Text)
    version: Mapped[int] = mapped_column(SmallInteger, default=1)
    status: Mapped[str] = mapped_column(
        String(20), default="candidate"
    )  # candidate / reviewed / canonical
    sort_order: Mapped[int] = mapped_column(SmallInteger, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_taxnode_dimension", "dimension"),
        Index("idx_taxnode_name", "name"),
        Index("idx_taxnode_status", "status"),
    )


class TaxonomyEdge(Base):
    """An edge in the taxonomy DAG.

    Supports multiple relation types: is_a, part_of, uses,
    optimizes, evaluates_on, applies_to, constrained_by, combines_with.
    """
    __tablename__ = "taxonomy_edges"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    parent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    child_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # is_a / part_of / uses / optimizes / evaluates_on / applies_to / constrained_by / combines_with
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    evidence_ids: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    __table_args__ = (
        UniqueConstraint("parent_id", "child_id", "relation_type",
                         name="uq_taxonomy_edge"),
        Index("idx_taxedge_parent", "parent_id"),
        Index("idx_taxedge_child", "child_id"),
    )


class PaperFacet(Base):
    """Links a paper to a taxonomy node with a specific facet role.

    A paper can have multiple facets (e.g., modality=video, task=VQA,
    paradigm=RL, mechanism=reward_design).
    """
    __tablename__ = "paper_facets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    facet_role: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    # primary_task / secondary_task / modality / paradigm / scenario
    # mechanism / baseline / dataset / benchmark / constraint

    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    source: Mapped[str | None] = mapped_column(String(30))
    # auto_cso / auto_llm / human
    evidence_refs: Mapped[dict | None] = mapped_column(JSONB)

    __table_args__ = (
        UniqueConstraint("paper_id", "node_id", "facet_role",
                         name="uq_paper_facet"),
        Index("idx_facet_paper", "paper_id"),
        Index("idx_facet_node", "node_id"),
        Index("idx_facet_role", "facet_role"),
    )


class ProblemNode(Base):
    """A common problem/bottleneck under a specific task.

    Replaces the old ProjectBottleneck as a top-level concept.
    Problems are now scoped under tasks in the taxonomy.
    """
    __tablename__ = "problem_nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    name_zh: Mapped[str | None] = mapped_column(Text)
    parent_task_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    # FK to taxonomy_nodes (dimension=task)

    scope_facet_ids: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    symptom: Mapped[str | None] = mapped_column(Text)
    root_cause: Mapped[str | None] = mapped_column(Text)
    why_common: Mapped[str | None] = mapped_column(Text)
    solution_families: Mapped[dict | None] = mapped_column(JSONB)
    # Schema: [{name, description, representative_paper_ids}]
    evidence_paper_ids: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    status: Mapped[str] = mapped_column(
        String(20), default="candidate"
    )  # candidate / canonical

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_problem_task", "parent_task_id"),
    )


class ProblemClaim(Base):
    """A paper's claim about a problem it addresses."""
    __tablename__ = "problem_claims"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    problem_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    claim_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # mentions / solves / partially_solves / reveals / worsens
    evidence_refs: Mapped[dict | None] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)

    __table_args__ = (
        Index("idx_claim_paper", "paper_id"),
        Index("idx_claim_problem", "problem_id"),
    )
