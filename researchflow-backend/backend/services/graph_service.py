"""Graph service — create IdeaDeltas, persist EvidenceUnits, manage edges.

Core workflow after L4 analysis:
  1. frame_assign → match paradigm
  2. idea_extract → create IdeaDelta from LLM output
  3. evidence_persist → save EvidenceUnits as independent DB rows
  4. edge_create → create GraphEdges (supported_by, changes_slot, etc.)
  5. publish_check → enforce evidence gate (evidence_count >= 2 → auto_published)
"""

import logging
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.evidence import EvidenceUnit
from backend.models.graph import GraphEdge, IdeaDelta, ImplementationUnit
from backend.models.paper import Paper

logger = logging.getLogger(__name__)

# Minimum evidence to auto-publish
MIN_EVIDENCE_FOR_PUBLISH = 2
AUTO_PUBLISH_CONFIDENCE = 0.85


async def create_idea_delta(
    session: AsyncSession,
    paper_id: UUID,
    analysis_id: UUID | None,
    paradigm_id: UUID | None,
    delta_statement: str,
    changed_slots: list[dict] | None = None,
    mechanism_family_ids: list[UUID] | None = None,
    structurality_score: float | None = None,
    transferability_score: float | None = None,
    confidence: float | None = None,
    is_structural: bool | None = None,
    primary_gain_source: str | None = None,
    bottleneck_id: UUID | None = None,
) -> IdeaDelta:
    """Create an IdeaDelta — the core knowledge graph object."""
    idea = IdeaDelta(
        paper_id=paper_id,
        analysis_id=analysis_id,
        primary_bottleneck_id=bottleneck_id,
        paradigm_id=paradigm_id,
        delta_statement=delta_statement,
        changed_slots=changed_slots,
        mechanism_family_ids=mechanism_family_ids,
        structurality_score=structurality_score,
        transferability_score=transferability_score,
        confidence=confidence,
        is_structural=is_structural,
        primary_gain_source=primary_gain_source,
        publish_status="draft",
        evidence_count=0,
    )
    session.add(idea)
    await session.flush()
    await session.refresh(idea)
    return idea


async def persist_evidence_units(
    session: AsyncSession,
    paper_id: UUID,
    analysis_id: UUID | None,
    idea_delta_id: UUID,
    evidence_data: list[dict],
) -> list[EvidenceUnit]:
    """Persist LLM-extracted evidence as independent DB rows.

    This fixes the critical GAP: evidence was only in JSONB before.
    Now each evidence unit is a first-class DB entity with FK to IdeaDelta.
    """
    units = []
    for ev in evidence_data:
        unit = EvidenceUnit(
            paper_id=paper_id,
            analysis_id=analysis_id,
            idea_delta_id=idea_delta_id,
            atom_type=ev.get("atom_type", "evidence"),
            claim=ev.get("claim", ""),
            evidence_type=ev.get("evidence_type"),
            causal_strength=ev.get("causal_strength"),
            confidence=ev.get("confidence"),
            source_section=ev.get("source_section"),
            source_page=ev.get("source_page"),
            source_quote=ev.get("source_quote"),
            conditions=ev.get("conditions"),
            failure_modes=ev.get("failure_modes"),
        )
        # Set basis enum if provided
        basis_str = ev.get("basis")
        if basis_str:
            from backend.models.enums import EvidenceBasis
            try:
                unit.basis = EvidenceBasis(basis_str)
            except ValueError:
                pass
        session.add(unit)
        units.append(unit)

    await session.flush()
    return units


async def create_edges_for_idea(
    session: AsyncSession,
    idea: IdeaDelta,
    evidence_units: list[EvidenceUnit],
    paradigm_slots: list[dict] | None = None,
) -> list[GraphEdge]:
    """Create graph edges after IdeaDelta + evidence are persisted.

    Edges created:
    - idea_delta → evidence_unit (supported_by)
    - idea_delta → slot (changes_slot) for each changed slot
    - idea_delta → mechanism_family (instance_of_mechanism)
    - idea_delta → bottleneck (targets_bottleneck)
    """
    edges = []

    # 1. supported_by: IdeaDelta → EvidenceUnit
    for eu in evidence_units:
        edge = GraphEdge(
            source_type="idea_delta",
            source_id=idea.id,
            target_type="evidence_unit",
            target_id=eu.id,
            edge_type="supported_by",
            assertion_source="inferred_by_system",
            confidence=eu.confidence,
            evidence_id=eu.id,
        )
        session.add(edge)
        edges.append(edge)

    # 2. changes_slot: IdeaDelta → Slot
    if idea.changed_slots and paradigm_slots:
        slot_name_to_id = {s["name"]: s["id"] for s in paradigm_slots}
        for slot_change in (idea.changed_slots if isinstance(idea.changed_slots, list) else []):
            slot_name = slot_change.get("slot_name") or slot_change.get("name", "")
            slot_id = slot_name_to_id.get(slot_name)
            if slot_id:
                edge = GraphEdge(
                    source_type="idea_delta",
                    source_id=idea.id,
                    target_type="slot",
                    target_id=slot_id,
                    edge_type="changes_slot",
                    assertion_source="inferred_by_system",
                    confidence=idea.confidence,
                    metadata_={"from": slot_change.get("from"), "to": slot_change.get("to"),
                               "change_type": slot_change.get("change_type")},
                )
                session.add(edge)
                edges.append(edge)

    # 3. instance_of_mechanism: IdeaDelta → MechanismFamily
    if idea.mechanism_family_ids:
        for mf_id in idea.mechanism_family_ids:
            edge = GraphEdge(
                source_type="idea_delta",
                source_id=idea.id,
                target_type="mechanism_family",
                target_id=mf_id,
                edge_type="instance_of_mechanism",
                assertion_source="inferred_by_system",
                confidence=idea.confidence,
            )
            session.add(edge)
            edges.append(edge)

    # 4. targets_bottleneck: IdeaDelta → Bottleneck
    if idea.primary_bottleneck_id:
        edge = GraphEdge(
            source_type="idea_delta",
            source_id=idea.id,
            target_type="bottleneck",
            target_id=idea.primary_bottleneck_id,
            edge_type="targets_bottleneck",
            assertion_source="inferred_by_system",
            confidence=idea.confidence,
        )
        session.add(edge)
        edges.append(edge)

    await session.flush()
    return edges


async def check_publish(session: AsyncSession, idea_id: UUID) -> str:
    """Enforce hard constraint: IdeaDelta needs evidence to publish.

    Rules:
    - evidence_count >= 2 AND confidence >= 0.85 → auto_published
    - evidence_count >= 2 → draft (but publishable)
    - Otherwise → stays draft
    """
    idea = await session.get(IdeaDelta, idea_id)
    if not idea:
        return "not_found"

    # Count linked evidence
    count_result = await session.execute(
        select(func.count()).select_from(EvidenceUnit).where(
            EvidenceUnit.idea_delta_id == idea_id
        )
    )
    evidence_count = count_result.scalar() or 0
    idea.evidence_count = evidence_count

    if evidence_count >= MIN_EVIDENCE_FOR_PUBLISH:
        if idea.confidence and idea.confidence >= AUTO_PUBLISH_CONFIDENCE:
            idea.publish_status = "auto_published"
        # If confidence is below threshold, keep draft (needs human review)

    await session.flush()
    return idea.publish_status


async def get_idea_deltas_for_paper(session: AsyncSession, paper_id: UUID) -> list[IdeaDelta]:
    result = await session.execute(
        select(IdeaDelta).where(IdeaDelta.paper_id == paper_id).order_by(IdeaDelta.created_at.desc())
    )
    return list(result.scalars().all())


async def get_edges_for_node(
    session: AsyncSession,
    node_type: str,
    node_id: UUID,
    direction: str = "outgoing",  # outgoing / incoming / both
) -> list[GraphEdge]:
    """Get all edges connected to a node."""
    conditions = []
    if direction in ("outgoing", "both"):
        conditions.append(
            (GraphEdge.source_type == node_type) & (GraphEdge.source_id == node_id)
        )
    if direction in ("incoming", "both"):
        conditions.append(
            (GraphEdge.target_type == node_type) & (GraphEdge.target_id == node_id)
        )

    from sqlalchemy import or_
    result = await session.execute(
        select(GraphEdge).where(or_(*conditions)).order_by(GraphEdge.created_at.desc())
    )
    return list(result.scalars().all())
