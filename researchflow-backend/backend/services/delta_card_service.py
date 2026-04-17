"""DeltaCard service — build the intermediate truth layer from L4 analysis.

Core workflow:
  L4 analysis_data → build_delta_card → persist evidence → derive IdeaDelta → propose assertions

DeltaCard is the structured "what changed" from a paper, aligned to ontology.
IdeaDelta is derived from DeltaCard as the reusable knowledge atom.
GraphAssertions are proposed from the DeltaCard's slot/mechanism/evidence links.

Publishing gate:
  - DeltaCard: frame + bottleneck + changed_slots + evidence_refs >= 2
  - IdeaDelta: min(extraction, linkage, evidence confidence) >= 0.85 → auto_published
  - High-value edges (contradicts, transferable_to, patch_of): candidate by default
"""

import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.assertion import GraphAssertion, GraphAssertionEvidence, GraphNode
from backend.models.delta_card import DeltaCard
from backend.models.evidence import EvidenceUnit
from backend.models.graph import IdeaDelta

logger = logging.getLogger(__name__)

# Thresholds
MIN_EVIDENCE_FOR_PUBLISH = 2
AUTO_PUBLISH_CONFIDENCE = 0.85
HIGH_VALUE_EDGE_TYPES = {"contradicts", "transferable_to", "patch_of"}


# ── DeltaCard construction ────────────────────────────────────────

async def build_delta_card(
    session: AsyncSession,
    paper_id: UUID,
    analysis_id: UUID | None,
    analysis_data: dict,
    paradigm_id: UUID | None = None,
    paradigm_name: str | None = None,
    slot_ids: list[UUID] | None = None,
    changed_slot_ids: list[UUID] | None = None,
    mechanism_family_ids: list[UUID] | None = None,
    bottleneck_id: UUID | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
) -> DeltaCard:
    """Build a DeltaCard from L4 analysis output.

    This is the central construction point: all downstream objects
    (IdeaDelta, GraphAssertions) derive from this card.
    """
    delta_card_data = analysis_data.get("delta_card", {})
    if not isinstance(delta_card_data, dict):
        delta_card_data = {}

    # Core delta statement
    core_intuition = analysis_data.get("core_intuition", "")
    delta_statement = (
        core_intuition
        or delta_card_data.get("primary_gain_source", "")
        or analysis_data.get("method_summary", "")[:500]
    )
    if not delta_statement:
        delta_statement = f"Analysis of paper {paper_id}"

    # Compute unchanged slots (all slots minus changed)
    unchanged_slot_ids = None
    if slot_ids and changed_slot_ids:
        changed_set = set(changed_slot_ids)
        unchanged_slot_ids = [s for s in slot_ids if s not in changed_set]

    # Extract key ideas from confidence notes
    key_ideas = _extract_key_ideas(analysis_data)

    # Extract assumptions and failure modes from evidence units
    evidence_data = analysis_data.get("evidence_units", [])
    assumptions = []
    failure_modes_list = []
    for ev in (evidence_data if isinstance(evidence_data, list) else []):
        if ev.get("conditions"):
            assumptions.append(ev["conditions"])
        if ev.get("failure_modes"):
            failure_modes_list.append(ev["failure_modes"])

    card = DeltaCard(
        paper_id=paper_id,
        analysis_id=analysis_id,
        frame_id=paradigm_id,
        baseline_paradigm=paradigm_name,
        primary_bottleneck_id=bottleneck_id,
        changed_slot_ids=changed_slot_ids,
        unchanged_slot_ids=unchanged_slot_ids,
        mechanism_family_ids=mechanism_family_ids,
        delta_statement=delta_statement[:2000],
        key_ideas_ranked=key_ideas if key_ideas else None,
        structurality_score=analysis_data.get("structurality_score"),
        extensionability_score=analysis_data.get("extensionability_score"),
        transferability_score=analysis_data.get("transferability_score"),
        assumptions=assumptions if assumptions else None,
        failure_modes=failure_modes_list if failure_modes_list else None,
        evaluation_context=analysis_data.get("evidence_summary"),
        extraction_confidence=analysis_data.get("confidence", 0.7),
        model_provider=model_provider,
        model_name=model_name,
        prompt_version="l4_v1",
        schema_version="v1",
        status="draft",
    )
    session.add(card)
    await session.flush()
    await session.refresh(card)
    return card


async def persist_evidence_for_card(
    session: AsyncSession,
    paper_id: UUID,
    analysis_id: UUID | None,
    delta_card_id: UUID,
    evidence_data: list[dict],
) -> list[EvidenceUnit]:
    """Persist evidence units linked to a DeltaCard.

    Each evidence unit gets both delta_card_id and (later) idea_delta_id.
    """
    units = []
    for ev in evidence_data:
        unit = EvidenceUnit(
            paper_id=paper_id,
            analysis_id=analysis_id,
            delta_card_id=delta_card_id,
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

    # Update delta_card evidence_refs
    if units:
        card = await session.get(DeltaCard, delta_card_id)
        if card:
            card.evidence_refs = [u.id for u in units]
            card.evidence_confidence = _compute_evidence_confidence(units)
            await session.flush()

    return units


async def derive_idea_delta(
    session: AsyncSession,
    delta_card: DeltaCard,
    evidence_units: list[EvidenceUnit],
    changed_slots_graph: list[dict] | None = None,
) -> IdeaDelta:
    """Derive an IdeaDelta from a DeltaCard.

    IdeaDelta is the reusable knowledge atom; DeltaCard is the raw extraction.
    """
    idea = IdeaDelta(
        paper_id=delta_card.paper_id,
        analysis_id=delta_card.analysis_id,
        delta_card_id=delta_card.id,
        primary_bottleneck_id=delta_card.primary_bottleneck_id,
        paradigm_id=delta_card.frame_id,
        delta_statement=delta_card.delta_statement,
        changed_slots=changed_slots_graph,
        mechanism_family_ids=delta_card.mechanism_family_ids,
        structurality_score=delta_card.structurality_score,
        transferability_score=delta_card.transferability_score,
        confidence=delta_card.extraction_confidence,
        is_structural=(
            delta_card.structurality_score >= 0.5
            if delta_card.structurality_score is not None
            else None
        ),
        publish_status="draft",
        evidence_count=len(evidence_units),
    )
    session.add(idea)
    await session.flush()
    await session.refresh(idea)

    # Link evidence units to idea_delta
    for eu in evidence_units:
        eu.idea_delta_id = idea.id
    await session.flush()

    return idea


# ── Graph node + assertion pipeline ───────────────────────────────

async def get_or_create_node(
    session: AsyncSession,
    node_type: str,
    ref_table: str,
    ref_id: UUID,
) -> GraphNode:
    """Get or create a GraphNode for an entity."""
    result = await session.execute(
        select(GraphNode).where(
            GraphNode.ref_table == ref_table,
            GraphNode.ref_id == ref_id,
        )
    )
    node = result.scalar_one_or_none()
    if node:
        return node

    node = GraphNode(
        node_type=node_type,
        ref_table=ref_table,
        ref_id=ref_id,
        status="active",
    )
    session.add(node)
    await session.flush()
    await session.refresh(node)
    return node


async def propose_assertions(
    session: AsyncSession,
    idea: IdeaDelta,
    evidence_units: list[EvidenceUnit],
    paradigm_slots: list[dict] | None = None,
) -> list[GraphAssertion]:
    """Propose graph assertions from an IdeaDelta.

    Creates GraphNodes as needed and proposes assertions with appropriate
    status (candidate for high-value, published for structural).
    """
    assertions = []

    # Get/create idea node
    idea_node = await get_or_create_node(
        session, "idea_delta", "idea_deltas", idea.id
    )

    # 1. supported_by: IdeaDelta → EvidenceUnit
    for eu in evidence_units:
        eu_node = await get_or_create_node(
            session, "evidence", "evidence_units", eu.id
        )
        assertion = GraphAssertion(
            from_node_id=idea_node.id,
            to_node_id=eu_node.id,
            edge_type="supported_by",
            assertion_source="system_inferred",
            confidence=eu.confidence,
            status="published",  # structural, auto-publish
        )
        session.add(assertion)
        assertions.append(assertion)

        # Link assertion to evidence
        await session.flush()
        link = GraphAssertionEvidence(
            assertion_id=assertion.id,
            evidence_unit_id=eu.id,
            role="supports",
            weight=1.0,
        )
        session.add(link)

    # 2. changes_slot: IdeaDelta → Slot
    if idea.changed_slots and paradigm_slots:
        slot_name_to_id = {s["name"]: s["id"] for s in paradigm_slots}
        for slot_change in (idea.changed_slots if isinstance(idea.changed_slots, list) else []):
            slot_name = slot_change.get("slot_name") or slot_change.get("name", "")
            slot_id = slot_name_to_id.get(slot_name)
            if slot_id:
                slot_node = await get_or_create_node(
                    session, "slot", "slots", slot_id
                )
                assertion = GraphAssertion(
                    from_node_id=idea_node.id,
                    to_node_id=slot_node.id,
                    edge_type="changes_slot",
                    assertion_source="system_inferred",
                    confidence=idea.confidence,
                    status="published",
                    metadata_={
                        "from": slot_change.get("from"),
                        "to": slot_change.get("to"),
                        "change_type": slot_change.get("change_type"),
                    },
                )
                session.add(assertion)
                assertions.append(assertion)

    # 3. instance_of_mechanism: IdeaDelta → MechanismFamily
    if idea.mechanism_family_ids:
        for mf_id in idea.mechanism_family_ids:
            mf_node = await get_or_create_node(
                session, "mechanism", "mechanism_families", mf_id
            )
            assertion = GraphAssertion(
                from_node_id=idea_node.id,
                to_node_id=mf_node.id,
                edge_type="instance_of_mechanism",
                assertion_source="system_inferred",
                confidence=idea.confidence,
                status="published",
            )
            session.add(assertion)
            assertions.append(assertion)

    # 4. targets_bottleneck: IdeaDelta → Bottleneck
    if idea.primary_bottleneck_id:
        bn_node = await get_or_create_node(
            session, "bottleneck", "project_bottlenecks", idea.primary_bottleneck_id
        )
        assertion = GraphAssertion(
            from_node_id=idea_node.id,
            to_node_id=bn_node.id,
            edge_type="targets_bottleneck",
            assertion_source="system_inferred",
            confidence=idea.confidence,
            status="published",
        )
        session.add(assertion)
        assertions.append(assertion)

    await session.flush()
    return assertions


async def check_and_publish(
    session: AsyncSession,
    delta_card: DeltaCard,
    idea: IdeaDelta,
) -> tuple[str, str]:
    """Check publishing gates for both DeltaCard and IdeaDelta.

    DeltaCard gate: frame + bottleneck + changed_slots + evidence_refs >= 2
    IdeaDelta gate: min(extraction, linkage, evidence confidence) >= 0.85

    Returns: (delta_card_status, idea_status)
    """
    # Count evidence
    count_result = await session.execute(
        select(func.count()).select_from(EvidenceUnit).where(
            EvidenceUnit.idea_delta_id == idea.id
        )
    )
    evidence_count = count_result.scalar() or 0
    idea.evidence_count = evidence_count

    # DeltaCard publish check
    dc_ready = (
        delta_card.frame_id is not None
        and delta_card.changed_slot_ids
        and len(delta_card.evidence_refs or []) >= MIN_EVIDENCE_FOR_PUBLISH
    )
    if dc_ready:
        delta_card.status = "published"

    # IdeaDelta publish check
    if evidence_count >= MIN_EVIDENCE_FOR_PUBLISH:
        confidences = [
            c for c in [
                delta_card.extraction_confidence,
                delta_card.linkage_confidence,
                delta_card.evidence_confidence,
            ] if c is not None
        ]
        if confidences and min(confidences) >= AUTO_PUBLISH_CONFIDENCE:
            idea.publish_status = "auto_published"

    await session.flush()
    return delta_card.status, idea.publish_status


# ── Full pipeline ─────────────────────────────────────────────────

async def run_delta_card_pipeline(
    session: AsyncSession,
    paper_id: UUID,
    analysis_id: UUID | None,
    analysis_data: dict,
    paradigm_id: UUID | None,
    paradigm_name: str | None,
    slots: list[dict] | None,
    changed_slots_graph: list[dict] | None,
    mechanism_family_ids: list[UUID] | None = None,
    bottleneck_id: UUID | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
) -> dict:
    """Run the full DeltaCard pipeline: build → evidence → idea → assertions → publish.

    Returns dict with all created objects for logging/debugging.
    """
    # Compute slot IDs from paradigm slots
    slot_ids = [s["id"] for s in slots] if slots else None
    changed_slot_ids = None
    if changed_slots_graph and slots:
        slot_name_to_id = {s["name"]: s["id"] for s in slots}
        changed_slot_ids = [
            slot_name_to_id[sc.get("slot_name") or sc.get("name", "")]
            for sc in changed_slots_graph
            if (sc.get("slot_name") or sc.get("name", "")) in slot_name_to_id
        ]

    # 1. Build DeltaCard
    delta_card = await build_delta_card(
        session,
        paper_id=paper_id,
        analysis_id=analysis_id,
        analysis_data=analysis_data,
        paradigm_id=paradigm_id,
        paradigm_name=paradigm_name,
        slot_ids=slot_ids,
        changed_slot_ids=changed_slot_ids if changed_slot_ids else None,
        mechanism_family_ids=mechanism_family_ids,
        bottleneck_id=bottleneck_id,
        model_provider=model_provider,
        model_name=model_name,
    )

    # 2. Persist evidence
    evidence_data = analysis_data.get("evidence_units", [])
    if not isinstance(evidence_data, list):
        evidence_data = []
    evidence_units = await persist_evidence_for_card(
        session, paper_id, analysis_id, delta_card.id, evidence_data,
    )

    # 3. Derive IdeaDelta
    idea = await derive_idea_delta(
        session, delta_card, evidence_units, changed_slots_graph,
    )

    # 4. Propose assertions
    slots_dicts = [{"id": s["id"], "name": s["name"]} for s in slots] if slots else []
    assertions = await propose_assertions(
        session, idea, evidence_units, slots_dicts,
    )

    # 5. Check publish gates
    dc_status, idea_status = await check_and_publish(session, delta_card, idea)

    logger.info(
        f"DeltaCard pipeline complete: paper={paper_id}, "
        f"delta_card={delta_card.id}({dc_status}), "
        f"idea={idea.id}({idea_status}), "
        f"evidence={len(evidence_units)}, assertions={len(assertions)}"
    )

    return {
        "delta_card": delta_card,
        "idea_delta": idea,
        "evidence_units": evidence_units,
        "assertions": assertions,
    }


# ── Helpers ───────────────────────────────────────────────────────

def _extract_key_ideas(analysis_data: dict) -> list[dict] | None:
    """Extract ranked key ideas from confidence notes."""
    notes = analysis_data.get("confidence_notes", [])
    if not isinstance(notes, list) or not notes:
        return None

    ranked = []
    for i, note in enumerate(notes[:5]):
        if isinstance(note, dict) and note.get("claim"):
            ranked.append({
                "rank": i + 1,
                "statement": note["claim"],
                "confidence": note.get("confidence", 0.5),
            })
    return ranked if ranked else None


def _compute_evidence_confidence(units: list[EvidenceUnit]) -> float:
    """Compute aggregate evidence confidence from individual units."""
    if not units:
        return 0.0
    confidences = [u.confidence for u in units if u.confidence is not None]
    if not confidences:
        return 0.5
    return sum(confidences) / len(confidences)
