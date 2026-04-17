"""Baseline evolution service — method lineage DAG + paradigm versioning.

Core problem: improvements are layered. GRPO → GRPO+LP → GRPO-LP+sampling.
Each improvement can become a new baseline. The knowledge graph must capture
this DAG structure, not just flat "paper changed slot X".

This service handles:
1. Linking a DeltaCard to its parent baselines (DAG edges)
2. Detecting when a method becomes an established baseline (citation-driven)
3. Evolving paradigm templates when a new baseline gains adoption
4. Computing lineage depth and downstream impact

Trigger: called after delta_card_build, using LLM-extracted baseline info
and citation data from Semantic Scholar.
"""

import logging
from uuid import UUID

from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.analysis import ParadigmTemplate
from backend.models.delta_card import DeltaCard
from backend.models.graph import IdeaDelta, Slot
from backend.models.paper import Paper

logger = logging.getLogger(__name__)

# A method becomes "established baseline" when N+ papers build on it
BASELINE_ADOPTION_THRESHOLD = 3


async def link_to_parent_baselines(
    session: AsyncSession,
    delta_card_id: UUID,
    analysis_data: dict,
) -> dict:
    """Link a DeltaCard to its parent baselines via the DAG.

    Uses LLM-extracted info from analysis_data.delta_card.paradigm
    and paper references to find parent DeltaCards.
    """
    card = await session.get(DeltaCard, delta_card_id)
    if not card:
        return {"error": "DeltaCard not found"}

    paper = await session.get(Paper, card.paper_id)
    if not paper:
        return {"error": "Paper not found"}

    parent_ids = []
    baseline_paper_ids = []

    # Strategy 1: Find parent DeltaCards by delta_card.paradigm name matching
    delta_data = analysis_data.get("delta_card", {})
    paradigm_name = delta_data.get("paradigm", "")
    if paradigm_name:
        # Find DeltaCards with same paradigm that are published
        parents = await session.execute(
            select(DeltaCard).where(
                DeltaCard.baseline_paradigm == paradigm_name,
                DeltaCard.status == "published",
                DeltaCard.paper_id != card.paper_id,
            ).order_by(desc(DeltaCard.downstream_count)).limit(5)
        )
        for parent in parents.scalars():
            parent_ids.append(parent.id)
            baseline_paper_ids.append(parent.paper_id)

    # Strategy 2: Match by mechanism families
    if card.mechanism_family_ids and not parent_ids:
        for mf_id in card.mechanism_family_ids[:3]:
            related = await session.execute(
                select(DeltaCard).where(
                    DeltaCard.mechanism_family_ids.contains([mf_id]),
                    DeltaCard.status == "published",
                    DeltaCard.paper_id != card.paper_id,
                ).order_by(desc(DeltaCard.downstream_count)).limit(3)
            )
            for r in related.scalars():
                if r.id not in parent_ids:
                    parent_ids.append(r.id)
                    baseline_paper_ids.append(r.paper_id)

    # Strategy 3: Use paper citation data if available
    if paper.cited_by_count and not parent_ids:
        # Find established baselines in the same category
        established = await session.execute(
            select(DeltaCard).where(
                DeltaCard.is_established_baseline.is_(True),
                DeltaCard.frame_id == card.frame_id,
            ).order_by(desc(DeltaCard.downstream_count)).limit(3)
        )
        for e in established.scalars():
            if e.id not in parent_ids:
                parent_ids.append(e.id)
                baseline_paper_ids.append(e.paper_id)

    # Compute lineage depth
    max_parent_depth = 0
    for pid in parent_ids:
        parent = await session.get(DeltaCard, pid)
        if parent and parent.lineage_depth is not None:
            max_parent_depth = max(max_parent_depth, parent.lineage_depth)

    # Update DeltaCard
    card.parent_delta_card_ids = parent_ids if parent_ids else None
    card.baseline_paper_ids = baseline_paper_ids if baseline_paper_ids else None
    card.lineage_depth = max_parent_depth + 1 if parent_ids else 0

    # Update downstream counts on parents
    for pid in parent_ids:
        parent = await session.get(DeltaCard, pid)
        if parent:
            parent.downstream_count = (parent.downstream_count or 0) + 1
            # Check if parent should become established baseline
            if (parent.downstream_count or 0) >= BASELINE_ADOPTION_THRESHOLD:
                parent.is_established_baseline = True

    await session.flush()

    # Create "builds_on" graph assertions
    from backend.services.delta_card_service import get_or_create_node
    from backend.models.assertion import GraphAssertion

    card_node = await get_or_create_node(session, "delta_card", "delta_cards", card.id)
    assertions_created = 0
    for pid in parent_ids:
        parent_node = await get_or_create_node(session, "delta_card", "delta_cards", pid)
        assertion = GraphAssertion(
            from_node_id=card_node.id,
            to_node_id=parent_node.id,
            edge_type="builds_on",
            assertion_source="system_inferred",
            confidence=0.8,
            status="published",
        )
        session.add(assertion)
        assertions_created += 1

    await session.flush()

    return {
        "delta_card_id": str(delta_card_id),
        "parent_count": len(parent_ids),
        "lineage_depth": card.lineage_depth,
        "assertions_created": assertions_created,
    }


async def check_paradigm_evolution(
    session: AsyncSession,
    domain: str | None = None,
) -> list[dict]:
    """Check if any DeltaCards have become established enough to warrant
    a new paradigm version.

    A paradigm evolves when:
    1. A DeltaCard has downstream_count >= BASELINE_ADOPTION_THRESHOLD
    2. It introduces structural changes (structurality_score >= 0.6)
    3. It's not already the anchor of a paradigm
    """
    conditions = [
        DeltaCard.is_established_baseline.is_(True),
        DeltaCard.structurality_score >= 0.6,
    ]
    if domain:
        conditions.append(DeltaCard.frame_id.in_(
            select(ParadigmTemplate.id).where(ParadigmTemplate.domain == domain)
        ))

    result = await session.execute(
        select(DeltaCard).where(*conditions)
        .order_by(desc(DeltaCard.downstream_count))
        .limit(10)
    )
    candidates = list(result.scalars().all())

    evolutions = []
    for card in candidates:
        # Check if already anchoring a paradigm
        existing = await session.execute(
            select(ParadigmTemplate).where(
                ParadigmTemplate.anchor_paper_id == card.paper_id
            )
        )
        if existing.scalar_one_or_none():
            continue  # Already a paradigm anchor

        paper = await session.get(Paper, card.paper_id)
        evolutions.append({
            "delta_card_id": str(card.id),
            "paper_title": paper.title if paper else "Unknown",
            "downstream_count": card.downstream_count,
            "structurality_score": card.structurality_score,
            "delta_statement": (card.delta_statement or "")[:200],
            "recommendation": "Create new paradigm version" if card.downstream_count >= 5 else "Monitor",
        })

    return evolutions


async def promote_to_paradigm(
    session: AsyncSession,
    delta_card_id: UUID,
    new_paradigm_name: str | None = None,
) -> ParadigmTemplate | None:
    """Promote an established baseline DeltaCard to a new paradigm version.

    Creates a new ParadigmTemplate that inherits the parent's slots
    but reflects the changes this DeltaCard introduces.
    """
    card = await session.get(DeltaCard, delta_card_id)
    if not card:
        return None

    paper = await session.get(Paper, card.paper_id)
    old_paradigm = await session.get(ParadigmTemplate, card.frame_id) if card.frame_id else None

    # Determine name
    if not new_paradigm_name:
        base_name = old_paradigm.name if old_paradigm else (paper.category if paper else "unknown")
        # Extract version number and increment
        old_version = old_paradigm.version if old_paradigm else "v0"
        version_num = int(old_version.replace("v", "")) if old_version.startswith("v") else 0
        new_paradigm_name = f"{base_name}_v{version_num + 1}"

    # Build new slot dict from old paradigm + DeltaCard changes
    new_slots = dict(old_paradigm.slots) if old_paradigm else {}

    # Create new paradigm
    new_paradigm = ParadigmTemplate(
        name=new_paradigm_name,
        version=f"v{int(old_paradigm.version.replace('v', '')) + 1}" if old_paradigm else "v1",
        domain=old_paradigm.domain if old_paradigm else (paper.category if paper else None),
        slots=new_slots,
        anchor_paper_id=card.paper_id,
    )
    session.add(new_paradigm)
    await session.flush()

    # Copy slots from old paradigm
    if old_paradigm:
        old_slots = await session.execute(
            select(Slot).where(Slot.paradigm_id == old_paradigm.id).order_by(Slot.sort_order)
        )
        for i, old_slot in enumerate(old_slots.scalars()):
            new_slot = Slot(
                paradigm_id=new_paradigm.id,
                name=old_slot.name,
                description=old_slot.description,
                slot_type=old_slot.slot_type,
                is_required=old_slot.is_required,
                sort_order=old_slot.sort_order,
            )
            session.add(new_slot)

        # Mark old paradigm as superseded
        old_paradigm.superseded_by = new_paradigm.id

    await session.flush()
    logger.info(f"Promoted DeltaCard {delta_card_id} to paradigm {new_paradigm_name}")
    return new_paradigm


async def get_lineage_tree(
    session: AsyncSession,
    delta_card_id: UUID,
    direction: str = "both",  # ancestors / descendants / both
    max_depth: int = 5,
) -> dict:
    """Get the method lineage tree for a DeltaCard.

    Returns the DAG of parent→child relationships showing
    how methods evolved.
    """
    card = await session.get(DeltaCard, delta_card_id)
    if not card:
        return {"error": "DeltaCard not found"}

    result = {
        "root": await _card_summary(session, card),
        "ancestors": [],
        "descendants": [],
    }

    # Walk ancestors
    if direction in ("ancestors", "both"):
        result["ancestors"] = await _walk_ancestors(session, card, max_depth)

    # Walk descendants
    if direction in ("descendants", "both"):
        result["descendants"] = await _walk_descendants(session, card, max_depth)

    return result


async def _walk_ancestors(session: AsyncSession, card: DeltaCard, max_depth: int) -> list[dict]:
    """Walk up the lineage DAG."""
    ancestors = []
    queue = [(card, 0)]
    seen = {card.id}

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            break
        for pid in (current.parent_delta_card_ids or []):
            if pid in seen:
                continue
            seen.add(pid)
            parent = await session.get(DeltaCard, pid)
            if parent:
                ancestors.append(await _card_summary(session, parent, depth=depth + 1))
                queue.append((parent, depth + 1))

    return ancestors


async def _walk_descendants(session: AsyncSession, card: DeltaCard, max_depth: int) -> list[dict]:
    """Walk down the lineage DAG."""
    descendants = []
    queue = [(card, 0)]
    seen = {card.id}

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            break
        children = await session.execute(
            select(DeltaCard).where(
                DeltaCard.parent_delta_card_ids.contains([current.id])
            )
        )
        for child in children.scalars():
            if child.id in seen:
                continue
            seen.add(child.id)
            descendants.append(await _card_summary(session, child, depth=depth + 1))
            queue.append((child, depth + 1))

    return descendants


async def _card_summary(session: AsyncSession, card: DeltaCard, depth: int = 0) -> dict:
    paper = await session.get(Paper, card.paper_id)
    return {
        "delta_card_id": str(card.id),
        "paper_id": str(card.paper_id),
        "title": paper.title if paper else "Unknown",
        "delta_statement": (card.delta_statement or "")[:200],
        "lineage_depth": card.lineage_depth,
        "downstream_count": card.downstream_count,
        "is_established_baseline": card.is_established_baseline,
        "structurality_score": card.structurality_score,
        "depth_in_tree": depth,
    }
