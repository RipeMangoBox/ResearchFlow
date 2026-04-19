"""Edge profile generation — contextual descriptions for graph connections.

Each edge gets a one-liner explaining why two nodes are connected,
written from the perspective of the paper that established the connection.
"""

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.kb import GraphEdgeCandidate, KBEdgeProfile
from backend.services.agent_runner import AgentRunner
from backend.services.context_pack_builder import ContextPackBuilder

logger = logging.getLogger(__name__)


async def generate_edge_profile(
    session: AsyncSession,
    source_entity_type: str,
    source_entity_id: UUID,
    target_entity_type: str,
    target_entity_id: UUID,
    relation_type: str,
    *,
    paper_id: UUID | None = None,
    lang: str = "zh",
) -> KBEdgeProfile:
    """Generate a contextual profile for a graph edge.

    Gathers metadata and evidence for both endpoints, runs the
    ``edge_profile`` agent, and saves/updates a :class:`KBEdgeProfile`.
    """
    # ── Gather context for both endpoints ───────────────────────────────
    run_items = await _gather_edge_context(
        session,
        source_entity_type, source_entity_id,
        target_entity_type, target_entity_id,
        relation_type,
        paper_id=paper_id,
    )

    builder = ContextPackBuilder(session)
    context = await builder.build(
        "edge_profile",
        paper_id=paper_id,
        run_items=run_items,
    )

    # ── Run agent ───────────────────────────────────────────────────────
    runner = AgentRunner(session)
    result = await runner.run_agent(
        "edge_profile",
        context,
        paper_id=paper_id,
    )

    # ── Persist ─────────────────────────────────────────────────────────
    profile = await get_edge_profile(
        session,
        source_entity_type, source_entity_id,
        target_entity_type, target_entity_id,
    )

    if profile is None:
        profile = KBEdgeProfile(
            source_entity_type=source_entity_type,
            source_entity_id=source_entity_id,
            target_entity_type=target_entity_type,
            target_entity_id=target_entity_id,
            relation_type=relation_type,
            lang=lang,
        )
        session.add(profile)

    profile.one_liner = result.get("one_liner")
    profile.relation_summary = result.get("relation_summary")
    profile.source_context = result.get("source_context")
    profile.target_context = result.get("target_context")
    profile.evidence_refs = result.get("evidence_refs")
    profile.review_status = "auto"

    await session.flush()
    logger.info(
        "Generated edge profile %s -> %s (%s)",
        source_entity_type, target_entity_type, relation_type,
    )
    return profile


async def batch_generate_for_paper(
    session: AsyncSession,
    paper_id: UUID,
) -> int:
    """Generate edge profiles for all high-confidence edge candidates of a paper.

    Selects edge candidates with ``confidence_score >= 0.70`` and generates
    a profile for each.  Returns the count of profiles generated.
    """
    edge_rows = (
        await session.execute(
            select(GraphEdgeCandidate).where(
                GraphEdgeCandidate.paper_id == paper_id,
                GraphEdgeCandidate.confidence_score >= 0.70,
                GraphEdgeCandidate.status != "rejected",
                GraphEdgeCandidate.source_entity_id.isnot(None),
                GraphEdgeCandidate.target_entity_id.isnot(None),
            )
        )
    ).scalars().all()

    count = 0
    for edge in edge_rows:
        try:
            await generate_edge_profile(
                session,
                source_entity_type=edge.source_entity_type,
                source_entity_id=edge.source_entity_id,
                target_entity_type=edge.target_entity_type,
                target_entity_id=edge.target_entity_id,
                relation_type=edge.relation_type,
                paper_id=paper_id,
            )
            count += 1
        except Exception:
            logger.exception(
                "Failed to generate edge profile for edge %s", edge.id,
            )

    logger.info(
        "Batch generated %d / %d edge profiles for paper %s",
        count, len(edge_rows), paper_id,
    )
    return count


async def get_edge_profile(
    session: AsyncSession,
    source_entity_type: str,
    source_entity_id: UUID,
    target_entity_type: str,
    target_entity_id: UUID,
) -> KBEdgeProfile | None:
    """Look up an existing edge profile by its endpoint types and IDs."""
    return (
        await session.execute(
            select(KBEdgeProfile).where(
                KBEdgeProfile.source_entity_type == source_entity_type,
                KBEdgeProfile.source_entity_id == source_entity_id,
                KBEdgeProfile.target_entity_type == target_entity_type,
                KBEdgeProfile.target_entity_id == target_entity_id,
            )
        )
    ).scalar_one_or_none()


# ── Private Helpers ─────────────────────────────────────────────────────


async def _gather_edge_context(
    session: AsyncSession,
    source_entity_type: str,
    source_entity_id: UUID,
    target_entity_type: str,
    target_entity_id: UUID,
    relation_type: str,
    *,
    paper_id: UUID | None = None,
) -> dict:
    """Build run_items for the edge_profile context pack.

    Populates: source_node, target_node, relation_type, evidence_snippets.
    """
    from backend.models.kb import GraphNodeCandidate

    run_items: dict = {
        "relation_type": relation_type,
    }

    # ── Source node info ────────────────────────────────────────────────
    run_items["source_node"] = await _load_node_summary(
        session, source_entity_type, source_entity_id,
    )

    # ── Target node info ────────────────────────────────────────────────
    run_items["target_node"] = await _load_node_summary(
        session, target_entity_type, target_entity_id,
    )

    # ── Evidence snippets ───────────────────────────────────────────────
    edge_candidate = (
        await session.execute(
            select(
                GraphEdgeCandidate.one_liner,
                GraphEdgeCandidate.evidence_refs,
                GraphEdgeCandidate.slot_name,
                GraphEdgeCandidate.confidence_score,
            ).where(
                GraphEdgeCandidate.source_entity_type == source_entity_type,
                GraphEdgeCandidate.source_entity_id == source_entity_id,
                GraphEdgeCandidate.target_entity_type == target_entity_type,
                GraphEdgeCandidate.target_entity_id == target_entity_id,
            )
            .limit(1)
        )
    ).first()

    if edge_candidate:
        run_items["evidence_snippets"] = {
            "one_liner": edge_candidate.one_liner,
            "evidence_refs": edge_candidate.evidence_refs,
            "slot_name": edge_candidate.slot_name,
            "confidence": edge_candidate.confidence_score,
        }
    else:
        run_items["evidence_snippets"] = {}

    # ── Paper context (if available) ────────────────────────────────────
    if paper_id:
        from backend.models.paper import Paper

        paper = (
            await session.execute(
                select(Paper.title, Paper.venue, Paper.year)
                .where(Paper.id == paper_id)
            )
        ).first()
        if paper:
            run_items["evidence_snippets"]["paper_title"] = paper.title
            run_items["evidence_snippets"]["paper_venue"] = paper.venue
            run_items["evidence_snippets"]["paper_year"] = paper.year

    return run_items


async def _load_node_summary(
    session: AsyncSession,
    entity_type: str,
    entity_id: UUID,
) -> dict:
    """Load a brief summary for a node from candidates or profiles."""
    from backend.models.kb import GraphNodeCandidate, KBNodeProfile

    # Try profile first (richer)
    profile = (
        await session.execute(
            select(
                KBNodeProfile.one_liner,
                KBNodeProfile.entity_type,
            ).where(
                KBNodeProfile.entity_type == entity_type,
                KBNodeProfile.entity_id == entity_id,
            )
            .limit(1)
        )
    ).first()

    if profile and profile.one_liner:
        return {
            "entity_type": entity_type,
            "entity_id": str(entity_id),
            "one_liner": profile.one_liner,
        }

    # Fallback to candidate
    candidate = (
        await session.execute(
            select(
                GraphNodeCandidate.name,
                GraphNodeCandidate.name_zh,
                GraphNodeCandidate.node_type,
                GraphNodeCandidate.one_liner,
            ).where(
                GraphNodeCandidate.promoted_entity_type == entity_type,
                GraphNodeCandidate.promoted_entity_id == entity_id,
            )
            .limit(1)
        )
    ).first()

    if candidate:
        return {
            "entity_type": entity_type,
            "entity_id": str(entity_id),
            "name": candidate.name,
            "name_zh": candidate.name_zh,
            "node_type": candidate.node_type,
            "one_liner": candidate.one_liner,
        }

    return {
        "entity_type": entity_type,
        "entity_id": str(entity_id),
    }
