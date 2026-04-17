"""Graph API router — knowledge graph queries and management.

Migrated: endpoints now use graph_query_service and assertion_service
instead of the legacy graph_service (which is kept for backward compat only).
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import assertion_service, graph_query_service, quality_service

router = APIRouter(prefix="/graph", tags=["graph"])


# ── Stats ───────────────────────────────────────────────────────

@router.get("/stats")
async def get_graph_stats(session: AsyncSession = Depends(get_session)):
    """Get knowledge graph statistics."""
    return await graph_query_service.graph_stats(session)


# ── IdeaDelta queries ───────────────────────────────────────────

@router.get("/ideas/{paper_id}")
async def get_ideas_for_paper(
    paper_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Get all IdeaDeltas extracted from a paper."""
    ideas = await graph_query_service.get_idea_deltas_for_paper(session, paper_id)
    return [graph_query_service._idea_to_dict(i) for i in ideas]


@router.get("/edges/{node_type}/{node_id}")
async def get_edges(
    node_type: str,
    node_id: UUID,
    direction: str = Query(default="both", pattern="^(outgoing|incoming|both)$"),
    session: AsyncSession = Depends(get_session),
):
    """Get all edges connected to a node.

    Returns a combined list of assertion-based edges (v3) and legacy
    graph_edges for any data not yet migrated.
    """
    edges = await graph_query_service.get_edges_for_node_compat(
        session, node_type, node_id, direction,
    )
    return edges


# ── 5-Route Query Router ───────────────────────────────────────

@router.get("/query/citations/{paper_id}")
async def query_citations(
    paper_id: UUID,
    direction: str = Query(default="both"),
    session: AsyncSession = Depends(get_session),
):
    """Route 1: Citation/factual query — who cites who."""
    return await graph_query_service.query_citations(session, paper_id, direction)


@router.get("/query/bottleneck")
async def query_by_bottleneck(
    bottleneck_id: UUID | None = None,
    keyword: str | None = None,
    limit: int = Query(default=20, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
):
    """Route 2: Bottleneck query — what ideas target a bottleneck."""
    return await graph_query_service.query_by_bottleneck(session, bottleneck_id, keyword, limit)


@router.get("/query/mechanism")
async def query_by_mechanism(
    name: str | None = None,
    mechanism_id: UUID | None = None,
    limit: int = Query(default=20, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
):
    """Route 3: Mechanism query — what approaches exist for a mechanism family."""
    return await graph_query_service.query_by_mechanism(session, name, mechanism_id, limit)


@router.get("/query/transfers")
async def query_transfers(
    source_domain: str | None = None,
    target_domain: str | None = None,
    limit: int = Query(default=20, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
):
    """Route 4: Transfer query — can insight X move to domain Y."""
    return await graph_query_service.query_transfers(session, source_domain, target_domain, limit)


@router.post("/query/synthesis")
async def query_for_synthesis(
    category: str | None = None,
    min_structurality: float | None = None,
    limit: int = Query(default=30, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
):
    """Route 5: Synthesis query — gather IdeaDeltas + evidence for reports."""
    return await graph_query_service.query_for_synthesis(
        session, category, min_structurality=min_structurality, limit=limit,
    )


# ── Paradigm & Mechanism listing ────────────────────────────────

@router.get("/paradigms")
async def list_paradigms(session: AsyncSession = Depends(get_session)):
    """List all paradigm frames with their slots."""
    from sqlalchemy import select, text
    from backend.models.analysis import ParadigmTemplate

    result = await session.execute(select(ParadigmTemplate).order_by(ParadigmTemplate.name))
    paradigms = []
    for p in result.scalars():
        slots_result = await session.execute(
            text("SELECT name, slot_type, description, is_required FROM slots WHERE paradigm_id = :pid ORDER BY sort_order"),
            {"pid": p.id},
        )
        paradigms.append({
            "id": str(p.id),
            "name": p.name,
            "domain": p.domain,
            "slots": [dict(row._mapping) for row in slots_result.fetchall()],
        })
    return paradigms


@router.get("/mechanisms")
async def list_mechanisms(
    domain: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """List mechanism families, optionally filtered by domain."""
    from sqlalchemy import select
    from backend.models.graph import MechanismFamily

    stmt = select(MechanismFamily).order_by(MechanismFamily.domain, MechanismFamily.name)
    if domain:
        stmt = stmt.where(MechanismFamily.domain == domain)
    result = await session.execute(stmt)
    return [
        {"id": str(m.id), "name": m.name, "domain": m.domain, "description": m.description}
        for m in result.scalars()
    ]


# ── Quality ────────────────────────────────────────────────────────

@router.get("/quality")
async def get_kb_quality_report(session: AsyncSession = Depends(get_session)):
    """Get aggregate quality report across all published IdeaDeltas and DeltaCards."""
    return await quality_service.compute_kb_quality_report(session)
