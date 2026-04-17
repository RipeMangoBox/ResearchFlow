"""Graph query service — 5-route query router + multi-path retrieval.

Route 1: Factual / citation  — who wrote this, who cites who
Route 2: Bottleneck          — what are the core bottlenecks, which ideas target them
Route 3: Mechanism           — what approaches exist for a mechanism family
Route 4: Transfer            — can insight X move to domain Y
Route 5: Synthesis / report  — aggregate for report generation

Each route combines graph traversal with existing keyword/semantic search.
"""

import logging
from uuid import UUID

from sqlalchemy import and_, desc, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.evidence import EvidenceUnit
from backend.models.graph import GraphEdge, IdeaDelta, MechanismFamily, Slot
from backend.models.paper import Paper
from backend.models.research import ProjectBottleneck

logger = logging.getLogger(__name__)


# ── Route 1: Factual / Citation ─────────────────────────────────

async def query_citations(
    session: AsyncSession,
    paper_id: UUID,
    direction: str = "both",  # outgoing (this cites) / incoming (cited by) / both
) -> dict:
    """Get citation graph for a paper via graph_edges."""
    results = {"cites": [], "cited_by": []}

    if direction in ("outgoing", "both"):
        out_edges = await session.execute(
            select(GraphEdge).where(
                GraphEdge.source_type == "paper",
                GraphEdge.source_id == paper_id,
                GraphEdge.edge_type == "cites",
            )
        )
        for edge in out_edges.scalars():
            target = await session.get(Paper, edge.target_id)
            if target:
                results["cites"].append({
                    "paper_id": str(target.id),
                    "title": target.title,
                    "venue": target.venue,
                    "year": target.year,
                })

    if direction in ("incoming", "both"):
        in_edges = await session.execute(
            select(GraphEdge).where(
                GraphEdge.target_type == "paper",
                GraphEdge.target_id == paper_id,
                GraphEdge.edge_type == "cites",
            )
        )
        for edge in in_edges.scalars():
            source = await session.get(Paper, edge.source_id)
            if source:
                results["cited_by"].append({
                    "paper_id": str(source.id),
                    "title": source.title,
                    "venue": source.venue,
                    "year": source.year,
                })

    return results


# ── Route 2: Bottleneck Query ───────────────────────────────────

async def query_by_bottleneck(
    session: AsyncSession,
    bottleneck_id: UUID | None = None,
    keyword: str | None = None,
    limit: int = 20,
) -> dict:
    """Find IdeaDeltas that target a specific bottleneck, or search bottlenecks by keyword."""
    results = {"bottlenecks": [], "idea_deltas": []}

    if bottleneck_id:
        bottleneck = await session.get(ProjectBottleneck, bottleneck_id)
        if bottleneck:
            results["bottlenecks"].append({
                "id": str(bottleneck.id),
                "title": bottleneck.title,
                "description": bottleneck.description,
                "domain": bottleneck.domain,
            })
        # Find ideas targeting this bottleneck
        ideas = await session.execute(
            select(IdeaDelta).where(
                IdeaDelta.primary_bottleneck_id == bottleneck_id
            ).order_by(desc(IdeaDelta.structurality_score)).limit(limit)
        )
        for idea in ideas.scalars():
            paper = await session.get(Paper, idea.paper_id)
            results["idea_deltas"].append(_idea_to_dict(idea, paper))

    elif keyword:
        # Search bottlenecks by keyword
        bns = await session.execute(
            select(ProjectBottleneck).where(
                or_(
                    ProjectBottleneck.title.ilike(f"%{keyword}%"),
                    ProjectBottleneck.description.ilike(f"%{keyword}%"),
                )
            ).limit(10)
        )
        for bn in bns.scalars():
            results["bottlenecks"].append({
                "id": str(bn.id),
                "title": bn.title,
                "description": bn.description,
                "domain": bn.domain,
            })

    return results


# ── Route 3: Mechanism Query ────────────────────────────────────

async def query_by_mechanism(
    session: AsyncSession,
    mechanism_name: str | None = None,
    mechanism_id: UUID | None = None,
    limit: int = 20,
) -> dict:
    """Find IdeaDeltas by mechanism family."""
    results = {"mechanism": None, "idea_deltas": []}

    # Resolve mechanism
    mf = None
    if mechanism_id:
        mf = await session.get(MechanismFamily, mechanism_id)
    elif mechanism_name:
        mf_result = await session.execute(
            select(MechanismFamily).where(
                or_(
                    MechanismFamily.name == mechanism_name,
                    MechanismFamily.aliases.contains([mechanism_name]) if mechanism_name else False,
                )
            ).limit(1)
        )
        mf = mf_result.scalar_one_or_none()

    if not mf:
        return results

    results["mechanism"] = {
        "id": str(mf.id),
        "name": mf.name,
        "domain": mf.domain,
        "description": mf.description,
    }

    # Find ideas linked via graph_edges
    edges = await session.execute(
        select(GraphEdge).where(
            GraphEdge.target_type == "mechanism_family",
            GraphEdge.target_id == mf.id,
            GraphEdge.edge_type == "instance_of_mechanism",
        ).limit(limit)
    )
    for edge in edges.scalars():
        idea = await session.get(IdeaDelta, edge.source_id)
        if idea:
            paper = await session.get(Paper, idea.paper_id)
            results["idea_deltas"].append(_idea_to_dict(idea, paper))

    # Also find by mechanism_family_ids array
    ideas_by_array = await session.execute(
        select(IdeaDelta).where(
            IdeaDelta.mechanism_family_ids.contains([mf.id])
        ).limit(limit)
    )
    seen = {d["id"] for d in results["idea_deltas"]}
    for idea in ideas_by_array.scalars():
        if str(idea.id) not in seen:
            paper = await session.get(Paper, idea.paper_id)
            results["idea_deltas"].append(_idea_to_dict(idea, paper))

    return results


# ── Route 4: Transfer Query ─────────────────────────────────────

async def query_transfers(
    session: AsyncSession,
    source_domain: str | None = None,
    target_domain: str | None = None,
    limit: int = 20,
) -> dict:
    """Find transferable ideas across domains."""
    results = {"transfers": []}

    conditions = [GraphEdge.edge_type == "transferable_to"]
    if source_domain:
        # Join with IdeaDelta to filter by domain via paradigm
        pass  # For now, return all transfer edges

    edges = await session.execute(
        select(GraphEdge).where(
            GraphEdge.edge_type == "transferable_to"
        ).limit(limit)
    )
    for edge in edges.scalars():
        source_idea = await session.get(IdeaDelta, edge.source_id)
        target_idea = await session.get(IdeaDelta, edge.target_id)
        if source_idea and target_idea:
            results["transfers"].append({
                "source": _idea_to_dict(source_idea),
                "target": _idea_to_dict(target_idea),
                "confidence": edge.confidence,
                "assertion_source": edge.assertion_source,
            })

    return results


# ── Route 5: Synthesis — aggregate for reports ──────────────────

async def query_for_synthesis(
    session: AsyncSession,
    category: str | None = None,
    paradigm_name: str | None = None,
    min_structurality: float | None = None,
    limit: int = 30,
) -> dict:
    """Gather IdeaDeltas + evidence for report/synthesis generation."""
    conditions = []
    if min_structurality:
        conditions.append(IdeaDelta.structurality_score >= min_structurality)

    stmt = select(IdeaDelta)
    if conditions:
        stmt = stmt.where(and_(*conditions))
    stmt = stmt.order_by(desc(IdeaDelta.structurality_score)).limit(limit)

    ideas_result = await session.execute(stmt)
    ideas = list(ideas_result.scalars().all())

    # Filter by category if specified
    if category:
        filtered = []
        for idea in ideas:
            paper = await session.get(Paper, idea.paper_id)
            if paper and paper.category == category:
                filtered.append(idea)
        ideas = filtered

    # Gather with evidence
    results = []
    for idea in ideas[:limit]:
        paper = await session.get(Paper, idea.paper_id)
        evidence = await session.execute(
            select(EvidenceUnit).where(
                EvidenceUnit.idea_delta_id == idea.id
            ).limit(5)
        )
        ev_list = [
            {"atom_type": eu.atom_type, "claim": eu.claim, "confidence": eu.confidence,
             "basis": eu.basis.value if eu.basis else None}
            for eu in evidence.scalars()
        ]
        entry = _idea_to_dict(idea, paper)
        entry["evidence"] = ev_list
        results.append(entry)

    return {
        "category": category,
        "total": len(results),
        "idea_deltas": results,
    }


# ── Graph stats ─────────────────────────────────────────────────

async def graph_stats(session: AsyncSession) -> dict:
    """Get overall graph statistics."""
    idea_count = (await session.execute(text("SELECT count(*) FROM idea_deltas"))).scalar()
    edge_count = (await session.execute(text("SELECT count(*) FROM graph_edges"))).scalar()
    evidence_linked = (await session.execute(text("SELECT count(*) FROM evidence_units WHERE idea_delta_id IS NOT NULL"))).scalar()
    slot_count = (await session.execute(text("SELECT count(*) FROM slots"))).scalar()
    mf_count = (await session.execute(text("SELECT count(*) FROM mechanism_families"))).scalar()

    # Edge type distribution
    edge_dist = (await session.execute(text(
        "SELECT edge_type, count(*) FROM graph_edges GROUP BY edge_type ORDER BY count(*) DESC"
    ))).fetchall()

    # Publish status distribution
    pub_dist = (await session.execute(text(
        "SELECT publish_status, count(*) FROM idea_deltas GROUP BY publish_status"
    ))).fetchall()

    return {
        "idea_deltas": idea_count,
        "graph_edges": edge_count,
        "evidence_linked": evidence_linked,
        "slots": slot_count,
        "mechanism_families": mf_count,
        "edge_types": {row[0]: row[1] for row in edge_dist},
        "publish_status": {row[0]: row[1] for row in pub_dist},
    }


# ── Helpers ─────────────────────────────────────────────────────

def _idea_to_dict(idea: IdeaDelta, paper: Paper | None = None) -> dict:
    d = {
        "id": str(idea.id),
        "delta_statement": idea.delta_statement,
        "structurality_score": idea.structurality_score,
        "transferability_score": idea.transferability_score,
        "confidence": idea.confidence,
        "publish_status": idea.publish_status,
        "evidence_count": idea.evidence_count,
        "is_structural": idea.is_structural,
        "changed_slots": idea.changed_slots,
    }
    if paper:
        d["paper"] = {
            "id": str(paper.id),
            "title": paper.title,
            "venue": paper.venue,
            "year": paper.year,
            "category": paper.category,
        }
    return d
