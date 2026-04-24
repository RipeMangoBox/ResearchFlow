"""Graph query service — 5-route query router + multi-path retrieval.

Route 1: Factual / citation  — who wrote this, who cites who
Route 2: Bottleneck          — what are the core bottlenecks, which ideas target them
Route 3: Mechanism           — what approaches exist for a mechanism family
Route 4: Transfer            — can insight X move to domain Y
Route 5: Synthesis / report  — aggregate for report generation

Updated for v3 architecture: queries use graph_assertions + graph_nodes
instead of the legacy graph_edges table. Falls back to graph_edges for
any data not yet migrated.
"""

import logging
from uuid import UUID

from sqlalchemy import and_, desc, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.assertion import GraphAssertion, GraphNode
from backend.models.delta_card import DeltaCard
from backend.models.evidence import EvidenceUnit
from backend.models.method import MethodNode
from backend.models.paper import Paper
from backend.models.research import ProjectBottleneck

logger = logging.getLogger(__name__)


# ── Node lookup helpers ───────────────────────────────────────────

async def _find_node(session: AsyncSession, ref_table: str, ref_id: UUID) -> GraphNode | None:
    """Find a graph node by its reference."""
    result = await session.execute(
        select(GraphNode).where(
            GraphNode.ref_table == ref_table,
            GraphNode.ref_id == ref_id,
        )
    )
    return result.scalar_one_or_none()


async def _resolve_node_ref(session: AsyncSession, node: GraphNode):
    """Resolve a graph node to its underlying entity."""
    table_map = {
        "papers": Paper,
        "delta_cards": DeltaCard,
        "evidence_units": EvidenceUnit,
        "slots": Slot,
        "method_nodes": MethodNode,
        "project_bottlenecks": ProjectBottleneck,
        "delta_cards": DeltaCard,
    }
    model = table_map.get(node.ref_table)
    if model:
        return await session.get(model, node.ref_id)
    return None


# ── Route 1: Factual / Citation ─────────────────────────────────

async def query_citations(
    session: AsyncSession,
    paper_id: UUID,
    direction: str = "both",
) -> dict:
    """Get citation graph for a paper via graph_assertions."""
    results = {"cites": [], "cited_by": []}

    paper_node = await _find_node(session, "papers", paper_id)

    if paper_node:
        if direction in ("outgoing", "both"):
            out = await session.execute(
                select(GraphAssertion).where(
                    GraphAssertion.from_node_id == paper_node.id,
                    GraphAssertion.edge_type == "cites",
                    GraphAssertion.status == "published",
                )
            )
            for a in out.scalars():
                to_node = await session.get(GraphNode, a.to_node_id)
                if to_node:
                    target = await session.get(Paper, to_node.ref_id)
                    if target:
                        results["cites"].append({
                            "paper_id": str(target.id),
                            "title": target.title,
                            "venue": target.venue,
                            "year": target.year,
                        })

        if direction in ("incoming", "both"):
            inc = await session.execute(
                select(GraphAssertion).where(
                    GraphAssertion.to_node_id == paper_node.id,
                    GraphAssertion.edge_type == "cites",
                    GraphAssertion.status == "published",
                )
            )
            for a in inc.scalars():
                from_node = await session.get(GraphNode, a.from_node_id)
                if from_node:
                    source = await session.get(Paper, from_node.ref_id)
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
    """Find DeltaCards that target a specific bottleneck."""
    results = {"bottlenecks": [], "delta_cards": []}

    if bottleneck_id:
        bottleneck = await session.get(ProjectBottleneck, bottleneck_id)
        if bottleneck:
            results["bottlenecks"].append({
                "id": str(bottleneck.id),
                "title": bottleneck.title,
                "description": bottleneck.description,
                "domain": bottleneck.domain,
            })
        ideas = await session.execute(
            select(DeltaCard).where(
                DeltaCard.primary_bottleneck_id == bottleneck_id
            ).order_by(desc(DeltaCard.structurality_score)).limit(limit)
        )
        for idea in ideas.scalars():
            paper = await session.get(Paper, idea.paper_id)
            results["delta_cards"].append(_idea_to_dict(idea, paper))

    elif keyword:
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
    """Find DeltaCards by mechanism family, using assertions + array fallback."""
    results = {"mechanism": None, "delta_cards": []}

    mf = None
    if mechanism_id:
        mf = await session.get(MethodNode, mechanism_id)
    elif mechanism_name:
        # Try exact match, then alias
        from backend.services.entity_resolution_service import resolve_mechanism
        mf = await resolve_mechanism(session, mechanism_name)

    if not mf:
        return results

    results["mechanism"] = {
        "id": str(mf.id),
        "name": mf.name,
        "domain": mf.domain,
        "description": mf.description,
    }

    # Primary: graph_assertions
    mf_node = await _find_node(session, "method_nodes", mf.id)
    seen_ids = set()
    if mf_node:
        edges = await session.execute(
            select(GraphAssertion).where(
                GraphAssertion.to_node_id == mf_node.id,
                GraphAssertion.edge_type == "instance_of_method",
                GraphAssertion.status == "published",
            ).limit(limit)
        )
        for a in edges.scalars():
            from_node = await session.get(GraphNode, a.from_node_id)
            if from_node and from_node.ref_table == "delta_cards":
                idea = await session.get(DeltaCard, from_node.ref_id)
                if idea:
                    paper = await session.get(Paper, idea.paper_id)
                    results["delta_cards"].append(_idea_to_dict(idea, paper))
                    seen_ids.add(idea.id)

    # Fallback: method_node_ids array on DeltaCard
    ideas_by_array = await session.execute(
        select(DeltaCard).where(
            DeltaCard.method_node_ids.contains([mf.id])
        ).limit(limit)
    )
    for idea in ideas_by_array.scalars():
        if idea.id not in seen_ids:
            paper = await session.get(Paper, idea.paper_id)
            results["delta_cards"].append(_idea_to_dict(idea, paper))

    return results


# ── Route 4: Transfer Query ────────────────────────────────────

async def query_transfers(
    session: AsyncSession,
    source_domain: str | None = None,
    target_domain: str | None = None,
    limit: int = 20,
) -> dict:
    """Find transferable ideas across domains via assertions."""
    results = {"transfers": []}

    # Primary: graph_assertions with transferable_to edge type
    stmt = select(GraphAssertion).where(
        GraphAssertion.edge_type == "transferable_to",
        GraphAssertion.status.in_(["published", "candidate"]),
    ).limit(limit)
    edges = await session.execute(stmt)

    for a in edges.scalars():
        from_node = await session.get(GraphNode, a.from_node_id)
        to_node = await session.get(GraphNode, a.to_node_id)
        if from_node and to_node:
            source = await _resolve_node_ref(session, from_node)
            target = await _resolve_node_ref(session, to_node)
            if isinstance(source, DeltaCard) and isinstance(target, DeltaCard):
                results["transfers"].append({
                    "source": _idea_to_dict(source),
                    "target": _idea_to_dict(target),
                    "confidence": a.confidence,
                    "assertion_source": a.assertion_source,
                    "status": a.status,
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
    """Gather DeltaCards + DeltaCards + evidence for report generation.

    v3: also includes DeltaCard data when available.
    """
    conditions = []
    if min_structurality:
        conditions.append(DeltaCard.structurality_score >= min_structurality)

    stmt = select(DeltaCard)
    if conditions:
        stmt = stmt.where(and_(*conditions))
    stmt = stmt.order_by(desc(DeltaCard.structurality_score)).limit(limit)

    ideas_result = await session.execute(stmt)
    ideas = list(ideas_result.scalars().all())

    if category:
        filtered = []
        for idea in ideas:
            paper = await session.get(Paper, idea.paper_id)
            if paper and paper.category == category:
                filtered.append(idea)
        ideas = filtered

    results = []
    for idea in ideas[:limit]:
        paper = await session.get(Paper, idea.paper_id)
        evidence = await session.execute(
            select(EvidenceUnit).where(
                EvidenceUnit.delta_card_id == idea.id
            ).limit(5)
        )
        ev_list = [
            {"atom_type": eu.atom_type, "claim": eu.claim, "confidence": eu.confidence,
             "basis": eu.basis.value if eu.basis else None}
            for eu in evidence.scalars()
        ]
        entry = _idea_to_dict(idea, paper)
        entry["evidence"] = ev_list

        # Attach DeltaCard if available
        if idea.delta_card_id:
            card = await session.get(DeltaCard, idea.delta_card_id)
            if card:
                entry["delta_card"] = {
                    "id": str(card.id),
                    "delta_statement": card.delta_statement,
                    "key_ideas_ranked": card.key_ideas_ranked,
                    "assumptions": card.assumptions,
                    "failure_modes": card.failure_modes,
                    "status": card.status,
                }

        results.append(entry)

    return {
        "category": category,
        "total": len(results),
        "delta_cards": results,
    }


# ── Paper → DeltaCard lookup ──────────────────────────────────────

async def get_delta_cards_for_paper(session: AsyncSession, paper_id: UUID) -> list[DeltaCard]:
    """Get all DeltaCards extracted from a paper (migrated from graph_service)."""
    result = await session.execute(
        select(DeltaCard).where(DeltaCard.paper_id == paper_id).order_by(desc(DeltaCard.created_at))
    )
    return list(result.scalars().all())


# ── Compatibility edge query (assertions + legacy) ──────────────

async def get_edges_for_node_compat(
    session: AsyncSession,
    node_type: str,
    node_id: UUID,
    direction: str = "both",
) -> list[dict]:
    """Get edges connected to a node, combining assertion-based and legacy edges.

    Returns a unified list of dicts so callers don't need to know the backend.
    Assertion-based edges are preferred; legacy graph_edges fill any gaps.
    """
    from backend.services import assertion_service

    results: list[dict] = []
    seen_keys: set[tuple] = set()  # (edge_type, source, target) dedup

    # ── 1. Assertion-based edges (v3) ──
    # Map node_type to ref_table for graph_nodes lookup
    _type_to_table = {
        "paper": "papers",
        "delta_card": "delta_cards",
        "evidence_unit": "evidence_units",
        "slot": "slots",
        "method_family": "method_nodes",
        "bottleneck": "project_bottlenecks",
        "delta_card": "delta_cards",
    }
    ref_table = _type_to_table.get(node_type)
    if ref_table:
        graph_node = await _find_node(session, ref_table, node_id)
        if graph_node:
            assertions = await assertion_service.get_assertions_for_node(
                session, graph_node.id, direction,
            )
            for a in assertions:
                from_node = await session.get(GraphNode, a.from_node_id)
                to_node = await session.get(GraphNode, a.to_node_id)
                key = (a.edge_type, str(a.from_node_id), str(a.to_node_id))
                seen_keys.add(key)
                results.append({
                    "id": str(a.id),
                    "source": "assertion",
                    "source_type": from_node.node_type if from_node else None,
                    "source_id": str(from_node.ref_id) if from_node else None,
                    "target_type": to_node.node_type if to_node else None,
                    "target_id": str(to_node.ref_id) if to_node else None,
                    "edge_type": a.edge_type,
                    "assertion_source": a.assertion_source,
                    "confidence": a.confidence,
                    "status": a.status,
                })

    # Legacy graph_edges removed — graph_assertions is the only edge source now.
    return results


# ── Graph stats ─────────────────────────────────────────────────

async def graph_stats(session: AsyncSession) -> dict:
    """Get overall graph statistics — includes both legacy and v3 counts."""
    idea_count = (await session.execute(text("SELECT count(*) FROM delta_cards"))).scalar()
    delta_card_count = (await session.execute(text("SELECT count(*) FROM delta_cards"))).scalar()

    # v3 assertions
    assertion_count = (await session.execute(text("SELECT count(*) FROM graph_assertions"))).scalar()
    node_count = (await session.execute(text("SELECT count(*) FROM graph_nodes"))).scalar()

    evidence_linked = (await session.execute(text(
        "SELECT count(*) FROM evidence_units WHERE delta_card_id IS NOT NULL"
    ))).scalar()
    mf_count = (await session.execute(text("SELECT count(*) FROM method_nodes"))).scalar()

    # Assertion status distribution
    assertion_dist = (await session.execute(text(
        "SELECT status, count(*) FROM graph_assertions GROUP BY status ORDER BY count(*) DESC"
    ))).fetchall()

    # Assertion edge type distribution
    edge_type_dist = (await session.execute(text(
        "SELECT edge_type, count(*) FROM graph_assertions GROUP BY edge_type ORDER BY count(*) DESC"
    ))).fetchall()

    # Publish status distribution
    pub_dist = (await session.execute(text(
        "SELECT publish_status, count(*) FROM delta_cards GROUP BY publish_status"
    ))).fetchall()

    # Review queue
    review_pending = (await session.execute(text(
        "SELECT count(*) FROM review_queue WHERE status IN ('pending', 'in_progress')"
    ))).scalar()

    return {
        "delta_cards": delta_card_count,
        "graph_assertions": assertion_count,
        "graph_nodes": node_count,
        "evidence_linked": evidence_linked,
        "method_nodes": mf_count,
        "review_pending": review_pending,
        "assertion_status": {row[0]: row[1] for row in assertion_dist},
        "assertion_edge_types": {row[0]: row[1] for row in edge_type_dist},
        "publish_status": {row[0]: row[1] for row in pub_dist},
    }


# ── Helpers ─────────────────────────────────────────────────────

def _idea_to_dict(idea: DeltaCard, paper: Paper | None = None) -> dict:
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
        "delta_card_id": str(idea.delta_card_id) if idea.delta_card_id else None,
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
