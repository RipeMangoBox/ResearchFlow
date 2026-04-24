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


# ── DeltaCard queries ───────────────────────────────────────────

@router.get("/ideas/{paper_id}")
async def get_ideas_for_paper(
    paper_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Get all DeltaCards extracted from a paper."""
    ideas = await graph_query_service.get_delta_cards_for_paper(session, paper_id)
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
    """Route 5: Synthesis query — gather DeltaCards + evidence for reports."""
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
    from backend.models.method import MethodNode

    stmt = select(MethodNode).order_by(MethodNode.domain, MethodNode.name)
    if domain:
        stmt = stmt.where(MethodNode.domain == domain)
    result = await session.execute(stmt)
    return [
        {"id": str(m.id), "name": m.name, "domain": m.domain, "description": m.description}
        for m in result.scalars()
    ]


# ── Quality ────────────────────────────────────────────────────────

@router.get("/quality")
async def get_kb_quality_report(session: AsyncSession = Depends(get_session)):
    """Get aggregate quality report across all published DeltaCards and DeltaCards."""
    return await quality_service.compute_kb_quality_report(session)


# ── Visualization data ──────────────────────────────────────────

@router.get("/admin-stats")
async def get_admin_stats(session: AsyncSession = Depends(get_session)):
    """Admin monitoring: pipeline stats, paper state distribution, analysis coverage, job health."""
    from sqlalchemy import text

    # Paper state distribution
    states = (await session.execute(text(
        "SELECT state, count(*) FROM papers GROUP BY state ORDER BY count(*) DESC"
    ))).fetchall()

    # Analysis coverage
    total_papers = (await session.execute(text("SELECT count(*) FROM papers"))).scalar()
    l4_count = (await session.execute(text(
        "SELECT count(DISTINCT paper_id) FROM paper_analyses WHERE level = 'l4_deep' AND is_current = true"
    ))).scalar()
    l3_count = (await session.execute(text(
        "SELECT count(DISTINCT paper_id) FROM paper_analyses WHERE level = 'l3_skim' AND is_current = true"
    ))).scalar()
    dc_count = (await session.execute(text(
        "SELECT count(*) FROM delta_cards WHERE status = 'published'"
    ))).scalar()

    # Review queue
    review_pending = (await session.execute(text(
        "SELECT count(*) FROM review_tasks WHERE status = 'pending'"
    ))).scalar()
    review_by_type = (await session.execute(text(
        "SELECT target_type, count(*) FROM review_tasks WHERE status = 'pending' GROUP BY target_type"
    ))).fetchall()

    # Enrichment coverage
    with_abstract = (await session.execute(text("SELECT count(*) FROM papers WHERE abstract IS NOT NULL"))).scalar()
    with_doi = (await session.execute(text("SELECT count(*) FROM papers WHERE doi IS NOT NULL"))).scalar()
    with_code = (await session.execute(text("SELECT count(*) FROM papers WHERE code_url IS NOT NULL"))).scalar()
    with_pdf = (await session.execute(text("SELECT count(*) FROM papers WHERE pdf_path_local IS NOT NULL OR pdf_object_key IS NOT NULL"))).scalar()

    # Candidate counts
    paradigm_cands = (await session.execute(text("SELECT count(*) FROM paradigm_candidates WHERE status = 'pending'"))).scalar()
    lineage_cands = (await session.execute(text("SELECT count(*) FROM delta_card_lineage WHERE status = 'candidate'"))).scalar()

    # Recent activity
    recent_imports = (await session.execute(text(
        "SELECT count(*) FROM papers WHERE created_at > now() - interval '7 days'"
    ))).scalar()
    recent_analyses = (await session.execute(text(
        "SELECT count(*) FROM paper_analyses WHERE generated_at > now() - interval '7 days'"
    ))).scalar()

    return {
        "paper_states": {r[0]: r[1] for r in states},
        "total_papers": total_papers,
        "analysis_coverage": {
            "l3_skim": l3_count,
            "l4_deep": l4_count,
            "delta_cards_published": dc_count,
            "coverage_pct": round((l4_count / total_papers * 100) if total_papers else 0, 1),
        },
        "enrichment": {
            "with_abstract": with_abstract,
            "with_doi": with_doi,
            "with_code": with_code,
            "with_pdf": with_pdf,
        },
        "review_queue": {
            "pending": review_pending,
            "by_type": {r[0]: r[1] for r in review_by_type},
        },
        "candidates": {
            "paradigms_pending": paradigm_cands,
            "lineage_pending": lineage_cands,
        },
        "recent_7d": {
            "imports": recent_imports,
            "analyses": recent_analyses,
        },
    }


@router.get("/vis-data")
async def get_graph_vis_data(
    limit: int = Query(default=200, ge=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    """Get graph data formatted for vis-network visualization.

    Returns nodes + edges for interactive rendering.
    Node types: paper, mechanism, bottleneck, paradigm.
    Edges from: graph_assertions (published), delta_card_lineage.
    """
    from sqlalchemy import text

    # Nodes: papers with delta cards
    paper_rows = (await session.execute(text("""
        SELECT p.id, p.title, p.venue, p.year, p.category,
               p.method_family, p.structurality_score,
               dc.baseline_paradigm
        FROM papers p
        LEFT JOIN delta_cards dc ON dc.id = p.current_delta_card_id
        WHERE p.state NOT IN ('archived_or_expired', 'skip')
        ORDER BY p.keep_score DESC NULLS LAST
        LIMIT :lim
    """), {"lim": limit})).fetchall()

    nodes = []
    edges = []
    node_ids = set()

    # Paper nodes
    for p in paper_rows:
        nid = f"paper:{p.id}"
        node_ids.add(nid)
        size = 10 + (float(p.structurality_score or 0) * 20)
        nodes.append({
            "id": nid,
            "label": p.title[:40],
            "title": f"{p.title}\n{p.venue} {p.year}\nStruct: {p.structurality_score}",
            "group": "paper",
            "size": round(size, 1),
            "category": p.category,
        })

        # Connect to mechanism
        if p.method_family:
            mech_id = f"mechanism:{p.method_family}"
            if mech_id not in node_ids:
                node_ids.add(mech_id)
                nodes.append({
                    "id": mech_id, "label": p.method_family,
                    "group": "mechanism", "size": 18,
                })
            edges.append({"from": nid, "to": mech_id, "label": "uses", "dashes": True})

        # Connect to paradigm
        if p.baseline_paradigm:
            para_id = f"paradigm:{p.baseline_paradigm}"
            if para_id not in node_ids:
                node_ids.add(para_id)
                nodes.append({
                    "id": para_id, "label": p.baseline_paradigm,
                    "group": "paradigm", "size": 22,
                })
            edges.append({"from": nid, "to": para_id, "label": "aligned_to", "dashes": True})

    # Lineage edges
    lineage_rows = (await session.execute(text("""
        SELECT dcl.relation_type, dcl.confidence, dcl.status,
               child_p.id AS child_id, parent_p.id AS parent_id
        FROM delta_card_lineage dcl
        JOIN delta_cards child_dc ON child_dc.id = dcl.child_delta_card_id
        JOIN papers child_p ON child_p.id = child_dc.paper_id
        JOIN delta_cards parent_dc ON parent_dc.id = dcl.parent_delta_card_id
        JOIN papers parent_p ON parent_p.id = parent_dc.paper_id
    """))).fetchall()

    for ln in lineage_rows:
        child_nid = f"paper:{ln.child_id}"
        parent_nid = f"paper:{ln.parent_id}"
        if child_nid in node_ids and parent_nid in node_ids:
            edges.append({
                "from": child_nid, "to": parent_nid,
                "label": ln.relation_type,
                "arrows": "to",
                "color": "#e74c3c" if ln.status == "published" else "#95a5a6",
            })

    # Bottleneck nodes (from paper claims)
    bn_rows = (await session.execute(text("""
        SELECT DISTINCT pb.id, pb.title, pb.domain
        FROM project_bottlenecks pb
        JOIN paper_bottleneck_claims pbc ON pbc.bottleneck_id = pb.id
        JOIN papers p ON p.id = pbc.paper_id
        WHERE p.state NOT IN ('archived_or_expired', 'skip')
        LIMIT 30
    """))).fetchall()

    for bn in bn_rows:
        bn_id = f"bottleneck:{bn.id}"
        if bn_id not in node_ids:
            node_ids.add(bn_id)
            nodes.append({
                "id": bn_id, "label": bn.title[:30],
                "title": bn.title, "group": "bottleneck", "size": 15,
            })

    # Bottleneck edges
    claim_rows = (await session.execute(text("""
        SELECT pbc.paper_id, pbc.bottleneck_id
        FROM paper_bottleneck_claims pbc
        JOIN papers p ON p.id = pbc.paper_id
        WHERE p.state NOT IN ('archived_or_expired', 'skip')
    """))).fetchall()

    for cl in claim_rows:
        p_nid = f"paper:{cl.paper_id}"
        bn_nid = f"bottleneck:{cl.bottleneck_id}"
        if p_nid in node_ids and bn_nid in node_ids:
            edges.append({"from": p_nid, "to": bn_nid, "label": "addresses", "dashes": [5, 5]})

    return {
        "nodes": nodes,
        "edges": edges,
        "groups": {
            "paper": {"color": "#3498db", "shape": "dot"},
            "mechanism": {"color": "#2ecc71", "shape": "diamond"},
            "paradigm": {"color": "#9b59b6", "shape": "triangle"},
            "bottleneck": {"color": "#e74c3c", "shape": "square"},
        },
    }
