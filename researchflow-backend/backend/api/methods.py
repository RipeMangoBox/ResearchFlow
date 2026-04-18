"""Methods API — CRUD for method nodes, edges, applications."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.models.method import MethodNode, MethodSlot, MethodEdge, MethodApplication

router = APIRouter(prefix="/methods", tags=["methods"])


@router.get("/nodes")
async def list_method_nodes(
    maturity: str | None = None,
    type_filter: str | None = None,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    """List method nodes, optionally filtered by maturity or type."""
    q = select(MethodNode)
    if maturity:
        q = q.where(MethodNode.maturity == maturity)
    if type_filter:
        q = q.where(MethodNode.type == type_filter)
    q = q.order_by(MethodNode.downstream_count.desc(), MethodNode.name).limit(limit)
    result = await session.execute(q)
    return [
        {
            "id": str(m.id), "name": m.name, "name_zh": m.name_zh,
            "type": m.type, "maturity": m.maturity,
            "downstream_count": m.downstream_count,
            "description": m.description,
            "canonical_paper_id": str(m.canonical_paper_id) if m.canonical_paper_id else None,
        }
        for m in result.scalars().all()
    ]


@router.get("/nodes/{method_id}")
async def get_method_detail(
    method_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Get method detail with slots and edges."""
    method = await session.get(MethodNode, method_id)
    if not method:
        return {"error": "Method not found"}

    # Get slots
    slots_result = await session.execute(
        select(MethodSlot).where(MethodSlot.method_id == method_id)
        .order_by(MethodSlot.sort_order)
    )
    slots = [{"name": s.slot_name, "description": s.default_description}
             for s in slots_result.scalars().all()]

    # Get parent edges (methods this builds on)
    parents_result = await session.execute(
        select(MethodEdge, MethodNode)
        .join(MethodNode, MethodEdge.parent_method_id == MethodNode.id)
        .where(MethodEdge.child_method_id == method_id)
    )
    parents = [
        {"parent": e_n.name, "relation": e.relation_type,
         "delta": e.delta_description, "status": e.status}
        for e, e_n in parents_result.all()
    ]

    # Get child edges (methods that build on this)
    children_result = await session.execute(
        select(MethodEdge, MethodNode)
        .join(MethodNode, MethodEdge.child_method_id == MethodNode.id)
        .where(MethodEdge.parent_method_id == method_id)
    )
    children = [
        {"child": e_n.name, "relation": e.relation_type,
         "delta": e.delta_description, "status": e.status}
        for e, e_n in children_result.all()
    ]

    # Get paper applications
    apps_result = await session.execute(
        select(MethodApplication).where(MethodApplication.method_id == method_id)
    )
    applications = [
        {"paper_id": str(a.paper_id), "role": a.role}
        for a in apps_result.scalars().all()
    ]

    return {
        "id": str(method.id), "name": method.name, "type": method.type,
        "maturity": method.maturity, "description": method.description,
        "downstream_count": method.downstream_count,
        "slots": slots, "parents": parents, "children": children,
        "applications": applications,
    }


@router.get("/lineage/{method_id}")
async def get_method_lineage(
    method_id: UUID,
    depth: int = 3,
    session: AsyncSession = Depends(get_session),
):
    """Get method evolution DAG (ancestors + descendants)."""
    method = await session.get(MethodNode, method_id)
    if not method:
        return {"error": "Method not found"}

    ancestors = await _trace_lineage(session, method_id, "up", depth)
    descendants = await _trace_lineage(session, method_id, "down", depth)

    return {
        "root": {"id": str(method.id), "name": method.name, "maturity": method.maturity},
        "ancestors": ancestors,
        "descendants": descendants,
    }


async def _trace_lineage(session, method_id: UUID, direction: str, max_depth: int) -> list:
    """Trace method lineage up (parents) or down (children)."""
    result = []
    visited = {method_id}
    frontier = [method_id]

    for depth in range(max_depth):
        next_frontier = []
        for mid in frontier:
            if direction == "up":
                edges = (await session.execute(
                    select(MethodEdge).where(MethodEdge.child_method_id == mid)
                )).scalars().all()
                neighbor_ids = [e.parent_method_id for e in edges]
            else:
                edges = (await session.execute(
                    select(MethodEdge).where(MethodEdge.parent_method_id == mid)
                )).scalars().all()
                neighbor_ids = [e.child_method_id for e in edges]

            for edge, nid in zip(edges, neighbor_ids):
                if nid in visited:
                    continue
                visited.add(nid)
                node = await session.get(MethodNode, nid)
                if node:
                    result.append({
                        "id": str(node.id), "name": node.name,
                        "maturity": node.maturity, "depth": depth + 1,
                        "relation": edge.relation_type,
                        "delta": edge.delta_description,
                    })
                    next_frontier.append(nid)

        frontier = next_frontier
        if not frontier:
            break

    return result
