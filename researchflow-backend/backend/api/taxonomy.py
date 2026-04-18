"""Taxonomy API — CRUD for taxonomy nodes, edges, paper facets."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.models.taxonomy import TaxonomyNode, TaxonomyEdge, PaperFacet, ProblemNode

router = APIRouter(prefix="/taxonomy", tags=["taxonomy"])


@router.get("/nodes")
async def list_nodes(
    dimension: str | None = None,
    status: str | None = None,
    limit: int = 100,
    session: AsyncSession = Depends(get_session),
):
    """List taxonomy nodes, optionally filtered by dimension and status."""
    q = select(TaxonomyNode)
    if dimension:
        q = q.where(TaxonomyNode.dimension == dimension)
    if status:
        q = q.where(TaxonomyNode.status == status)
    q = q.order_by(TaxonomyNode.dimension, TaxonomyNode.sort_order, TaxonomyNode.name)
    q = q.limit(limit)
    result = await session.execute(q)
    nodes = result.scalars().all()
    return [
        {
            "id": str(n.id), "name": n.name, "name_zh": n.name_zh,
            "dimension": n.dimension, "status": n.status,
            "aliases": n.aliases, "description": n.description,
        }
        for n in nodes
    ]


@router.get("/tree")
async def get_tree(
    root_dimension: str = "domain",
    session: AsyncSession = Depends(get_session),
):
    """Get taxonomy tree starting from a dimension (domain → tasks → subtasks)."""
    # Get all nodes
    nodes_result = await session.execute(select(TaxonomyNode))
    all_nodes = {n.id: n for n in nodes_result.scalars().all()}

    # Get all edges
    edges_result = await session.execute(select(TaxonomyEdge))
    all_edges = edges_result.scalars().all()

    # Build children map
    children: dict[UUID, list] = {}
    for edge in all_edges:
        children.setdefault(edge.parent_id, []).append({
            "child_id": str(edge.child_id),
            "relation": edge.relation_type,
        })

    # Build tree from root dimension
    roots = [n for n in all_nodes.values() if n.dimension == root_dimension]

    def build_subtree(node_id: UUID, depth: int = 0) -> dict:
        node = all_nodes.get(node_id)
        if not node or depth > 5:
            return None
        result = {
            "id": str(node.id), "name": node.name, "name_zh": node.name_zh,
            "dimension": node.dimension, "children": [],
        }
        for child_info in children.get(node_id, []):
            child = build_subtree(UUID(child_info["child_id"]), depth + 1)
            if child:
                child["relation"] = child_info["relation"]
                result["children"].append(child)
        return result

    return [build_subtree(r.id) for r in roots]


@router.get("/dimensions")
async def list_dimensions(session: AsyncSession = Depends(get_session)):
    """List all taxonomy dimensions with node counts."""
    result = await session.execute(
        select(TaxonomyNode.dimension, func.count(TaxonomyNode.id))
        .group_by(TaxonomyNode.dimension)
        .order_by(TaxonomyNode.dimension)
    )
    return [{"dimension": dim, "count": count} for dim, count in result.all()]


@router.get("/paper/{paper_id}/facets")
async def get_paper_facets(
    paper_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Get all taxonomy facets assigned to a paper."""
    result = await session.execute(
        select(PaperFacet, TaxonomyNode)
        .join(TaxonomyNode, PaperFacet.node_id == TaxonomyNode.id)
        .where(PaperFacet.paper_id == paper_id)
    )
    return [
        {
            "facet_role": f.facet_role,
            "node_name": n.name,
            "node_name_zh": n.name_zh,
            "dimension": n.dimension,
            "confidence": f.confidence,
            "source": f.source,
        }
        for f, n in result.all()
    ]


@router.get("/problems")
async def list_problems(
    task_id: UUID | None = None,
    session: AsyncSession = Depends(get_session),
):
    """List problem nodes, optionally filtered by parent task."""
    q = select(ProblemNode)
    if task_id:
        q = q.where(ProblemNode.parent_task_id == task_id)
    result = await session.execute(q)
    return [
        {
            "id": str(p.id), "name": p.name, "name_zh": p.name_zh,
            "symptom": p.symptom, "root_cause": p.root_cause,
            "status": p.status,
        }
        for p in result.scalars().all()
    ]
