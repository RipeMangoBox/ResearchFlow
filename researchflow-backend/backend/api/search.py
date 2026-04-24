"""Search + embedding + reading plan + intent-based query router API."""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import embedding_service, reading_planner, search_service, query_router_service

router = APIRouter(tags=["search"])


# ── Hybrid search ───────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    category: str | None = None
    venue: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    tags: list[str] | None = None
    min_structurality: float | None = None
    must_not_method_categories: list[str] | None = None
    must_have_open_code: bool | None = None
    exclude_tags: list[str] | None = None
    semantic: bool = True
    limit: int = Field(default=20, ge=1, le=100)


@router.post("/search/hybrid")
async def hybrid_search(
    data: SearchRequest,
    session: AsyncSession = Depends(get_session),
):
    """Hybrid search: keyword (tsvector) + semantic (pgvector) + structured filters.

    Returns ranked results with text_score, vector_score, and combined_score.
    """
    results = await search_service.hybrid_search(
        session,
        query=data.query,
        category=data.category,
        venue=data.venue,
        year_min=data.year_min,
        year_max=data.year_max,
        tags=data.tags,
        min_structurality=data.min_structurality,
        must_not_method_categories=data.must_not_method_categories,
        must_have_open_code=data.must_have_open_code,
        exclude_tags=data.exclude_tags,
        semantic=data.semantic,
        limit=data.limit,
    )
    return {"query": data.query, "total": len(results), "results": results}


# ── Idea-centric search ────────────────────────────────────────

class IdeaSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    category: str | None = None
    min_structurality: float | None = None
    min_evidence: int | None = None
    limit: int = Field(default=20, ge=1, le=100)


@router.post("/search/ideas")
async def idea_search(
    data: IdeaSearchRequest,
    session: AsyncSession = Depends(get_session),
):
    """Idea-centric search across DeltaCards and DeltaCards.

    Searches delta_statement, key_ideas, and assumptions.
    Returns structured delta info with paper context.
    """
    results = await search_service.idea_search(
        session,
        query=data.query,
        category=data.category,
        min_structurality=data.min_structurality,
        min_evidence=data.min_evidence,
        limit=data.limit,
    )
    return {"query": data.query, "total": len(results), "results": results}


# ── Bottleneck search ─────────────────────────────────────────

class BottleneckSearchRequest(BaseModel):
    keyword: str = Field(..., min_length=1)
    limit: int = Field(default=20, ge=1, le=100)


@router.post("/search/bottlenecks")
async def bottleneck_search(
    data: BottleneckSearchRequest,
    session: AsyncSession = Depends(get_session),
):
    """Search bottlenecks by keyword and return linked DeltaCards."""
    results = await search_service.bottleneck_search(
        session,
        keyword=data.keyword,
        limit=data.limit,
    )
    return results


# ── Mechanism search ──────────────────────────────────────────

class MechanismSearchRequest(BaseModel):
    name: str = Field(..., min_length=1)
    limit: int = Field(default=20, ge=1, le=100)


@router.post("/search/mechanisms")
async def mechanism_search(
    data: MechanismSearchRequest,
    session: AsyncSession = Depends(get_session),
):
    """Resolve a mechanism name via entity resolution and return linked ideas."""
    results = await search_service.mechanism_search(
        session,
        name=data.name,
        limit=data.limit,
    )
    return results


# ── Transfer search ───────────────────────────────────────────

class TransferSearchRequest(BaseModel):
    source_domain: str | None = None
    target_domain: str | None = None
    limit: int = Field(default=20, ge=1, le=100)


@router.post("/search/transfers")
async def transfer_search(
    data: TransferSearchRequest,
    session: AsyncSession = Depends(get_session),
):
    """Find transferable_to assertions across domains."""
    results = await search_service.transfer_search(
        session,
        source_domain=data.source_domain,
        target_domain=data.target_domain,
        limit=data.limit,
    )
    return results


# ── Embeddings ──────────────────────────────────────────────────

@router.post("/embeddings/generate")
async def generate_embeddings(
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    """Generate embeddings for papers that don't have one yet."""
    count = await embedding_service.embed_batch(session, limit=limit)
    await session.commit()
    return {"embedded": count}


# ── Reading plan ────────────────────────────────────────────────

@router.post("/reading-plan")
async def create_reading_plan(
    category: str | None = Query(default=None),
    topic: str | None = Query(default=None),
    max_papers: int = Query(default=15, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
):
    """Generate a tiered reading plan.

    Tiers: canonical baselines → structural improvements →
    strong follow-ups → patches & boundary.
    Each paper has recommended reading depth (30s / 5min / 20min).
    """
    plan = await reading_planner.generate_reading_plan(
        session,
        category=category,
        topic=topic,
        max_papers=max_papers,
    )
    return plan


# ── Intent-based query router ────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    intent: str | None = Field(
        default=None,
        description="Force intent: bottleneck | mechanism | lineage | evidence. Auto-detected if omitted.",
    )
    constraints: dict | None = Field(
        default=None,
        description="Negative/positive constraints: min_structurality_score, must_have_open_code, "
                    "must_have_evidence_count, must_not_method_categories, exclude_tags, year_min, year_max",
    )
    limit: int = Field(default=20, ge=1, le=100)


@router.post("/search/query")
async def intent_query(
    data: QueryRequest,
    session: AsyncSession = Depends(get_session),
):
    """Intent-based query router — auto-routes to the best retrieval path.

    Four intents:
    - **bottleneck**: What's blocking progress? (project focus → paper claims → delta cards)
    - **mechanism**: What approaches exist? (canonical ideas → mechanism families → contributions)
    - **lineage**: How did this method evolve? (lineage DAG → ancestors/descendants)
    - **evidence**: Where's the proof/code? (evidence units → implementations → delta cards)

    Supports negative constraints: `must_not_method_categories`, `min_structurality_score`,
    `must_have_open_code`, `must_have_evidence_count`, `exclude_tags`.
    """
    return await query_router_service.route_query(
        session,
        query=data.query,
        intent=data.intent,
        constraints=data.constraints,
        limit=data.limit,
    )


# ── Materialized view refresh ────────────────────────────────────

@router.post("/search/refresh-views")
async def refresh_materialized_views(
    session: AsyncSession = Depends(get_session),
):
    """Refresh all CQRS-lite materialized views (paper_search_docs, idea_search_docs, lineage_view, review_queue_view)."""
    from sqlalchemy import text
    views = ["paper_search_docs", "idea_search_docs", "lineage_view", "review_queue_view"]
    refreshed = []
    for view in views:
        try:
            await session.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}"))
            refreshed.append(view)
        except Exception as e:
            # Fallback to non-concurrent refresh if no unique index or first time
            try:
                await session.execute(text(f"REFRESH MATERIALIZED VIEW {view}"))
                refreshed.append(view)
            except Exception as e2:
                refreshed.append(f"{view}: FAILED ({str(e2)[:80]})")
    await session.commit()
    return {"refreshed": refreshed}
