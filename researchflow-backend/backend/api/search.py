"""Search + embedding + reading plan API router."""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import embedding_service, reading_planner, search_service

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
    """Idea-centric search across DeltaCards and IdeaDeltas.

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
