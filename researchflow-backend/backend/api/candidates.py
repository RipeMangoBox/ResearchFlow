# NOTE: Register this router in main.py:
#   from backend.api.candidates import router as candidates_router
#   app.include_router(candidates_router, prefix="/api")

"""Candidates API router — candidate pool CRUD, scoring, promotion."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import candidate_service

router = APIRouter(prefix="/candidates", tags=["candidates"])

Session = Annotated[AsyncSession, Depends(get_session)]


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------

class CandidateBrief(BaseModel):
    id: UUID
    title: str
    arxiv_id: str | None = None
    doi: str | None = None
    venue: str | None = None
    year: int | None = None
    status: str
    absorption_level: int
    discovery_source: str
    discovery_reason: str | None = None
    relation_hint: str | None = None
    citation_count: int | None = None
    code_url: str | None = None

    model_config = {"from_attributes": True}


class CandidateDetail(CandidateBrief):
    abstract: str | None = None
    authors_json: dict | None = None
    paper_link: str | None = None
    metadata_json: dict | None = None
    discovered_from_paper_id: UUID | None = None
    discovered_from_domain_id: UUID | None = None
    ingested_paper_id: UUID | None = None
    reject_reason: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ScoreResponse(BaseModel):
    id: UUID
    candidate_id: UUID
    discovery_score: float | None = None
    discovery_breakdown: dict | None = None
    decision: str | None = None
    decision_reason: str | None = None
    score_version: int = 1

    model_config = {"from_attributes": True}


class PaperBrief(BaseModel):
    id: UUID
    title: str
    venue: str | None = None
    year: int | None = None
    state: str
    category: str

    model_config = {"from_attributes": True}


class DiscoverResponse(BaseModel):
    candidates_created: int
    candidates_existing: int
    total_discovered: int


class ScoreBatchRequest(BaseModel):
    limit: int = Field(default=50, ge=1, le=200)
    domain_id: UUID | None = None


class ScoreBatchResponse(BaseModel):
    scored_count: int


class PromoteRequest(BaseModel):
    absorption_level: int = Field(default=1, ge=1, le=3)


class RejectRequest(BaseModel):
    reason: str


class AutoPromoteRequest(BaseModel):
    threshold: float = Field(default=75.0, ge=0, le=100)
    limit: int = Field(default=20, ge=1, le=100)
    domain_id: UUID | None = None


class AutoPromoteResponse(BaseModel):
    promoted_count: int
    papers: list[PaperBrief]


class ColdStartRequest(BaseModel):
    name: str
    display_name_zh: str | None = None
    scope: dict  # {modalities, tasks, paradigms, seed_methods, ...}


class StatsResponse(BaseModel):
    total: int
    by_status: dict[str, int]
    by_absorption_level: dict[int, int]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/discover/{paper_id}", response_model=DiscoverResponse)
async def discover_candidates(
    session: Session,
    paper_id: UUID,
    domain_id: UUID | None = None,
    max_references: int = Query(default=30, ge=1, le=100),
    max_citations: int = Query(default=50, ge=1, le=200),
    max_related: int = Query(default=10, ge=1, le=50),
):
    """Trigger neighborhood discovery for a paper.

    Discovers related papers from Semantic Scholar (references, citations,
    recommendations) and creates scored candidates for each.
    """
    from backend.services.ingest_workflow import IngestWorkflow

    workflow = IngestWorkflow(session)
    try:
        result = await workflow.discover_neighborhood(
            paper_id,
            max_references=max_references,
            max_citations=max_citations,
            max_related=max_related,
            domain_id=domain_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Discovery failed: {str(e)[:200]}",
        )

    if "error" in result and result.get("total_discovered", 0) == 0:
        raise HTTPException(status_code=404, detail=result["error"])

    await session.commit()

    return DiscoverResponse(
        candidates_created=result.get("candidates_created", 0),
        candidates_existing=result.get("candidates_existing", 0),
        total_discovered=result.get("total_discovered", 0),
    )


@router.get("", response_model=list[CandidateBrief])
async def list_candidates(
    session: Session,
    domain_id: UUID | None = None,
    status: str | None = None,
    min_score: float | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List candidates with optional filters, ordered by score desc."""
    candidates = await candidate_service.list_candidates(
        session,
        domain_id=domain_id,
        status=status,
        min_score=min_score,
        limit=limit,
        offset=offset,
    )
    return [CandidateBrief.model_validate(c) for c in candidates]


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    session: Session,
    domain_id: UUID | None = None,
):
    """Candidate pool statistics grouped by status and absorption level."""
    stats = await candidate_service.get_stats(session, domain_id=domain_id)
    return StatsResponse(**stats)


@router.get("/{candidate_id}", response_model=CandidateDetail)
async def get_candidate(session: Session, candidate_id: UUID):
    """Get full candidate detail including metadata."""
    candidate = await candidate_service.get_candidate(session, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return CandidateDetail.model_validate(candidate)


@router.post("/{candidate_id}/score", response_model=ScoreResponse)
async def score_candidate(session: Session, candidate_id: UUID):
    """Trigger scoring for a single candidate."""
    candidate = await candidate_service.get_candidate(session, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    score = await candidate_service.score_candidate(session, candidate_id)
    await session.commit()
    return ScoreResponse.model_validate(score)


@router.post("/score-batch", response_model=ScoreBatchResponse)
async def score_batch(session: Session, body: ScoreBatchRequest):
    """Batch-score unscored candidates."""
    count = await candidate_service.score_batch(
        session,
        limit=body.limit,
        domain_id=body.domain_id,
    )
    await session.commit()
    return ScoreBatchResponse(scored_count=count)


@router.post("/{candidate_id}/promote", response_model=PaperBrief)
async def promote_candidate(
    session: Session,
    candidate_id: UUID,
    body: PromoteRequest,
):
    """Promote a candidate to a full Paper in the knowledge base."""
    candidate = await candidate_service.get_candidate(session, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    paper = await candidate_service.promote_candidate(
        session, candidate_id, absorption_level=body.absorption_level,
    )
    await session.commit()
    return PaperBrief.model_validate(paper)


@router.post("/{candidate_id}/reject", response_model=CandidateBrief)
async def reject_candidate(
    session: Session,
    candidate_id: UUID,
    body: RejectRequest,
):
    """Reject a candidate with a reason."""
    candidate = await candidate_service.get_candidate(session, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    updated = await candidate_service.reject_candidate(
        session, candidate_id, reason=body.reason,
    )
    await session.commit()
    return CandidateBrief.model_validate(updated)


@router.post("/auto-promote", response_model=AutoPromoteResponse)
async def auto_promote(session: Session, body: AutoPromoteRequest):
    """Auto-promote candidates scoring above a threshold."""
    papers = await candidate_service.auto_promote_batch(
        session,
        threshold=body.threshold,
        limit=body.limit,
        domain_id=body.domain_id,
    )
    await session.commit()
    return AutoPromoteResponse(
        promoted_count=len(papers),
        papers=[PaperBrief.model_validate(p) for p in papers],
    )


@router.post("/domains/cold-start")
async def cold_start_domain_endpoint(body: ColdStartRequest, session: Session):
    """Bootstrap a new domain knowledge base from a manifest.

    Creates DomainSpec, skeleton nodes, harvests candidates from arXiv + S2,
    scores them, and auto-promotes top anchors. Deep ingest is deferred to workers.
    """
    from backend.services.cold_start_service import cold_start_domain as _cold_start

    result = await _cold_start(session, body.model_dump())
    await session.commit()
    return result
