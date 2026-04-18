"""Papers API router — CRUD + list + filter + search."""

import math
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.models.enums import Importance, PaperState, Tier
from backend.schemas.paper import (
    AnalysisBrief,
    PaperBrief,
    PaperCreate,
    PaperDetail,
    PaperFilter,
    PaperListResponse,
    PaperResponse,
    PaperUpdate,
)
from backend.services import enrich_service, paper_service, triage_service

router = APIRouter(prefix="/papers", tags=["papers"])

Session = Annotated[AsyncSession, Depends(get_session)]


@router.get("", response_model=PaperListResponse)
async def list_papers(
    session: Session,
    q: str | None = None,
    state: PaperState | None = None,
    category: str | None = None,
    venue: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    importance: Importance | None = None,
    tier: Tier | None = None,
    tags: Annotated[list[str] | None, Query()] = None,
    is_ephemeral: bool | None = None,
    min_keep_score: float | None = None,
    min_structurality: float | None = None,
    min_extensionability: float | None = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    sort_by: str = "updated_at",
    sort_order: str = "desc",
):
    filters = PaperFilter(
        q=q, state=state, category=category, venue=venue,
        year_min=year_min, year_max=year_max, importance=importance,
        tier=tier, tags=tags, is_ephemeral=is_ephemeral,
        min_keep_score=min_keep_score, min_structurality=min_structurality,
        min_extensionability=min_extensionability,
        page=page, size=size, sort_by=sort_by, sort_order=sort_order,
    )
    papers, total = await paper_service.list_papers(session, filters)
    return PaperListResponse(
        items=[PaperBrief.model_validate(p) for p in papers],
        total=total,
        page=page,
        size=size,
        pages=math.ceil(total / size) if total > 0 else 0,
    )


@router.get("/{paper_id}", response_model=PaperDetail)
async def get_paper(session: Session, paper_id: UUID):
    paper, analysis = await paper_service.get_paper_with_analysis(session, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    result = PaperDetail.model_validate(paper)
    if analysis:
        result.latest_analysis = AnalysisBrief.model_validate(analysis)
    return result


@router.post("", response_model=PaperResponse, status_code=201)
async def create_paper(session: Session, data: PaperCreate):
    paper = await paper_service.create_paper(session, data)
    await session.commit()
    return PaperResponse.model_validate(paper)


@router.patch("/{paper_id}", response_model=PaperResponse)
async def update_paper(session: Session, paper_id: UUID, data: PaperUpdate):
    paper = await paper_service.update_paper(session, paper_id, data)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    await session.commit()
    return PaperResponse.model_validate(paper)


@router.delete("/{paper_id}", status_code=204)
async def delete_paper(session: Session, paper_id: UUID):
    deleted = await paper_service.delete_paper(session, paper_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Paper not found")
    await session.commit()


@router.post("/{paper_id}/triage", response_model=PaperResponse)
async def triage_paper(session: Session, paper_id: UUID):
    """Compute 4-dimension scores for a single paper."""
    paper = await triage_service.triage_paper(session, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    await session.commit()
    return PaperResponse.model_validate(paper)


@router.post("/triage-all")
async def triage_all(session: Session):
    """Score all unscored papers in the database."""
    count = await triage_service.triage_all_unscored(session)
    await session.commit()
    return {"scored": count}


@router.post("/enrich")
async def enrich_papers(
    session: Session,
    limit: int = Query(default=10, ge=1, le=50),
):
    """Enrich papers missing metadata (abstract, authors, doi) from arXiv/Crossref."""
    results = await enrich_service.enrich_batch(session, limit=limit)
    await session.commit()
    return {"processed": len(results), "results": results}


@router.post("/search", response_model=list[PaperBrief])
async def search_papers(
    session: Session,
    q: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=100),
):
    papers = await paper_service.search_papers(session, q, limit)
    return [PaperBrief.model_validate(p) for p in papers]


@router.get("/{paper_id}/download-pdf")
async def download_paper_pdf(
    paper_id: UUID,
    session: Session,
):
    """Download a paper's PDF. Checks object storage first, then local path."""
    from fastapi.responses import FileResponse, Response
    from backend.models.paper import Paper

    paper = await session.get(Paper, paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found")

    # Try object storage first
    if paper.pdf_object_key:
        from backend.services.object_storage import get_storage
        storage = get_storage()
        data = await storage.get(paper.pdf_object_key)
        if data:
            filename = f"{paper.title_sanitized or str(paper.id)}.pdf"
            return Response(
                content=data,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

    # Fallback to local path
    if paper.pdf_path_local:
        from pathlib import Path
        from backend.config import settings
        local_path = Path(settings.paper_pdfs_dir) / paper.pdf_path_local
        if local_path.exists():
            return FileResponse(
                str(local_path),
                media_type="application/pdf",
                filename=f"{paper.title_sanitized or str(paper.id)}.pdf",
            )

    raise HTTPException(404, "PDF not available")
