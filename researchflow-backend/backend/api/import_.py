"""Import API router — ingest papers from various sources."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.schemas.import_ import (
    ImportResponse,
    LinkImportRequest,
)
from backend.schemas.paper import PaperResponse
from backend.services import ingestion_service

router = APIRouter(prefix="/import", tags=["import"])


@router.post("/links", response_model=ImportResponse)
async def import_links(
    data: LinkImportRequest,
    session: AsyncSession = Depends(get_session),
):
    """Import paper links in batch.

    Each link is normalized, deduped, and ingested. Duplicates are
    reported but not created again.
    """
    results = await ingestion_service.ingest_links(
        session,
        data.items,
        default_category=data.default_category,
        is_ephemeral=data.is_ephemeral,
        retention_days=data.retention_days,
    )
    await session.commit()

    created = sum(1 for r in results if r.status == "created")
    duplicates = sum(1 for r in results if r.status == "duplicate")
    errors = sum(1 for r in results if r.status == "error")

    return ImportResponse(
        total=len(results),
        created=created,
        duplicates=duplicates,
        errors=errors,
        items=results,
    )


@router.post("/{paper_id}/accept", response_model=PaperResponse)
async def accept_to_kb(
    paper_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Promote an ephemeral paper to the main knowledge base."""
    paper = await ingestion_service.accept_to_kb(session, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    await session.commit()
    return PaperResponse.model_validate(paper)


@router.post("/cleanup-expired")
async def cleanup_expired(
    session: AsyncSession = Depends(get_session),
):
    """Archive expired ephemeral papers."""
    count = await ingestion_service.cleanup_expired(session)
    await session.commit()
    return {"archived": count}
