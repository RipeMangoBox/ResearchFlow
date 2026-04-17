"""Digests API router — daily/weekly/monthly summaries."""

from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import digest_service

router = APIRouter(prefix="/digests", tags=["digests"])


@router.post("/generate")
async def generate_digest(
    period_type: str = Query(..., pattern="^(day|week|month)$"),
    target_date: date | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
):
    """Generate a digest for the specified period.

    - day: today's new papers, top 3 to read, pending judgments
    - week: direction trends, structural vs patch, shifted branches
    - month: strategy review, bottleneck ranking, start/stop directions
    """
    digest = await digest_service.generate_digest(session, period_type, target_date)
    await session.commit()
    return {
        "id": str(digest.id),
        "period_type": digest.period_type.value,
        "period_start": str(digest.period_start),
        "period_end": str(digest.period_end),
        "content": digest.rendered_text,
        "metadata": digest.metadata_,
    }


@router.get("/latest")
async def get_latest_digests(
    session: AsyncSession = Depends(get_session),
):
    """Get the latest digest of each period type."""
    results = {}
    for pt in ("day", "week", "month"):
        digest = await digest_service.get_latest_digest(session, pt)
        if digest:
            results[pt] = {
                "id": str(digest.id),
                "period_start": str(digest.period_start),
                "period_end": str(digest.period_end),
                "content": digest.rendered_text[:500] + "..." if len(digest.rendered_text) > 500 else digest.rendered_text,
            }
        else:
            results[pt] = None
    return results
