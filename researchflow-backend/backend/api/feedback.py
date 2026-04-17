"""Feedback, bookmarks, events API router."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import feedback_service

router = APIRouter(tags=["feedback"])


# ── Feedback ────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    target_type: str  # paper, analysis, direction
    target_id: str
    feedback_type: str  # correction, confirmation, rejection, tag_edit
    comment: str | None = None
    old_value: dict | None = None
    new_value: dict | None = None


@router.post("/feedback")
async def record_feedback(
    data: FeedbackRequest,
    session: AsyncSession = Depends(get_session),
):
    fb = await feedback_service.record_feedback(
        session, data.target_type, UUID(data.target_id),
        data.feedback_type, data.comment, data.old_value, data.new_value,
    )
    await session.commit()
    return {"id": str(fb.id), "status": "recorded"}


@router.get("/feedback")
async def list_feedback(
    target_type: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    items = await feedback_service.list_feedback(session, target_type, limit)
    return [
        {
            "id": str(f.id),
            "target_type": f.target_type,
            "target_id": str(f.target_id),
            "feedback_type": f.feedback_type.value,
            "comment": f.comment,
            "created_at": str(f.created_at),
        }
        for f in items
    ]


# ── Bookmarks ───────────────────────────────────────────────────

class BookmarkRequest(BaseModel):
    target_type: str  # paper, direction, report
    target_id: str
    note: str | None = None


@router.post("/bookmarks")
async def add_bookmark(
    data: BookmarkRequest,
    session: AsyncSession = Depends(get_session),
):
    bm = await feedback_service.add_bookmark(
        session, data.target_type, UUID(data.target_id), data.note,
    )
    await session.commit()
    return {"id": str(bm.id), "target_type": bm.target_type, "target_id": str(bm.target_id)}


@router.delete("/bookmarks/{bookmark_id}")
async def remove_bookmark(
    bookmark_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    removed = await feedback_service.remove_bookmark(session, bookmark_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    await session.commit()
    return {"status": "removed"}


@router.get("/bookmarks")
async def list_bookmarks(
    target_type: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    items = await feedback_service.list_bookmarks(session, target_type, limit)
    return [
        {
            "id": str(b.id),
            "target_type": b.target_type,
            "target_id": str(b.target_id),
            "note": b.note,
            "created_at": str(b.created_at),
        }
        for b in items
    ]
