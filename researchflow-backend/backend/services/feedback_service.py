"""Feedback service — record user corrections, bookmarks, events.

Tracks structured user interactions for:
1. Recording corrections/confirmations on papers/analyses
2. Managing bookmarks (papers, directions, reports)
3. Logging behavioral events for system improvement
"""

import logging
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.direction import UserBookmark, UserEvent
from backend.models.enums import FeedbackType
from backend.models.system import UserFeedback

logger = logging.getLogger(__name__)


# ── Feedback ────────────────────────────────────────────────────

async def record_feedback(
    session: AsyncSession,
    target_type: str,
    target_id: UUID,
    feedback_type: str,
    comment: str | None = None,
    old_value: dict | None = None,
    new_value: dict | None = None,
) -> UserFeedback:
    fb = UserFeedback(
        target_type=target_type,
        target_id=target_id,
        feedback_type=FeedbackType(feedback_type),
        comment=comment,
        old_value=old_value,
        new_value=new_value,
    )
    session.add(fb)
    await session.flush()
    await session.refresh(fb)
    return fb


async def list_feedback(
    session: AsyncSession,
    target_type: str | None = None,
    limit: int = 50,
) -> list[UserFeedback]:
    stmt = select(UserFeedback).order_by(UserFeedback.created_at.desc()).limit(limit)
    if target_type:
        stmt = stmt.where(UserFeedback.target_type == target_type)
    result = await session.execute(stmt)
    return list(result.scalars().all())


# ── Bookmarks ───────────────────────────────────────────────────

async def add_bookmark(
    session: AsyncSession,
    target_type: str,
    target_id: UUID,
    note: str | None = None,
) -> UserBookmark:
    # Check for existing
    existing = await session.execute(
        select(UserBookmark).where(
            and_(UserBookmark.target_type == target_type, UserBookmark.target_id == target_id)
        )
    )
    bm = existing.scalar_one_or_none()
    if bm:
        bm.note = note
        await session.flush()
        await session.refresh(bm)
        return bm

    bm = UserBookmark(target_type=target_type, target_id=target_id, note=note)
    session.add(bm)
    await session.flush()
    await session.refresh(bm)
    return bm


async def remove_bookmark(session: AsyncSession, bookmark_id: UUID) -> bool:
    bm = await session.get(UserBookmark, bookmark_id)
    if not bm:
        return False
    await session.delete(bm)
    await session.flush()
    return True


async def list_bookmarks(
    session: AsyncSession,
    target_type: str | None = None,
    limit: int = 50,
) -> list[UserBookmark]:
    stmt = select(UserBookmark).order_by(UserBookmark.created_at.desc()).limit(limit)
    if target_type:
        stmt = stmt.where(UserBookmark.target_type == target_type)
    result = await session.execute(stmt)
    return list(result.scalars().all())


# ── Events ──────────────────────────────────────────────────────

async def log_event(
    session: AsyncSession,
    event_type: str,
    target_type: str | None = None,
    target_id: UUID | None = None,
    payload: dict | None = None,
) -> UserEvent:
    ev = UserEvent(
        event_type=event_type,
        target_type=target_type,
        target_id=target_id,
        payload=payload,
    )
    session.add(ev)
    await session.flush()
    return ev
