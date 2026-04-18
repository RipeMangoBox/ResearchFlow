"""Review queue + human override + candidate management API."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import review_service

router = APIRouter(prefix="/reviews", tags=["reviews"])


# ── Review queue ─────────────────────────────────────────────────

@router.get("")
async def list_reviews(
    status: str | None = Query(default="pending"),
    target_type: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """List review tasks with filtering."""
    return await review_service.list_reviews(session, status, target_type, limit, offset)


@router.get("/stats")
async def queue_stats(session: AsyncSession = Depends(get_session)):
    """Get review queue statistics by status and target type."""
    return await review_service.queue_stats(session)


@router.get("/{task_id}")
async def get_review(
    task_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Get a single review task with its target object details."""
    task = await review_service.get_review(session, task_id)
    if not task:
        raise HTTPException(404, "Review task not found")
    detail = await review_service.get_review_detail(session, task)
    return detail


class ReviewActionRequest(BaseModel):
    reviewer: str = Field(..., min_length=1)
    notes: str | None = None


@router.post("/{task_id}/approve")
async def approve_review(
    task_id: UUID,
    data: ReviewActionRequest,
    session: AsyncSession = Depends(get_session),
):
    """Approve a review task — cascades to publish/verify the target object."""
    try:
        result = await review_service.approve_review(session, task_id, data.reviewer, data.notes)
        if result.get("status") == "not_found":
            raise HTTPException(404, "Review task not found")
        await session.commit()
        return result
    except HTTPException:
        raise
    except Exception:
        await session.rollback()
        raise


class RejectRequest(BaseModel):
    reviewer: str = Field(..., min_length=1)
    reason: str | None = None


@router.post("/{task_id}/reject")
async def reject_review(
    task_id: UUID,
    data: RejectRequest,
    session: AsyncSession = Depends(get_session),
):
    """Reject a review task — cascades rejection to the target object."""
    try:
        result = await review_service.reject_review(session, task_id, data.reviewer, data.reason)
        if result.get("status") == "not_found":
            raise HTTPException(404, "Review task not found")
        await session.commit()
        return result
    except HTTPException:
        raise
    except Exception:
        await session.rollback()
        raise


class AssignRequest(BaseModel):
    assigned_to: str = Field(..., min_length=1)


@router.post("/{task_id}/assign")
async def assign_review(
    task_id: UUID,
    data: AssignRequest,
    session: AsyncSession = Depends(get_session),
):
    """Assign a review task to a reviewer."""
    task = await review_service.assign_review(session, task_id, data.assigned_to)
    if not task:
        raise HTTPException(404, "Review task not found")
    await session.commit()
    return {"task_id": str(task_id), "assigned_to": data.assigned_to, "status": "in_progress"}


# ── Human overrides ──────────────────────────────────────────────

@router.get("/overrides")
async def list_overrides(
    target_type: str | None = None,
    target_id: UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    """List human overrides, optionally filtered by target."""
    return await review_service.list_overrides(session, target_type, target_id, limit)


class OverrideRequest(BaseModel):
    target_type: str = Field(..., description="delta_card / assertion / idea_delta / paper")
    target_id: UUID
    field_name: str
    new_value: dict | str | float | int | bool | list | None
    reason: str | None = None
    overridden_by: str | None = None


@router.post("/override")
async def create_override(
    data: OverrideRequest,
    session: AsyncSession = Depends(get_session),
):
    """Apply a human override — records the change AND applies it to the target object."""
    try:
        result = await review_service.apply_override(
            session,
            target_type=data.target_type,
            target_id=data.target_id,
            field_name=data.field_name,
            new_value=data.new_value,
            reason=data.reason,
            overridden_by=data.overridden_by,
        )
        await session.commit()
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        await session.rollback()
        raise


# ── Paradigm candidates ──────────────────────────────────────────

@router.get("/candidates/paradigms")
async def list_paradigm_candidates(
    status: str = Query(default="pending"),
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
):
    """List paradigm candidates pending review."""
    return await review_service.list_paradigm_candidates(session, status, limit)


class PromoteParadigmRequest(BaseModel):
    reviewer: str = Field(..., min_length=1)
    name_override: str | None = None


@router.post("/candidates/paradigms/{candidate_id}/promote")
async def promote_paradigm_candidate(
    candidate_id: UUID,
    data: PromoteParadigmRequest,
    session: AsyncSession = Depends(get_session),
):
    """Promote a paradigm candidate to a live ParadigmTemplate."""
    try:
        result = await review_service.promote_paradigm_candidate(
            session, candidate_id, data.reviewer, data.name_override,
        )
        if not result:
            raise HTTPException(404, "Candidate not found")
        await session.commit()
        return result
    except HTTPException:
        raise
    except Exception:
        await session.rollback()
        raise


@router.post("/candidates/paradigms/{candidate_id}/reject")
async def reject_paradigm_candidate(
    candidate_id: UUID,
    data: RejectRequest,
    session: AsyncSession = Depends(get_session),
):
    """Reject a paradigm candidate."""
    try:
        result = await review_service.reject_paradigm_candidate(
            session, candidate_id, data.reviewer, data.reason,
        )
        if not result:
            raise HTTPException(404, "Candidate not found")
        await session.commit()
        return result
    except HTTPException:
        raise
    except Exception:
        await session.rollback()
        raise


# ── Lineage review ───────────────────────────────────────────────

@router.get("/candidates/lineage")
async def list_lineage_candidates(
    status: str = Query(default="candidate"),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    """List lineage edges pending review."""
    return await review_service.list_lineage_candidates(session, status, limit)
