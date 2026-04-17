"""Review service — audit queue management.

Manages ReviewTask lifecycle:
  - Create review tasks (auto or manual)
  - List/filter pending reviews by type, priority
  - Assign to reviewer
  - Approve/reject with cascading effects on target objects
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.assertion import GraphAssertion
from backend.models.delta_card import DeltaCard
from backend.models.graph import IdeaDelta
from backend.models.review import HumanOverride, ReviewTask

logger = logging.getLogger(__name__)


# ── Review task CRUD ──────────────────────────────────────────────

async def create_review_task(
    session: AsyncSession,
    target_type: str,
    target_id: UUID,
    task_type: str = "human_review",
    priority: int = 3,
    notes: str | None = None,
) -> ReviewTask:
    """Create a new review task."""
    task = ReviewTask(
        target_type=target_type,
        target_id=target_id,
        task_type=task_type,
        priority=priority,
        notes=notes,
    )
    session.add(task)
    await session.flush()
    await session.refresh(task)
    return task


async def list_reviews(
    session: AsyncSession,
    status: str | None = "pending",
    target_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List review tasks with filtering."""
    stmt = select(ReviewTask)
    if status:
        stmt = stmt.where(ReviewTask.status == status)
    if target_type:
        stmt = stmt.where(ReviewTask.target_type == target_type)
    stmt = stmt.order_by(ReviewTask.priority.asc(), ReviewTask.created_at.asc())

    # Count total
    count_stmt = select(func.count()).select_from(ReviewTask)
    if status:
        count_stmt = count_stmt.where(ReviewTask.status == status)
    if target_type:
        count_stmt = count_stmt.where(ReviewTask.target_type == target_type)
    total = (await session.execute(count_stmt)).scalar() or 0

    result = await session.execute(stmt.offset(offset).limit(limit))
    tasks = list(result.scalars().all())

    return {
        "total": total,
        "tasks": [_task_to_dict(t) for t in tasks],
    }


async def get_review(session: AsyncSession, task_id: UUID) -> ReviewTask | None:
    return await session.get(ReviewTask, task_id)


async def assign_review(
    session: AsyncSession,
    task_id: UUID,
    assigned_to: str,
) -> ReviewTask | None:
    """Assign a review task to a reviewer."""
    task = await session.get(ReviewTask, task_id)
    if not task:
        return None
    task.assigned_to = assigned_to
    task.status = "in_progress"
    await session.flush()
    return task


async def approve_review(
    session: AsyncSession,
    task_id: UUID,
    reviewer: str,
    notes: str | None = None,
) -> dict:
    """Approve a review task, cascading to the target object.

    Effects:
    - assertion → publish assertion
    - delta_card → publish delta_card
    - idea_delta → set human_verified
    """
    task = await session.get(ReviewTask, task_id)
    if not task:
        return {"status": "not_found"}

    task.status = "approved"
    task.completed_at = datetime.now(timezone.utc)
    if notes:
        task.notes = (task.notes or "") + f"\n[approved] {notes}"

    # Cascade to target
    cascade_result = await _cascade_approval(session, task.target_type, task.target_id, reviewer)

    await session.flush()
    return {
        "task_id": str(task_id),
        "status": "approved",
        "target_type": task.target_type,
        "target_id": str(task.target_id),
        "cascade": cascade_result,
    }


async def reject_review(
    session: AsyncSession,
    task_id: UUID,
    reviewer: str,
    reason: str | None = None,
) -> dict:
    """Reject a review task, cascading rejection to target."""
    task = await session.get(ReviewTask, task_id)
    if not task:
        return {"status": "not_found"}

    task.status = "rejected"
    task.completed_at = datetime.now(timezone.utc)
    if reason:
        task.notes = (task.notes or "") + f"\n[rejected] {reason}"

    cascade_result = await _cascade_rejection(session, task.target_type, task.target_id, reason)

    await session.flush()
    return {
        "task_id": str(task_id),
        "status": "rejected",
        "target_type": task.target_type,
        "target_id": str(task.target_id),
        "cascade": cascade_result,
    }


# ── Human overrides ──────────────────────────────────────────────

async def create_override(
    session: AsyncSession,
    target_type: str,
    target_id: UUID,
    field_name: str,
    old_value,
    new_value,
    reason: str | None = None,
    overridden_by: str | None = None,
) -> HumanOverride:
    """Record a human override on any entity field."""
    override = HumanOverride(
        target_type=target_type,
        target_id=target_id,
        field_name=field_name,
        old_value={"value": old_value} if not isinstance(old_value, dict) else old_value,
        new_value={"value": new_value} if not isinstance(new_value, dict) else new_value,
        reason=reason,
        overridden_by=overridden_by,
    )
    session.add(override)
    await session.flush()
    return override


async def list_overrides(
    session: AsyncSession,
    target_type: str | None = None,
    target_id: UUID | None = None,
    limit: int = 50,
) -> list[dict]:
    """List human overrides, optionally filtered."""
    stmt = select(HumanOverride)
    if target_type:
        stmt = stmt.where(HumanOverride.target_type == target_type)
    if target_id:
        stmt = stmt.where(HumanOverride.target_id == target_id)
    stmt = stmt.order_by(desc(HumanOverride.created_at)).limit(limit)

    result = await session.execute(stmt)
    return [
        {
            "id": str(o.id),
            "target_type": o.target_type,
            "target_id": str(o.target_id),
            "field_name": o.field_name,
            "old_value": o.old_value,
            "new_value": o.new_value,
            "reason": o.reason,
            "overridden_by": o.overridden_by,
            "created_at": o.created_at.isoformat() if o.created_at else None,
        }
        for o in result.scalars()
    ]


# ── Queue stats ───────────────────────────────────────────────────

async def queue_stats(session: AsyncSession) -> dict:
    """Get review queue statistics."""
    result = await session.execute(
        select(ReviewTask.status, func.count())
        .group_by(ReviewTask.status)
    )
    by_status = {row[0]: row[1] for row in result}

    type_result = await session.execute(
        select(ReviewTask.target_type, func.count())
        .where(ReviewTask.status.in_(["pending", "in_progress"]))
        .group_by(ReviewTask.target_type)
    )
    by_type = {row[0]: row[1] for row in type_result}

    return {
        "by_status": by_status,
        "by_target_type": by_type,
        "total_pending": by_status.get("pending", 0) + by_status.get("in_progress", 0),
    }


# ── Cascade helpers ───────────────────────────────────────────────

async def _cascade_approval(
    session: AsyncSession,
    target_type: str,
    target_id: UUID,
    reviewer: str,
) -> dict:
    """Cascade approval to target object."""
    now = datetime.now(timezone.utc)

    if target_type == "assertion":
        assertion = await session.get(GraphAssertion, target_id)
        if assertion:
            assertion.status = "published"
            assertion.reviewed_by = reviewer
            assertion.reviewed_at = now
            assertion.assertion_source = "human_verified"
            return {"assertion_status": "published"}

    elif target_type == "delta_card":
        card = await session.get(DeltaCard, target_id)
        if card:
            card.status = "published"
            return {"delta_card_status": "published"}

    elif target_type == "idea_delta":
        idea = await session.get(IdeaDelta, target_id)
        if idea:
            idea.publish_status = "human_verified"
            return {"idea_status": "human_verified"}

    return {"status": "no_cascade"}


async def _cascade_rejection(
    session: AsyncSession,
    target_type: str,
    target_id: UUID,
    reason: str | None,
) -> dict:
    """Cascade rejection to target object."""
    if target_type == "assertion":
        assertion = await session.get(GraphAssertion, target_id)
        if assertion:
            assertion.status = "rejected"
            if reason:
                assertion.metadata_ = {**(assertion.metadata_ or {}), "rejection_reason": reason}
            return {"assertion_status": "rejected"}

    elif target_type == "delta_card":
        card = await session.get(DeltaCard, target_id)
        if card:
            card.status = "deprecated"
            return {"delta_card_status": "deprecated"}

    return {"status": "no_cascade"}


def _task_to_dict(task: ReviewTask) -> dict:
    return {
        "id": str(task.id),
        "target_type": task.target_type,
        "target_id": str(task.target_id),
        "task_type": task.task_type,
        "status": task.status,
        "priority": task.priority,
        "assigned_to": task.assigned_to,
        "notes": task.notes,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }
