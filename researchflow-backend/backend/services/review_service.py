"""Review service — audit queue management + human overrides + candidate promotion.

Manages ReviewTask lifecycle:
  - Create review tasks (auto or manual)
  - List/filter pending reviews by type, priority
  - Assign to reviewer
  - Approve/reject with cascading effects on target objects
  - Apply human overrides (record + apply change)
  - Promote/reject paradigm, slot, mechanism candidates
  - List lineage candidates for review
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


# ── Review detail (with target object) ───────────────────────────

async def get_review_detail(session: AsyncSession, task: ReviewTask) -> dict:
    """Get review task with its target object summary."""
    detail = _task_to_dict(task)
    target = None

    if task.target_type == "assertion":
        obj = await session.get(GraphAssertion, task.target_id)
        if obj:
            target = {
                "edge_type": obj.edge_type,
                "status": obj.status,
                "confidence": obj.confidence,
                "assertion_source": obj.assertion_source,
            }
    elif task.target_type == "delta_card":
        obj = await session.get(DeltaCard, task.target_id)
        if obj:
            from backend.models.paper import Paper
            paper = await session.get(Paper, obj.paper_id)
            target = {
                "paper_title": paper.title if paper else "Unknown",
                "delta_statement": (obj.delta_statement or "")[:200],
                "status": obj.status,
                "structurality_score": obj.structurality_score,
            }
    elif task.target_type == "idea_delta":
        obj = await session.get(IdeaDelta, task.target_id)
        if obj:
            target = {
                "delta_statement": (obj.delta_statement or "")[:200],
                "publish_status": obj.publish_status,
                "confidence": obj.confidence,
                "evidence_count": obj.evidence_count,
            }
    elif task.target_type == "lineage":
        from backend.models.lineage import DeltaCardLineage
        obj = await session.get(DeltaCardLineage, task.target_id)
        if obj:
            target = {
                "relation_type": obj.relation_type,
                "confidence": obj.confidence,
                "status": obj.status,
            }
    elif task.target_type == "paradigm_candidate":
        from backend.models.candidates import ParadigmCandidate
        obj = await session.get(ParadigmCandidate, task.target_id)
        if obj:
            target = {
                "name": obj.name,
                "domain": obj.domain,
                "trigger_count": obj.trigger_count,
                "status": obj.status,
            }

    detail["target"] = target
    return detail


# ── Apply human overrides ────────────────────────────────────────

# Map of target_type → (model_class, allowed_fields)
_OVERRIDE_TARGETS = {
    "delta_card": ("backend.models.delta_card", "DeltaCard", {
        "delta_statement", "structurality_score", "extensionability_score",
        "transferability_score", "status", "baseline_paradigm",
    }),
    "idea_delta": ("backend.models.graph", "IdeaDelta", {
        "delta_statement", "publish_status", "confidence",
        "structurality_score", "transferability_score",
    }),
    "assertion": ("backend.models.assertion", "GraphAssertion", {
        "status", "confidence", "edge_type", "assertion_source",
    }),
    "paper": ("backend.models.paper", "Paper", {
        "importance", "category", "venue", "year", "method_family",
        "tags", "role_in_kb",
    }),
}


async def apply_override(
    session: AsyncSession,
    target_type: str,
    target_id: UUID,
    field_name: str,
    new_value,
    reason: str | None = None,
    overridden_by: str | None = None,
) -> dict:
    """Record a human override AND apply it to the target object."""
    if target_type not in _OVERRIDE_TARGETS:
        raise ValueError(f"Unknown target_type: {target_type}")

    module_path, class_name, allowed_fields = _OVERRIDE_TARGETS[target_type]
    if field_name not in allowed_fields:
        raise ValueError(f"Field '{field_name}' not overridable on {target_type}. Allowed: {allowed_fields}")

    # Dynamic import
    import importlib
    mod = importlib.import_module(module_path)
    model_class = getattr(mod, class_name)

    obj = await session.get(model_class, target_id)
    if not obj:
        raise ValueError(f"{target_type} {target_id} not found")

    # Capture old value
    old_value = getattr(obj, field_name, None)

    # Apply the change
    setattr(obj, field_name, new_value)

    # Record the override
    override = await create_override(
        session,
        target_type=target_type,
        target_id=target_id,
        field_name=field_name,
        old_value=old_value,
        new_value=new_value,
        reason=reason,
        overridden_by=overridden_by,
    )

    await session.flush()
    return {
        "override_id": str(override.id),
        "target_type": target_type,
        "target_id": str(target_id),
        "field_name": field_name,
        "old_value": old_value if not isinstance(old_value, (dict, list)) else old_value,
        "new_value": new_value,
        "applied": True,
    }


# ── Paradigm candidate management ───────────────────────────────

async def list_paradigm_candidates(
    session: AsyncSession,
    status: str = "pending",
    limit: int = 20,
) -> list[dict]:
    from backend.models.candidates import ParadigmCandidate
    result = await session.execute(
        select(ParadigmCandidate)
        .where(ParadigmCandidate.status == status)
        .order_by(desc(ParadigmCandidate.trigger_count))
        .limit(limit)
    )
    return [
        {
            "id": str(c.id),
            "name": c.name,
            "domain": c.domain,
            "description": c.description,
            "slots_json": c.slots_json,
            "trigger_count": c.trigger_count,
            "status": c.status,
            "max_similarity_to_existing": c.max_similarity_to_existing,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in result.scalars()
    ]


async def promote_paradigm_candidate(
    session: AsyncSession,
    candidate_id: UUID,
    reviewer: str,
    name_override: str | None = None,
) -> dict | None:
    """Promote a paradigm candidate to a live ParadigmTemplate + Slots."""
    from backend.models.candidates import ParadigmCandidate
    from backend.models.analysis import ParadigmTemplate
    from backend.models.graph import Slot

    cand = await session.get(ParadigmCandidate, candidate_id)
    if not cand:
        return None

    paradigm_name = name_override or cand.name

    # Check no duplicate live paradigm
    existing = await session.execute(
        select(ParadigmTemplate).where(ParadigmTemplate.name == paradigm_name).limit(1)
    )
    if existing.scalar_one_or_none():
        raise ValueError(f"Paradigm '{paradigm_name}' already exists")

    # Create live paradigm
    slots_dict = {}
    raw_slots = cand.slots_json or []
    if isinstance(raw_slots, list):
        slots_dict = {s["name"]: s.get("description", "") for s in raw_slots if isinstance(s, dict)}

    paradigm = ParadigmTemplate(
        name=paradigm_name,
        version="v1",
        domain=cand.domain,
        slots=slots_dict,
    )
    session.add(paradigm)
    await session.flush()

    # Create slot records
    slot_count = 0
    for i, s in enumerate(raw_slots if isinstance(raw_slots, list) else []):
        if not isinstance(s, dict) or not s.get("name"):
            continue
        slot = Slot(
            paradigm_id=paradigm.id,
            name=s["name"],
            description=s.get("description"),
            slot_type=s.get("slot_type", "architecture"),
            is_required=s.get("is_required", True),
            sort_order=i,
        )
        session.add(slot)
        slot_count += 1

    # Update candidate status
    cand.status = "approved"
    cand.promoted_paradigm_id = paradigm.id
    cand.reviewed_by = reviewer
    cand.reviewed_at = datetime.now(timezone.utc)

    # Record taxonomy version
    from backend.models.review import TaxonomyVersion
    tv = TaxonomyVersion(
        entity_type="paradigm_template",
        entity_id=paradigm.id,
        action="created_from_candidate",
        version_label=f"{paradigm_name}_v1",
        changed_by=reviewer,
        change_summary=f"Promoted from candidate (trigger_count={cand.trigger_count})",
    )
    session.add(tv)

    await session.flush()
    return {
        "paradigm_id": str(paradigm.id),
        "name": paradigm_name,
        "domain": cand.domain,
        "slots_created": slot_count,
        "candidate_trigger_count": cand.trigger_count,
    }


async def reject_paradigm_candidate(
    session: AsyncSession,
    candidate_id: UUID,
    reviewer: str,
    reason: str | None = None,
) -> dict | None:
    from backend.models.candidates import ParadigmCandidate
    cand = await session.get(ParadigmCandidate, candidate_id)
    if not cand:
        return None

    cand.status = "rejected"
    cand.reviewed_by = reviewer
    cand.reviewed_at = datetime.now(timezone.utc)
    cand.review_notes = reason

    await session.flush()
    return {"candidate_id": str(candidate_id), "status": "rejected"}


# ── Lineage candidates ──────────────────────────────────────────

async def list_lineage_candidates(
    session: AsyncSession,
    status: str = "candidate",
    limit: int = 50,
) -> list[dict]:
    from backend.models.lineage import DeltaCardLineage
    from backend.models.delta_card import DeltaCard
    from backend.models.paper import Paper
    from sqlalchemy.orm import aliased

    child_dc = aliased(DeltaCard, name="child_dc")
    parent_dc = aliased(DeltaCard, name="parent_dc")
    child_paper = aliased(Paper, name="child_paper")
    parent_paper = aliased(Paper, name="parent_paper")

    result = await session.execute(
        select(
            DeltaCardLineage,
            child_paper.title.label("child_title"),
            parent_paper.title.label("parent_title"),
        )
        .join(child_dc, DeltaCardLineage.child_delta_card_id == child_dc.id)
        .join(child_paper, child_dc.paper_id == child_paper.id)
        .join(parent_dc, DeltaCardLineage.parent_delta_card_id == parent_dc.id)
        .join(parent_paper, parent_dc.paper_id == parent_paper.id)
        .where(DeltaCardLineage.status == status)
        .order_by(DeltaCardLineage.created_at)
        .limit(limit)
    )
    return [
        {
            "id": str(ln.id),
            "child_title": (ct or "Unknown")[:80],
            "parent_title": (pt or "Unknown")[:80],
            "relation_type": ln.relation_type,
            "confidence": ln.confidence,
            "status": ln.status,
            "created_at": ln.created_at.isoformat() if ln.created_at else None,
        }
        for ln, ct, pt in result
    ]
