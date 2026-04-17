"""Assertions, reviews, and overrides API router."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import assertion_service, review_service, entity_resolution_service

router = APIRouter(prefix="/assertions", tags=["assertions"])


# ── Pydantic models ───────────────────────────────────────────────

class ProposeAssertionRequest(BaseModel):
    from_node_id: UUID
    to_node_id: UUID
    edge_type: str
    assertion_source: str = "system_inferred"
    confidence: float | None = None
    metadata: dict | None = None
    evidence_unit_ids: list[UUID] | None = None


class ReviewDecisionRequest(BaseModel):
    reviewer: str
    notes: str | None = None
    reason: str | None = None


class OverrideRequest(BaseModel):
    target_type: str
    target_id: UUID
    field_name: str
    old_value: dict | None = None
    new_value: dict | None = None
    reason: str | None = None
    overridden_by: str | None = None


class AliasRequest(BaseModel):
    entity_type: str
    entity_id: UUID
    alias: str
    confidence: float | None = None


# ── Review endpoints (BEFORE /{assertion_id} to avoid conflict) ───

@router.get("/reviews/queue")
async def list_review_queue(
    status: str | None = Query(default="pending"),
    target_type: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """List review tasks."""
    return await review_service.list_reviews(session, status, target_type, limit, offset)


@router.get("/reviews/stats")
async def review_stats(session: AsyncSession = Depends(get_session)):
    """Get review queue statistics."""
    return await review_service.queue_stats(session)


@router.post("/reviews/{task_id}/assign")
async def assign_review(
    task_id: UUID,
    assigned_to: str = Query(...),
    session: AsyncSession = Depends(get_session),
):
    """Assign a review task."""
    task = await review_service.assign_review(session, task_id, assigned_to)
    if not task:
        raise HTTPException(404, "Review task not found")
    await session.commit()
    return {"id": str(task.id), "status": task.status, "assigned_to": task.assigned_to}


@router.post("/reviews/{task_id}/approve")
async def approve_review(
    task_id: UUID,
    req: ReviewDecisionRequest,
    session: AsyncSession = Depends(get_session),
):
    """Approve a review task (cascades to target)."""
    result = await review_service.approve_review(session, task_id, req.reviewer, req.notes)
    await session.commit()
    return result


@router.post("/reviews/{task_id}/reject")
async def reject_review(
    task_id: UUID,
    req: ReviewDecisionRequest,
    session: AsyncSession = Depends(get_session),
):
    """Reject a review task (cascades to target)."""
    result = await review_service.reject_review(session, task_id, req.reviewer, req.reason)
    await session.commit()
    return result


# ── Override endpoints (BEFORE /{assertion_id}) ───────────────────

@router.post("/overrides")
async def create_override(
    req: OverrideRequest,
    session: AsyncSession = Depends(get_session),
):
    """Record a human override."""
    override = await review_service.create_override(
        session,
        target_type=req.target_type,
        target_id=req.target_id,
        field_name=req.field_name,
        old_value=req.old_value,
        new_value=req.new_value,
        reason=req.reason,
        overridden_by=req.overridden_by,
    )
    await session.commit()
    return {"id": str(override.id)}


@router.get("/overrides")
async def list_overrides(
    target_type: str | None = None,
    target_id: UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    """List human overrides."""
    return await review_service.list_overrides(session, target_type, target_id, limit)


# ── Alias endpoints (BEFORE /{assertion_id}) ──────────────────────

@router.post("/aliases")
async def register_alias(
    req: AliasRequest,
    session: AsyncSession = Depends(get_session),
):
    """Register a new entity alias."""
    alias = await entity_resolution_service.register_alias(
        session, req.entity_type, req.entity_id, req.alias, "manual", req.confidence,
    )
    await session.commit()
    return {"id": str(alias.id)}


@router.get("/aliases")
async def list_aliases(
    entity_type: str | None = None,
    entity_id: UUID | None = None,
    session: AsyncSession = Depends(get_session),
):
    """List entity aliases."""
    return await entity_resolution_service.list_aliases(session, entity_type, entity_id)


# ── Node query (BEFORE /{assertion_id}) ───────────────────────────

@router.get("/node/{node_id}")
async def get_assertions_for_node(
    node_id: UUID,
    direction: str = Query(default="both", pattern="^(outgoing|incoming|both)$"),
    status: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Get assertions connected to a graph node."""
    assertions = await assertion_service.get_assertions_for_node(
        session, node_id, direction, status,
    )
    return [
        {
            "id": str(a.id),
            "from_node_id": str(a.from_node_id),
            "to_node_id": str(a.to_node_id),
            "edge_type": a.edge_type,
            "assertion_source": a.assertion_source,
            "confidence": a.confidence,
            "status": a.status,
        }
        for a in assertions
    ]


# ── Assertion CRUD (path params LAST) ────────────────────────────

@router.post("")
async def propose_assertion(
    req: ProposeAssertionRequest,
    session: AsyncSession = Depends(get_session),
):
    """Propose a new graph assertion."""
    assertion = await assertion_service.propose_assertion(
        session,
        from_node_id=req.from_node_id,
        to_node_id=req.to_node_id,
        edge_type=req.edge_type,
        assertion_source=req.assertion_source,
        confidence=req.confidence,
        metadata=req.metadata,
        evidence_unit_ids=req.evidence_unit_ids,
    )
    await session.commit()
    return {"id": str(assertion.id), "status": assertion.status, "edge_type": assertion.edge_type}


@router.get("/{assertion_id}")
async def get_assertion(
    assertion_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Get assertion detail with evidence and node info."""
    result = await assertion_service.get_assertion_with_evidence(session, assertion_id)
    if not result:
        raise HTTPException(404, "Assertion not found")
    return result


@router.get("/{assertion_id}/audit")
async def audit_assertion(
    assertion_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Audit an assertion's evidence backing."""
    return await assertion_service.audit_assertion(session, assertion_id)


@router.post("/{assertion_id}/publish")
async def publish_assertion(
    assertion_id: UUID,
    reviewed_by: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Publish an assertion."""
    assertion = await assertion_service.publish_assertion(session, assertion_id, reviewed_by)
    if not assertion:
        raise HTTPException(404, "Assertion not found")
    await session.commit()
    return {"id": str(assertion.id), "status": assertion.status}


@router.post("/{assertion_id}/reject")
async def reject_assertion(
    assertion_id: UUID,
    req: ReviewDecisionRequest,
    session: AsyncSession = Depends(get_session),
):
    """Reject an assertion."""
    assertion = await assertion_service.reject_assertion(
        session, assertion_id, req.reviewer, req.reason,
    )
    if not assertion:
        raise HTTPException(404, "Assertion not found")
    await session.commit()
    return {"id": str(assertion.id), "status": assertion.status}
