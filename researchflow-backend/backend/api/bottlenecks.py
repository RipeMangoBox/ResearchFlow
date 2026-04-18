"""Bottleneck management API — normalization, focus, claims."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import bottleneck_normalization_service as bn_service

router = APIRouter(prefix="/bottlenecks", tags=["bottlenecks"])


@router.post("/normalize")
async def normalize_claims(session: AsyncSession = Depends(get_session)):
    """Cluster unlinked paper bottleneck claims into canonical bottlenecks."""
    try:
        result = await bn_service.cluster_unlinked_claims(session)
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise


@router.post("/merge-duplicates")
async def merge_duplicates(session: AsyncSession = Depends(get_session)):
    """Merge near-duplicate bottlenecks by embedding similarity."""
    try:
        result = await bn_service.normalize_bottlenecks(session)
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise


@router.get("/unlinked-claims")
async def list_unlinked_claims(
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    """List paper bottleneck claims not yet linked to a canonical bottleneck."""
    return await bn_service.get_unlinked_claims(session, limit)


class FocusRequest(BaseModel):
    bottleneck_id: UUID
    project_name: str | None = None
    user_description: str | None = None
    priority: int = Field(default=3, ge=1, le=5)
    negative_constraints: list[str] | None = None


@router.post("/focus")
async def create_focus(
    data: FocusRequest,
    session: AsyncSession = Depends(get_session),
):
    """Create a project-level focus bottleneck (user decision: what I care about)."""
    try:
        focus = await bn_service.create_focus_bottleneck(
            session,
            bottleneck_id=data.bottleneck_id,
            project_name=data.project_name,
            user_description=data.user_description,
            priority=data.priority,
            negative_constraints=data.negative_constraints,
        )
        await session.commit()
        return {"id": str(focus.id), "status": "created"}
    except Exception:
        await session.rollback()
        raise


@router.get("/focus")
async def list_focus(session: AsyncSession = Depends(get_session)):
    """List active project focus bottlenecks."""
    return await bn_service.list_focus_bottlenecks(session)
