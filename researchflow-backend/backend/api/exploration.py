"""Exploration API — research session tracking + smart search."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import exploration_service

router = APIRouter(prefix="/explore", tags=["explore"])


class StartExplorationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    context: str | None = None


class ExplorationStepRequest(BaseModel):
    query: str = Field(..., min_length=1)
    step_type: str = Field(default="refine", pattern="^(refine|pivot|deepen|broaden)$")
    rejected_reason: str | None = None
    insight: str | None = None


@router.post("/start")
async def start_exploration(
    data: StartExplorationRequest,
    session: AsyncSession = Depends(get_session),
):
    """Start a new research exploration session."""
    try:
        search = await exploration_service.start_exploration(
            session, data.query, data.context,
        )
        await session.commit()
        return {"session_id": str(search.id), "query": data.query}
    except Exception:
        await session.rollback()
        raise


@router.post("/{session_id}/step")
async def add_step(
    session_id: UUID,
    data: ExplorationStepRequest,
    session: AsyncSession = Depends(get_session),
):
    """Add an exploration step (refine/pivot/deepen/broaden)."""
    try:
        result = await exploration_service.add_exploration_step(
            session, session_id, data.query,
            step_type=data.step_type,
            rejected_reason=data.rejected_reason,
            insight=data.insight,
        )
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise


@router.get("/{session_id}")
async def get_exploration(
    session_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Get full exploration path with insights and paper classifications."""
    return await exploration_service.get_exploration_summary(session, session_id)


@router.post("/{session_id}/search")
async def smart_search(
    session_id: UUID,
    data: StartExplorationRequest,
    session: AsyncSession = Depends(get_session),
):
    """Smart search within an exploration session.

    Searches KB, classifies results by method type (structural vs plugin),
    identifies gaps, and suggests next exploration direction.
    """
    try:
        result = await exploration_service.smart_explore(
            session, session_id, data.query,
        )
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise
