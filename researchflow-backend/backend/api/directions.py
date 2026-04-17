"""Directions API router — propose and expand research directions."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import direction_service

router = APIRouter(prefix="/directions", tags=["directions"])


class ProposeRequest(BaseModel):
    topic: str
    category: str | None = None
    max_directions: int = 3


@router.post("/propose")
async def propose_directions(
    data: ProposeRequest,
    session: AsyncSession = Depends(get_session),
):
    """Propose 1-3 research direction cards for a given topic."""
    cards = await direction_service.propose_directions(
        session, data.topic, data.category, data.max_directions,
    )
    await session.commit()
    return {
        "topic": data.topic,
        "directions": [
            {
                "id": str(c.id),
                "title": c.title,
                "rationale": c.rationale,
                "is_structural": c.is_structural,
                "estimated_cost": c.estimated_cost,
                "max_risk": c.max_risk,
                "confidence": c.confidence,
                "required_assets": c.required_assets,
                "has_feasibility_plan": c.feasibility_plan_md is not None,
            }
            for c in cards
        ],
    }


@router.post("/{direction_id}/expand")
async def expand_direction(
    direction_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Expand a direction card into a detailed feasibility plan."""
    card = await direction_service.expand_direction(session, direction_id)
    if not card:
        raise HTTPException(status_code=404, detail="Direction not found")
    await session.commit()
    return {
        "id": str(card.id),
        "title": card.title,
        "feasibility_plan": card.feasibility_plan_md,
    }


@router.get("")
async def list_directions(
    limit: int = Query(default=20, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
):
    cards = await direction_service.list_directions(session, limit)
    return [
        {
            "id": str(c.id),
            "title": c.title,
            "rationale": c.rationale,
            "is_structural": c.is_structural,
            "confidence": c.confidence,
            "source_topic": c.source_topic,
            "has_feasibility_plan": c.feasibility_plan_md is not None,
            "created_at": str(c.created_at),
        }
        for c in cards
    ]
