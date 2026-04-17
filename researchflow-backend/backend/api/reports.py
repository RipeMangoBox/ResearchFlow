"""Reports API router — generate briefing reports."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import report_service

router = APIRouter(prefix="/reports", tags=["reports"])


class ReportRequest(BaseModel):
    paper_ids: list[UUID] = Field(..., min_length=1, max_length=20)
    report_type: str = Field(default="briefing", pattern="^(quick|briefing|deep_compare)$")
    topic: str | None = None


@router.post("/generate")
async def generate_report(
    data: ReportRequest,
    session: AsyncSession = Depends(get_session),
):
    """Generate a briefing report from a set of papers.

    Report types:
    - quick: 30-second summary (~200 words)
    - briefing: 5-minute structured report with delta card table
    - deep_compare: full cross-paper analysis with evidence
    """
    try:
        result = await report_service.generate_report(
            session,
            paper_ids=data.paper_ids,
            report_type=data.report_type,
            topic=data.topic,
        )
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise
