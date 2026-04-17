"""Analysis API router — trigger L3/L4 analysis, get results."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import analysis_service

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("/{paper_id}/skim")
async def skim_paper(
    paper_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Run L3 skim analysis on a paper. Returns lightweight card."""
    try:
        analysis = await analysis_service.skim_paper(session, paper_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Paper not found")
        await session.commit()

        return {
            "paper_id": str(paper_id),
            "analysis_id": str(analysis.id),
            "level": analysis.level.value,
            "model": f"{analysis.model_provider}/{analysis.model_name}",
            "problem_summary": analysis.problem_summary,
            "method_summary": analysis.method_summary,
            "evidence_summary": analysis.evidence_summary,
            "core_intuition": analysis.core_intuition,
            "changed_slots": analysis.changed_slots,
            "is_plugin_patch": analysis.is_plugin_patch,
            "worth_deep_read": analysis.worth_deep_read,
            "confidence_notes": analysis.confidence_notes,
        }
    except HTTPException:
        raise
    except Exception:
        await session.rollback()
        raise


@router.post("/{paper_id}/deep")
async def deep_analyze_paper(
    paper_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Run L4 deep analysis on a paper. Returns full report + delta card."""
    try:
        analysis = await analysis_service.deep_analyze_paper(session, paper_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Paper not found")
        await session.commit()

        return {
            "paper_id": str(paper_id),
            "analysis_id": str(analysis.id),
            "level": analysis.level.value,
            "model": f"{analysis.model_provider}/{analysis.model_name}",
            "problem_summary": analysis.problem_summary,
            "method_summary": analysis.method_summary,
            "evidence_summary": analysis.evidence_summary,
            "core_intuition": analysis.core_intuition,
            "changed_slots": analysis.changed_slots,
            "is_plugin_patch": analysis.is_plugin_patch,
            "full_report_md": analysis.full_report_md[:500] + "..." if analysis.full_report_md and len(analysis.full_report_md) > 500 else analysis.full_report_md,
            "confidence_notes": analysis.confidence_notes,
        }
    except HTTPException:
        raise
    except Exception:
        await session.rollback()
        raise


@router.post("/skim-batch")
async def skim_batch(
    limit: int = Query(default=5, ge=1, le=20),
    session: AsyncSession = Depends(get_session),
):
    """Run L3 skim on a batch of papers ready for analysis."""
    try:
        results = await analysis_service.skim_batch(session, limit=limit)
        await session.commit()
        return {"processed": len(results), "results": results}
    except Exception:
        await session.rollback()
        raise
