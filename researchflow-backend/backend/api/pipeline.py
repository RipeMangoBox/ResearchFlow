"""Pipeline + discovery API router."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import pipeline_service, discovery_service

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


# ── Full pipeline ─────────────────────────────────────────────────

@router.post("/{paper_id}/run")
async def run_full_pipeline(
    paper_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Run complete pipeline for a paper: download → enrich → parse → skim → deep → graph."""
    try:
        result = await pipeline_service.run_full_pipeline(session, paper_id)
        await session.commit()
        return result
    except HTTPException:
        raise
    except Exception:
        await session.rollback()
        raise


@router.post("/batch")
async def run_pipeline_batch(
    limit: int = Query(default=5, ge=1, le=20),
    session: AsyncSession = Depends(get_session),
):
    """Run full pipeline on papers that need processing."""
    try:
        results = await pipeline_service.run_pipeline_batch(session, limit)
        await session.commit()
        return {"processed": len(results), "results": results}
    except Exception:
        await session.rollback()
        raise


# ── Paper discovery ───────────────────────────────────────────────

@router.post("/{paper_id}/discover")
async def discover_related(
    paper_id: UUID,
    max_references: int = Query(default=10, ge=1, le=50),
    max_citations: int = Query(default=10, ge=1, le=50),
    auto_ingest: bool = Query(default=True),
    session: AsyncSession = Depends(get_session),
):
    """Discover related papers via Semantic Scholar and auto-ingest."""
    try:
        result = await discovery_service.discover_related_papers(
            session, paper_id,
            max_references=max_references,
            max_citations=max_citations,
            auto_ingest=auto_ingest,
        )
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise


class BuildDomainRequest(BaseModel):
    depth: int = Field(default=1, ge=1, le=3)
    max_per_hop: int = Field(default=10, ge=1, le=30)
    run_pipeline: bool = False


@router.post("/{paper_id}/build-domain")
async def build_domain_from_seed(
    paper_id: UUID,
    data: BuildDomainRequest,
    session: AsyncSession = Depends(get_session),
):
    """Build a domain knowledge graph from a seed paper.

    Discovers references + citations + related papers, ingests them,
    and optionally runs the full analysis pipeline.
    """
    try:
        result = await discovery_service.build_domain_from_seed(
            session, paper_id,
            depth=data.depth,
            max_per_hop=data.max_per_hop,
            run_pipeline=data.run_pipeline,
        )
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise


# ── PDF download ──────────────────────────────────────────────────

@router.post("/{paper_id}/download-pdf")
async def download_pdf(
    paper_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Download PDF from arxiv for a paper."""
    try:
        ok = await pipeline_service.download_arxiv_pdf(session, paper_id)
        if not ok:
            raise HTTPException(400, "PDF download failed (no arxiv_id or download error)")
        await session.commit()
        return {"status": "downloaded"}
    except HTTPException:
        raise
    except Exception:
        await session.rollback()
        raise
