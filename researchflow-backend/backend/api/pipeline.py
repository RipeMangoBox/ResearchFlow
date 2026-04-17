"""Pipeline + discovery API router."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.services import pipeline_service, discovery_service, domain_init_service, evolution_service

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


# ── Domain initialization ─────────────────────────────────────────

class InitDomainRequest(BaseModel):
    domain: str = Field(..., min_length=1)
    repo_url: str | None = None
    max_papers: int = Field(default=50, ge=1, le=200)
    category: str | None = None


@router.post("/init-domain")
async def init_domain(
    data: InitDomainRequest,
    session: AsyncSession = Depends(get_session),
):
    """Initialize a domain KB from awesome-list repos.

    Finds the best awesome-list for the domain, extracts papers,
    ingests them, and returns a priority queue for analysis.
    """
    try:
        result = await domain_init_service.init_domain_from_awesome(
            session, data.domain,
            repo_url=data.repo_url,
            max_papers=data.max_papers,
            category=data.category,
        )
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise


@router.get("/awesome-repos")
async def search_awesome_repos(
    domain: str = Query(..., min_length=1),
    limit: int = Query(default=5, ge=1, le=20),
):
    """Search GitHub for awesome-list repos matching a domain."""
    repos = await domain_init_service.find_awesome_repos(domain, limit)
    return {"domain": domain, "repos": repos}


# ── Lineage / evolution ───────────────────────────────────────────

@router.get("/{paper_id}/lineage")
async def get_lineage(
    paper_id: UUID,
    direction: str = Query(default="both", pattern="^(ancestors|descendants|both)$"),
    session: AsyncSession = Depends(get_session),
):
    """Get the method lineage tree for a paper's DeltaCard."""
    from sqlalchemy import desc, select
    from backend.models.delta_card import DeltaCard
    dc_result = await session.execute(
        select(DeltaCard).where(
            DeltaCard.paper_id == paper_id,
            DeltaCard.status != "deprecated",
        ).order_by(desc(DeltaCard.created_at)).limit(1)
    )
    dc = dc_result.scalar_one_or_none()
    if not dc:
        raise HTTPException(404, "No DeltaCard for this paper")
    return await evolution_service.get_lineage_tree(session, dc.id, direction)


@router.get("/evolution/candidates")
async def evolution_candidates(
    domain: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Find DeltaCards that might be ready to become new paradigm versions."""
    return await evolution_service.check_paradigm_evolution(session, domain)


@router.post("/evolution/promote/{delta_card_id}")
async def promote_to_paradigm(
    delta_card_id: UUID,
    name: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Promote an established baseline DeltaCard to a new paradigm version."""
    try:
        paradigm = await evolution_service.promote_to_paradigm(session, delta_card_id, name)
        if not paradigm:
            raise HTTPException(404, "DeltaCard not found")
        await session.commit()
        return {"paradigm_id": str(paradigm.id), "name": paradigm.name, "version": paradigm.version}
    except HTTPException:
        raise
    except Exception:
        await session.rollback()
        raise
