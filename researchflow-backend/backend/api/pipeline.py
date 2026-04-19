"""Pipeline + discovery API router."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import get_session
from backend.models.paper import Paper
from backend.services import pipeline_service, discovery_service, domain_init_service, evolution_service

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


# ── Full pipeline ─────────────────────────────────────────────────

@router.post("/{paper_id}/run")
async def run_full_pipeline(
    paper_id: UUID,
    sync: bool = Query(default=False, description="Run synchronously (slow, for debugging only)"),
    session: AsyncSession = Depends(get_session),
):
    """Run complete pipeline for a paper.

    Default: enqueues to worker (async, returns immediately).
    ?sync=true: runs in API process (slow, may OOM on large PDFs).
    """
    paper = await session.get(Paper, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    if sync:
        # Synchronous mode — for debugging only
        try:
            result = await pipeline_service.run_full_pipeline(session, paper_id)
            await session.commit()
            return result
        except Exception:
            await session.rollback()
            raise
    else:
        # Async mode — enqueue to worker (default)
        from arq import create_pool
        from backend.workers.arq_app import _parse_redis_url
        redis = await create_pool(_parse_redis_url(settings.redis_url))
        job = await redis.enqueue_job("task_pipeline_run", str(paper_id))
        await redis.close()
        return {
            "paper_id": str(paper_id),
            "status": "enqueued",
            "job_id": job.job_id if job else None,
            "message": "Pipeline running in background worker. Check paper state for progress.",
        }


@router.post("/batch")
async def run_pipeline_batch(
    limit: int = Query(default=10, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
):
    """Enqueue pipeline for papers that need processing."""
    from arq import create_pool
    from backend.workers.arq_app import _parse_redis_url

    # Find papers needing processing
    from backend.models.enums import PaperState
    papers = (await session.execute(
        select(Paper).where(
            Paper.state.in_([
                PaperState.WAIT, PaperState.DOWNLOADED,
                PaperState.L1_METADATA, PaperState.ENRICHED,
                PaperState.L2_PARSED, PaperState.L3_SKIMMED,
            ]),
            Paper.arxiv_id.isnot(None),
        ).order_by(Paper.analysis_priority.desc().nullsfirst())
        .limit(limit)
    )).scalars().all()

    if not papers:
        return {"enqueued": 0, "message": "No papers need processing"}

    redis = await create_pool(_parse_redis_url(settings.redis_url))
    enqueued = []
    for paper in papers:
        job = await redis.enqueue_job("task_pipeline_run", str(paper.id))
        enqueued.append({
            "paper_id": str(paper.id),
            "title": paper.title[:60],
            "state": paper.state.value if paper.state else "?",
            "job_id": job.job_id if job else None,
        })
    await redis.close()
    return {"enqueued": len(enqueued), "papers": enqueued}


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
    domain: str = Field(..., min_length=1, alias="domain_name")
    seed_papers: list[str] | None = Field(default=None, description="arXiv IDs or URLs of seed papers")
    seed_repos: list[str] | None = Field(default=None, description="Awesome-list repo URLs")
    openalex_topic_ids: list[str] | None = None
    constraints: dict | None = None
    negative_constraints: list[str] | None = None
    max_papers: int = Field(default=50, ge=1, le=200)
    category: str | None = None
    # Legacy compat
    repo_url: str | None = None

    class Config:
        populate_by_name = True


@router.post("/init-domain")
async def init_domain(
    data: InitDomainRequest,
    session: AsyncSession = Depends(get_session),
):
    """Initialize a domain KB from multiple sources.

    Sources (in priority order): seed_papers → awesome repos → OpenAlex topics → S2 expansion.
    Creates DomainSpec, triages papers into rings (baseline/structural/plugin).
    """
    try:
        # Use multi-source if any new params provided, else legacy
        if data.seed_papers or data.openalex_topic_ids or data.seed_repos:
            result = await domain_init_service.init_domain_multi_source(
                session,
                domain_name=data.domain,
                seed_papers=data.seed_papers,
                seed_repos=data.seed_repos or ([data.repo_url] if data.repo_url else None),
                openalex_topic_ids=data.openalex_topic_ids,
                constraints=data.constraints,
                negative_constraints=data.negative_constraints,
                max_papers=data.max_papers,
                category=data.category,
            )
        else:
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


@router.post("/export/sync-analyses")
async def sync_analyses(
    limit: int = Query(default=50, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
):
    """Export all L4-analyzed papers to paperAnalysis/ Markdown files."""
    from backend.services.export_service import export_paper_analysis
    from sqlalchemy import select
    from backend.models.paper import Paper
    from backend.models.enums import PaperState

    result = await session.execute(
        select(Paper.id).where(
            Paper.state.in_([PaperState.L4_DEEP, PaperState.CHECKED])
        ).limit(limit)
    )
    exported = 0
    for (pid,) in result:
        try:
            path = await export_paper_analysis(session, pid)
            if path:
                exported += 1
        except Exception:
            pass
    return {"exported": exported}


@router.post("/sync-domain/{domain_id}")
async def sync_domain(
    domain_id: UUID,
    mode: str = Query(default="hot", pattern="^(hot|weekly|monthly)$"),
    max_new: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
):
    """Incrementally sync a domain's KB from registered sources.

    Modes: hot (daily, OpenAlex new works), weekly (+S2 expansion),
    monthly (+re-analyze low-confidence + ontology audit).
    """
    from backend.services import domain_sync_service
    try:
        result = await domain_sync_service.sync_domain(session, domain_id, mode, max_new)
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise


@router.post("/refresh-connections")
async def refresh_connections(
    paper_id: UUID | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Re-link papers: update same_family, downstream_count, baseline promotions.

    Call after importing new papers to update relationships across the KB.
    If paper_id given, refresh that paper + neighbors only.
    """
    try:
        result = await evolution_service.refresh_connections(session, paper_id)
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise


@router.post("/export/obsidian-vault")
async def export_obsidian_vault(
    session: AsyncSession = Depends(get_session),
):
    """Export full knowledge base as Obsidian-ready vault with [[wikilinks]] + concept hub pages.

    Generates: papers/ (per-paper with wikilinks), concepts/mechanisms/, concepts/bottlenecks/,
    concepts/paradigms/, _Index.md (Dataview queries), _Graph.md (graph view guide).
    """
    from backend.services.export_service import export_obsidian_vault as _export
    result = await _export(session)
    return result


@router.post("/export/obsidian-vault-v6")
async def export_obsidian_vault_v6(
    session: AsyncSession = Depends(get_session),
):
    """Export v6 vault: v5 base + node/edge profiles + Lab pages."""
    from backend.services.vault_export_v6 import export_vault_v6
    result = await export_vault_v6(session)
    return result


@router.post("/export/build-collection-index")
async def build_collection_index(
    session: AsyncSession = Depends(get_session),
):
    """Build paperCollection/index.jsonl + navigation pages from DB."""
    from backend.services.export_service import build_collection_index as _build
    result = await _build(session)
    return result


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


# ── V6 pipeline endpoints ───────────────────────────────────────

class V6PipelineRequest(BaseModel):
    source: str  # URL or arxiv_id
    domain_id: UUID | None = None


@router.post("/v6/run")
async def run_v6_pipeline(
    body: V6PipelineRequest,
    session: AsyncSession = Depends(get_session),
):
    """Run the V6 pipeline for a paper URL."""
    from backend.services.ingest_workflow import IngestWorkflow
    try:
        workflow = IngestWorkflow(session)
        result = await workflow.run_full_v6_pipeline(body.source, domain_id=body.domain_id)
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise


@router.post("/v6/shallow/{candidate_id}")
async def run_v6_shallow(candidate_id: UUID, session: AsyncSession = Depends(get_session)):
    """Run shallow ingest on a candidate."""
    from backend.services.ingest_workflow import IngestWorkflow
    try:
        workflow = IngestWorkflow(session)
        result = await workflow.shallow_ingest(candidate_id)
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise


@router.post("/v6/deep/{paper_id}")
async def run_v6_deep(paper_id: UUID, session: AsyncSession = Depends(get_session)):
    """Run deep ingest on a paper."""
    from backend.services.ingest_workflow import IngestWorkflow
    try:
        workflow = IngestWorkflow(session)
        result = await workflow.deep_ingest(paper_id)
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise
