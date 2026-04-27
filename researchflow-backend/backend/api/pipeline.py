"""Pipeline + discovery API router."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import get_session
from backend.models.paper import Paper
from backend.services import pipeline_service, discovery_service, evolution_service

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


@router.post("/export/obsidian-vault")
async def export_obsidian_vault_v6(
    session: AsyncSession = Depends(get_session),
):
    """Export v6 vault: v5 base + node/edge profiles + Lab pages."""
    from backend.services.vault_export_v6 import export_vault
    result = await export_vault(session)
    return result


@router.post("/export/build-collection-index")
async def build_collection_index(
    session: AsyncSession = Depends(get_session),
):
    """Build paperCollection/index.jsonl + navigation pages from DB."""
    from backend.services.export_service import build_collection_index as _build
    result = await _build(session)
    return result


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


