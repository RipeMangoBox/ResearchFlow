"""arq worker configuration and task definitions."""

import logging

from arq import cron
from arq.connections import RedisSettings

from backend.config import settings

logger = logging.getLogger(__name__)


# ── Task functions ──────────────────────────────────────────────

async def task_triage_all(ctx: dict):
    """Score all unscored papers."""
    from backend.database import async_session
    from backend.services import triage_service

    async with async_session() as session:
        count = await triage_service.triage_all_unscored(session)
        await session.commit()
    return {"scored": count}


async def task_enrich_batch(ctx: dict, limit: int = 20):
    """Enrich papers missing metadata from arXiv/Crossref."""
    from backend.database import async_session
    from backend.services import enrich_service

    async with async_session() as session:
        results = await enrich_service.enrich_batch(session, limit=limit)
        await session.commit()
    return {"processed": len(results)}


async def task_daily_digest(ctx: dict):
    """Generate daily research digest."""
    from backend.database import async_session
    from backend.services import digest_service

    async with async_session() as session:
        digest = await digest_service.generate_digest(session, "day")
        await session.commit()
    return {"digest_id": str(digest.id)}


async def task_weekly_digest(ctx: dict):
    """Generate weekly research digest."""
    from backend.database import async_session
    from backend.services import digest_service

    async with async_session() as session:
        digest = await digest_service.generate_digest(session, "week")
        await session.commit()
    return {"digest_id": str(digest.id)}


async def task_cleanup_expired(ctx: dict):
    """Archive expired ephemeral papers."""
    from backend.database import async_session
    from backend.services import ingestion_service

    async with async_session() as session:
        count = await ingestion_service.cleanup_expired(session)
        await session.commit()
    return {"archived": count}


async def task_sync_domains_hot(ctx: dict):
    """Daily hot sync — check OpenAlex for new papers in all active domains."""
    from backend.database import async_session
    from backend.services import domain_sync_service
    from backend.models.domain import DomainSpec
    from sqlalchemy import select

    async with async_session() as session:
        domains = (await session.execute(
            select(DomainSpec).where(DomainSpec.status == "active")
        )).scalars().all()
        results = []
        for d in domains:
            try:
                r = await domain_sync_service.sync_domain(session, d.id, "hot", 10)
                results.append({"domain": d.name, "new": r.get("papers_new", 0)})
            except Exception as e:
                results.append({"domain": d.name, "error": str(e)[:80]})
        await session.commit()
    return {"synced": len(results), "results": results}


async def task_sync_domains_weekly(ctx: dict):
    """Weekly sync — OpenAlex + S2 expansion + awesome diff."""
    from backend.database import async_session
    from backend.services import domain_sync_service
    from backend.models.domain import DomainSpec
    from sqlalchemy import select

    async with async_session() as session:
        domains = (await session.execute(
            select(DomainSpec).where(DomainSpec.status == "active")
        )).scalars().all()
        results = []
        for d in domains:
            try:
                r = await domain_sync_service.sync_domain(session, d.id, "weekly", 20)
                results.append({"domain": d.name, "new": r.get("papers_new", 0)})
            except Exception as e:
                results.append({"domain": d.name, "error": str(e)[:80]})
        await session.commit()
    return {"synced": len(results), "results": results}


async def task_refresh_materialized_views(ctx: dict):
    """Refresh all CQRS-lite materialized views for read-optimized queries."""
    from backend.database import async_session
    from sqlalchemy import text

    views = ["paper_search_docs", "idea_search_docs", "lineage_view", "review_queue_view"]
    refreshed = []
    async with async_session() as session:
        for view in views:
            try:
                await session.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}"))
                refreshed.append(view)
            except Exception:
                try:
                    await session.execute(text(f"REFRESH MATERIALIZED VIEW {view}"))
                    refreshed.append(view)
                except Exception as e2:
                    refreshed.append(f"{view}: FAILED")
        await session.commit()
    return {"refreshed": refreshed}


# ── v2 tasks ─────────────────────────────────────────────────────

async def task_venue_resolve_batch(ctx: dict, limit: int = 10):
    """Resolve venue/acceptance status for papers missing acceptance_type."""
    from backend.database import async_session
    from backend.models.paper import Paper
    from backend.models.enums import PaperState
    from backend.services.venue_resolver_service import resolve_venue
    from sqlalchemy import select

    async with async_session() as session:
        papers = (await session.execute(
            select(Paper).where(
                Paper.acceptance_type.is_(None),
                Paper.state.notin_([PaperState.SKIP, PaperState.ARCHIVED_OR_EXPIRED]),
            ).order_by(Paper.analysis_priority.desc().nullsfirst())
            .limit(limit)
        )).scalars().all()

        results = []
        for paper in papers:
            try:
                authors = [a.get("name", "") for a in (paper.authors or []) if isinstance(a, dict)]
                r = await resolve_venue(
                    session, paper.id, title=paper.title,
                    authors=authors or None,
                    arxiv_id=paper.arxiv_id or "",
                    current_venue=paper.venue or "",
                    current_year=paper.year or 0,
                )
                results.append({"paper_id": str(paper.id), "venue": r.get("venue"), "status": r.get("acceptance_status")})
            except Exception as e:
                results.append({"paper_id": str(paper.id), "error": str(e)[:80]})
        await session.commit()
    return {"processed": len(results), "results": results}


async def task_fetch_hf_daily_papers(ctx: dict):
    """Fetch today's trending papers from HuggingFace Daily Papers."""
    import httpx
    from backend.database import async_session
    from backend.services import ingestion_service
    from backend.schemas.import_ import LinkImportItem

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get("https://huggingface.co/api/daily_papers", params={"limit": "30"})
        if resp.status_code != 200:
            return {"error": f"HF API returned {resp.status_code}"}
        papers = resp.json()

    # Convert to import items
    items = []
    for p in papers:
        paper_data = p.get("paper", {})
        arxiv_id = paper_data.get("id", "")
        if arxiv_id:
            items.append(LinkImportItem(url=f"https://arxiv.org/abs/{arxiv_id}"))

    if not items:
        return {"fetched": 0, "ingested": 0}

    async with async_session() as session:
        results = await ingestion_service.ingest_links(
            session, items, "HF_Daily", False, 30,
        )
        await session.commit()
    return {"fetched": len(papers), "ingested": len(results)}


async def task_parse_batch(ctx: dict, limit: int = 5):
    """Run L2 parse (PyMuPDF + VLM formula extraction) on unprocessed papers."""
    from backend.database import async_session
    from backend.services import parse_service

    async with async_session() as session:
        results = await parse_service.parse_all_unprocessed(session, limit=limit)
        await session.commit()
    return {"processed": len(results), "results": results}


async def task_pipeline_run(ctx: dict, paper_id: str):
    """Run the FULL pipeline for an existing paper.

    Use run_for_existing_paper — it runs enrich → parse → shallow agents
    (shallow_extractor + reference_role) → deep agents → materialize. The
    earlier version skipped shallow_ingest, so structural papers ended up
    with empty shallow_extract blackboard, no delta_card, and no facets.
    """
    from uuid import UUID
    from backend.database import async_session
    from backend.services.ingest_workflow import IngestWorkflow
    from backend.models.paper import Paper
    from backend.models.enums import PaperState

    async with async_session() as session:
        try:
            pid = UUID(paper_id)
            # Idempotency: arq retries from worker restarts and dup-enqueue
            # produce many redundant invocations for papers already at
            # l4_deep with a published DeltaCard. Skip them — re-running
            # the full LLM pipeline costs ~$0.05 per paper for no gain.
            paper = await session.get(Paper, pid)
            if paper is None:
                return {"paper_id": paper_id, "skipped": "not_found"}
            if (paper.state == PaperState.L4_DEEP
                    and paper.current_delta_card_id is not None):
                return {"paper_id": paper_id, "skipped": "already_l4_deep"}

            workflow = IngestWorkflow(session)
            result = await workflow.run_for_existing_paper(pid)
            await session.commit()
            logger.info(f"Pipeline completed for {paper_id}")
            return result
        except Exception as e:
            await session.rollback()
            logger.error(f"Pipeline failed for {paper_id}: {e}")
            return {"paper_id": paper_id, "error": str(e)[:200]}


async def task_pipeline_recover(ctx: dict, limit: int = 5):
    """Auto-recover papers stuck in intermediate states.

    Uses IngestWorkflow for recovery (enrich + parse + deep analysis).
    """
    from backend.database import async_session
    from backend.models.paper import Paper
    from backend.models.enums import PaperState
    from backend.services.ingest_workflow import IngestWorkflow
    from sqlalchemy import select
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
    stuck_states = [
        PaperState.WAIT,
        PaperState.DOWNLOADED,
        PaperState.L1_METADATA,
        PaperState.ENRICHED,
        PaperState.L2_PARSED,
        PaperState.L3_SKIMMED,
    ]

    async with async_session() as session:
        # Pre-extract (id, state) tuples — never reuse ORM instances across
        # a commit cycle. After session.commit() the instance may be expired
        # and next attribute access triggers a lazy load OUTSIDE the greenlet
        # context, throwing MissingGreenlet. Plain Python tuples are immune.
        rows = (await session.execute(
            select(Paper.id, Paper.state).where(
                Paper.state.in_(stuck_states),
                Paper.arxiv_id.isnot(None),
                Paper.updated_at < cutoff,
            ).order_by(Paper.analysis_priority.desc().nullsfirst())
            .limit(limit)
        )).all()

        if not rows:
            return {"recovered": 0}

        targets = [(pid, st.value if st else "?") for pid, st in rows]

        results = []
        for pid, state_val in targets:
            pid_str = str(pid)
            try:
                logger.info(f"Auto-recovering paper {pid} (state={state_val})")
                workflow = IngestWorkflow(session)
                await workflow.enrich_and_prepare(pid)
                await workflow.deep_ingest(pid)
                await session.commit()
                # Re-read the new state via raw query — no ORM instance reuse.
                new_state = (await session.execute(
                    select(Paper.state).where(Paper.id == pid)
                )).scalar_one_or_none()
                results.append({
                    "paper_id": pid_str,
                    "from_state": state_val,
                    "to_state": new_state.value if new_state else "?",
                })
            except Exception as e:
                await session.rollback()
                logger.error(f"Recovery failed for {pid}: {e}")
                results.append({"paper_id": pid_str, "error": str(e)[:100]})

    return {"recovered": len(results), "results": results}


# ── V6 tasks ───────────────────────────────────────────────────────

async def task_score_candidates_v6(ctx: dict, limit: int = 50):
    """Score unscored paper candidates."""
    from backend.database import async_session
    from backend.services import candidate_service

    async with async_session() as session:
        count = await candidate_service.score_batch(session, limit=limit)
        await session.commit()
    return {"scored": count}


async def task_auto_promote_v6(ctx: dict, threshold: float = 75.0, limit: int = 20):
    """Auto-promote high-scoring candidates to papers."""
    from backend.database import async_session
    from backend.services import candidate_service

    async with async_session() as session:
        papers = await candidate_service.auto_promote_batch(session, threshold=threshold, limit=limit)
        await session.commit()
    return {"promoted": len(papers)}


async def task_refresh_stale_profiles_v6(ctx: dict, threshold: int = 3, limit: int = 20):
    """Refresh node profiles that have become stale."""
    from backend.database import async_session
    from backend.services import node_profile_service

    async with async_session() as session:
        count = await node_profile_service.refresh_stale_profiles(session, threshold=threshold, limit=limit)
        await session.commit()
    return {"refreshed": count}



# ── V6 incremental sync tasks ─────────────────────────────────────

# V6 arXiv daily sync — daily at 10:00
async def task_arxiv_daily_sync_v6(ctx: dict):
    """Sync new arXiv papers for all active domains."""
    from backend.database import async_session
    async with async_session() as session:
        from backend.services.incremental_sync_service import sync_arxiv_daily
        from sqlalchemy import select
        from backend.models.domain import DomainSpec
        domains = (await session.execute(
            select(DomainSpec).where(DomainSpec.status == "active")
        )).scalars().all()
        total = 0
        for domain in domains:
            try:
                result = await sync_arxiv_daily(session, domain.id)
                total += result.get("candidates_created", 0)
            except Exception as e:
                logger.error(f"arXiv sync failed for {domain.name}: {e}")
        await session.commit()
        return {"domains_synced": len(domains), "total_candidates": total}


# V6 citation refresh — weekly on Wednesday 03:00
async def task_citation_refresh_v6(ctx: dict, limit: int = 50):
    """Refresh citation counts for papers via S2 API."""
    from backend.database import async_session
    async with async_session() as session:
        from backend.services.incremental_sync_service import refresh_citation_counts
        result = await refresh_citation_counts(session, limit=limit)
        await session.commit()
        return result


# V6 awesome repo diff — weekly on Thursday 04:00
async def task_awesome_repo_diff_v6(ctx: dict):
    """Detect new papers in tracked awesome repos."""
    from backend.database import async_session
    async with async_session() as session:
        from backend.services.incremental_sync_service import detect_awesome_repo_changes
        from sqlalchemy import select
        from backend.models.domain import DomainSpec
        domains = (await session.execute(
            select(DomainSpec).where(DomainSpec.status == "active")
        )).scalars().all()
        total = 0
        for domain in domains:
            try:
                result = await detect_awesome_repo_changes(session, domain.id)
                total += result.get("candidates_created", 0)
            except Exception as e:
                logger.error(f"Awesome repo diff failed for {domain.name}: {e}")
        await session.commit()
        return {"domains_checked": len(domains), "total_candidates": total}


# V6 lineage detection — weekly on Friday 05:00
async def task_lineage_detection_v6(ctx: dict):
    """Detect method evolution lineage chains."""
    from backend.database import async_session
    async with async_session() as session:
        from backend.services.incremental_sync_service import detect_lineage_chains
        result = await detect_lineage_chains(session)
        await session.commit()
        return result


# V6 node score recomputation — weekly on Saturday 04:00
async def task_recompute_node_scores_v6(ctx: dict, limit: int = 100):
    """Recompute promotion scores for graph node candidates."""
    from backend.database import async_session
    async with async_session() as session:
        from backend.services.incremental_sync_service import recompute_node_scores
        result = await recompute_node_scores(session, limit=limit)
        await session.commit()
        return result


# V6 duplicate detection — monthly on 1st at 02:00
async def task_detect_duplicates_v6(ctx: dict):
    """Detect duplicate nodes in taxonomy and method tables."""
    from backend.database import async_session
    async with async_session() as session:
        from backend.services.incremental_sync_service import detect_duplicate_nodes
        result = await detect_duplicate_nodes(session)
        await session.commit()
        return result


# V6 stale candidate cleanup — monthly on 15th at 02:00
async def task_cleanup_stale_candidates_v6(ctx: dict, days: int = 90):
    """Archive stale candidates that haven't progressed."""
    from backend.database import async_session
    async with async_session() as session:
        from backend.services.incremental_sync_service import cleanup_stale_candidates
        result = await cleanup_stale_candidates(session, days=days)
        await session.commit()
        return result


# ── Startup / shutdown ──────────────────────────────────────────

async def startup(ctx: dict):
    pass


async def shutdown(ctx: dict):
    pass


# ── Worker settings ─────────────────────────────────────────────

async def task_supplement_venue_papers(ctx: dict, conf_years: list[str] | None = None):
    """Batch supplement venue_papers metadata (pdf_url, arxiv_id, doi)."""
    from backend.database import async_session
    from backend.services.venue_supplement_service import run_full_supplement

    async with async_session() as session:
        result = await run_full_supplement(session, conf_years)
        await session.commit()
    return result


async def task_download_paper_pdf(ctx: dict, paper_id: str):
    """Download PDF for a single paper to OSS."""
    from uuid import UUID
    from backend.database import async_session
    from backend.services.pdf_download_service import download_pdf_to_oss

    async with async_session() as session:
        ok = await download_pdf_to_oss(session, UUID(paper_id))
        await session.commit()
    return {"paper_id": paper_id, "success": ok}


async def task_download_pdfs_batch(ctx: dict, limit: int = 20):
    """Download PDFs for papers that have been promoted but lack PDF."""
    from backend.database import async_session
    from backend.models.paper import Paper
    from backend.services.pdf_download_service import download_pdf_to_oss
    from sqlalchemy import select

    async with async_session() as session:
        papers = (await session.execute(
            select(Paper.id).where(
                Paper.pdf_object_key.is_(None),
                Paper.pdf_path_local.is_(None),
                Paper.state.notin_(["skip", "archived_or_expired"]),
            ).order_by(Paper.analysis_priority.desc().nullsfirst())
            .limit(limit)
        )).scalars().all()

        results = []
        for pid in papers:
            ok = await download_pdf_to_oss(session, pid)
            results.append({"paper_id": str(pid), "success": ok})
            await session.commit()
    return {"processed": len(results), "success": sum(1 for r in results if r["success"])}


def _parse_redis_url(url: str) -> RedisSettings:
    """Parse redis://host:port/db into RedisSettings."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int(parsed.path.lstrip("/") or 0),
    )


class WorkerSettings:
    functions = [
        task_triage_all,
        task_enrich_batch,
        task_daily_digest,
        task_weekly_digest,
        task_cleanup_expired,
        task_refresh_materialized_views,
        task_sync_domains_hot,
        task_sync_domains_weekly,
        task_venue_resolve_batch,
        task_fetch_hf_daily_papers,
        task_parse_batch,
        task_pipeline_run,
        task_pipeline_recover,
        # Venue supplement + PDF download
        task_supplement_venue_papers,
        task_download_paper_pdf,
        task_download_pdfs_batch,
        # V6
        task_score_candidates_v6,
        task_auto_promote_v6,
        task_refresh_stale_profiles_v6,
        # V6 incremental sync
        task_arxiv_daily_sync_v6,
        task_citation_refresh_v6,
        task_awesome_repo_diff_v6,
        task_lineage_detection_v6,
        task_recompute_node_scores_v6,
        task_detect_duplicates_v6,
        task_cleanup_stale_candidates_v6,
    ]

    cron_jobs = [
        # Refresh materialized views every 30 minutes
        cron(task_refresh_materialized_views, minute={0, 30}),
        # Domain sync: daily hot at 06:00, weekly on Monday at 04:00
        cron(task_sync_domains_hot, hour=6, minute=0),
        cron(task_sync_domains_weekly, weekday=0, hour=4, minute=0),
        # Daily digest at 23:00
        cron(task_daily_digest, hour=23, minute=0),
        # Weekly digest on Sunday at 22:00
        cron(task_weekly_digest, weekday=6, hour=22, minute=0),
        # Cleanup expired ephemeral papers daily at 03:00
        cron(task_cleanup_expired, hour=3, minute=0),
        # v2: HuggingFace daily papers at 08:00 and 20:00
        cron(task_fetch_hf_daily_papers, hour={8, 20}, minute=0),
        # v2: Venue resolution batch at 07:00
        cron(task_venue_resolve_batch, hour=7, minute=0),
        # v2: Parse unprocessed PDFs every 2 hours
        cron(task_parse_batch, hour={2, 4, 8, 12, 16, 20}, minute=30),
        # Auto-recover stuck papers every 30 minutes
        cron(task_pipeline_recover, minute={15, 45}),
        # V6: Score candidates every 2 hours
        cron(task_score_candidates_v6, hour={0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}, minute=15),
        # V6: Auto-promote daily at 09:00
        cron(task_auto_promote_v6, hour=9, minute=0),
        # V6: Refresh stale profiles daily at 05:00
        cron(task_refresh_stale_profiles_v6, hour=5, minute=0),
        # V6 incremental: arXiv daily sync at 10:00
        cron(task_arxiv_daily_sync_v6, hour=10, minute=0),
        # V6 incremental: Citation refresh Wednesday 03:00
        cron(task_citation_refresh_v6, weekday=2, hour=3, minute=0),
        # V6 incremental: Awesome repo diff Thursday 04:00
        cron(task_awesome_repo_diff_v6, weekday=3, hour=4, minute=0),
        # V6 incremental: Lineage detection Friday 05:00
        cron(task_lineage_detection_v6, weekday=4, hour=5, minute=0),
        # V6 incremental: Node score recomputation Saturday 04:00
        cron(task_recompute_node_scores_v6, weekday=5, hour=4, minute=0),
        # V6 incremental: Duplicate detection — Sunday 02:00 (weekly, arq has no month_day)
        cron(task_detect_duplicates_v6, weekday=6, hour=2, minute=0),
        # V6 incremental: Stale candidate cleanup — Sunday 03:00
        cron(task_cleanup_stale_candidates_v6, weekday=6, hour=3, minute=0),
    ]

    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _parse_redis_url(settings.redis_url)
    # Concurrency 4: tried 6 but Kimi proxy throttles under that load —
    # 19 LLM timeouts in 60min, each forcing 3x retry → effective rate
    # actually DROPPED from ~50/h to ~14/h. 4 keeps Kimi happy and
    # lets every call succeed first try, which is cheaper end-to-end.
    max_jobs = 4
    job_timeout = 1200  # 20 min per job (L4 with max_tokens=16K can take 5-10 min)
