"""arq worker configuration and task definitions."""

from arq import cron
from arq.connections import RedisSettings

from backend.config import settings


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
    """Run L2 parse (GROBID + PyMuPDF + formula extraction) on unprocessed papers."""
    from backend.database import async_session
    from backend.services import parse_service

    async with async_session() as session:
        results = await parse_service.parse_all_unprocessed(session, limit=limit)
        await session.commit()
    return {"processed": len(results), "results": results}


# ── Startup / shutdown ──────────────────────────────────────────

async def startup(ctx: dict):
    pass


async def shutdown(ctx: dict):
    pass


# ── Worker settings ─────────────────────────────────────────────

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
    ]

    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _parse_redis_url(settings.redis_url)
    max_jobs = 2
    job_timeout = 300  # 5 min per job
