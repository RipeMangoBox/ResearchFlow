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
    ]

    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _parse_redis_url(settings.redis_url)
    max_jobs = 2
    job_timeout = 300  # 5 min per job
