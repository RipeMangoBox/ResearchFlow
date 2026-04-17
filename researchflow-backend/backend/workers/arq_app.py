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


async def task_cleanup_expired(ctx: dict):
    """Archive expired ephemeral papers."""
    from backend.database import async_session
    from backend.services import ingestion_service

    async with async_session() as session:
        count = await ingestion_service.cleanup_expired(session)
        await session.commit()
    return {"archived": count}


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
        task_cleanup_expired,
    ]

    cron_jobs = [
        # Cleanup expired ephemeral papers daily at 03:00
        cron(task_cleanup_expired, hour=3, minute=0),
    ]

    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _parse_redis_url(settings.redis_url)
    max_jobs = 2
    job_timeout = 300  # 5 min per job
