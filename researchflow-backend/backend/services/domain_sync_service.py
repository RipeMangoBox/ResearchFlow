"""Domain sync service — incremental knowledge base updates.

Three modes:
  hot (daily): OpenAlex new works since last checkpoint
  weekly: + S2 recs expansion + awesome README diff
  monthly: + low-confidence re-analysis + ontology candidate audit
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

import httpx
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.domain import DomainSpec, DomainSourceRegistry, IncrementalCheckpoint
from backend.models.paper import Paper
from backend.schemas.import_ import LinkImportItem
from backend.services.ingestion_service import ingest_link

logger = logging.getLogger(__name__)


async def sync_domain(
    session: AsyncSession,
    domain_id: UUID,
    mode: str = "hot",
    max_new: int = 20,
) -> dict:
    """Sync a domain's knowledge base from registered sources.

    Modes:
      hot: daily — new works from OpenAlex since last sync
      weekly: + S2 recs + awesome README diff
      monthly: + re-analyze low-confidence + audit candidates
    """
    domain = await session.get(DomainSpec, domain_id)
    if not domain:
        return {"error": "Domain not found"}

    result = {
        "domain_id": str(domain_id),
        "mode": mode,
        "sources_synced": 0,
        "papers_found": 0,
        "papers_new": 0,
    }

    # Get registered sources
    sources = (await session.execute(
        select(DomainSourceRegistry).where(
            DomainSourceRegistry.domain_id == domain_id,
            DomainSourceRegistry.is_active.is_(True),
        )
    )).scalars().all()

    category = domain.name.replace(" ", "_")

    for src in sources:
        # Check sync frequency
        if mode == "hot" and src.sync_frequency not in ("daily", "hot"):
            if src.source_type != "openalex_topic":
                continue
        elif mode == "weekly" and src.sync_frequency == "monthly":
            continue

        try:
            if src.source_type == "openalex_topic":
                found, new = await _sync_openalex_topic(session, src, category, max_new)
            elif src.source_type == "awesome_repo" and mode in ("weekly", "monthly"):
                found, new = await _sync_awesome_repo(session, src, category, max_new)
            else:
                continue

            result["sources_synced"] += 1
            result["papers_found"] += found
            result["papers_new"] += new

            # Update last synced
            src.last_synced_at = datetime.now(timezone.utc)

            # Record checkpoint
            cp = IncrementalCheckpoint(
                source_registry_id=src.id,
                checkpoint_value=datetime.now(timezone.utc).isoformat(),
                papers_found=found,
                papers_new=new,
                sync_mode=mode,
            )
            session.add(cp)
        except Exception as e:
            logger.warning(f"Sync failed for {src.source_type}:{src.source_ref}: {e}")

    # Weekly: S2 expansion from recent high-score papers
    if mode in ("weekly", "monthly"):
        recent = (await session.execute(
            select(Paper).where(
                Paper.domain_id == domain_id,
                Paper.ring == "structural",
            ).order_by(desc(Paper.created_at)).limit(3)
        )).scalars().all()

        from backend.services import discovery_service
        for p in recent:
            try:
                disc = await discovery_service.discover_related_papers(
                    session, p.id, max_references=3, max_citations=3, auto_ingest=True,
                )
                result["papers_new"] += disc.get("ingested", 0)
            except Exception:
                pass

    # Monthly: re-analyze low-confidence DeltaCards
    if mode == "monthly":
        from backend.models.delta_card import DeltaCard
        low_conf = (await session.execute(
            select(DeltaCard).where(
                DeltaCard.extraction_confidence < 0.5,
                DeltaCard.status != "deprecated",
            ).limit(10)
        )).scalars().all()

        for dc in low_conf:
            paper = await session.get(Paper, dc.paper_id)
            if paper:
                paper.state = "l2_parsed"  # Reset to re-analyze
                result["papers_found"] += 1

    # Update domain stats
    total = (await session.execute(
        select(Paper).where(Paper.domain_id == domain_id)
    )).scalars().all()
    domain.paper_count = len(total)

    await session.flush()
    return result


async def _sync_openalex_topic(
    session: AsyncSession,
    src: DomainSourceRegistry,
    category: str,
    max_new: int,
) -> tuple[int, int]:
    """Sync new works from an OpenAlex topic since last checkpoint."""
    # Get last checkpoint date
    last_cp = (await session.execute(
        select(IncrementalCheckpoint).where(
            IncrementalCheckpoint.source_registry_id == src.id
        ).order_by(desc(IncrementalCheckpoint.created_at)).limit(1)
    )).scalar_one_or_none()

    from_date = "2024-01-01"
    if last_cp and last_cp.checkpoint_value:
        from_date = last_cp.checkpoint_value[:10]

    found, new = 0, 0
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.get(
                "https://api.openalex.org/works",
                params={
                    "filter": f"topics.id:{src.source_ref},from_publication_date:{from_date}",
                    "sort": "publication_date:desc",
                    "per_page": str(min(max_new, 25)),
                },
                headers={"User-Agent": "ResearchFlow/0.1"},
            )
            if resp.status_code == 200:
                works = resp.json().get("results", [])
                found = len(works)
                for w in works:
                    doi = w.get("doi", "")
                    if doi:
                        url = f"https://doi.org/{doi.replace('https://doi.org/', '')}"
                        try:
                            item = LinkImportItem(url=url, title=w.get("title", ""), category=category)
                            res = await ingest_link(session, item)
                            if res.status == "created":
                                new += 1
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"OpenAlex sync failed for topic {src.source_ref}: {e}")

    return found, new


async def _sync_awesome_repo(
    session: AsyncSession,
    src: DomainSourceRegistry,
    category: str,
    max_new: int,
) -> tuple[int, int]:
    """Check awesome repo for new papers since last sync."""
    from backend.services.domain_init_service import extract_papers_from_readme

    papers = await extract_papers_from_readme(src.source_ref, max_new)
    found, new = len(papers), 0

    for p in papers:
        try:
            item = LinkImportItem(url=p["url"], title=p.get("title"), category=category)
            res = await ingest_link(session, item)
            if res.status == "created":
                new += 1
        except Exception:
            pass

    return found, new
