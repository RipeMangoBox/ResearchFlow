"""Promote venue_papers rows into the papers table for ingest.

The bulk venue crawl populates `venue_papers` (42k+ rows) with metadata +
OSS PDF object_key, but does NOT enter the analysis pipeline. This script
copies a curated subset into `papers` so they become eligible for
`task_pipeline_run` (enrich → parse → L4 → paper_report → relations).

Selection knobs:
    --venue ICCV_2025
    --year 2025
    --limit 30
    --strategy small | random | recent  (default: small for fast smoke)

Sets:
    papers.pdf_object_key  ← venue_papers.pdf_object_key  (no re-download)
    papers.title / title_sanitized / venue / year / etc.
    papers.source = 'venue_papers'
    papers.state = 'metadata_only'  (so task_pipeline_run starts at enrich)
    papers.category = '__pending__'  (will be set by L3 facet assignment)

Idempotent: skips rows whose normalized title is already in papers.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import uuid
from typing import Iterable

from sqlalchemy import text as sa_text

from backend.database import async_session

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _slug(raw: str, max_len: int = 200) -> str:
    s = re.sub(r"\s+", " ", raw or "").strip()
    s = re.sub(r"[^\w\s\-\.]", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s[:max_len] or "Untitled"


def _build_candidates_sql(venue: str | None, year: int | None, strategy: str) -> tuple[str, dict]:
    where = ["vp.pdf_object_key <> ''",
             "vp.title_normalized NOT IN (SELECT lower(p.title) FROM papers p WHERE p.title IS NOT NULL)"]
    params: dict = {"limit": 0}
    if venue is not None:
        where.append("vp.venue = :venue")
        params["venue"] = venue
    if year is not None:
        where.append("vp.year = :year")
        params["year"] = year
    if strategy == "small":
        order = "vp.pdf_size_bytes ASC NULLS LAST"
    elif strategy == "recent":
        order = "vp.pdf_downloaded_at DESC NULLS LAST"
    else:
        order = "random()"
    sql = f"""
        SELECT vp.id, vp.title, vp.title_normalized, vp.venue, vp.year,
               vp.conf_year, vp.pdf_object_key, vp.pdf_size_bytes,
               vp.arxiv_id, vp.doi, vp.openreview_forum_id,
               vp.acceptance_type, vp.authors, vp.pdf_url
        FROM venue_papers vp
        WHERE {' AND '.join(where)}
        ORDER BY {order}
        LIMIT :limit
    """
    return sql, params


SQL_INSERT_PAPER = sa_text("""
    INSERT INTO papers (
      id, title, title_sanitized, venue, year, category, state,
      pdf_object_key, arxiv_id, doi, source, source_ref,
      acceptance_type, authors, source_quality, created_at, updated_at
    )
    VALUES (
      :id, :title, :title_sanitized, :venue, :year, :category, :state,
      :pdf_object_key, :arxiv_id, :doi, :source, :source_ref,
      :acceptance_type, CAST(:authors AS jsonb), :source_quality, now(), now()
    )
    ON CONFLICT DO NOTHING
""")


async def main(venue: str | None, year: int | None, limit: int,
               strategy: str, dry_run: bool) -> None:
    async with async_session() as session:
        sql, params = _build_candidates_sql(venue, year, strategy)
        params["limit"] = limit
        rows = (await session.execute(sa_text(sql), params)).fetchall()
        logger.info("Found %d candidates (venue=%s year=%s strategy=%s, dry=%s)",
                    len(rows), venue, year, strategy, dry_run)
        if not rows:
            return

        promoted_ids: list[str] = []
        for r in rows:
            pid = uuid.uuid4()
            payload = {
                "id": str(pid),
                "title": r.title,
                "title_sanitized": _slug(r.title),
                "venue": r.venue,
                "year": r.year,
                # category = venue+year so the paper joins the venue page in
                # the vault export and never falls into a placeholder bucket.
                # `__pending__` was the old placeholder which created a single
                # mega-node connecting to dozens of papers — see fix #11.
                "category": (r.venue or "Unknown") if r.venue else "Unknown",
                # `enriched` = metadata done but not yet parsed; ingest_workflow.deep_ingest
                # uses pdf_object_key directly so we don't need to re-download.
                "state": "enriched",
                "pdf_object_key": r.pdf_object_key,
                "arxiv_id": r.arxiv_id,
                "doi": r.doi,
                "source": "venue_papers",
                "source_ref": str(r.id),
                "acceptance_type": r.acceptance_type or None,
                "authors": _json_dumps(r.authors),
                "source_quality": "normal",
            }
            if dry_run:
                logger.info("DRY: would promote %s — %s",
                            payload["id"][:8], (r.title or "")[:80])
                continue
            await session.execute(SQL_INSERT_PAPER, payload)
            promoted_ids.append(payload["id"])

        if not dry_run:
            await session.commit()
            logger.info("Promoted %d papers — paper_ids:", len(promoted_ids))
            for pid in promoted_ids:
                print(pid)


def _json_dumps(v):
    import json
    if v is None:
        return "[]"
    if isinstance(v, str):
        # already serialized?
        try:
            json.loads(v)
            return v
        except Exception:
            return json.dumps([v], ensure_ascii=False)
    return json.dumps(v, ensure_ascii=False, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--strategy", choices=["small", "random", "recent"],
                        default="small")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.venue, args.year, args.limit, args.strategy, args.dry_run))
