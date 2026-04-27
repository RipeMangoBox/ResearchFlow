"""Baseline-priority paper promotion.

When `reference_role` agent identifies a paper's direct_baseline /
comparison_baseline / method_source / formula_source references, this service
makes sure each such baseline ends up either:

  (a) already in `papers` table (then `paper_relation_service` builds the edge), or
  (b) in `paper_candidates` with a high-priority flag, ready to be promoted, or
  (c) explicitly skipped with a logged reason (true unfindable).

Search ladder (per missing baseline):
  1. Title fuzzy-match against `papers`.            (cheap, in-process)
  2. Title fuzzy-match against `venue_papers`.       (broader pool, in-process)
  3. Semantic Scholar `find_paper_on_s2(title)`.    (network)
  4. arXiv search-by-title.                          (network)
  5. OpenReview search by title.                    (network, last resort)

When (3-5) succeed we create a candidate and (optionally) enqueue an arq
`task_pipeline_run` so the baseline gets fully ingested (recursively, one
level deep — we cap recursion to avoid explosion).

Public entry: `promote_for_paper(session, paper_id, *, max_promote=8,
recursive=False)`.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.agent import AgentBlackboardItem
from backend.models.paper import Paper
from backend.services import candidate_service
from backend.services.paper_relation_service import _tokens, _jaccard

logger = logging.getLogger(__name__)


# Roles we will actively pursue ingest for. Order matters — direct_baseline
# is highest priority; same_task_prior_work is excluded (too speculative).
PRIORITY_ROLES = ("direct_baseline", "method_source", "formula_source",
                  "comparison_baseline", "benchmark_source", "dataset_source")


# ── Local lookup helpers ────────────────────────────────────────────────

async def _papers_index(session: AsyncSession) -> list[tuple[UUID, str, set[str]]]:
    rows = (await session.execute(
        select(Paper.id, Paper.title)
        .where(Paper.title.isnot(None))
    )).all()
    return [(pid, t, _tokens(t)) for pid, t in rows if len(_tokens(t)) >= 3]


async def _venue_papers_index(session: AsyncSession, conf_filter: list[str] | None = None
                              ) -> list[tuple[str, str, set[str], dict]]:
    """Returns (venue_paper_id, title, tokens, meta) tuples.
    Meta carries arxiv_id / pdf_object_key / openreview_forum_id so we can
    promote with full metadata in one shot.
    """
    sql = """
        SELECT id, title, arxiv_id, doi, pdf_object_key,
               openreview_forum_id, venue, year, conf_year
        FROM venue_papers
    """
    if conf_filter:
        sql += " WHERE conf_year = ANY(:cf)"
    rows = (await session.execute(sa_text(sql),
                                  {"cf": conf_filter} if conf_filter else {})).all()
    out = []
    for r in rows:
        if not r.title:
            continue
        tok = _tokens(r.title)
        if len(tok) < 3:
            continue
        meta = {
            "arxiv_id": r.arxiv_id, "doi": r.doi,
            "pdf_object_key": r.pdf_object_key,
            "openreview_forum_id": r.openreview_forum_id,
            "venue": r.venue, "year": r.year, "conf_year": r.conf_year,
        }
        out.append((str(r.id), r.title, tok, meta))
    return out


def _best_match(ref_title: str, index: list, threshold: float = 0.65) -> tuple | None:
    """Best-fit Jaccard match in (id, title, tokens, ...) tuples.

    Two-stage: (1) substring fast-path — if the ref tokens are a strict subset
    of an indexed title's tokens AND >=4 tokens overlap, we accept it
    immediately (handles ICLR-style short titles like "GRPO" appearing inside
    "Group Relative Policy Optimization (GRPO) for ..."). (2) Jaccard >= 0.65.
    """
    ref_tok = _tokens(ref_title)
    if len(ref_tok) < 3:
        return None
    best, best_score = None, 0.0
    for entry in index:
        idx_tok = entry[2]
        # 1. Subset fast-path — covers acronym + truncation cases
        if len(ref_tok) >= 4 and ref_tok.issubset(idx_tok):
            return entry + (1.0,)
        s = _jaccard(ref_tok, idx_tok)
        if s > best_score:
            best, best_score = entry, s
    if best_score >= threshold:
        return best + (best_score,)
    return None


# ── External search ────────────────────────────────────────────────────

async def _search_s2(client: httpx.AsyncClient, title: str) -> dict | None:
    try:
        from backend.services.discovery_service import find_paper_on_s2
        return await find_paper_on_s2(client, title)
    except Exception as e:
        logger.debug("S2 lookup failed for %r: %s", title[:50], e)
        return None


async def _search_arxiv(client: httpx.AsyncClient, title: str) -> dict | None:
    """Query arxiv API for a paper by title — returns first match dict."""
    try:
        # arxiv API: search_query uses key:value with quoted phrases for title
        # Strip punctuation for safety
        q = re.sub(r"[^\w\s]", " ", title).strip()
        url = "https://export.arxiv.org/api/query"
        r = await client.get(url, params={
            "search_query": f"ti:\"{q[:200]}\"",
            "max_results": "3",
        }, timeout=15)
        if r.status_code != 200:
            return None
        # Cheap parsing — extract first <entry><id>
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.find("atom:id", ns).text.rsplit("/", 1)[-1]
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
            etitle = (entry.find("atom:title", ns).text or "").strip()
            if _jaccard(_tokens(title), _tokens(etitle)) >= 0.7:
                return {
                    "title": etitle,
                    "arxiv_id": arxiv_id,
                    "paper_link": f"https://arxiv.org/abs/{arxiv_id}",
                }
        return None
    except Exception as e:
        logger.debug("arxiv lookup failed for %r: %s", title[:50], e)
        return None


async def _search_openreview(title: str) -> dict | None:
    """Use openreview-py SDK to search by title (anonymous)."""
    try:
        import openreview
        # Anonymous client — sufficient for searching public notes
        c = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")
        notes = c.search_notes(term=title[:200], type="all", limit=3)
        for n in notes or []:
            ntitle = (n.content or {}).get("title", {})
            if isinstance(ntitle, dict):
                ntitle = ntitle.get("value", "")
            if _jaccard(_tokens(title), _tokens(str(ntitle))) >= 0.7:
                return {
                    "title": str(ntitle),
                    "openreview_id": n.id,
                    "paper_link": f"https://openreview.net/forum?id={n.id}",
                }
        return None
    except Exception as e:
        logger.debug("OpenReview lookup failed for %r: %s", title[:50], e)
        return None


# ── arq enqueue helper ─────────────────────────────────────────────────

async def _enqueue_pipeline_run(paper_id: UUID) -> bool:
    """Best-effort enqueue of task_pipeline_run via arq. Silent on failure."""
    try:
        from arq import create_pool
        from backend.workers.arq_app import _parse_redis_url
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        pool = await create_pool(_parse_redis_url(redis_url))
        await pool.enqueue_job("task_pipeline_run", str(paper_id))
        await pool.close()
        return True
    except Exception as e:
        logger.warning("arq enqueue failed for %s: %s", paper_id, e)
        return False


# ── Promote a venue_paper into papers (cheap, no re-download) ───────────

async def _promote_venue_paper(session: AsyncSession, vp_meta: dict, raw_title: str
                               ) -> UUID | None:
    """Insert a single venue_papers row into papers using its OSS PDF.
    Returns the new paper_id, or None on duplicate / failure.
    """
    import uuid as _uuid
    from backend.models.paper import Paper as _Paper
    pid = _uuid.uuid4()
    try:
        # Build a minimal sanitized title — keep it short and filesystem-safe
        sanitized = re.sub(r"\s+", "_",
                           re.sub(r"[^\w\s\-\.]", "_", raw_title or "")).strip("_")[:200]
        await session.execute(sa_text("""
            INSERT INTO papers (
              id, title, title_sanitized, venue, year, category, state,
              pdf_object_key, arxiv_id, doi, source, source_quality,
              created_at, updated_at
            ) VALUES (
              :id, :title, :sanit, :venue, :year, :cat, 'enriched',
              :pok, :arxiv, :doi, 'baseline_promoter', 'normal',
              now(), now()
            )
            ON CONFLICT DO NOTHING
        """), {
            "id": str(pid),
            "title": raw_title,
            "sanit": sanitized,
            "venue": vp_meta.get("venue"),
            "year": vp_meta.get("year"),
            "cat": "__pending__",
            "pok": vp_meta.get("pdf_object_key"),
            "arxiv": vp_meta.get("arxiv_id"),
            "doi": vp_meta.get("doi"),
        })
        return pid
    except Exception as e:
        logger.warning("venue→paper promote failed for %r: %s", raw_title[:50], e)
        return None


# ── Public API ─────────────────────────────────────────────────────────

async def promote_for_paper(
    session: AsyncSession,
    paper_id: UUID,
    *,
    max_promote: int = 8,
    auto_enqueue: bool = True,
) -> dict:
    """Walk this paper's reference_role_map; ensure each baseline ref is
    either already in `papers`, queued in `paper_candidates`, or has been
    promoted (with full pipeline enqueued)."""
    rr = (await session.execute(
        select(AgentBlackboardItem)
        .where(
            AgentBlackboardItem.paper_id == paper_id,
            AgentBlackboardItem.item_type == "reference_role_map",
        )
        .order_by(AgentBlackboardItem.created_at.desc())
        .limit(1)
    )).scalar_one_or_none()
    if not rr or not rr.value_json:
        return {"paper_id": str(paper_id), "skipped": "no_reference_role_map"}

    classifications = (rr.value_json.get("classifications") or [])
    # Only pursue priority roles, sort baseline first
    targets = [c for c in classifications if c.get("role") in PRIORITY_ROLES]
    targets.sort(key=lambda c: PRIORITY_ROLES.index(c.get("role", PRIORITY_ROLES[-1])))
    targets = [c for c in targets if (c.get("ref_title") or "").strip()][:max_promote]

    if not targets:
        return {"paper_id": str(paper_id), "skipped": "no_priority_targets",
                "total_classifications": len(classifications)}

    # Build lookup indexes once
    p_index = await _papers_index(session)
    vp_index = await _venue_papers_index(session)
    logger.info("baseline_promoter: %d targets, papers_idx=%d, vp_idx=%d",
                len(targets), len(p_index), len(vp_index))

    stats = {"already_in_papers": 0, "promoted_from_venue": 0,
             "candidate_created": 0, "queued": 0, "not_found": 0}
    enqueued: list[UUID] = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
        for c in targets:
            ref_title = c["ref_title"].strip()

            # 1. Already in papers?
            m = _best_match(ref_title, p_index, threshold=0.7)
            if m:
                stats["already_in_papers"] += 1
                continue

            # 2. In venue_papers? Promote inline (cheapest pipeline trigger)
            m = _best_match(ref_title, vp_index, threshold=0.7)
            if m:
                vp_id, vp_title, _tok, vp_meta, _score = m
                pid = await _promote_venue_paper(session, vp_meta, vp_title)
                if pid:
                    stats["promoted_from_venue"] += 1
                    if auto_enqueue and len(enqueued) < max_promote:
                        if await _enqueue_pipeline_run(pid):
                            enqueued.append(pid)
                            stats["queued"] += 1
                continue

            # 3. Already a candidate?
            from backend.services.candidate_service import _normalize_title as _nt
            existing = await candidate_service.find_duplicate(
                session, normalized_title=_nt(ref_title)
            )
            if existing:
                continue

            # 4. External search ladder
            meta = await _search_s2(client, ref_title)
            source = "s2_baseline_lookup"
            if not meta:
                meta = await _search_arxiv(client, ref_title)
                source = "arxiv_baseline_lookup"
            if not meta:
                meta = await _search_openreview(ref_title)
                source = "openreview_baseline_lookup"

            if not meta:
                stats["not_found"] += 1
                continue

            # 5. Create paper_candidate. s2_paper_id / openreview_id ride along
            # in metadata_json since the API doesn't expose them as kwargs.
            try:
                extra = {}
                if meta.get("paper_id") or meta.get("paperId"):
                    extra["s2_paper_id"] = meta.get("paper_id") or meta.get("paperId")
                if meta.get("openreview_id"):
                    extra["openreview_id"] = meta["openreview_id"]
                await candidate_service.create_candidate(
                    session,
                    title=meta.get("title", ref_title),
                    discovery_source=source,
                    discovery_reason=c.get("role", "baseline"),
                    discovered_from_paper_id=paper_id,
                    relation_hint=c.get("role"),
                    arxiv_id=meta.get("arxiv_id"),
                    doi=meta.get("doi"),
                    paper_link=meta.get("paper_link") or meta.get("url"),
                    abstract=meta.get("abstract"),
                    venue=meta.get("venue"),
                    year=meta.get("year"),
                    metadata_json=extra or None,
                )
                stats["candidate_created"] += 1
                # Cron auto_promote will pick this up; we don't inline-enqueue
                # discovered candidates to bound load.
            except Exception as e:
                logger.warning("create_candidate failed for %r: %s", ref_title[:50], e)
                stats["not_found"] += 1

    await session.flush()
    stats["paper_id"] = str(paper_id)
    stats["enqueued_paper_ids"] = [str(p) for p in enqueued]
    return stats
