"""Batch supplement missing fields in venue_papers.

Strategies:
  1. OpenReview forum_id → pdf_url (zero API cost, pure SQL)
  2. S2 batch API → arxiv_id, doi, oa_pdf_url (500 papers/request)
  3. CVF rule-based pdf_url construction (CVPR/ICCV)

All operations are idempotent — only fill empty fields, never overwrite.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.utils.api_clients import limiters

logger = logging.getLogger(__name__)

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_SIZE = 500


def _s2_headers() -> dict:
    h = {"User-Agent": "ResearchFlow/0.1"}
    if settings.s2_api_key:
        h["x-api-key"] = settings.s2_api_key
    return h


# ── Strategy 1: OpenReview forum_id → pdf_url ────────────────────

async def supplement_openreview_pdf_urls(session: AsyncSession) -> dict:
    """Set pdf_url from openreview_forum_id for rows that have forum_id but no pdf_url.

    Pattern: https://openreview.net/pdf?id={forum_id}
    Zero API calls — pure SQL update.
    """
    result = await session.execute(text("""
        UPDATE venue_papers
        SET pdf_url = 'https://openreview.net/pdf?id=' || openreview_forum_id,
            updated_at = now()
        WHERE pdf_url = ''
          AND openreview_forum_id != ''
    """))
    count = result.rowcount
    logger.info(f"[supplement] OpenReview pdf_url: updated {count} rows")
    return {"strategy": "openreview_pdf_url", "updated": count}


# ── Strategy 2: S2 batch → arxiv_id, doi, oa_pdf_url ────────────

async def supplement_via_s2_batch(
    session: AsyncSession,
    conf_years: list[str] | None = None,
    batch_size: int = S2_BATCH_SIZE,
    max_batches: int = 200,
) -> dict:
    """Batch-fetch arxiv_id, doi, oa_pdf_url from Semantic Scholar.

    Builds S2 IDs from existing data:
      - Has openreview_forum_id → URL:https://openreview.net/forum?id={id}
      - Has arxiv_id already → ARXIV:{id} (for doi/pdf supplement)
      - Otherwise → skip (title search is unreliable for batch)

    Only processes rows where at least one target field is empty.
    """
    # Build filter — only rows that have an S2-resolvable identifier
    where_clause = """(arxiv_id = '' OR doi = '' OR pdf_url = '')
        AND (openreview_forum_id != '' OR arxiv_id != '')"""
    params: dict = {}
    if conf_years:
        where_clause += " AND conf_year = ANY(:conf_years)"
        params["conf_years"] = conf_years

    # Count eligible rows
    total_row = (await session.execute(
        text(f"SELECT count(*) FROM venue_papers WHERE {where_clause}"), params
    )).scalar()
    logger.info(f"[supplement] S2 batch: {total_row} rows eligible (have forum_id or arxiv_id)")

    stats = {"batches_sent": 0, "papers_matched": 0, "arxiv_id": 0, "doi": 0, "pdf_url": 0}
    offset = 0

    async with httpx.AsyncClient(timeout=30) as client:
        for batch_num in range(max_batches):
            # Fetch a batch of rows needing supplement
            rows = (await session.execute(text(f"""
                SELECT id, arxiv_id, doi, openreview_forum_id, pdf_url
                FROM venue_papers
                WHERE {where_clause}
                ORDER BY id
                LIMIT :limit OFFSET :offset
            """), {**params, "limit": batch_size, "offset": offset})).fetchall()

            if not rows:
                break
            offset += len(rows)

            # Build S2 IDs
            s2_ids = []
            id_map: dict[int, int] = {}  # index_in_batch → venue_papers.id
            row_map: dict[int, dict] = {}  # venue_papers.id → current fields

            for row in rows:
                vp_id = row[0]
                arxiv_id = row[1] or ""
                doi = row[2] or ""
                orf_id = row[3] or ""
                pdf_url = row[4] or ""
                row_map[vp_id] = {"arxiv_id": arxiv_id, "doi": doi, "pdf_url": pdf_url}

                # Build S2 identifier
                if arxiv_id:
                    s2_id = f"ARXIV:{re.sub(r'v[0-9]+$', '', arxiv_id)}"
                elif orf_id:
                    s2_id = f"URL:https://openreview.net/forum?id={orf_id}"
                else:
                    continue  # Can't build S2 ID without arxiv or openreview

                idx = len(s2_ids)
                s2_ids.append(s2_id)
                id_map[idx] = vp_id

            if not s2_ids:
                continue

            # Rate limit
            await limiters["s2"].acquire()

            # POST batch
            try:
                resp = await client.post(
                    S2_BATCH_URL,
                    json={"ids": s2_ids},
                    params={"fields": "externalIds,openAccessPdf"},
                    headers=_s2_headers(),
                    timeout=30,
                )
                if resp.status_code == 429:
                    logger.warning("[supplement] S2 rate limited, waiting 30s")
                    await asyncio.sleep(30)
                    continue
                if resp.status_code != 200:
                    logger.warning(f"[supplement] S2 batch HTTP {resp.status_code}: {resp.text[:200]}")
                    continue

                items = resp.json()
                stats["batches_sent"] += 1

                for idx, item in enumerate(items):
                    if item is None:
                        continue
                    vp_id = id_map.get(idx)
                    if vp_id is None:
                        continue

                    current = row_map[vp_id]
                    updates = []
                    update_params: dict = {"vp_id": vp_id}

                    ext = item.get("externalIds") or {}
                    if not current["arxiv_id"] and ext.get("ArXiv"):
                        updates.append("arxiv_id = :new_arxiv")
                        update_params["new_arxiv"] = ext["ArXiv"]
                        stats["arxiv_id"] += 1
                    if not current["doi"] and ext.get("DOI"):
                        updates.append("doi = :new_doi")
                        update_params["new_doi"] = ext["DOI"]
                        stats["doi"] += 1

                    oa = item.get("openAccessPdf") or {}
                    if not current["pdf_url"] and oa.get("url"):
                        updates.append("pdf_url = :new_pdf")
                        update_params["new_pdf"] = oa["url"][:500]
                        stats["pdf_url"] += 1

                    if updates:
                        updates.append("updated_at = now()")
                        await session.execute(
                            text(f"UPDATE venue_papers SET {', '.join(updates)} WHERE id = :vp_id"),
                            update_params,
                        )
                        stats["papers_matched"] += 1

            except Exception as e:
                logger.warning(f"[supplement] S2 batch error: {e}")

            # Commit after each batch for resumability
            await session.flush()

            if batch_num % 10 == 0:
                logger.info(f"[supplement] S2 batch progress: {batch_num + 1} batches, "
                            f"{stats['papers_matched']} matched")

    logger.info(f"[supplement] S2 batch done: {stats}")
    return {"strategy": "s2_batch", **stats}


# ── Strategy 3: CVF rule-based pdf_url ───────────────────────────

async def supplement_cvf_pdf_urls(session: AsyncSession) -> dict:
    """Construct pdf_url for CVF conferences (CVPR/ICCV/ECCV) from paper_link.

    CVF HTML page pattern:
      https://openaccess.thecvf.com/content/CVPR2025/html/Author_Title_CVPR_2025_paper.html
    Corresponding PDF:
      https://openaccess.thecvf.com/content/CVPR2025/papers/Author_Title_CVPR_2025_paper.pdf
    """
    result = await session.execute(text("""
        UPDATE venue_papers
        SET pdf_url = replace(replace(paper_link, '/html/', '/papers/'), '_paper.html', '_paper.pdf'),
            updated_at = now()
        WHERE pdf_url = ''
          AND paper_link LIKE '%openaccess.thecvf.com%/html/%_paper.html'
          AND venue IN ('CVPR', 'ICCV', 'ECCV')
    """))
    count = result.rowcount
    logger.info(f"[supplement] CVF pdf_url: updated {count} rows")
    return {"strategy": "cvf_pdf_url", "updated": count}


# ── Strategy 4: OpenReview API v2 → forum_id + pdf_url ───────────

# Venues whose accepted papers live on OpenReview
_OPENREVIEW_GROUPS = {
    "NeurIPS_2024": ("NeurIPS.cc/2024/Conference", ["NeurIPS 2024"]),
    "ICML_2024":    ("ICML.cc/2024/Conference",    ["ICML 2024"]),
    "NeurIPS_2025": ("NeurIPS.cc/2025/Conference", ["NeurIPS 2025"]),
    "ICML_2025":    ("ICML.cc/2025/Conference",    ["ICML 2025"]),
    "ICLR_2024":    ("ICLR.cc/2024/Conference",    ["ICLR 2024"]),
    "ICLR_2025":    ("ICLR.cc/2025/Conference",    ["ICLR 2025"]),
    "ICLR_2026":    ("ICLR.cc/2026/Conference",    ["ICLR 2026"]),
}


async def supplement_via_openreview_api(
    session: AsyncSession,
    conf_years: list[str] | None = None,
) -> dict:
    """Crawl OpenReview API v2 to fill forum_id and pdf_url for conferences
    that have accepted papers on OpenReview but are missing these fields.

    Flow:
      1. For each conf_year with missing forum_id, query OR API v2
      2. Build title→forum_id+pdf_url mapping from OR results
      3. Match against venue_papers by title_normalized
      4. UPDATE matched rows
    """
    from backend.services.venue_index.normalize import normalize_title

    targets = conf_years or list(_OPENREVIEW_GROUPS.keys())
    total_updated = 0
    stats_per_venue: dict[str, int] = {}

    for conf_year in targets:
        if conf_year not in _OPENREVIEW_GROUPS:
            continue

        # Check how many rows are missing forum_id
        missing = (await session.execute(text("""
            SELECT count(*) FROM venue_papers
            WHERE conf_year = :cy AND openreview_forum_id = ''
        """), {"cy": conf_year})).scalar()

        if not missing:
            logger.info(f"[supplement] OR API: {conf_year} — no rows missing forum_id, skip")
            continue

        group, venue_prefixes = _OPENREVIEW_GROUPS[conf_year]
        logger.info(f"[supplement] OR API: crawling {conf_year} (group={group}, {missing} rows missing)")

        # Paginate through OpenReview API
        or_papers: dict[str, dict] = {}  # title_normalized → {forum_id, pdf_url}
        offset = 0
        page_size = 1000

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                try:
                    resp = await client.get(
                        "https://api2.openreview.net/notes/search",
                        params={"query": "*", "group": group, "limit": page_size, "offset": offset},
                        headers={"User-Agent": "ResearchFlow/0.1", "Accept": "application/json"},
                    )
                    if resp.status_code != 200:
                        logger.warning(f"[supplement] OR API {resp.status_code} for {group}")
                        break

                    data = resp.json()
                    notes = data.get("notes", [])
                    total_count = data.get("count", 0)

                    if offset == 0:
                        logger.info(f"[supplement] OR API: {conf_year} total={total_count}")

                    for note in notes:
                        content = note.get("content", {})
                        venue_val = content.get("venue", {}).get("value", "")
                        # Filter accepted papers
                        if not any(venue_val.lower().startswith(p.lower()) for p in venue_prefixes):
                            continue

                        title_raw = content.get("title", {}).get("value", "")
                        forum_id = note.get("forum", "") or note.get("id", "")
                        pdf_path = content.get("pdf", {}).get("value", "")
                        pdf_url = f"https://openreview.net{pdf_path}" if pdf_path else ""

                        tn = normalize_title(title_raw)
                        if tn and forum_id:
                            or_papers[tn] = {"forum_id": forum_id, "pdf_url": pdf_url}

                    offset += len(notes)
                    if not notes or offset >= total_count:
                        break
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.warning(f"[supplement] OR API error for {conf_year}: {e}")
                    break

        logger.info(f"[supplement] OR API: {conf_year} fetched {len(or_papers)} accepted papers from OpenReview")

        if not or_papers:
            continue

        # Match and update venue_papers by title_normalized
        rows = (await session.execute(text("""
            SELECT id, title_normalized FROM venue_papers
            WHERE conf_year = :cy AND openreview_forum_id = ''
        """), {"cy": conf_year})).fetchall()

        updated = 0
        for row_id, title_norm in rows:
            match = or_papers.get(title_norm)
            if not match:
                continue

            updates = ["openreview_forum_id = :fid", "updated_at = now()"]
            params: dict = {"vp_id": row_id, "fid": match["forum_id"]}

            if match["pdf_url"]:
                updates.append("pdf_url = CASE WHEN pdf_url = '' THEN :purl ELSE pdf_url END")
                params["purl"] = match["pdf_url"][:500]

            await session.execute(
                text(f"UPDATE venue_papers SET {', '.join(updates)} WHERE id = :vp_id"),
                params,
            )
            updated += 1

        stats_per_venue[conf_year] = updated
        total_updated += updated
        logger.info(f"[supplement] OR API: {conf_year} matched {updated}/{len(rows)} rows")
        await session.flush()

    return {"strategy": "openreview_api", "total_updated": total_updated, "per_venue": stats_per_venue}


# ── Strategy 5: Re-crawl CVF OpenAccess for ICCV pdf_url ────────

async def supplement_iccv_cvf_pdf_urls(session: AsyncSession) -> dict:
    """Crawl CVF OpenAccess HTML to extract pdf_url for ICCV_2025 papers.

    The VC JSON primary source had 0% pdf_url, but CVF HTML has direct PDF links.
    """
    from backend.services.venue_index.normalize import normalize_title

    url = "https://openaccess.thecvf.com/ICCV2025?day=all"

    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        try:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            logger.error(f"[supplement] CVF ICCV_2025 fetch failed: {e}")
            return {"strategy": "iccv_cvf", "error": str(e)}

    # Parse pdf_url + arxiv_id from CVF HTML
    import re as _re
    block_re = _re.compile(
        r"<dt class=\"ptitle\"><br><a href=\"([^\"]+)\">([^<]+)</a></dt>"
        r"\s*<dd>(.*?)</dd>"
        r"\s*<dd>(.*?)</dd>",
        _re.I | _re.S,
    )
    OPENACCESS_BASE = "https://openaccess.thecvf.com"
    cvf_papers: dict[str, dict] = {}  # title_norm → {pdf_url, arxiv_id}

    import html as html_mod
    for href, raw_title, _, dd_links in block_re.findall(html):
        title = html_mod.unescape(raw_title).strip()
        tn = normalize_title(title)
        if not tn:
            continue

        # Extract PDF URL
        pdf_m = _re.search(r'href="(/content/ICCV\d{4}/papers/[^"]+\.pdf)"', dd_links)
        pdf_url = f"{OPENACCESS_BASE}{pdf_m.group(1)}" if pdf_m else ""

        # Extract arxiv_id
        arxiv_m = _re.search(r'href="https?://arxiv\.org/abs/([^"]+)"', dd_links)
        arxiv_id = arxiv_m.group(1).strip() if arxiv_m else ""
        if arxiv_id:
            arxiv_id = _re.sub(r"v\d+$", "", arxiv_id)

        if pdf_url or arxiv_id:
            cvf_papers[tn] = {"pdf_url": pdf_url, "arxiv_id": arxiv_id}

    logger.info(f"[supplement] CVF ICCV_2025: parsed {len(cvf_papers)} papers from HTML")

    if not cvf_papers:
        return {"strategy": "iccv_cvf", "parsed": 0, "updated": 0}

    # Match and update
    rows = (await session.execute(text("""
        SELECT id, title_normalized FROM venue_papers
        WHERE conf_year = 'ICCV_2025' AND (pdf_url = '' OR arxiv_id = '')
    """))).fetchall()

    updated = 0
    for row_id, title_norm in rows:
        match = cvf_papers.get(title_norm)
        if not match:
            continue

        updates = ["updated_at = now()"]
        params: dict = {"vp_id": row_id}
        if match.get("pdf_url"):
            updates.append("pdf_url = CASE WHEN pdf_url = '' THEN :purl ELSE pdf_url END")
            params["purl"] = match["pdf_url"][:500]
        if match.get("arxiv_id"):
            updates.append("arxiv_id = CASE WHEN arxiv_id = '' THEN :aid ELSE arxiv_id END")
            params["aid"] = match["arxiv_id"][:30]

        if len(updates) > 1:
            await session.execute(
                text(f"UPDATE venue_papers SET {', '.join(updates)} WHERE id = :vp_id"),
                params,
            )
            updated += 1

    logger.info(f"[supplement] CVF ICCV_2025: updated {updated}/{len(rows)} rows")
    await session.flush()
    return {"strategy": "iccv_cvf", "parsed": len(cvf_papers), "updated": updated}


# ── Orchestrator ─────────────────────────────────────────────────

async def run_full_supplement(
    session: AsyncSession,
    conf_years: list[str] | None = None,
) -> dict:
    """Run all supplement strategies in order. Returns combined stats."""
    results = {}

    # Step 1: OpenReview forum_id → pdf_url (instant, zero API)
    results["openreview_sql"] = await supplement_openreview_pdf_urls(session)
    await session.flush()

    # Step 2: Crawl OpenReview API for venues missing forum_id
    results["openreview_api"] = await supplement_via_openreview_api(session, conf_years)
    await session.flush()

    # Step 2b: Now that forum_id is filled, re-run pdf_url from forum_id
    results["openreview_sql_2"] = await supplement_openreview_pdf_urls(session)
    await session.flush()

    # Step 3: CVF rule-based pdf_url
    results["cvf_rule"] = await supplement_cvf_pdf_urls(session)
    await session.flush()

    # Step 4: ICCV CVF HTML crawl for pdf_url
    results["iccv_cvf"] = await supplement_iccv_cvf_pdf_urls(session)
    await session.flush()

    # Step 5: S2 batch (arxiv_id + doi + fallback pdf_url)
    results["s2_batch"] = await supplement_via_s2_batch(session, conf_years)
    await session.flush()

    return results
