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

                # Build S2 identifier — only ARXIV: and DOI: are reliably
                # supported by the S2 batch API. URL: format returns null
                # for most papers, causing 400 "No valid paper ids" errors.
                if arxiv_id:
                    s2_id = f"ARXIV:{re.sub(r'v[0-9]+$', '', arxiv_id)}"
                elif doi:
                    s2_id = f"DOI:{doi}"
                else:
                    continue  # forum_id alone can't be used with S2 batch

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


# ── Strategy 5b: OpenReview API → arxiv_id extraction ────────────

async def supplement_arxiv_id_from_openreview(
    session: AsyncSession,
    conf_years: list[str] | None = None,
) -> dict:
    """Extract arxiv_id from OpenReview API notes for papers that have forum_id.

    OpenReview notes contain arxiv references in multiple places:
      - content.pdf.value may be an arxiv URL (/pdf/XXXX.XXXXX)
      - content._bibtex.value may contain arxiv ID
      - the forum page itself may link to arxiv

    This strategy fetches all notes for each conf_year group, extracts
    arxiv_id, and matches by forum_id (not title — avoids normalization issues).
    """
    from backend.services.venue_index.normalize import extract_arxiv_id

    targets = conf_years or list(_OPENREVIEW_GROUPS.keys())
    total_updated = 0
    stats_per_venue: dict[str, int] = {}

    for conf_year in targets:
        if conf_year not in _OPENREVIEW_GROUPS:
            continue

        missing = (await session.execute(text("""
            SELECT count(*) FROM venue_papers
            WHERE conf_year = :cy AND (arxiv_id = '' OR arxiv_id IS NULL)
              AND openreview_forum_id != ''
        """), {"cy": conf_year})).scalar()

        if not missing:
            logger.info(f"[supplement] OR arxiv_id: {conf_year} — none missing, skip")
            continue

        group, venue_prefixes = _OPENREVIEW_GROUPS[conf_year]
        logger.info(f"[supplement] OR arxiv_id: crawling {conf_year} ({missing} missing)")

        # Fetch notes from OR API — collect forum_id → arxiv_id mapping
        or_arxiv: dict[str, str] = {}  # forum_id → arxiv_id
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
                        logger.warning(f"[supplement] OR arxiv_id {resp.status_code} for {group}")
                        break

                    data = resp.json()
                    notes = data.get("notes", [])
                    total_count = data.get("count", 0)

                    for note in notes:
                        content = note.get("content", {})
                        forum_id = note.get("forum", "") or note.get("id", "")
                        if not forum_id:
                            continue

                        # Try extracting arxiv_id from multiple fields
                        arxiv_id = ""

                        # 1. content.pdf.value — sometimes points to arxiv
                        pdf_val = str(content.get("pdf", {}).get("value", ""))
                        if "arxiv" in pdf_val.lower():
                            arxiv_id = extract_arxiv_id(pdf_val)

                        # 2. content._bibtex.value
                        if not arxiv_id:
                            bibtex = str(content.get("_bibtex", {}).get("value", ""))
                            m = re.search(r"arxiv[.:]\s*(\d{4}\.\d{4,5})", bibtex, re.I)
                            if m:
                                arxiv_id = m.group(1)

                        # 3. content.code.value — sometimes contains arxiv URL
                        if not arxiv_id:
                            code_val = str(content.get("code", {}).get("value", ""))
                            arxiv_id = extract_arxiv_id(code_val)

                        # 4. content.venue or paper URL — unlikely but check
                        if not arxiv_id:
                            paper_url = str(content.get("pdf", {}).get("value", ""))
                            if paper_url.startswith("/pdf/"):
                                # OR internal PDF, not arxiv — skip
                                pass

                        if arxiv_id and forum_id:
                            # Strip version suffix
                            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
                            or_arxiv[forum_id] = arxiv_id

                    offset += len(notes)
                    if not notes or offset >= total_count:
                        break
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.warning(f"[supplement] OR arxiv_id error: {e}")
                    break

        logger.info(f"[supplement] OR arxiv_id: fetched {len(or_arxiv)} papers with arxiv from {conf_year}")

        if not or_arxiv:
            stats_per_venue[conf_year] = 0
            continue

        # Update venue_papers by forum_id match
        rows = (await session.execute(text("""
            SELECT id, openreview_forum_id FROM venue_papers
            WHERE conf_year = :cy AND (arxiv_id = '' OR arxiv_id IS NULL)
              AND openreview_forum_id != ''
        """), {"cy": conf_year})).fetchall()

        updated = 0
        for vp_id, forum_id in rows:
            aid = or_arxiv.get(forum_id)
            if not aid:
                continue
            await session.execute(
                text("UPDATE venue_papers SET arxiv_id = :aid, updated_at = now() WHERE id = :id"),
                {"aid": aid[:30], "id": vp_id},
            )
            updated += 1

        stats_per_venue[conf_year] = updated
        total_updated += updated
        await session.flush()
        logger.info(f"[supplement] OR arxiv_id: {conf_year} updated {updated}/{len(rows)}")

    logger.info(f"[supplement] OR arxiv_id: total updated {total_updated}")
    return {"strategy": "or_arxiv_id", "total_updated": total_updated, "per_venue": stats_per_venue}


# ── Strategy 5c: S2 single-paper search → arxiv_id for non-OR venues ─

async def supplement_arxiv_id_from_s2_title_search(
    session: AsyncSession,
    conf_years: list[str] | None = None,
    max_papers: int = 5000,
) -> dict:
    """Fill arxiv_id for papers that have no forum_id/doi — use S2 title search.

    Targets: CVPR, ICCV, ECCV rows where arxiv_id is empty and no OR forum_id.
    Uses S2 single-paper search (slower but works for any paper).
    Rate: ~1 req/s with API key.
    """
    from backend.services.venue_index.normalize import normalize_title

    where_clause = "(arxiv_id = '' OR arxiv_id IS NULL) AND openreview_forum_id = '' AND doi = ''"
    params: dict = {}
    if conf_years:
        where_clause += " AND conf_year = ANY(:conf_years)"
        params["conf_years"] = conf_years

    total = (await session.execute(
        text(f"SELECT count(*) FROM venue_papers WHERE {where_clause}"), params
    )).scalar()
    if max_papers > 0:
        total = min(total, max_papers)
    logger.info(f"[supplement] S2 title search: {total} rows eligible")

    updated = 0
    offset = 0

    async with httpx.AsyncClient(timeout=15) as client:
        while offset < total:
            rows = (await session.execute(text(f"""
                SELECT id, title FROM venue_papers
                WHERE {where_clause}
                ORDER BY id LIMIT 100 OFFSET :offset
            """), {**params, "offset": offset})).fetchall()
            if not rows:
                break
            offset += len(rows)

            for vp_id, title in rows:
                if not title:
                    continue
                await limiters["s2"].acquire()
                try:
                    resp = await client.get(
                        "https://api.semanticscholar.org/graph/v1/paper/search/match",
                        params={"query": title[:200], "fields": "externalIds"},
                        headers=_s2_headers(),
                    )
                    if resp.status_code == 404:
                        continue
                    if resp.status_code == 429:
                        await asyncio.sleep(30)
                        continue
                    if resp.status_code != 200:
                        continue

                    data = resp.json().get("data", [])
                    if not data:
                        continue

                    ext = data[0].get("externalIds", {}) or {}
                    arxiv_id = (ext.get("ArXiv") or "").strip()
                    if arxiv_id:
                        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
                        await session.execute(
                            text("UPDATE venue_papers SET arxiv_id = :aid, updated_at = now() WHERE id = :id"),
                            {"aid": arxiv_id[:30], "id": vp_id},
                        )
                        updated += 1

                        # Also fill DOI if available
                        doi = (ext.get("DOI") or "").strip()
                        if doi:
                            await session.execute(
                                text("UPDATE venue_papers SET doi = CASE WHEN doi = '' THEN :doi ELSE doi END WHERE id = :id"),
                                {"doi": doi[:200], "id": vp_id},
                            )

                except Exception as e:
                    logger.warning(f"[supplement] S2 title search error: {e}")

            await session.flush()
            if offset % 500 == 0:
                logger.info(f"[supplement] S2 title search: {offset}/{total}, updated {updated}")

    logger.info(f"[supplement] S2 title search: done, updated {updated}/{total}")
    return {"strategy": "s2_title_search", "total": total, "updated": updated}


# ── Strategy 6: PWC dump → code_url ─────────────────────────────

async def supplement_via_pwc_dump(
    session: AsyncSession,
    conf_years: list[str] | None = None,
    batch_size: int = 1000,
) -> dict:
    """Fill code_url from Papers With Code local dump (zero API cost).

    Matches by arxiv_id first, then by normalized title.
    """
    from backend.utils.metadata_helpers import load_pwc_dump
    from backend.services.venue_index.normalize import normalize_title

    pwc_path = settings.pwc_dump_path
    if not pwc_path:
        logger.info("[supplement] PWC dump path not configured, skipping")
        return {"strategy": "pwc_dump", "skipped": True}

    links, arxiv_to_url, title_to_url = load_pwc_dump(pwc_path)
    if not links:
        logger.info("[supplement] PWC dump empty or not found")
        return {"strategy": "pwc_dump", "loaded": 0}

    where_clause = "code_url = ''"
    params: dict = {}
    if conf_years:
        where_clause += " AND conf_year = ANY(:conf_years)"
        params["conf_years"] = conf_years

    total = (await session.execute(
        text(f"SELECT count(*) FROM venue_papers WHERE {where_clause}"), params
    )).scalar()
    logger.info(f"[supplement] PWC: {total} rows missing code_url")

    updated = 0
    offset = 0
    while offset < total:
        rows = (await session.execute(text(f"""
            SELECT id, title, arxiv_id FROM venue_papers
            WHERE {where_clause}
            ORDER BY id LIMIT :limit OFFSET :offset
        """), {**params, "limit": batch_size, "offset": offset})).fetchall()
        if not rows:
            break
        offset += len(rows)

        for vp_id, title, arxiv_id in rows:
            paper_url = None
            # Try arxiv_id first
            if arxiv_id:
                clean_id = re.sub(r"v\d+$", "", arxiv_id)
                paper_url = arxiv_to_url.get(clean_id)
            # Try normalized title
            if not paper_url and title:
                paper_url = title_to_url.get(normalize_title(title))
            if not paper_url or paper_url not in links:
                continue
            repos = links[paper_url]
            # Prefer github.com
            code_url = next((r for r in repos if "github.com" in r), repos[0])
            await session.execute(
                text("UPDATE venue_papers SET code_url = :url, updated_at = now() WHERE id = :id"),
                {"url": code_url[:500], "id": vp_id},
            )
            updated += 1

        await session.flush()

    logger.info(f"[supplement] PWC: updated {updated}/{total} rows with code_url")
    return {"strategy": "pwc_dump", "total_missing": total, "updated": updated}


# ── Strategy 7: OpenReview API → keywords batch ─────────────────

async def supplement_openreview_keywords(
    session: AsyncSession,
    conf_years: list[str] | None = None,
) -> dict:
    """Batch-fill keywords from OpenReview API for papers that have forum_id but no keywords.

    Uses the same _OPENREVIEW_GROUPS mapping. Queries bulk OR notes.
    """
    from backend.services.venue_index.normalize import normalize_title

    targets = conf_years or list(_OPENREVIEW_GROUPS.keys())
    total_updated = 0
    stats_per_venue: dict[str, int] = {}

    for conf_year in targets:
        if conf_year not in _OPENREVIEW_GROUPS:
            continue

        missing = (await session.execute(text("""
            SELECT count(*) FROM venue_papers
            WHERE conf_year = :cy AND (keywords = '' OR keywords IS NULL)
              AND openreview_forum_id != ''
        """), {"cy": conf_year})).scalar()

        if not missing:
            logger.info(f"[supplement] OR keywords: {conf_year} — none missing, skip")
            continue

        group, venue_prefixes = _OPENREVIEW_GROUPS[conf_year]
        logger.info(f"[supplement] OR keywords: crawling {conf_year} ({missing} missing)")

        # Fetch from OR API with keywords field
        or_keywords: dict[str, str] = {}  # forum_id → keywords string
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
                        logger.warning(f"[supplement] OR keywords {resp.status_code} for {group}")
                        break

                    data = resp.json()
                    notes = data.get("notes", [])
                    total_count = data.get("count", 0)

                    for note in notes:
                        content = note.get("content", {})
                        forum_id = note.get("forum", "") or note.get("id", "")
                        kw_raw = content.get("keywords", {}).get("value", [])
                        if isinstance(kw_raw, list) and kw_raw and forum_id:
                            or_keywords[forum_id] = "; ".join(str(k) for k in kw_raw)

                    offset += len(notes)
                    if not notes or offset >= total_count:
                        break
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.warning(f"[supplement] OR keywords error: {e}")
                    break

        logger.info(f"[supplement] OR keywords: fetched {len(or_keywords)} papers with keywords from {conf_year}")

        if not or_keywords:
            continue

        # Update venue_papers by forum_id match
        rows = (await session.execute(text("""
            SELECT id, openreview_forum_id FROM venue_papers
            WHERE conf_year = :cy AND (keywords = '' OR keywords IS NULL)
              AND openreview_forum_id != ''
        """), {"cy": conf_year})).fetchall()

        updated = 0
        still_missing: list[tuple[int, str]] = []
        for vp_id, forum_id in rows:
            kw = or_keywords.get(forum_id)
            if not kw:
                still_missing.append((vp_id, forum_id))
                continue
            await session.execute(
                text("UPDATE venue_papers SET keywords = :kw, updated_at = now() WHERE id = :id"),
                {"kw": kw[:5000], "id": vp_id},
            )
            updated += 1

        # Fallback: direct per-note fetch for remaining rows (handles venues
        # where group search didn't return keywords, e.g. ICML_2024)
        if still_missing and len(still_missing) <= 5000:
            logger.info(f"[supplement] OR keywords: {conf_year} direct fetch for {len(still_missing)} remaining")
            async with httpx.AsyncClient(timeout=15) as client:
                for vp_id, forum_id in still_missing:
                    try:
                        resp = await client.get(
                            f"https://api2.openreview.net/notes?id={forum_id}",
                            headers={"User-Agent": "ResearchFlow/0.1"},
                        )
                        if resp.status_code != 200:
                            continue
                        notes = resp.json().get("notes", [])
                        if not notes:
                            continue
                        kw_raw = notes[0].get("content", {}).get("keywords", {}).get("value", [])
                        if isinstance(kw_raw, list) and kw_raw:
                            kw = "; ".join(str(k) for k in kw_raw)
                            await session.execute(
                                text("UPDATE venue_papers SET keywords = :kw, updated_at = now() WHERE id = :id"),
                                {"kw": kw[:5000], "id": vp_id},
                            )
                            updated += 1
                        await asyncio.sleep(0.3)  # polite rate for per-note
                    except Exception as e:
                        logger.warning(f"[supplement] OR keywords direct fetch error for {forum_id}: {e}")

        stats_per_venue[conf_year] = updated
        total_updated += updated
        await session.flush()

    logger.info(f"[supplement] OR keywords: total updated {total_updated}")
    return {"strategy": "openreview_keywords", "total_updated": total_updated, "per_venue": stats_per_venue}


# ── Strategy 8: OpenAlex → citation_count ────────────────────────

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


async def supplement_via_openalex_citations(
    session: AsyncSession,
    conf_years: list[str] | None = None,
    batch_size: int = 200,
    max_pages: int = 500,
) -> dict:
    """Batch-fill citation_count into extra_data via OpenAlex DOI lookup.

    Uses filter=doi:DOI1|DOI2|... with up to 50 DOIs per request.
    Only targets rows that have a DOI and no citation_count in extra_data.
    """
    where_clause = "doi != ''"
    params: dict = {}
    if conf_years:
        where_clause += " AND conf_year = ANY(:conf_years)"
        params["conf_years"] = conf_years

    total = (await session.execute(
        text(f"SELECT count(*) FROM venue_papers WHERE {where_clause}"), params
    )).scalar()
    logger.info(f"[supplement] OpenAlex citations: {total} rows with DOI")

    updated = 0
    offset = 0
    oa_batch_size = 50  # OpenAlex supports ~50 DOIs per filter

    # Read OpenAlex key if available
    oa_headers: dict[str, str] = {"User-Agent": "ResearchFlow/0.1 (mailto:researchflow@example.com)"}

    async with httpx.AsyncClient(timeout=30) as client:
        page = 0
        while offset < total and page < max_pages:
            rows = (await session.execute(text(f"""
                SELECT id, doi, extra_data FROM venue_papers
                WHERE {where_clause}
                ORDER BY id LIMIT :limit OFFSET :offset
            """), {**params, "limit": oa_batch_size, "offset": offset})).fetchall()

            if not rows:
                break
            offset += len(rows)
            page += 1

            # Filter out rows that already have citation_count
            need_rows = []
            for vp_id, doi, extra_data in rows:
                ed = extra_data or {}
                if "citation_count" not in ed:
                    need_rows.append((vp_id, doi))

            if not need_rows:
                continue

            # Build DOI filter
            doi_filter = "|".join(f"https://doi.org/{doi}" for _, doi in need_rows)

            try:
                resp = await client.get(
                    OPENALEX_WORKS_URL,
                    params={
                        "filter": f"doi:{doi_filter}",
                        "select": "doi,cited_by_count",
                        "per_page": oa_batch_size,
                    },
                    headers=oa_headers,
                )
                if resp.status_code != 200:
                    logger.warning(f"[supplement] OpenAlex HTTP {resp.status_code}")
                    await asyncio.sleep(1)
                    continue

                results = resp.json().get("results", [])
                doi_to_count: dict[str, int] = {}
                for w in results:
                    raw_doi = (w.get("doi") or "").replace("https://doi.org/", "")
                    if raw_doi:
                        doi_to_count[raw_doi.lower()] = w.get("cited_by_count", 0)

                for vp_id, doi in need_rows:
                    count = doi_to_count.get(doi.lower())
                    if count is not None:
                        await session.execute(
                            text("""
                                UPDATE venue_papers
                                SET extra_data = coalesce(extra_data, '{}'::jsonb) || jsonb_build_object('citation_count', CAST(:cc AS int)),
                                    updated_at = now()
                                WHERE id = :id
                            """),
                            {"cc": count, "id": vp_id},
                        )
                        updated += 1

            except Exception as e:
                logger.warning(f"[supplement] OpenAlex error: {e}")

            await session.flush()
            await asyncio.sleep(0.2)  # polite rate

    logger.info(f"[supplement] OpenAlex citations: updated {updated}/{total}")
    return {"strategy": "openalex_citations", "total_with_doi": total, "updated": updated}


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

    # Step 5b: OR API → arxiv_id (for papers with forum_id but no arxiv)
    results["or_arxiv_id"] = await supplement_arxiv_id_from_openreview(session, conf_years)
    await session.flush()

    # Step 6: Papers With Code dump → code_url (zero API)
    results["pwc_dump"] = await supplement_via_pwc_dump(session, conf_years)
    await session.flush()

    # Step 7: OpenReview keywords batch
    results["openreview_keywords"] = await supplement_openreview_keywords(session, conf_years)
    await session.flush()

    # Step 8: OpenAlex → citation_count
    results["openalex_citations"] = await supplement_via_openalex_citations(session, conf_years)
    await session.flush()

    return results
