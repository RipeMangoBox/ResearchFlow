"""Generalized PDF download service — fetch from multiple sources, validate, store to OSS.

Handles: OpenReview, CVF OpenAccess, arXiv, generic OA URLs.
Proxy-aware: httpx reads HTTP_PROXY/HTTPS_PROXY from environment.
Quality checks: ≥50KB, %PDF- magic, retry with exponential backoff.

OSS key convention:
  papers/raw-pdf/{category}/{venue_year}/{title_sanitized}.pdf
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from uuid import UUID

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.models.enums import AssetType, PaperState
from backend.models.paper import Paper, PaperAsset

logger = logging.getLogger(__name__)

MIN_PDF_SIZE = 50 * 1024  # 50 KB — smaller is likely an error page
MAX_PDF_SIZE = 50 * 1024 * 1024  # 50 MB — larger is likely a wrong file
MAX_RETRIES = 3


# ── PDF URL resolution ───────────────────────────────────────────

async def resolve_best_pdf_url(session: AsyncSession, paper: Paper) -> str | None:
    """Determine the best PDF download URL for a paper.

    Priority:
      1. Existing pdf_url from venue_papers lookup
      2. OpenReview: https://openreview.net/pdf?id={forum_id}
      3. arXiv: https://arxiv.org/pdf/{arxiv_id}.pdf
      4. venue_papers.pdf_url (secondary lookup by title)
    """
    # 1. Try paper_link for OpenReview forum_id
    link = paper.paper_link or ""
    m = re.search(r'openreview\.net/(?:forum|pdf)\?id=([a-zA-Z0-9_-]+)', link)
    if m:
        return f"https://openreview.net/pdf?id={m.group(1)}"

    # 2. arXiv
    if paper.arxiv_id:
        base_id = re.sub(r"v\d+$", "", paper.arxiv_id)
        return f"https://arxiv.org/pdf/{base_id}.pdf"

    # 3. Lookup from venue_papers
    from backend.services.venue_index.service import lookup_paper
    vi = await lookup_paper(
        session, title=paper.title or "", arxiv_id=paper.arxiv_id or "", doi=paper.doi or ""
    )
    if vi and vi.get("pdf_url"):
        return vi["pdf_url"]

    return None


# ── Download + validate ──────────────────────────────────────────

async def _download_with_retry(
    client: httpx.AsyncClient,
    url: str,
    max_retries: int = MAX_RETRIES,
) -> bytes | None:
    """Download URL with retry and exponential backoff. Returns bytes or None."""
    for attempt in range(max_retries):
        try:
            resp = await client.get(url)
            if resp.status_code == 429:
                wait = min(30 * (2 ** attempt), 120)
                logger.warning(f"PDF download 429 for {url}, waiting {wait}s")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.content
        except httpx.TimeoutException:
            logger.warning(f"PDF download timeout (attempt {attempt + 1}): {url}")
        except httpx.HTTPStatusError as e:
            logger.warning(f"PDF download HTTP {e.response.status_code} (attempt {attempt + 1}): {url}")
            if e.response.status_code in (403, 404):
                return None  # Don't retry client errors
        except Exception as e:
            logger.warning(f"PDF download error (attempt {attempt + 1}): {url}: {e}")

        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)

    return None


def _validate_pdf(data: bytes, url: str) -> bool:
    """Check that downloaded content is a valid PDF."""
    if len(data) < MIN_PDF_SIZE:
        logger.warning(f"PDF too small ({len(data)} bytes): {url}")
        return False
    if len(data) > MAX_PDF_SIZE:
        logger.warning(f"PDF too large ({len(data)} bytes): {url}")
        return False
    if data[:5] != b"%PDF-":
        logger.warning(f"PDF magic check failed: {url} (got {data[:20]!r})")
        return False
    return True


# ── Main download function ───────────────────────────────────────

async def download_pdf_to_oss(
    session: AsyncSession,
    paper_id: UUID,
    *,
    pdf_url: str | None = None,
    force: bool = False,
) -> bool:
    """Download PDF from best available URL, validate, and store to OSS.

    Steps:
      1. Resolve PDF URL (if not provided)
      2. Download with retry
      3. Validate: ≥50KB, %PDF- magic
      4. Upload to OSS
      5. Optionally save local copy
      6. Set paper.pdf_object_key, paper.state = DOWNLOADED
      7. Create PaperAsset record

    Returns True on success, False on failure.
    """
    paper = await session.get(Paper, paper_id)
    if not paper:
        logger.error(f"Paper {paper_id} not found")
        return False

    # Skip if already has PDF (unless forced)
    if not force and (paper.pdf_object_key or paper.pdf_path_local):
        return True

    # Resolve URL
    if not pdf_url:
        pdf_url = await resolve_best_pdf_url(session, paper)
    if not pdf_url:
        logger.warning(f"No PDF URL found for paper {paper_id} ({paper.title[:60]})")
        return False

    # Download
    async with httpx.AsyncClient(follow_redirects=True, timeout=180) as client:
        data = await _download_with_retry(client, pdf_url)

    if not data:
        logger.error(f"PDF download failed after retries: {pdf_url}")
        return False

    if not _validate_pdf(data, pdf_url):
        return False

    # Build storage key
    category = paper.category or "Uncategorized"
    venue_year = f"{paper.venue}_{paper.year}" if paper.venue and paper.year else "Unknown"
    filename = f"{paper.title_sanitized or paper.arxiv_id or str(paper_id)[:8]}.pdf"
    rel_path = f"{category}/{venue_year}/{filename}"
    object_key = f"papers/raw-pdf/{rel_path}"

    # Upload to OSS (primary storage — no local copy to save disk)
    from backend.services.object_storage import get_storage, compute_checksum
    storage = get_storage()
    try:
        await storage.put(object_key, data)
        paper.pdf_object_key = object_key
        logger.info(f"Uploaded PDF to OSS: {object_key} ({len(data)} bytes)")
    except Exception as e:
        logger.error(f"OSS upload failed for {paper_id}: {e}")
        return False

    # Set pdf_path_local to the OSS key so downstream code can resolve via storage.get_local_path()
    paper.pdf_path_local = rel_path

    # Update state
    paper.state = PaperState.DOWNLOADED

    # Create asset record
    checksum = compute_checksum(data)
    # Check if asset already exists
    existing = (await session.execute(
        text("SELECT id FROM paper_assets WHERE paper_id = :pid AND asset_type = :atype LIMIT 1"),
        {"pid": paper_id, "atype": AssetType.RAW_PDF.value},
    )).first()

    if not existing:
        asset = PaperAsset(
            paper_id=paper.id,
            asset_type=AssetType.RAW_PDF,
            object_key=object_key,
            mime_type="application/pdf",
            size_bytes=len(data),
            checksum=checksum,
        )
        session.add(asset)

    await session.flush()
    logger.info(f"PDF stored for paper {paper_id}: {object_key} ({len(data)} bytes, {pdf_url[:80]})")
    return True


# ── Venue papers bulk PDF download ─────────────────────────────

def _vp_object_key(conf_year: str, arxiv_id: str, title: str, vp_id: int) -> str:
    """Build OSS key for venue_papers PDF.

    Pattern: venue-papers/pdf/{conf_year}/{identifier}.pdf
    Identifier priority: arxiv_id > title_slug > id_hash
    """
    import hashlib

    if arxiv_id:
        ident = re.sub(r"v\d+$", "", arxiv_id).replace("/", "_")
    elif title:
        # Slugify: first 40 chars, ascii-safe
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", title[:60]).strip("_")[:40]
        hash_suffix = hashlib.md5(title.encode()).hexdigest()[:8]
        ident = f"{slug}_{hash_suffix}"
    else:
        ident = f"vp_{vp_id}"

    return f"venue-papers/pdf/{conf_year}/{ident}.pdf"


def _resolve_vp_pdf_urls(pdf_url: str, arxiv_id: str, openreview_forum_id: str, doi: str = "") -> list[str]:
    """Build a prioritized list of PDF download URLs for a venue_paper row.

    Returns multiple URLs for fallback — caller tries each in order.
    Priority:
      1. OpenReview PDF (most reliable for conference papers)
      2. arXiv PDF (fast via proxy)
      3. Original pdf_url (various sources — CVF, S2, etc.)
      4. Unpaywall via DOI (OA papers)
    """
    urls: list[str] = []

    if openreview_forum_id:
        urls.append(f"https://openreview.net/pdf?id={openreview_forum_id}")
    if arxiv_id:
        base_id = re.sub(r"v\d+$", "", arxiv_id)
        urls.append(f"https://arxiv.org/pdf/{base_id}.pdf")
    if pdf_url and pdf_url not in urls:
        urls.append(pdf_url)

    return urls


async def download_venue_papers_pdfs(
    session: AsyncSession,
    conf_years: list[str] | None = None,
    *,
    concurrency: int = 5,
    batch_size: int = 100,
    max_total: int = 0,
) -> dict:
    """Bulk-download PDFs for venue_papers → OSS.

    Features:
      - Resume from where we left off (WHERE pdf_object_key = '')
      - Concurrent downloads (default 5 workers)
      - Validates %PDF- magic + ≥50KB
      - Stores OSS key, size, checksum, timestamp back to DB
      - Never stores PDFs locally (OSS-primary)

    Returns download stats dict.
    """
    from backend.services.object_storage import get_storage, compute_checksum

    where_clause = "pdf_url != '' AND (pdf_object_key IS NULL OR pdf_object_key = '')"
    params: dict = {}
    if conf_years:
        where_clause += " AND conf_year = ANY(:conf_years)"
        params["conf_years"] = conf_years

    total = (await session.execute(
        text(f"SELECT count(*) FROM venue_papers WHERE {where_clause}"), params
    )).scalar()
    logger.info(f"[vp-pdf] {total} venue_papers pending PDF download")

    if max_total > 0:
        total = min(total, max_total)

    storage = get_storage()
    stats = {"total": total, "downloaded": 0, "failed": 0, "skipped": 0}
    sem = asyncio.Semaphore(concurrency)
    offset = 0

    async def _download_one(
        client: httpx.AsyncClient,
        vp_id: int,
        pdf_url: str,
        arxiv_id: str,
        orf_id: str,
        conf_year: str,
        title: str,
        doi: str = "",
    ):
        async with sem:
            urls = _resolve_vp_pdf_urls(pdf_url, arxiv_id, orf_id, doi)
            if not urls:
                stats["skipped"] += 1
                return

            # Try each URL in priority order until one succeeds
            data = None
            for url in urls:
                data = await _download_with_retry(client, url)
                if data and _validate_pdf(data, url):
                    break
                data = None  # invalid, try next

            if not data:
                stats["failed"] += 1
                await session.execute(
                    text("""
                        UPDATE venue_papers
                        SET extra_data = coalesce(extra_data, '{}')::jsonb
                            || jsonb_build_object('pdf_error', :err),
                            updated_at = now()
                        WHERE id = :id
                    """),
                    {"err": f"download_failed:{','.join(u[:60] for u in urls)}", "id": vp_id},
                )
                return

            # Upload to OSS
            object_key = _vp_object_key(conf_year, arxiv_id, title, vp_id)
            try:
                await storage.put(object_key, data)
            except Exception as e:
                logger.error(f"[vp-pdf] OSS upload failed for vp {vp_id}: {e}")
                stats["failed"] += 1
                return

            checksum = compute_checksum(data)
            await session.execute(
                text("""
                    UPDATE venue_papers
                    SET pdf_object_key = :key,
                        pdf_size_bytes = :sz,
                        pdf_checksum = :ck,
                        pdf_downloaded_at = now(),
                        updated_at = now()
                    WHERE id = :id
                """),
                {"key": object_key, "sz": len(data), "ck": checksum, "id": vp_id},
            )
            stats["downloaded"] += 1

    async with httpx.AsyncClient(follow_redirects=True, timeout=180) as client:
        while offset < total:
            rows = (await session.execute(text(f"""
                SELECT id, pdf_url, arxiv_id, openreview_forum_id, conf_year, title, doi
                FROM venue_papers
                WHERE {where_clause}
                ORDER BY id
                LIMIT :limit OFFSET :offset
            """), {**params, "limit": batch_size, "offset": offset})).fetchall()

            if not rows:
                break

            tasks = [
                _download_one(client, r[0], r[1] or "", r[2] or "", r[3] or "", r[4] or "", r[5] or "", r[6] or "")
                for r in rows
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Commit after each batch
            await session.commit()
            offset += len(rows)

            logger.info(
                f"[vp-pdf] progress: {offset}/{total} "
                f"(ok={stats['downloaded']} fail={stats['failed']} skip={stats['skipped']})"
            )

    logger.info(f"[vp-pdf] done: {stats}")
    return stats
