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
