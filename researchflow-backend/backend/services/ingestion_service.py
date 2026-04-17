"""Ingestion service — normalize, dedup, canonical identity, state machine.

Handles all input types: links, PDFs, awesome lists, repos.
All inputs flow through the same pipeline:
  receive → canonicalize (identity + dedup) → enrich → skim → accept_to_kb
"""

import re
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.enums import PaperState
from backend.models.paper import Paper
from backend.schemas.import_ import ImportResultItem, LinkImportItem
from backend.utils.sanitize import extract_arxiv_id, parse_venue_year, sanitize_filename


async def _find_duplicate(
    session: AsyncSession,
    title_sanitized: str,
    arxiv_id: str | None,
    doi: str | None,
) -> Paper | None:
    """Find existing paper by arxiv_id, doi, or title_sanitized."""
    # Priority 1: arxiv_id exact match
    if arxiv_id:
        result = await session.execute(
            select(Paper).where(Paper.arxiv_id == arxiv_id).limit(1)
        )
        existing = result.scalar_one_or_none()
        if existing:
            return existing

    # Priority 2: doi exact match
    if doi:
        result = await session.execute(
            select(Paper).where(Paper.doi == doi).limit(1)
        )
        existing = result.scalar_one_or_none()
        if existing:
            return existing

    # Priority 3: title_sanitized match (case-insensitive)
    result = await session.execute(
        select(Paper).where(
            func.lower(Paper.title_sanitized) == title_sanitized.lower()
        ).limit(1)
    )
    return result.scalar_one_or_none()


def _normalize_url(url: str) -> tuple[str, str | None]:
    """Normalize a paper URL. Returns (paper_link, arxiv_id | None)."""
    url = url.strip()

    # Normalize arxiv URLs to abs format
    arxiv_id = extract_arxiv_id(url)
    if arxiv_id:
        # Strip version suffix for dedup
        base_id = re.sub(r"v\d+$", "", arxiv_id)
        return f"https://arxiv.org/abs/{base_id}", base_id

    return url, None


def _infer_title_from_url(url: str) -> str:
    """Best-effort title extraction from URL path."""
    from urllib.parse import unquote, urlparse
    path = unquote(urlparse(url).path)
    # Try to get filename without extension
    parts = path.rstrip("/").split("/")
    if parts:
        name = parts[-1]
        name = re.sub(r"\.(pdf|html|htm)$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"[-_]", " ", name)
        if len(name) > 5:
            return name.strip()
    return url


async def ingest_link(
    session: AsyncSession,
    item: LinkImportItem,
    default_category: str,
    is_ephemeral: bool,
    retention_days: int,
) -> ImportResultItem:
    """Ingest a single paper link through the canonical pipeline."""

    # 1. Normalize URL
    paper_link, arxiv_id = _normalize_url(item.url)

    # 2. Resolve title
    title = item.title or _infer_title_from_url(item.url)
    title_sanitized = sanitize_filename(title)

    # 3. Venue/year parsing
    venue = item.venue
    year = item.year
    if venue and not year:
        venue, year = parse_venue_year(venue)

    # 4. Dedup check
    existing = await _find_duplicate(session, title_sanitized, arxiv_id, None)
    if existing:
        return ImportResultItem(
            paper_id=existing.id,
            title=existing.title,
            status="duplicate",
            message=f"Matches existing paper (state={existing.state.value})",
        )

    # 5. Create paper
    category = item.category or default_category
    now = datetime.now(timezone.utc)

    paper = Paper(
        title=title,
        title_sanitized=title_sanitized,
        venue=venue or "",
        year=year,
        category=category,
        state=PaperState.EPHEMERAL_RECEIVED if is_ephemeral else PaperState.WAIT,
        paper_link=paper_link,
        project_link=item.project_link,
        arxiv_id=arxiv_id,
        tags=item.tags or [category],
        is_ephemeral=is_ephemeral,
        retention_days=retention_days,
        expires_at=(now + timedelta(days=retention_days)) if is_ephemeral else None,
        source="import_link",
        source_ref=item.url,
    )
    session.add(paper)
    await session.flush()
    await session.refresh(paper)

    # 6. Auto-advance: ephemeral → canonicalized (identity resolved)
    if is_ephemeral:
        paper.state = PaperState.CANONICALIZED
        await session.flush()

    return ImportResultItem(
        paper_id=paper.id,
        title=paper.title,
        status="created",
        message=f"State: {paper.state.value}",
    )


async def ingest_links(
    session: AsyncSession,
    items: list[LinkImportItem],
    default_category: str,
    is_ephemeral: bool,
    retention_days: int,
) -> list[ImportResultItem]:
    """Ingest a batch of paper links."""
    results = []
    for item in items:
        try:
            result = await ingest_link(
                session, item, default_category, is_ephemeral, retention_days
            )
            results.append(result)
        except Exception as e:
            results.append(ImportResultItem(
                paper_id="00000000-0000-0000-0000-000000000000",
                title=item.title or item.url,
                status="error",
                message=str(e)[:200],
            ))
    return results


async def accept_to_kb(session: AsyncSession, paper_id) -> Paper | None:
    """Promote an ephemeral paper to the main knowledge base."""
    paper = await session.get(Paper, paper_id)
    if not paper:
        return None

    paper.is_ephemeral = False
    paper.expires_at = None
    paper.state = PaperState.WAIT  # enters main pipeline
    await session.flush()
    await session.refresh(paper)
    return paper


async def cleanup_expired(session: AsyncSession) -> int:
    """Archive expired ephemeral papers. Returns count archived."""
    now = datetime.now(timezone.utc)
    result = await session.execute(
        select(Paper).where(
            Paper.is_ephemeral.is_(True),
            Paper.expires_at.isnot(None),
            Paper.expires_at < now,
            Paper.state != PaperState.ARCHIVED_OR_EXPIRED,
        )
    )
    papers = result.scalars().all()
    for paper in papers:
        paper.state = PaperState.ARCHIVED_OR_EXPIRED
    await session.flush()
    return len(papers)
