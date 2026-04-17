"""Enrich service — auto-complete paper metadata from external APIs.

Sources:
  1. arXiv API (for papers with arxiv_id)
  2. Crossref API (for DOI lookup by title)

Only fills in missing fields — never overwrites existing data.
"""

import asyncio
import logging
import re
import xml.etree.ElementTree as ET

import httpx

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.enums import PaperState
from backend.models.paper import Paper

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"

# Polite headers for Crossref
CROSSREF_HEADERS = {
    "User-Agent": "ResearchFlow/0.1 (mailto:researchflow@example.com)",
}


# ── arXiv enrichment ────────────────────────────────────────────

async def _fetch_arxiv(client: httpx.AsyncClient, arxiv_id: str) -> dict | None:
    """Fetch metadata from arXiv API for a given arxiv_id."""
    try:
        # Strip version suffix
        base_id = re.sub(r"v\d+$", "", arxiv_id)
        resp = await client.get(
            ARXIV_API,
            params={"id_list": base_id, "max_results": "1"},
            timeout=15,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning(f"arXiv API error for {arxiv_id}: {e}")
        return None

    # Parse Atom XML
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    try:
        root = ET.fromstring(resp.text)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
        abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")

        authors = []
        for author_el in entry.findall("atom:author", ns):
            name = author_el.findtext("atom:name", "", ns).strip()
            if name:
                authors.append({"name": name})

        # Published date → year
        published = entry.findtext("atom:published", "", ns)
        year = int(published[:4]) if published and len(published) >= 4 else None

        # Categories
        categories = []
        for cat_el in entry.findall("atom:category", ns):
            term = cat_el.get("term", "")
            if term:
                categories.append(term)

        # DOI (sometimes in arxiv:doi)
        doi = entry.findtext("arxiv:doi", None, ns)

        return {
            "title": title if title else None,
            "abstract": abstract if abstract else None,
            "authors": authors if authors else None,
            "year": year,
            "doi": doi,
            "keywords": categories if categories else None,
        }
    except ET.ParseError as e:
        logger.warning(f"arXiv XML parse error for {arxiv_id}: {e}")
        return None


# ── Crossref enrichment ─────────────────────────────────────────

async def _fetch_crossref(client: httpx.AsyncClient, title: str) -> dict | None:
    """Search Crossref by title to get DOI, authors, venue, year."""
    try:
        resp = await client.get(
            CROSSREF_API,
            params={"query.title": title, "rows": "1", "select": "DOI,title,author,container-title,published-print,published-online,type"},
            headers=CROSSREF_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning(f"Crossref API error for '{title[:50]}': {e}")
        return None

    items = data.get("message", {}).get("items", [])
    if not items:
        return None

    item = items[0]

    # Verify title similarity (basic check)
    cr_title = " ".join(item.get("title", [""]))
    if not _titles_similar(title, cr_title):
        return None

    # Extract fields
    doi = item.get("DOI")

    authors = []
    for a in item.get("author", []):
        name_parts = []
        if a.get("given"):
            name_parts.append(a["given"])
        if a.get("family"):
            name_parts.append(a["family"])
        if name_parts:
            authors.append({"name": " ".join(name_parts)})

    venue = None
    container = item.get("container-title", [])
    if container:
        venue = container[0]

    year = None
    for date_key in ("published-print", "published-online"):
        date_parts = item.get(date_key, {}).get("date-parts", [[]])
        if date_parts and date_parts[0] and date_parts[0][0]:
            year = date_parts[0][0]
            break

    return {
        "doi": doi,
        "authors": authors if authors else None,
        "venue_crossref": venue,
        "year": year,
    }


def _titles_similar(a: str, b: str) -> bool:
    """Check if two titles are roughly the same paper."""
    def normalize(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    # Check if shorter is mostly contained in longer
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    return shorter[:30] in longer or longer[:30] in shorter


# ── Main enrich logic ───────────────────────────────────────────

async def enrich_paper(session: AsyncSession, paper: Paper, client: httpx.AsyncClient) -> dict[str, bool]:
    """Enrich a single paper. Returns dict of which fields were updated."""
    updated = {}

    # 1. Try arXiv if we have arxiv_id
    if paper.arxiv_id:
        arxiv_data = await _fetch_arxiv(client, paper.arxiv_id)
        if arxiv_data:
            if not paper.abstract and arxiv_data.get("abstract"):
                paper.abstract = arxiv_data["abstract"]
                updated["abstract"] = True
            if not paper.authors and arxiv_data.get("authors"):
                paper.authors = arxiv_data["authors"]
                updated["authors"] = True
            if not paper.year and arxiv_data.get("year"):
                paper.year = arxiv_data["year"]
                updated["year"] = True
            if not paper.doi and arxiv_data.get("doi"):
                paper.doi = arxiv_data["doi"]
                updated["doi"] = True
            if not paper.keywords and arxiv_data.get("keywords"):
                paper.keywords = arxiv_data["keywords"]
                updated["keywords"] = True

    # 2. Try Crossref for DOI / author supplementation
    if not paper.doi and paper.title:
        # Small delay to be polite to Crossref
        await asyncio.sleep(0.5)
        cr_data = await _fetch_crossref(client, paper.title)
        if cr_data:
            if not paper.doi and cr_data.get("doi"):
                paper.doi = cr_data["doi"]
                updated["doi"] = True
            if not paper.authors and cr_data.get("authors"):
                paper.authors = cr_data["authors"]
                updated["authors"] = True
            if not paper.year and cr_data.get("year"):
                paper.year = cr_data["year"]
                updated["year"] = True

    # 3. Update state if enrichment happened and paper is in ephemeral flow
    if updated and paper.state == PaperState.CANONICALIZED:
        paper.state = PaperState.ENRICHED

    if updated:
        await session.flush()

    return updated


async def enrich_batch(
    session: AsyncSession,
    limit: int = 20,
) -> list[dict]:
    """Enrich papers that are missing abstract/authors/doi.

    Targets papers with arxiv_id but no abstract, or papers with
    title but no doi. Processes up to `limit` papers.
    """
    # Find candidates: have arxiv_id but missing abstract, or missing doi
    result = await session.execute(
        select(Paper)
        .where(
            Paper.state.notin_([
                PaperState.ARCHIVED_OR_EXPIRED,
                PaperState.SKIP,
            ]),
            # At least one enrichable field missing
            (Paper.abstract.is_(None)) | (Paper.doi.is_(None)) | (Paper.authors.is_(None)),
        )
        .order_by(Paper.analysis_priority.desc().nullsfirst())
        .limit(limit)
    )
    papers = list(result.scalars().all())

    if not papers:
        return []

    results = []
    async with httpx.AsyncClient() as client:
        for paper in papers:
            try:
                updated = await enrich_paper(session, paper, client)
                results.append({
                    "paper_id": str(paper.id),
                    "title": paper.title[:60],
                    "fields_updated": list(updated.keys()),
                })
            except Exception as e:
                logger.error(f"Enrich error for {paper.id}: {e}")
                results.append({
                    "paper_id": str(paper.id),
                    "title": paper.title[:60],
                    "error": str(e)[:100],
                })
            # Rate limit: ~1 paper/sec
            await asyncio.sleep(1.0)

    await session.flush()
    return results
