"""Enrich service — auto-complete paper metadata from external APIs.

Sources:
  1. arXiv API (title, abstract, authors, year, DOI, categories, **comments**)
  2. Crossref API (DOI, authors, venue, year)
  3. OpenAlex API (venue, citation count, year, open access, DOI)
  4. GitHub API (code repository discovery)
  5. HuggingFace API (model/dataset discovery)
  6. Semantic Scholar API (citation count, references, related papers)

v2 changes:
  - Writes metadata_observations for every source (observation ledger)
  - Extracts arXiv comments field ("Accepted at ICLR 2025" etc.)
  - Connects HuggingFace discovery
  - Adds Semantic Scholar enrichment for citation count
  - Dataset link discovery
  - Still fills Paper fields directly as backward compat
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

# Lazy import to avoid circular dependency at module load
def _record_obs():
    from backend.services.metadata_resolver_service import record_observation
    return record_observation

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

        # Comments field — often contains "Accepted at ICLR 2025", "12 pages, 5 figures"
        comment = entry.findtext("arxiv:comment", None, ns)

        # Journal reference (sometimes set for published papers)
        journal_ref = entry.findtext("arxiv:journal_ref", None, ns)

        return {
            "title": title if title else None,
            "abstract": abstract if abstract else None,
            "authors": authors if authors else None,
            "year": year,
            "doi": doi,
            "keywords": categories if categories else None,
            "comment": comment.strip() if comment else None,
            "journal_ref": journal_ref.strip() if journal_ref else None,
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


# ── OpenAlex enrichment ────────────────────────────────────────

OPENALEX_API = "https://api.openalex.org"

async def _fetch_openalex(client: httpx.AsyncClient, paper: "Paper") -> dict | None:
    """Fetch metadata from OpenAlex — venue verification, citation count, type."""
    try:
        # Try DOI first (most reliable)
        if paper.doi:
            resp = await client.get(
                f"{OPENALEX_API}/works/doi:{paper.doi}",
                headers={"User-Agent": "ResearchFlow/0.1 (mailto:researchflow@example.com)"},
                timeout=15,
            )
            if resp.status_code == 200:
                return _parse_openalex_work(resp.json())

        # Try arXiv ID
        if paper.arxiv_id:
            base_id = re.sub(r"v\d+$", "", paper.arxiv_id)
            resp = await client.get(
                f"{OPENALEX_API}/works",
                params={"filter": f"ids.openalex_id:https://openalex.org/W*,doi:https://doi.org/10.48550/arXiv.{base_id}"},
                headers={"User-Agent": "ResearchFlow/0.1"},
                timeout=15,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    return _parse_openalex_work(results[0])

        # Try title search (fallback)
        if paper.title:
            resp = await client.get(
                f"{OPENALEX_API}/works",
                params={"search": paper.title[:200], "per_page": "1"},
                headers={"User-Agent": "ResearchFlow/0.1"},
                timeout=15,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results and _titles_similar(paper.title, results[0].get("title", "")):
                    return _parse_openalex_work(results[0])

    except httpx.HTTPError as e:
        logger.warning(f"OpenAlex API error for '{paper.title[:50]}': {e}")
    return None


def _parse_openalex_work(work: dict) -> dict:
    """Parse an OpenAlex work into enrichment fields."""
    result = {}

    # OpenAlex ID
    oa_id = work.get("id", "")
    if oa_id:
        result["openalex_id"] = oa_id.replace("https://openalex.org/", "")

    # Venue / source
    primary_loc = work.get("primary_location", {}) or {}
    source = primary_loc.get("source", {}) or {}
    if source.get("display_name"):
        result["venue"] = source["display_name"]

    # Publication type (journal-article vs posted-content)
    result["work_type"] = work.get("type", "")

    # Is it actually published (not just a preprint)?
    result["is_published"] = work.get("type", "") != "posted-content"

    # Citation count
    result["cited_by_count"] = work.get("cited_by_count", 0)

    # Year
    if work.get("publication_year"):
        result["year"] = work["publication_year"]

    # DOI
    if work.get("doi"):
        result["doi"] = work["doi"].replace("https://doi.org/", "")

    # Abstract (reconstructed from inverted index)
    abstract_inv = work.get("abstract_inverted_index")
    if abstract_inv and isinstance(abstract_inv, dict):
        # Reconstruct from inverted index
        words = []
        for word, positions in abstract_inv.items():
            for pos in positions:
                words.append((pos, word))
        words.sort()
        result["abstract"] = " ".join(w for _, w in words)

    # Open access
    oa = work.get("open_access", {}) or {}
    if oa.get("is_oa"):
        result["is_oa"] = True

    return result


# ── GitHub code discovery ──────────────────────────────────────

async def _discover_github_links(client: httpx.AsyncClient, title: str) -> dict | None:
    """Search GitHub for paper implementation repos."""
    try:
        # Search for repos mentioning the paper title
        query = f'"{title[:80]}" in:readme'
        resp = await client.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": "3", "sort": "stars"},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None

        items = resp.json().get("items", [])
        if not items:
            return None

        best = items[0]
        return {
            "code_url": best.get("html_url"),
            "open_code": True,
        }
    except Exception as e:
        logger.debug(f"GitHub search failed for '{title[:40]}': {e}")
    return None


# ── HuggingFace discovery ──────────────────────────────────────

async def _discover_huggingface(client: httpx.AsyncClient, title: str) -> dict | None:
    """Search HuggingFace for models/datasets related to a paper."""
    try:
        resp = await client.get(
            "https://huggingface.co/api/models",
            params={"search": title[:100], "limit": "3", "sort": "downloads"},
            timeout=15,
        )
        if resp.status_code == 200:
            models = resp.json()
            if models:
                return {"huggingface_url": f"https://huggingface.co/{models[0].get('id', '')}"}
    except Exception:
        pass
    return None


# ── Semantic Scholar enrichment ───────────────────────────────

S2_API = "https://api.semanticscholar.org/graph/v1"

async def _fetch_semantic_scholar(client: httpx.AsyncClient, paper: "Paper") -> dict | None:
    """Fetch citation count + venue from Semantic Scholar."""
    # Find paper on S2
    s2_id = None
    if paper.arxiv_id:
        s2_id = f"arXiv:{re.sub(r'v[0-9]+$', '', paper.arxiv_id)}"
    elif paper.doi:
        s2_id = f"DOI:{paper.doi}"

    if not s2_id:
        return None

    try:
        resp = await client.get(
            f"{S2_API}/paper/{s2_id}",
            params={"fields": "title,venue,year,citationCount,referenceCount,externalIds,url"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        return {
            "citation_count": data.get("citationCount", 0),
            "reference_count": data.get("referenceCount", 0),
            "venue": data.get("venue", ""),
            "year": data.get("year"),
            "s2_url": data.get("url", ""),
            "external_ids": data.get("externalIds", {}),
        }
    except Exception as e:
        logger.debug(f"S2 API error for {s2_id}: {e}")
        return None


# ── Acceptance status from arXiv comments ────────────────────

def _parse_acceptance_from_comment(comment: str) -> dict | None:
    """Parse arXiv comment for conference acceptance info.

    Examples:
      "Accepted at ICLR 2025"
      "Published in NeurIPS 2024"
      "To appear at CVPR 2025"
      "AAAI 2025 oral"
    """
    if not comment:
        return None

    comment_lower = comment.lower()

    # Patterns for acceptance
    import re
    acceptance_patterns = [
        r'(?:accepted|published|to appear|appeared)\s+(?:at|in|by)\s+([A-Za-z]+)\s*(\d{4})',
        r'(ICLR|NeurIPS|ICML|CVPR|ICCV|ECCV|AAAI|IJCAI|ACL|EMNLP|NAACL|SIGGRAPH|CHI|KDD|WWW)\s*[\'"]?(\d{4})',
    ]

    for pattern in acceptance_patterns:
        m = re.search(pattern, comment, re.IGNORECASE)
        if m:
            venue = m.group(1).upper()
            year = int(m.group(2))
            # Check for acceptance type
            acc_type = ""
            if "oral" in comment_lower:
                acc_type = "oral"
            elif "spotlight" in comment_lower:
                acc_type = "spotlight"
            elif "poster" in comment_lower:
                acc_type = "poster"
            elif "workshop" in comment_lower:
                acc_type = "workshop"

            return {
                "venue": venue,
                "year": year,
                "acceptance_status": "accepted",
                "acceptance_type": acc_type,
            }

    return None


# ── Main enrich logic ───────────────────────────────────────────

async def enrich_paper(session: AsyncSession, paper: Paper, client: httpx.AsyncClient) -> dict[str, bool]:
    """Enrich a single paper from multiple external sources.

    v2: Writes observations to metadata_observations table AND fills Paper fields.
    Returns dict of which fields were updated.
    """
    record_observation = _record_obs()
    updated = {}

    # ── 1. arXiv ──────────────────────────────────────────────
    if paper.arxiv_id:
        arxiv_data = await _fetch_arxiv(client, paper.arxiv_id)
        if arxiv_data:
            # Record observations
            if arxiv_data.get("abstract"):
                await record_observation(session, entity_type="paper", entity_id=paper.id,
                    field_name="abstract", value=arxiv_data["abstract"][:500],
                    source="arxiv", source_url=f"https://arxiv.org/abs/{paper.arxiv_id}")
            if arxiv_data.get("authors"):
                await record_observation(session, entity_type="paper", entity_id=paper.id,
                    field_name="authors", value=arxiv_data["authors"],
                    source="arxiv", source_url=f"https://arxiv.org/abs/{paper.arxiv_id}")
            if arxiv_data.get("year"):
                await record_observation(session, entity_type="paper", entity_id=paper.id,
                    field_name="year", value=arxiv_data["year"],
                    source="arxiv")

            # Parse acceptance from arXiv comment field
            if arxiv_data.get("comment"):
                acceptance = _parse_acceptance_from_comment(arxiv_data["comment"])
                if acceptance:
                    await record_observation(session, entity_type="paper", entity_id=paper.id,
                        field_name="acceptance_status", value=acceptance["acceptance_status"],
                        source="arxiv", confidence=0.7,
                        source_url=f"https://arxiv.org/abs/{paper.arxiv_id}")
                    if acceptance.get("venue"):
                        await record_observation(session, entity_type="paper", entity_id=paper.id,
                            field_name="venue", value=acceptance["venue"],
                            source="arxiv", confidence=0.7)

            # Also from journal_ref
            if arxiv_data.get("journal_ref"):
                await record_observation(session, entity_type="paper", entity_id=paper.id,
                    field_name="venue", value=arxiv_data["journal_ref"],
                    source="arxiv", confidence=0.6)

            # Fill Paper fields (backward compat)
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

    # ── 2. Crossref ───────────────────────────────────────────
    if not paper.doi and paper.title:
        await asyncio.sleep(0.5)
        cr_data = await _fetch_crossref(client, paper.title)
        if cr_data:
            if cr_data.get("venue_crossref"):
                await record_observation(session, entity_type="paper", entity_id=paper.id,
                    field_name="venue", value=cr_data["venue_crossref"],
                    source="crossref")
            if cr_data.get("authors"):
                await record_observation(session, entity_type="paper", entity_id=paper.id,
                    field_name="authors", value=cr_data["authors"],
                    source="crossref")

            if not paper.doi and cr_data.get("doi"):
                paper.doi = cr_data["doi"]
                updated["doi"] = True
            if not paper.authors and cr_data.get("authors"):
                paper.authors = cr_data["authors"]
                updated["authors"] = True
            if not paper.year and cr_data.get("year"):
                paper.year = cr_data["year"]
                updated["year"] = True

    # ── 3. OpenAlex ───────────────────────────────────────────
    oa_data = await _fetch_openalex(client, paper)
    if oa_data:
        if oa_data.get("venue"):
            await record_observation(session, entity_type="paper", entity_id=paper.id,
                field_name="venue", value=oa_data["venue"],
                source="openalex")
        if oa_data.get("cited_by_count"):
            await record_observation(session, entity_type="paper", entity_id=paper.id,
                field_name="citation_count", value=oa_data["cited_by_count"],
                source="openalex")

        if oa_data.get("openalex_id") and not paper.openalex_id:
            paper.openalex_id = oa_data["openalex_id"]
            updated["openalex_id"] = True
        if oa_data.get("venue") and not paper.venue:
            paper.venue = oa_data["venue"][:100]
            updated["venue"] = True
        if oa_data.get("cited_by_count") and (not paper.cited_by_count or paper.cited_by_count == 0):
            paper.cited_by_count = min(oa_data["cited_by_count"], 32767)
            updated["cited_by_count"] = True
        if not paper.abstract and oa_data.get("abstract"):
            paper.abstract = oa_data["abstract"]
            updated["abstract"] = True
        if not paper.doi and oa_data.get("doi"):
            paper.doi = oa_data["doi"]
            updated["doi"] = True
        if not paper.year and oa_data.get("year"):
            paper.year = oa_data["year"]
            updated["year"] = True

    # ── 4. Semantic Scholar (citation count + venue) ──────────
    s2_data = await _fetch_semantic_scholar(client, paper)
    if s2_data:
        if s2_data.get("citation_count"):
            await record_observation(session, entity_type="paper", entity_id=paper.id,
                field_name="citation_count", value=s2_data["citation_count"],
                source="semantic_scholar", source_url=s2_data.get("s2_url"))
            # S2 is more authoritative for citations than OA
            if not paper.cited_by_count or paper.cited_by_count == 0:
                paper.cited_by_count = min(s2_data["citation_count"], 32767)
                updated["cited_by_count"] = True
        if s2_data.get("venue"):
            await record_observation(session, entity_type="paper", entity_id=paper.id,
                field_name="venue", value=s2_data["venue"],
                source="semantic_scholar", source_url=s2_data.get("s2_url"))

    # ── 5. GitHub code discovery ──────────────────────────────
    if not paper.code_url and paper.title:
        await asyncio.sleep(0.3)
        gh_data = await _discover_github_links(client, paper.title)
        if gh_data:
            paper.code_url = gh_data["code_url"]
            paper.open_code = True
            updated["code_url"] = True
            await record_observation(session, entity_type="paper", entity_id=paper.id,
                field_name="code_url", value=gh_data["code_url"],
                source="github")

    # ── 6. HuggingFace model/dataset discovery ────────────────
    if paper.title:
        await asyncio.sleep(0.3)
        hf_data = await _discover_huggingface(client, paper.title)
        if hf_data and hf_data.get("huggingface_url"):
            await record_observation(session, entity_type="paper", entity_id=paper.id,
                field_name="huggingface_url", value=hf_data["huggingface_url"],
                source="github")  # recording as a data link
            updated["huggingface_url"] = True

    # ── 7. Update state ──────────────────────────────────────
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
