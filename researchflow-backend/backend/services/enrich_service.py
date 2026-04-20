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

from backend.config import settings
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
            "_arxiv_id": base_id,  # For verification
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

        # Try arXiv ID via DOI format
        if paper.arxiv_id:
            base_id = re.sub(r"v\d+$", "", paper.arxiv_id)
            resp = await client.get(
                f"{OPENALEX_API}/works",
                params={"filter": f"doi:https://doi.org/10.48550/arXiv.{base_id}"},
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

async def _discover_github_links(
    client: httpx.AsyncClient, title: str, *, arxiv_id: str | None = None,
) -> dict | None:
    """Search GitHub for paper implementation repos.

    Also fetches README to check for acceptance info like
    "Accepted at CVPR 2025" or "NeurIPS 2024 Oral".
    """
    try:
        query = f'"{title[:80]}" in:readme'
        resp = await client.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": "5", "sort": "stars"},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None

        items = resp.json().get("items", [])
        if not items:
            return None

        # Verify the repo is actually for THIS paper
        best = None
        title_lower = title.lower()
        # Extract key words from title (skip common words)
        stop = {"a","an","the","of","for","in","on","with","and","to","from","by","is","are","its","via","based","using"}
        title_keywords = {w for w in title_lower.split() if w not in stop and len(w) > 2}

        for item in items:
            repo_name = item.get("name", "").lower()
            repo_desc = (item.get("description") or "").lower()
            full_name = item.get("full_name", "").lower()
            stars = item.get("stargazers_count", 0)

            # If we searched by arXiv ID, accept if description contains the paper title
            if arxiv_id and arxiv_id in (item.get("description") or ""):
                best = item
                break

            # Score: how many title keywords appear in repo name or description
            name_words = set(repo_name.replace("-", " ").replace("_", " ").split())
            desc_words = set(repo_desc.replace("-", " ").replace("_", " ").split())
            name_match = len(title_keywords & name_words)
            desc_match = len(title_keywords & desc_words)

            # Must have at least 2 keyword matches in name, or 3 in description
            if name_match >= 2 or (name_match >= 1 and desc_match >= 2):
                best = item
                break

            # If repo description closely matches paper title (>60% keyword overlap)
            if title_keywords and desc_words:
                overlap = len(title_keywords & desc_words) / len(title_keywords)
                if overlap >= 0.6:
                    best = item
                    break

            # Skip generic repos (awesome-lists, frameworks with >5K stars but low match)
            if stars > 5000 and name_match < 2:
                continue
            if "awesome" in repo_name and name_match < 3:
                continue

        if not best:
            return None  # No confident match — don't guess

        result = {
            "code_url": best.get("html_url"),
            "open_code": True,
            "stars": best.get("stargazers_count", 0),
        }

        # Fetch README to check for acceptance info + dataset links
        owner_repo = best.get("full_name", "")
        if owner_repo:
            readme_data = await _fetch_github_readme(client, owner_repo)
            if readme_data:
                result.update(readme_data)

        return result
    except Exception as e:
        logger.debug(f"GitHub search failed for '{title[:40]}': {e}")
    return None


async def _fetch_github_readme(client: httpx.AsyncClient, owner_repo: str) -> dict | None:
    """Fetch GitHub repo README and extract acceptance info + dataset links.

    Checks for patterns like:
      - "Accepted at ICLR 2025"
      - "This paper has been accepted by NeurIPS"
      - Dataset download links (HuggingFace datasets, Google Drive, etc.)
    """
    try:
        resp = await client.get(
            f"https://api.github.com/repos/{owner_repo}/readme",
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        import base64
        content_b64 = resp.json().get("content", "")
        if not content_b64:
            return None

        readme_text = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
        result = {}

        # Check for acceptance info
        acceptance = _parse_acceptance_from_comment(readme_text)
        if acceptance:
            result["acceptance_from_readme"] = acceptance

        # Extract dataset links
        dataset_urls = _extract_dataset_links(readme_text)
        if dataset_urls:
            result["dataset_urls"] = dataset_urls

        # Extract project page link (skip badge/shield URLs)
        import re
        project_patterns = [
            r'\[(?:Project\s*Page|Demo|Homepage|Website)\]\((https?://[^\)]+)\)',
            r'(?:project\s*page|demo|homepage)[:\s]*\[?[^\]]*\]?\(?(https?://[^\s\)]+)',
        ]
        for pat in project_patterns:
            m = re.search(pat, readme_text, re.IGNORECASE)
            if m:
                url = m.group(1)
                # Filter out badge/shield URLs
                if not any(skip in url.lower() for skip in
                           ["img.shields.io", "badge", "shields.io", "travis-ci",
                            "codecov.io", "github.com/workflows"]):
                    result["project_url"] = url
                    break

        return result if result else None

    except Exception as e:
        logger.debug(f"README fetch failed for {owner_repo}: {e}")
        return None


def _extract_dataset_links(text: str) -> list[str]:
    """Extract dataset download links from text (README, project page, etc.)."""
    import re
    urls = set()

    # HuggingFace datasets
    for m in re.finditer(r'https?://huggingface\.co/datasets/[\w\-\.\/]+', text):
        urls.add(m.group(0))

    # Google Drive links
    for m in re.finditer(r'https?://drive\.google\.com/[^\s\)\"\']+', text):
        urls.add(m.group(0))

    # Zenodo
    for m in re.finditer(r'https?://zenodo\.org/record[s]?/\d+', text):
        urls.add(m.group(0))

    # Direct download patterns with "dataset" in context
    for m in re.finditer(r'(?:dataset|data|benchmark)[\s\S]{0,100}?(https?://[^\s\)\"\']+(?:\.zip|\.tar|\.gz|\.json|\.csv))', text, re.IGNORECASE):
        urls.add(m.group(1))

    return list(urls)[:10]


async def _check_project_page_acceptance(client: httpx.AsyncClient, project_url: str) -> dict | None:
    """Fetch a paper's project page and check for acceptance info.

    Many papers have project pages like xxx.github.io that mention
    "Accepted at CVPR 2025" in the page content.
    """
    if not project_url:
        return None
    try:
        resp = await client.get(project_url, follow_redirects=True, timeout=10)
        if resp.status_code != 200:
            return None

        text = resp.text[:5000]  # Only check first 5KB
        acceptance = _parse_acceptance_from_comment(text)
        if acceptance:
            return {"acceptance_from_project_page": acceptance, "source_url": project_url}

        # Also extract dataset links from project page
        dataset_urls = _extract_dataset_links(text)
        if dataset_urls:
            return {"dataset_urls": dataset_urls, "source_url": project_url}

        return None
    except Exception as e:
        logger.debug(f"Project page fetch failed for {project_url}: {e}")
        return None


# ── HuggingFace discovery ──────────────────────────────────────

async def _discover_huggingface(client: httpx.AsyncClient, title: str) -> dict | None:
    """Search HuggingFace for models AND datasets related to a paper."""
    result = {}

    # Search models
    try:
        resp = await client.get(
            "https://huggingface.co/api/models",
            params={"search": title[:100], "limit": "3", "sort": "downloads"},
            timeout=15,
        )
        if resp.status_code == 200:
            models = resp.json()
            if models:
                result["huggingface_model_url"] = f"https://huggingface.co/{models[0].get('id', '')}"
    except Exception:
        pass

    # Search datasets
    try:
        resp = await client.get(
            "https://huggingface.co/api/datasets",
            params={"search": title[:100], "limit": "3", "sort": "downloads"},
            timeout=15,
        )
        if resp.status_code == 200:
            datasets = resp.json()
            if datasets:
                result["huggingface_dataset_url"] = f"https://huggingface.co/datasets/{datasets[0].get('id', '')}"
    except Exception:
        pass

    # Backward compat: keep huggingface_url
    if result.get("huggingface_model_url"):
        result["huggingface_url"] = result["huggingface_model_url"]

    return result if result else None


# ── Semantic Scholar enrichment ───────────────────────────────

S2_API = "https://api.semanticscholar.org/graph/v1"

def _s2_headers() -> dict:
    h = {"User-Agent": "ResearchFlow/0.1"}
    if settings.s2_api_key:
        h["x-api-key"] = settings.s2_api_key
    return h

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
            headers=_s2_headers(),
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

    # Reject patterns — these mean NOT accepted
    reject_patterns = [
        r'under\s+review',
        r'submitted\s+to',
        r'in\s+submission',
        r'preprint',
        r'work\s+in\s+progress',
        r'rejected',
        r'withdrawn',
    ]
    import re as _re
    for rp in reject_patterns:
        if _re.search(rp, comment_lower):
            return None

    # Positive acceptance patterns — must contain an acceptance verb
    acceptance_patterns = [
        # "Published as a conference paper at ICLR 2026"
        r'(?:accepted|published|to appear|appeared)\s+(?:as\s+)?(?:a\s+)?(?:\w+\s+)?(?:paper\s+)?(?:at|in|by|to)\s+([A-Za-z]+)\s*[\'"]?(\d{4})',
        # "Accepted to NeurIPS 2025", "Accepted at ECCV 2024"
        r'(?:accepted|published|to appear|appeared)\s+(?:at|in|by|to)\s+([A-Za-z]+)\s*[\'"]?(\d{4})',
        # "Extended NeurIPS submission", "NeurIPS 2023 paper"
        r'(?:extended|final)\s+([A-Za-z]+)\s+(?:submission|paper|version)\s*[\'"]?(\d{4})?',
        # "CVPR 2024 camera-ready", "NeurIPS 2023 oral"
        r'\b([A-Za-z]+)\s+(\d{4})\s+(?:oral|spotlight|poster|paper|submission|camera.?ready)',
        # "camera-ready version for ICLR 2025"
        r'camera.?ready\s+(?:\w+\s+)?(?:for|at|in)\s+([A-Za-z]+)\s*(\d{4})',
    ]

    for pattern in acceptance_patterns:
        m = _re.search(pattern, comment, _re.IGNORECASE)
        if m:
            venue = m.group(1).upper()
            year_str = m.group(2) if m.lastindex >= 2 else None
            year = int(year_str) if year_str else 0

            # Validate venue name is a known conference/journal
            known_venues = {
                "ICLR", "NEURIPS", "NIPS", "ICML", "CVPR", "ICCV", "ECCV",
                "AAAI", "IJCAI", "ACL", "EMNLP", "NAACL", "COLING",
                "SIGGRAPH", "CHI", "KDD", "WWW", "SIGIR", "ICRA", "IROS",
                "INTERSPEECH", "MICCAI", "WACV", "BMVC", "ACCV",
            }
            if venue not in known_venues:
                continue

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
            # VERIFY title match — arXiv API may return wrong paper for wrong ID
            # BUT skip verification if paper.title is just an arxiv ID placeholder
            title_is_placeholder = bool(re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', paper.title or ''))
            if not title_is_placeholder and arxiv_data.get("title") and paper.title:
                if not _titles_similar(paper.title, arxiv_data["title"]):
                    logger.warning(
                        f"arXiv title mismatch for {paper.arxiv_id}: "
                        f"expected '{paper.title[:40]}', got '{arxiv_data['title'][:40]}'"
                    )
                    arxiv_data = None  # Discard mismatched data

            # If title is placeholder, always update with arXiv title
            if title_is_placeholder and arxiv_data and arxiv_data.get("title"):
                paper.title = arxiv_data["title"]
                paper.title_sanitized = re.sub(r'[^a-zA-Z0-9]+', '_', arxiv_data["title"])[:80]

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
                        # Write venue to paper if not already set
                        if not paper.venue:
                            paper.venue = acceptance["venue"][:100]
                            updated["venue"] = True
                    if not paper.acceptance_type:
                        # Use acceptance_type if available, otherwise fallback to acceptance_status
                        acc_type = acceptance.get("acceptance_type") or acceptance.get("acceptance_status", "")
                        if acc_type:
                            paper.acceptance_type = acc_type
                            updated["acceptance_type"] = True

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
    # Skip Crossref if title is still a placeholder (arxiv ID) — would match wrong papers
    title_is_placeholder = bool(re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', paper.title or ''))
    if not paper.doi and paper.title and not title_is_placeholder:
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
    # Only search OpenAlex if we have a real title, DOI, or arxiv_id (not placeholder)
    title_is_placeholder = bool(re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', paper.title or ''))
    oa_data = None
    if paper.doi or paper.arxiv_id or not title_is_placeholder:
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
            # S2 is more authoritative for citations — always overwrite
            s2_count = min(s2_data["citation_count"], 32767)
            if s2_count > (paper.cited_by_count or 0):
                paper.cited_by_count = s2_count
                updated["cited_by_count"] = True
        if s2_data.get("venue"):
            await record_observation(session, entity_type="paper", entity_id=paper.id,
                field_name="venue", value=s2_data["venue"],
                source="semantic_scholar", source_url=s2_data.get("s2_url"))
            if not paper.venue and s2_data["venue"].strip():
                paper.venue = s2_data["venue"][:100]
                updated["venue"] = True

    # ── 5. GitHub code discovery ──────────────────────────────
    title_is_placeholder = bool(re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', paper.title or ''))
    if not paper.code_url and paper.title and not title_is_placeholder:
        await asyncio.sleep(0.3)
        gh_data = await _discover_github_links(client, paper.title, arxiv_id=paper.arxiv_id)
        # Fallback: search by arXiv ID if title search fails
        if not gh_data and paper.arxiv_id:
            await asyncio.sleep(0.3)
            gh_data = await _discover_github_links(
                client, paper.arxiv_id, arxiv_id=paper.arxiv_id,
            )
        if gh_data:
            paper.code_url = gh_data["code_url"]
            paper.open_code = True
            updated["code_url"] = True
            await record_observation(session, entity_type="paper", entity_id=paper.id,
                field_name="code_url", value=gh_data["code_url"],
                source="github")

    # ── 6. HuggingFace model/dataset discovery ────────────────
    if paper.title and not title_is_placeholder:
        await asyncio.sleep(0.3)
        hf_data = await _discover_huggingface(client, paper.title)
        if hf_data:
            if hf_data.get("huggingface_model_url"):
                await record_observation(session, entity_type="paper", entity_id=paper.id,
                    field_name="huggingface_model_url", value=hf_data["huggingface_model_url"],
                    source="huggingface")
                updated["huggingface_model_url"] = True
            if hf_data.get("huggingface_dataset_url"):
                await record_observation(session, entity_type="paper", entity_id=paper.id,
                    field_name="huggingface_dataset_url", value=hf_data["huggingface_dataset_url"],
                    source="huggingface")
                if not paper.data_url:
                    paper.data_url = hf_data["huggingface_dataset_url"]
                updated["huggingface_dataset_url"] = True

    # ── 7. GitHub README acceptance + dataset links ──────────
    # If GitHub discovered, check README for acceptance info
    if paper.code_url and not paper.acceptance_type:
        gh_readme = gh_data if 'gh_data' in dir() and gh_data else None
        if gh_readme and gh_readme.get("acceptance_from_readme"):
            acc = gh_readme["acceptance_from_readme"]
            await record_observation(session, entity_type="paper", entity_id=paper.id,
                field_name="acceptance_status", value=acc["acceptance_status"],
                source="github", source_url=paper.code_url, confidence=0.75)
            if acc.get("venue"):
                await record_observation(session, entity_type="paper", entity_id=paper.id,
                    field_name="venue", value=acc["venue"],
                    source="github", source_url=paper.code_url, confidence=0.75)
                if not paper.venue:
                    paper.venue = acc["venue"][:100]
                    updated["venue"] = True
            updated["acceptance_from_github"] = True

        # Dataset links from README
        if gh_readme and gh_readme.get("dataset_urls"):
            await record_observation(session, entity_type="paper", entity_id=paper.id,
                field_name="dataset_urls", value=gh_readme["dataset_urls"],
                source="github", source_url=paper.code_url)
            if not paper.data_url and gh_readme["dataset_urls"]:
                paper.data_url = gh_readme["dataset_urls"][0]
                updated["data_url"] = True

        # Project page URL from README
        if gh_readme and gh_readme.get("project_url"):
            paper.project_link = gh_readme["project_url"]
            updated["project_link"] = True

    # ── 8. Project page acceptance check ─────────────────────
    if paper.project_link and not paper.acceptance_type:
        await asyncio.sleep(0.3)
        page_data = await _check_project_page_acceptance(client, paper.project_link)
        if page_data:
            if page_data.get("acceptance_from_project_page"):
                acc = page_data["acceptance_from_project_page"]
                await record_observation(session, entity_type="paper", entity_id=paper.id,
                    field_name="acceptance_status", value=acc["acceptance_status"],
                    source="official_conf", source_url=paper.project_link, confidence=0.8)
                if acc.get("venue"):
                    await record_observation(session, entity_type="paper", entity_id=paper.id,
                        field_name="venue", value=acc["venue"],
                        source="official_conf", source_url=paper.project_link, confidence=0.8)
                updated["acceptance_from_project_page"] = True

            if page_data.get("dataset_urls"):
                await record_observation(session, entity_type="paper", entity_id=paper.id,
                    field_name="dataset_urls", value=page_data["dataset_urls"],
                    source="official_conf", source_url=paper.project_link)

    # ── 9. PDF first-page acceptance detection ──────────────
    # Many accepted papers have "Published as a conference paper at ICLR 2026"
    # or "Accepted at CVPR 2025" on the first page of the PDF itself
    if not paper.acceptance_type and paper.pdf_path_local:
        try:
            import fitz
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            pdf_full_path = os.path.join(project_root, paper.pdf_path_local)
            if os.path.exists(pdf_full_path):
                doc = fitz.open(pdf_full_path)
                try:
                    first_page_text = doc[0].get_text()[:500] if len(doc) > 0 else ""
                finally:
                    doc.close()
                if first_page_text:
                    pdf_acceptance = _parse_acceptance_from_comment(first_page_text)
                    if pdf_acceptance:
                        await record_observation(session, entity_type="paper", entity_id=paper.id,
                            field_name="acceptance_status", value=pdf_acceptance["acceptance_status"],
                            source="pymupdf", confidence=0.9)  # PDF text is very reliable
                        if pdf_acceptance.get("venue"):
                            await record_observation(session, entity_type="paper", entity_id=paper.id,
                                field_name="venue", value=pdf_acceptance["venue"],
                                source="pymupdf", confidence=0.9)
                            if not paper.venue:
                                paper.venue = pdf_acceptance["venue"][:100]
                                updated["venue"] = True
                        paper.acceptance_type = pdf_acceptance.get("acceptance_type") or pdf_acceptance.get("acceptance_status", "")
                        updated["acceptance_from_pdf"] = True
                        logger.info(f"PDF first-page acceptance for {paper.id}: {pdf_acceptance}")
        except Exception as e:
            logger.debug(f"PDF acceptance detection failed for {paper.id}: {e}")

    # ── 10. Update state ──────────────────────────────────────
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
