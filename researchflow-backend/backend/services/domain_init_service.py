"""Domain initialization service — awesome-first KB bootstrapping.

When a user says "I want to study domain X", the system should:
1. Find the best awesome-list GitHub repo for that domain
2. Import papers from the awesome list
3. Triage and prioritize (open data > open code > accepted > preprint)
4. Run analysis pipeline on highest-priority papers first
5. Build the initial knowledge graph with method lineage

This is the "cold start" solution for new domains.
"""

import json
import logging
import re
from uuid import UUID

import httpx
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.paper import Paper
from backend.models.enums import PaperState
from backend.schemas.import_ import LinkImportItem
from backend.services.ingestion_service import ingest_link

logger = logging.getLogger(__name__)

GITHUB_SEARCH_API = "https://api.github.com/search/repositories"


async def find_awesome_repos(
    domain: str,
    max_results: int = 5,
) -> list[dict]:
    """Search GitHub for awesome-list repos matching a domain.

    Searches for "awesome {domain}" repos, sorted by stars.
    """
    queries = [
        f"awesome {domain}",
        f"awesome-{domain.replace(' ', '-')}",
        f"{domain} paper list",
    ]

    repos = []
    seen_urls = set()

    async with httpx.AsyncClient(timeout=20) as client:
        for q in queries:
            try:
                resp = await client.get(
                    GITHUB_SEARCH_API,
                    params={"q": q, "sort": "stars", "per_page": str(max_results)},
                    headers={"Accept": "application/vnd.github.v3+json"},
                )
                if resp.status_code == 200:
                    for item in resp.json().get("items", []):
                        url = item.get("html_url", "")
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                        repos.append({
                            "name": item.get("full_name"),
                            "url": url,
                            "description": item.get("description", "")[:200],
                            "stars": item.get("stargazers_count", 0),
                            "updated_at": item.get("updated_at"),
                        })
            except Exception as e:
                logger.warning(f"GitHub search failed for '{q}': {e}")

    # Sort by stars
    repos.sort(key=lambda x: x.get("stars", 0), reverse=True)
    return repos[:max_results]


async def extract_papers_from_readme(
    repo_url: str,
    max_papers: int = 100,
) -> list[dict]:
    """Fetch a GitHub repo's README and extract paper links.

    Parses arxiv links, paper titles from markdown tables and lists.
    """
    # Convert repo URL to raw README URL
    parts = repo_url.rstrip("/").split("/")
    if len(parts) >= 2:
        owner, repo = parts[-2], parts[-1]
    else:
        return []

    readme_urls = [
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md",
    ]

    readme_text = ""
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        for url in readme_urls:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    readme_text = resp.text
                    break
            except Exception:
                continue

    if not readme_text:
        return []

    return _parse_papers_from_markdown(readme_text, max_papers)


def _parse_papers_from_markdown(text: str, max_papers: int) -> list[dict]:
    """Extract paper entries from markdown text.

    Handles:
    - arxiv links: https://arxiv.org/abs/2401.12345
    - Markdown links: [Paper Title](https://arxiv.org/abs/...)
    - Table rows with paper info
    """
    papers = []
    seen_urls = set()

    # Pattern 1: Markdown links with arxiv
    link_pattern = re.compile(
        r'\[([^\]]+)\]\((https?://(?:arxiv\.org/(?:abs|pdf)/[\w.]+|doi\.org/[^\s)]+))\)',
        re.IGNORECASE,
    )

    for match in link_pattern.finditer(text):
        title = match.group(1).strip()
        url = match.group(2).strip()
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Clean arxiv PDF links to abs
        url = re.sub(r'arxiv\.org/pdf/', 'arxiv.org/abs/', url)
        url = re.sub(r'\.pdf$', '', url)

        # Try to detect venue/year from surrounding context
        context = text[max(0, match.start() - 200):match.end() + 100]
        venue, year = _extract_venue_year(context)

        # Detect code/data links nearby
        has_code = bool(re.search(r'(?:github\.com|code|implementation)', context, re.I))

        papers.append({
            "title": title[:300],
            "url": url,
            "venue": venue,
            "year": year,
            "has_code": has_code,
        })

        if len(papers) >= max_papers:
            break

    # Pattern 2: Bare arxiv URLs not in markdown links
    bare_pattern = re.compile(r'(https?://arxiv\.org/(?:abs|pdf)/[\w.]+)', re.IGNORECASE)
    for match in bare_pattern.finditer(text):
        url = match.group(1).strip()
        url = re.sub(r'arxiv\.org/pdf/', 'arxiv.org/abs/', url)
        url = re.sub(r'\.pdf$', '', url)
        if url in seen_urls:
            continue
        seen_urls.add(url)

        papers.append({
            "title": f"arxiv:{url.split('/')[-1]}",
            "url": url,
        })

        if len(papers) >= max_papers:
            break

    return papers


def _extract_venue_year(context: str) -> tuple[str | None, int | None]:
    """Extract venue and year from nearby text."""
    year_match = re.search(r'(20[12]\d)', context)
    year = int(year_match.group(1)) if year_match else None

    venues = ["ICLR", "NeurIPS", "ICML", "CVPR", "ICCV", "ECCV", "SIGGRAPH",
              "ACL", "EMNLP", "NAACL", "AAAI", "IJCAI", "KDD", "WWW"]
    venue = None
    for v in venues:
        if v.lower() in context.lower():
            venue = v
            break

    return venue, year


async def init_domain_from_awesome(
    session: AsyncSession,
    domain: str,
    repo_url: str | None = None,
    max_papers: int = 50,
    category: str | None = None,
) -> dict:
    """Initialize a domain KB from an awesome-list repo.

    1. Find/use awesome repo
    2. Extract papers from README
    3. Ingest all papers
    4. Triage and sort by priority
    5. Return ingestion summary with priority queue

    Does NOT auto-run analysis pipeline (caller decides based on budget).
    """
    result = {
        "domain": domain,
        "repo": None,
        "papers_found": 0,
        "papers_ingested": 0,
        "papers_duplicate": 0,
        "priority_queue": [],
    }

    # 1. Find awesome repo if not provided
    if not repo_url:
        repos = await find_awesome_repos(domain, max_results=3)
        if repos:
            repo_url = repos[0]["url"]
            result["repo"] = repos[0]
            result["alternative_repos"] = repos[1:] if len(repos) > 1 else []
        else:
            return {**result, "error": f"No awesome repo found for '{domain}'"}
    else:
        result["repo"] = {"url": repo_url}

    # 2. Extract papers from README
    papers = await extract_papers_from_readme(repo_url, max_papers)
    result["papers_found"] = len(papers)

    if not papers:
        return {**result, "error": "No papers found in repo README"}

    # 3. Ingest all papers
    cat = category or domain.replace(" ", "_")
    ingested_ids = []
    for p in papers:
        try:
            item = LinkImportItem(
                url=p["url"],
                title=p.get("title"),
                venue=p.get("venue"),
                year=p.get("year"),
            )
            res = await ingest_link(session, item, cat, is_ephemeral=False, retention_days=30)
            if res.status == "created":
                result["papers_ingested"] += 1
                ingested_ids.append(res.paper_id)
            elif res.status == "duplicate":
                result["papers_duplicate"] += 1
        except Exception as e:
            logger.debug(f"Ingest skip: {e}")

    await session.commit()

    # 4. Triage all newly ingested papers
    from backend.services import triage_service
    for pid in ingested_ids:
        await triage_service.triage_paper(session, pid)
    await session.commit()

    # 5. Build priority queue (sorted by analysis_priority desc)
    if ingested_ids:
        papers_result = await session.execute(
            select(Paper).where(
                Paper.id.in_(ingested_ids)
            ).order_by(desc(Paper.analysis_priority))
        )
        for p in papers_result.scalars():
            result["priority_queue"].append({
                "paper_id": str(p.id),
                "title": p.title[:100],
                "tier": p.tier.value if p.tier else None,
                "keep_score": p.keep_score,
                "analysis_priority": p.analysis_priority,
                "venue": p.venue,
                "year": p.year,
            })

    return result
