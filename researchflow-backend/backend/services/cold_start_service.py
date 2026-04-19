"""V6 Cold Start — builds a domain knowledge base from a manifest.

Flow: Save Domain → Build Skeleton → Query Expand → Wide Harvest → Score → Anchor Select → Deep Ingest → Graph Build
"""

import itertools
import logging
import xml.etree.ElementTree as ET
from urllib.parse import quote
from uuid import UUID

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.domain import DomainSpec
from backend.models.method import MethodNode
from backend.models.taxonomy import TaxonomyNode
from backend.services import candidate_service

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
S2_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"


# ---------------------------------------------------------------------------
# 1. Main entry point
# ---------------------------------------------------------------------------

async def cold_start_domain(session: AsyncSession, manifest: dict) -> dict:
    """Bootstrap a domain knowledge base from a manifest.

    The manifest dict has: name, display_name_zh, scope
    (scope contains modalities, tasks, paradigms, seed_methods,
     seed_models, seed_datasets, negative_scope)

    Steps:
        a. Create DomainSpec from manifest
        b. Build skeleton taxonomy + method nodes
        c. Expand search queries from scope cross-product
        d. Wide metadata harvest (arXiv + S2)
        e. Batch score all candidates
        f. Anchor selection (auto-promote top K)
        g. Return summary (deep ingest is deferred to workers)
    """
    scope = manifest.get("scope", {})

    # ── a. Create DomainSpec ──────────────────────────────────────
    domain = DomainSpec(
        name=manifest["name"],
        description=manifest.get("display_name_zh"),
        scope_modalities=scope.get("modalities", []),
        scope_tasks=scope.get("tasks", []),
        scope_paradigms=scope.get("paradigms", []),
        scope_seed_methods=scope.get("seed_methods", []),
        scope_seed_models=scope.get("seed_models", []),
        scope_seed_datasets=scope.get("seed_datasets", []),
        negative_scope=scope.get("negative_scope", []),
        status="active",
    )
    session.add(domain)
    await session.flush()  # get domain.id

    domain_id = domain.id
    skeleton_count = 0

    # ── b. Build skeleton nodes ───────────────────────────────────
    for task_name in scope.get("tasks", []):
        node = TaxonomyNode(
            name=task_name,
            dimension="task",
            status="candidate",
        )
        session.add(node)
        skeleton_count += 1

    for dataset_name in scope.get("seed_datasets", []):
        node = TaxonomyNode(
            name=dataset_name,
            dimension="dataset",
            status="candidate",
        )
        session.add(node)
        skeleton_count += 1

    for model_name in scope.get("seed_models", []):
        node = TaxonomyNode(
            name=model_name,
            dimension="model_family",
            status="candidate",
        )
        session.add(node)
        skeleton_count += 1

    for method_name in scope.get("seed_methods", []):
        node = MethodNode(
            name=method_name,
            type="algorithm",
            maturity="seed",
        )
        session.add(node)
        skeleton_count += 1

    await session.flush()

    # ── c. Query expansion ────────────────────────────────────────
    queries = await expand_search_queries(scope)

    # ── d. Wide metadata harvest ──────────────────────────────────
    arxiv_count = await harvest_from_arxiv(session, queries, domain_id)
    s2_count = await harvest_from_s2(session, queries, domain_id)
    dblp_count = await harvest_from_dblp(session, queries, domain_id)
    total_candidates = arxiv_count + s2_count + dblp_count

    await session.flush()

    # ── e. Batch score ────────────────────────────────────────────
    scored = await candidate_service.score_batch(session, limit=total_candidates or 500)

    # ── f. Anchor selection ───────────────────────────────────────
    budget_k = domain.budget_deep_ingest or 50
    promoted = await candidate_service.auto_promote_batch(
        session,
        threshold=0.0,  # take top K regardless of threshold
        limit=budget_k,
        domain_id=domain_id,
    )

    return {
        "domain_id": str(domain_id),
        "skeleton_nodes_created": skeleton_count,
        "candidates_created": total_candidates,
        "candidates_scored": scored,
        "anchors_promoted": len(promoted),
        "search_queries_used": len(queries),
    }


# ---------------------------------------------------------------------------
# 2. Query expansion
# ---------------------------------------------------------------------------

async def expand_search_queries(scope: dict) -> list[str]:
    """Build search queries from the cross-product of tasks x methods x modalities.

    Also adds individual seed terms (methods, datasets) as direct queries.
    Returns a deduplicated list.
    """
    tasks = scope.get("tasks", [])
    methods = scope.get("seed_methods", [])
    modalities = scope.get("modalities", [])
    datasets = scope.get("seed_datasets", [])

    queries: list[str] = []

    # Cross-product: tasks x methods x modalities
    if tasks and methods and modalities:
        for t, m, mod in itertools.product(tasks, methods, modalities):
            queries.append(f"{t} {m} {mod}")
    elif tasks and methods:
        for t, m in itertools.product(tasks, methods):
            queries.append(f"{t} {m}")
    elif tasks and modalities:
        for t, mod in itertools.product(tasks, modalities):
            queries.append(f"{t} {mod}")
    elif tasks:
        queries.extend(tasks)

    # Direct seed terms
    for m in methods:
        queries.append(m)
    for d in datasets:
        queries.append(d)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q_lower = q.strip().lower()
        if q_lower and q_lower not in seen:
            seen.add(q_lower)
            unique.append(q.strip())

    return unique


# ---------------------------------------------------------------------------
# 3. arXiv harvest
# ---------------------------------------------------------------------------

async def harvest_from_arxiv(
    session: AsyncSession,
    queries: list[str],
    domain_id: UUID,
    *,
    max_per_query: int = 20,
) -> int:
    """Search arXiv API for each query and create candidates.

    Returns total number of candidates created.
    """
    total_created = 0
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    async with httpx.AsyncClient(timeout=30) as client:
        for query in queries:
            try:
                # Build structured arXiv query
                words = query.split()
                if len(words) >= 3:
                    # First word → title, rest → abstract
                    search_query = f"ti:{quote(words[0])} AND abs:{quote(' '.join(words[1:]))}"
                elif len(words) == 2:
                    search_query = f"ti:{quote(words[0])} AND abs:{quote(words[1])}"
                else:
                    search_query = f"ti:{quote(query)} OR abs:{quote(query)}"

                resp = await client.get(
                    ARXIV_API,
                    params={
                        "search_query": search_query,
                        "sortBy": "relevance",
                        "max_results": str(max_per_query),
                    },
                )
                if resp.status_code != 200:
                    logger.warning("arXiv search failed for '%s': HTTP %d", query, resp.status_code)
                    continue

                root = ET.fromstring(resp.text)
                entries = root.findall("atom:entry", ns)

                for entry in entries:
                    try:
                        title_el = entry.find("atom:title", ns)
                        abstract_el = entry.find("atom:summary", ns)
                        published_el = entry.find("atom:published", ns)

                        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
                        if not title:
                            continue

                        abstract = (abstract_el.text or "").strip().replace("\n", " ") if abstract_el is not None else None
                        published = (published_el.text or "").strip() if published_el is not None else None
                        year = int(published[:4]) if published and len(published) >= 4 else None

                        # Extract arxiv_id from entry id
                        id_el = entry.find("atom:id", ns)
                        arxiv_url = (id_el.text or "").strip() if id_el is not None else ""
                        arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else None

                        # Extract authors
                        author_els = entry.findall("atom:author", ns)
                        authors = []
                        for a in author_els:
                            name_el = a.find("atom:name", ns)
                            if name_el is not None and name_el.text:
                                authors.append({"name": name_el.text.strip()})

                        await candidate_service.create_candidate(
                            session,
                            title=title,
                            discovery_source="arxiv_cold_start",
                            discovery_reason=f"query: {query}",
                            arxiv_id=arxiv_id,
                            paper_link=arxiv_url or None,
                            abstract=abstract,
                            authors_json={"authors": authors} if authors else None,
                            year=year,
                            discovered_from_domain_id=domain_id,
                        )
                        total_created += 1
                    except Exception:
                        logger.debug("Failed to parse arXiv entry for query '%s'", query, exc_info=True)
                        continue

            except Exception:
                logger.warning("arXiv API call failed for query '%s'", query, exc_info=True)
                continue

    return total_created


# ---------------------------------------------------------------------------
# 4. Semantic Scholar harvest
# ---------------------------------------------------------------------------

async def harvest_from_s2(
    session: AsyncSession,
    queries: list[str],
    domain_id: UUID,
    *,
    max_per_query: int = 20,
) -> int:
    """Search Semantic Scholar API for each query and create candidates.

    Returns total number of candidates created.
    """
    total_created = 0
    fields = "title,abstract,year,authors,citationCount,externalIds,url"

    async with httpx.AsyncClient(timeout=30) as client:
        for query in queries:
            try:
                resp = await client.get(
                    S2_SEARCH_API,
                    params={
                        "query": query,
                        "limit": str(max_per_query),
                        "fields": fields,
                    },
                )
                if resp.status_code != 200:
                    logger.warning("S2 search failed for '%s': HTTP %d", query, resp.status_code)
                    continue

                data = resp.json()
                papers = data.get("data", [])

                for paper in papers:
                    try:
                        title = (paper.get("title") or "").strip()
                        if not title:
                            continue

                        abstract = paper.get("abstract")
                        year = paper.get("year")
                        citation_count = paper.get("citationCount")
                        url = paper.get("url")

                        # Extract IDs
                        ext_ids = paper.get("externalIds") or {}
                        arxiv_id = ext_ids.get("ArXiv")
                        doi = ext_ids.get("DOI")

                        # Authors
                        raw_authors = paper.get("authors") or []
                        authors = [{"name": a.get("name", "")} for a in raw_authors if a.get("name")]

                        await candidate_service.create_candidate(
                            session,
                            title=title,
                            discovery_source="s2_cold_start",
                            discovery_reason=f"query: {query}",
                            arxiv_id=arxiv_id,
                            doi=doi,
                            paper_link=url,
                            abstract=abstract,
                            authors_json={"authors": authors} if authors else None,
                            year=year,
                            citation_count=citation_count,
                            discovered_from_domain_id=domain_id,
                        )
                        total_created += 1
                    except Exception:
                        logger.debug("Failed to process S2 result for query '%s'", query, exc_info=True)
                        continue

            except Exception:
                logger.warning("S2 API call failed for query '%s'", query, exc_info=True)
                continue

    return total_created


# ---------------------------------------------------------------------------
# 5. DBLP harvest
# ---------------------------------------------------------------------------

async def harvest_from_dblp(
    session: AsyncSession,
    queries: list[str],
    domain_id: UUID,
    *,
    max_per_query: int = 20,
) -> int:
    """Search DBLP for papers matching queries and create candidates."""
    total_created = 0
    dblp_api = "https://dblp.org/search/publ/api"

    async with httpx.AsyncClient(timeout=30) as client:
        for query in queries[:5]:  # Limit to avoid rate limits
            try:
                resp = await client.get(
                    dblp_api,
                    params={"q": query, "format": "json", "h": str(max_per_query)},
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                hits = data.get("result", {}).get("hits", {}).get("hit", [])

                for hit in hits:
                    info = hit.get("info", {})
                    title = info.get("title", "").strip()
                    if not title:
                        continue
                    venue = info.get("venue", "")
                    year_str = info.get("year", "")
                    year = int(year_str) if year_str.isdigit() else None
                    doi = info.get("doi", "")
                    url = info.get("ee", "") or info.get("url", "")

                    authors = []
                    author_info = info.get("authors", {}).get("author", [])
                    if isinstance(author_info, dict):
                        author_info = [author_info]
                    for a in author_info:
                        name = a.get("text", "") if isinstance(a, dict) else str(a)
                        if name:
                            authors.append({"name": name})

                    try:
                        await candidate_service.create_candidate(
                            session,
                            title=title,
                            discovery_source="dblp_proceedings",
                            discovery_reason=f"dblp_search: {query}",
                            doi=doi if doi else None,
                            paper_link=url if url else None,
                            venue=venue if venue else None,
                            year=year,
                            authors_json=authors if authors else None,
                            discovered_from_domain_id=domain_id,
                        )
                        total_created += 1
                    except Exception:
                        continue
            except Exception as e:
                logger.warning("DBLP search failed for '%s': %s", query, e)

    return total_created
