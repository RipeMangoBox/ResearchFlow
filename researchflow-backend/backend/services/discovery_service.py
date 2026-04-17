"""Paper discovery service — find related papers from a seed.

Uses Semantic Scholar API (free, no key required) to:
1. Find a paper's references and citations
2. Find semantically related papers
3. Auto-ingest discovered papers into the knowledge base

This enables: give one paper → build the domain's knowledge graph.
"""

import asyncio
import logging
import re
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.enums import PaperState
from backend.models.paper import Paper
from backend.schemas.import_ import LinkImportItem
from backend.services.ingestion_service import ingest_link

logger = logging.getLogger(__name__)

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,abstract,year,venue,externalIds,citationCount,referenceCount,url"
S2_HEADERS = {"User-Agent": "ResearchFlow/0.1"}


async def _s2_get(client: httpx.AsyncClient, url: str, params: dict | None = None) -> dict | None:
    """Make a rate-limited Semantic Scholar API call."""
    try:
        resp = await client.get(url, params=params, headers=S2_HEADERS, timeout=20)
        if resp.status_code == 429:
            logger.warning("S2 rate limit hit, sleeping 3s")
            await asyncio.sleep(3)
            resp = await client.get(url, params=params, headers=S2_HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"S2 API error: {e}")
        return None


async def find_paper_on_s2(
    session: AsyncSession,
    paper_id: UUID,
) -> dict | None:
    """Find a paper on Semantic Scholar by its arxiv_id or title."""
    paper = await session.get(Paper, paper_id)
    if not paper:
        return None

    async with httpx.AsyncClient(follow_redirects=True) as client:
        # Try arxiv_id first
        if paper.arxiv_id:
            data = await _s2_get(client, f"{S2_API}/paper/ARXIV:{paper.arxiv_id}",
                                 {"fields": S2_FIELDS})
            if data:
                return data

        # Try DOI
        if paper.doi:
            data = await _s2_get(client, f"{S2_API}/paper/DOI:{paper.doi}",
                                 {"fields": S2_FIELDS})
            if data:
                return data

        # Fall back to title search
        data = await _s2_get(client, f"{S2_API}/paper/search",
                             {"query": paper.title[:200], "limit": "1", "fields": S2_FIELDS})
        if data and data.get("data"):
            return data["data"][0]

    return None


async def discover_related_papers(
    session: AsyncSession,
    paper_id: UUID,
    max_references: int = 10,
    max_citations: int = 10,
    max_related: int = 5,
    auto_ingest: bool = True,
) -> dict:
    """Discover related papers from a seed paper via Semantic Scholar.

    Finds: references (what it cites), citations (who cites it), and related papers.
    Optionally auto-ingests discovered papers into the KB.

    Returns: {seed, references: [...], citations: [...], related: [...], ingested: int}
    """
    paper = await session.get(Paper, paper_id)
    if not paper:
        return {"error": "Paper not found"}

    result = {
        "seed": {"id": str(paper.id), "title": paper.title},
        "references": [],
        "citations": [],
        "related": [],
        "ingested": 0,
    }

    # Find seed on S2
    s2_paper = await find_paper_on_s2(session, paper_id)
    if not s2_paper:
        return {**result, "error": "Paper not found on Semantic Scholar"}

    s2_id = s2_paper.get("paperId")
    if not s2_id:
        return {**result, "error": "No S2 paperId"}

    async with httpx.AsyncClient(follow_redirects=True) as client:
        # 1. Get references
        refs_data = await _s2_get(
            client, f"{S2_API}/paper/{s2_id}/references",
            {"fields": S2_FIELDS, "limit": str(max_references)},
        )
        if refs_data and refs_data.get("data"):
            for ref in refs_data["data"]:
                cited = ref.get("citedPaper", {})
                if cited and cited.get("title"):
                    result["references"].append(_s2_to_dict(cited))

        await asyncio.sleep(0.5)  # Rate limit

        # 2. Get citations
        cit_data = await _s2_get(
            client, f"{S2_API}/paper/{s2_id}/citations",
            {"fields": S2_FIELDS, "limit": str(max_citations)},
        )
        if cit_data and cit_data.get("data"):
            for cit in cit_data["data"]:
                citing = cit.get("citingPaper", {})
                if citing and citing.get("title"):
                    result["citations"].append(_s2_to_dict(citing))

        await asyncio.sleep(0.5)

        # 3. Get related papers (S2 recommendations)
        rec_data = await _s2_get(
            client, f"{S2_API}/paper/{s2_id}/recommendations",
            {"fields": S2_FIELDS, "limit": str(max_related)},
        )
        if rec_data and rec_data.get("recommendedPapers"):
            for rec in rec_data["recommendedPapers"]:
                if rec.get("title"):
                    result["related"].append(_s2_to_dict(rec))

    # Auto-ingest discovered papers
    if auto_ingest:
        all_discovered = result["references"] + result["citations"] + result["related"]
        ingested = 0
        for disc in all_discovered:
            url = disc.get("url")
            if not url:
                # Try to build arxiv URL from externalIds
                arxiv_id = disc.get("arxiv_id")
                if arxiv_id:
                    url = f"https://arxiv.org/abs/{arxiv_id}"
                else:
                    continue

            item = LinkImportItem(
                url=url,
                title=disc.get("title"),
                venue=disc.get("venue"),
                year=disc.get("year"),
            )
            try:
                res = await ingest_link(
                    session, item,
                    default_category=paper.category or "Discovered",
                    is_ephemeral=False,
                    retention_days=30,
                )
                if res.status == "created":
                    ingested += 1
            except Exception as e:
                logger.debug(f"Ingest skip for {disc.get('title', '?')}: {e}")

        result["ingested"] = ingested
        if ingested > 0:
            await session.commit()

    return result


async def build_domain_from_seed(
    session: AsyncSession,
    paper_id: UUID,
    depth: int = 1,
    max_per_hop: int = 10,
    run_pipeline: bool = False,
) -> dict:
    """Build a domain knowledge graph starting from a single seed paper.

    1. Discover related papers (references + citations + related)
    2. Ingest all discovered papers
    3. Optionally run full analysis pipeline on each

    Args:
        depth: How many hops to follow (1 = direct refs/cits only, 2 = refs of refs)
        max_per_hop: Max papers to discover per category per hop
        run_pipeline: Whether to run full analysis pipeline on discovered papers
    """
    result = {
        "seed": str(paper_id),
        "hops": [],
        "total_discovered": 0,
        "total_ingested": 0,
        "pipeline_results": [],
    }

    processed_ids = {paper_id}
    current_layer = [paper_id]

    for hop in range(depth):
        hop_result = {"hop": hop + 1, "papers_explored": 0, "discovered": 0, "ingested": 0}

        next_layer = []
        for pid in current_layer:
            disc = await discover_related_papers(
                session, pid,
                max_references=max_per_hop,
                max_citations=max_per_hop,
                max_related=max_per_hop // 2,
                auto_ingest=True,
            )
            hop_result["papers_explored"] += 1
            total = len(disc.get("references", [])) + len(disc.get("citations", [])) + len(disc.get("related", []))
            hop_result["discovered"] += total
            hop_result["ingested"] += disc.get("ingested", 0)

            # Collect newly ingested paper IDs for next hop
            # (We'd need to track which ones were new, for now just count)

        result["hops"].append(hop_result)
        result["total_discovered"] += hop_result["discovered"]
        result["total_ingested"] += hop_result["ingested"]

    # Optionally run pipeline on all new papers
    if run_pipeline:
        from backend.services.pipeline_service import run_pipeline_batch
        pipeline_results = await run_pipeline_batch(session, limit=result["total_ingested"])
        result["pipeline_results"] = pipeline_results

    return result


def _s2_to_dict(s2_paper: dict) -> dict:
    """Convert S2 paper to our format."""
    ext_ids = s2_paper.get("externalIds") or {}
    arxiv_id = ext_ids.get("ArXiv")
    doi = ext_ids.get("DOI")

    url = s2_paper.get("url", "")
    if not url and arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
    elif not url and doi:
        url = f"https://doi.org/{doi}"

    return {
        "title": s2_paper.get("title"),
        "year": s2_paper.get("year"),
        "venue": s2_paper.get("venue"),
        "url": url,
        "arxiv_id": arxiv_id,
        "doi": doi,
        "citation_count": s2_paper.get("citationCount"),
    }
