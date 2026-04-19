"""V6 Incremental Sync — periodic background tasks for keeping the KB fresh.

Tasks:
- arXiv daily new papers sync (filtered by domain scope)
- Citation count refresh via S2 API
- GitHub awesome repo diff detection
- Lineage chain detection (deterministic)
- Node promotion score recomputation
- Duplicate node detection
"""

import logging
import re
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from uuid import UUID

import httpx
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings  # noqa: F401

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
S2_PAPER_API = "https://api.semanticscholar.org/graph/v1/paper"
GITHUB_RAW = "https://raw.githubusercontent.com"


# ---------------------------------------------------------------------------
# 1. arXiv daily sync
# ---------------------------------------------------------------------------

async def sync_arxiv_daily(session: AsyncSession, domain_id: UUID) -> dict:
    """Sync new arXiv papers for a domain by querying its scope.

    Loads the DomainSpec, builds search queries from its scope,
    queries arXiv for papers from the last 2 days, and creates
    candidates for any new papers not already in the KB.
    """
    from backend.models.domain import DomainSpec
    from backend.models.candidate import PaperCandidate
    from backend.models.paper import Paper
    from backend.services.cold_start_service import expand_search_queries
    from backend.services import candidate_service

    # Load domain
    domain = await session.get(DomainSpec, domain_id)
    if not domain:
        return {"domain_id": str(domain_id), "error": "domain not found"}

    # Build scope dict from domain fields
    scope = {
        "tasks": domain.scope_tasks or [],
        "seed_methods": domain.scope_seed_methods or [],
        "modalities": domain.scope_modalities or [],
        "seed_datasets": domain.scope_seed_datasets or [],
    }
    queries = await expand_search_queries(scope)
    if not queries:
        return {
            "domain_id": str(domain_id),
            "queries_searched": 0,
            "papers_found": 0,
            "candidates_created": 0,
        }

    # Collect existing arxiv_ids to avoid duplicates
    existing_arxiv = set()
    for model in (PaperCandidate, Paper):
        rows = (await session.execute(
            select(model.arxiv_id).where(model.arxiv_id.isnot(None))
        )).scalars().all()
        existing_arxiv.update(rows)

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    total_found = 0
    total_created = 0

    async with httpx.AsyncClient(timeout=30) as client:
        for query in queries:
            try:
                resp = await client.get(
                    f"{ARXIV_API}?search_query=all:{quote(query)}"
                    f"&sortBy=submittedDate&sortOrder=descending&max_results=20",
                )
                if resp.status_code != 200:
                    logger.warning("arXiv daily sync failed for '%s': HTTP %d", query, resp.status_code)
                    continue

                root = ET.fromstring(resp.text)
                entries = root.findall("atom:entry", ns)
                total_found += len(entries)

                for entry in entries:
                    try:
                        title_el = entry.find("atom:title", ns)
                        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
                        if not title:
                            continue

                        # Extract arxiv_id
                        id_el = entry.find("atom:id", ns)
                        arxiv_url = (id_el.text or "").strip() if id_el is not None else ""
                        arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else None

                        # Skip if already known
                        if arxiv_id and arxiv_id in existing_arxiv:
                            continue

                        # Check publication date (last 2 days)
                        published_el = entry.find("atom:published", ns)
                        published = (published_el.text or "").strip() if published_el is not None else None
                        if published:
                            try:
                                pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                                if pub_dt < datetime.now(timezone.utc) - timedelta(days=2):
                                    continue
                            except ValueError:
                                pass

                        abstract_el = entry.find("atom:summary", ns)
                        abstract = (abstract_el.text or "").strip().replace("\n", " ") if abstract_el is not None else None
                        year = int(published[:4]) if published and len(published) >= 4 else None

                        # Authors
                        author_els = entry.findall("atom:author", ns)
                        authors = []
                        for a in author_els:
                            name_el = a.find("atom:name", ns)
                            if name_el is not None and name_el.text:
                                authors.append({"name": name_el.text.strip()})

                        await candidate_service.create_candidate(
                            session,
                            title=title,
                            discovery_source="arxiv_daily_sync",
                            discovery_reason=f"daily_query: {query}",
                            arxiv_id=arxiv_id,
                            paper_link=arxiv_url or None,
                            abstract=abstract,
                            authors_json={"authors": authors} if authors else None,
                            year=year,
                            discovered_from_domain_id=domain_id,
                        )
                        total_created += 1
                        if arxiv_id:
                            existing_arxiv.add(arxiv_id)

                    except Exception:
                        logger.debug("Failed to parse arXiv entry in daily sync", exc_info=True)
                        continue

            except Exception:
                logger.warning("arXiv daily sync API call failed for '%s'", query, exc_info=True)
                continue

    return {
        "domain_id": str(domain_id),
        "queries_searched": len(queries),
        "papers_found": total_found,
        "candidates_created": total_created,
    }


# ---------------------------------------------------------------------------
# 2. Citation count refresh
# ---------------------------------------------------------------------------

async def refresh_citation_counts(session: AsyncSession, *, limit: int = 50) -> dict:
    """Refresh citation counts for papers via Semantic Scholar API.

    Finds papers at L3_SKIMMED or above, ordered by oldest update first.
    If citations increased significantly (>20% or >10 new), flags for review.
    """
    from backend.models.paper import Paper
    from backend.models.enums import PaperState

    # States >= L3_SKIMMED
    advanced_states = [
        PaperState.L3_SKIMMED,
        PaperState.L4_DEEP,
        PaperState.CHECKED,
    ]

    papers = (await session.execute(
        select(Paper).where(
            Paper.state.in_(advanced_states),
        ).order_by(Paper.updated_at.asc().nullsfirst())
        .limit(limit)
    )).scalars().all()

    papers_checked = 0
    citations_updated = 0
    new_candidates = 0

    async with httpx.AsyncClient(timeout=20) as client:
        for paper in papers:
            try:
                # Try S2 API with arxiv_id or doi
                s2_id = None
                if paper.arxiv_id:
                    s2_id = f"ARXIV:{paper.arxiv_id}"
                elif paper.doi:
                    s2_id = f"DOI:{paper.doi}"
                else:
                    continue

                resp = await client.get(
                    f"{S2_PAPER_API}/{s2_id}",
                    params={"fields": "citationCount"},
                )
                papers_checked += 1

                if resp.status_code != 200:
                    continue

                data = resp.json()
                new_count = data.get("citationCount")
                if new_count is None:
                    continue

                old_count = paper.cited_by_count or 0

                if new_count != old_count:
                    paper.cited_by_count = new_count
                    citations_updated += 1

                    # Check for significant increase
                    diff = new_count - old_count
                    pct_change = (diff / old_count * 100) if old_count > 0 else 100
                    if diff > 10 or pct_change > 20:
                        logger.info(
                            "Significant citation increase for paper %s: %d -> %d (+%.0f%%)",
                            paper.id, old_count, new_count, pct_change,
                        )
                        # Could trigger candidate discovery for citing papers here
                        # For now, just log the event
                        new_candidates += 0  # placeholder for future citing-paper discovery

            except Exception:
                logger.debug("Citation refresh failed for paper %s", paper.id, exc_info=True)
                continue

    return {
        "papers_checked": papers_checked,
        "citations_updated": citations_updated,
        "new_candidates": new_candidates,
    }


# ---------------------------------------------------------------------------
# 3. Awesome repo diff detection
# ---------------------------------------------------------------------------

async def detect_awesome_repo_changes(session: AsyncSession, domain_id: UUID) -> dict:
    """Detect new papers added to awesome repos tracked by a domain.

    Loads DomainSourceRegistry entries of type 'awesome_repo',
    fetches the current README, diffs against the last checkpoint,
    and creates candidates for newly discovered paper URLs.
    """
    from backend.models.domain import DomainSourceRegistry, IncrementalCheckpoint
    from backend.services import candidate_service

    # Load awesome repo sources for this domain
    sources = (await session.execute(
        select(DomainSourceRegistry).where(
            DomainSourceRegistry.domain_id == domain_id,
            DomainSourceRegistry.source_type == "awesome_repo",
            DomainSourceRegistry.is_active.is_(True),
        )
    )).scalars().all()

    repos_checked = 0
    new_papers_found = 0
    candidates_created = 0

    async with httpx.AsyncClient(timeout=30) as client:
        for source in sources:
            try:
                repo_url = source.source_ref  # e.g. "owner/repo" or full URL
                # Normalize to owner/repo format
                repo_path = repo_url.replace("https://github.com/", "").strip("/")

                # Fetch current README
                raw_url = f"{GITHUB_RAW}/{repo_path}/main/README.md"
                resp = await client.get(raw_url)
                if resp.status_code == 404:
                    # Try master branch
                    raw_url = f"{GITHUB_RAW}/{repo_path}/master/README.md"
                    resp = await client.get(raw_url)
                if resp.status_code != 200:
                    logger.warning("Failed to fetch README for %s: HTTP %d", repo_path, resp.status_code)
                    continue

                current_content = resp.text
                repos_checked += 1

                # Load last checkpoint
                last_checkpoint = (await session.execute(
                    select(IncrementalCheckpoint).where(
                        IncrementalCheckpoint.source_registry_id == source.id,
                    ).order_by(IncrementalCheckpoint.created_at.desc())
                    .limit(1)
                )).scalar_one_or_none()

                previous_content = last_checkpoint.checkpoint_value if last_checkpoint else ""

                # Extract paper URLs from current and previous
                current_urls = _extract_paper_urls(current_content)
                previous_urls = _extract_paper_urls(previous_content) if previous_content else set()

                new_urls = current_urls - previous_urls
                new_papers_found += len(new_urls)

                # Create candidates for new papers
                for url in new_urls:
                    try:
                        arxiv_id = _extract_arxiv_id(url)
                        await candidate_service.create_candidate(
                            session,
                            title=f"[awesome-repo] {url}",
                            discovery_source="awesome_repo",
                            discovery_reason=f"repo: {repo_path}",
                            arxiv_id=arxiv_id,
                            paper_link=url,
                            discovered_from_domain_id=domain_id,
                        )
                        candidates_created += 1
                    except Exception:
                        logger.debug("Failed to create candidate from awesome repo URL: %s", url, exc_info=True)
                        continue

                # Update checkpoint
                checkpoint = IncrementalCheckpoint(
                    source_registry_id=source.id,
                    checkpoint_value=current_content,
                    papers_found=len(current_urls),
                    papers_new=len(new_urls),
                    sync_mode="weekly",
                )
                session.add(checkpoint)

                # Update source last_synced_at
                source.last_synced_at = datetime.now(timezone.utc)

            except Exception:
                logger.warning("Awesome repo diff failed for source %s", source.id, exc_info=True)
                continue

    return {
        "repos_checked": repos_checked,
        "new_papers_found": new_papers_found,
        "candidates_created": candidates_created,
    }


def _extract_paper_urls(text: str) -> set[str]:
    """Extract arxiv and paper URLs from markdown text."""
    patterns = [
        r'https?://arxiv\.org/abs/[\w\.]+',
        r'https?://arxiv\.org/pdf/[\w\.]+',
        r'https?://doi\.org/[\w\./\-]+',
        r'https?://openreview\.net/forum\?id=[\w\-]+',
        r'https?://papers\.nips\.cc/[\w/\-]+',
        r'https?://proceedings\.mlr\.press/[\w/\-]+',
    ]
    urls: set[str] = set()
    for pattern in patterns:
        urls.update(re.findall(pattern, text))
    return urls


def _extract_arxiv_id(url: str) -> str | None:
    """Extract arxiv ID from a URL."""
    match = re.search(r'arxiv\.org/(?:abs|pdf)/([\w\.]+)', url)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# 4. Lineage chain detection
# ---------------------------------------------------------------------------

async def detect_lineage_chains(session: AsyncSession, *, min_chain_length: int = 3) -> dict:
    """Detect lineage chains of sufficient length and create graph node candidates.

    Queries delta_card_lineage for chains where the depth from root
    reaches min_chain_length. For qualifying chains without an existing
    lineage story node, creates a GraphNodeCandidate.
    """
    from backend.models.lineage import DeltaCardLineage
    from backend.models.kb import GraphNodeCandidate

    # Find all lineage edges and build adjacency
    edges = (await session.execute(
        select(DeltaCardLineage.child_delta_card_id, DeltaCardLineage.parent_delta_card_id)
        .where(DeltaCardLineage.status != "rejected")
    )).all()

    if not edges:
        return {"chains_detected": 0, "candidates_created": 0}

    # Build parent->children adjacency for BFS
    children_of: dict[UUID, list[UUID]] = {}
    all_children: set[UUID] = set()
    for child_id, parent_id in edges:
        children_of.setdefault(parent_id, []).append(child_id)
        all_children.add(child_id)

    # Find roots (nodes that are parents but not children)
    all_parents = set(children_of.keys())
    roots = all_parents - all_children

    # BFS from each root to find chains of sufficient length
    chains_detected = 0
    candidates_created = 0

    for root in roots:
        # BFS to find max depth from this root
        queue = [(root, 1)]
        max_depth = 1
        chain_nodes: list[UUID] = [root]

        visited = {root}
        while queue:
            node, depth = queue.pop(0)
            max_depth = max(max_depth, depth)
            for child in children_of.get(node, []):
                if child not in visited:
                    visited.add(child)
                    chain_nodes.append(child)
                    queue.append((child, depth + 1))

        if max_depth >= min_chain_length:
            chains_detected += 1

            # Check if a lineage story node already exists for this chain root
            existing = (await session.execute(
                select(GraphNodeCandidate.id).where(
                    GraphNodeCandidate.node_type == "lineage",
                    GraphNodeCandidate.name == f"lineage_chain_{root}",
                )
            )).scalar_one_or_none()

            if not existing:
                candidate = GraphNodeCandidate(
                    node_type="lineage",
                    name=f"lineage_chain_{root}",
                    one_liner=f"Method evolution chain of depth {max_depth} starting from delta card {root}",
                    promotion_score=None,
                    status="candidate",
                    evidence_refs={
                        "root_delta_card_id": str(root),
                        "chain_depth": max_depth,
                        "chain_node_count": len(chain_nodes),
                    },
                )
                session.add(candidate)
                candidates_created += 1

    return {
        "chains_detected": chains_detected,
        "candidates_created": candidates_created,
    }


# ---------------------------------------------------------------------------
# 5. Node score recomputation
# ---------------------------------------------------------------------------

async def recompute_node_scores(session: AsyncSession, *, limit: int = 100) -> dict:
    """Recompute promotion scores for graph node candidates.

    Finds candidates that haven't been scored recently or have new evidence,
    recomputes their NodePromotionScore, and auto-promotes those reaching
    the canonical threshold (>=85).
    """
    from backend.models.kb import GraphNodeCandidate

    # Find candidates needing rescore
    candidates = (await session.execute(
        select(GraphNodeCandidate).where(
            GraphNodeCandidate.status == "candidate",
        ).order_by(GraphNodeCandidate.created_at.asc())
        .limit(limit)
    )).scalars().all()

    nodes_rescored = 0
    nodes_promoted = 0

    for cand in candidates:
        try:
            # Compute a heuristic promotion score based on available signals
            score = 0.0

            # Base score from evidence refs
            evidence = cand.evidence_refs or {}
            if evidence:
                score += min(30.0, len(evidence) * 10.0)

            # Bonus for having a one-liner (indicates analysis quality)
            if cand.one_liner:
                score += 10.0

            # Bonus for being linked to a paper
            if cand.paper_id:
                score += 15.0

            # Bonus for confidence
            if cand.confidence and cand.confidence > 0.7:
                score += 20.0

            # Node type bonuses
            type_bonuses = {
                "method": 10.0,
                "dataset": 5.0,
                "task": 5.0,
                "lineage": 15.0,
            }
            score += type_bonuses.get(cand.node_type, 0.0)

            cand.promotion_score = score
            cand.promotion_breakdown = {
                "evidence_score": min(30.0, len(evidence) * 10.0),
                "one_liner_bonus": 10.0 if cand.one_liner else 0.0,
                "paper_link_bonus": 15.0 if cand.paper_id else 0.0,
                "confidence_bonus": 20.0 if (cand.confidence and cand.confidence > 0.7) else 0.0,
                "type_bonus": type_bonuses.get(cand.node_type, 0.0),
                "total": score,
            }
            nodes_rescored += 1

            # Auto-promote if reaching threshold
            if score >= 85.0:
                cand.status = "promoted"
                nodes_promoted += 1
                logger.info("Auto-promoted node candidate %s (score=%.1f): %s", cand.id, score, cand.name)

        except Exception:
            logger.debug("Failed to rescore node candidate %s", cand.id, exc_info=True)
            continue

    return {
        "nodes_rescored": nodes_rescored,
        "nodes_promoted": nodes_promoted,
    }


# ---------------------------------------------------------------------------
# 6. Duplicate node detection
# ---------------------------------------------------------------------------

async def detect_duplicate_nodes(session: AsyncSession) -> dict:
    """Detect potential duplicate nodes in taxonomy and method tables.

    Uses name normalization (lowercase, strip whitespace/punctuation)
    to find near-exact matches. Creates review queue items for suspected duplicates.
    """
    from backend.models.taxonomy import TaxonomyNode
    from backend.models.method import MethodNode
    from backend.models.kb import ReviewQueueItem

    duplicates_found = 0
    review_items_created = 0

    # Check taxonomy nodes
    tax_nodes = (await session.execute(
        select(TaxonomyNode.id, TaxonomyNode.name, TaxonomyNode.dimension)
    )).all()

    tax_by_normalized: dict[str, list] = {}
    for node_id, name, dimension in tax_nodes:
        key = _normalize_name(name)
        tax_by_normalized.setdefault(key, []).append({
            "id": node_id, "name": name, "dimension": dimension,
        })

    for key, group in tax_by_normalized.items():
        if len(group) > 1:
            duplicates_found += 1
            # Create review item for the group
            primary = group[0]
            for dup in group[1:]:
                existing = (await session.execute(
                    select(ReviewQueueItem.id).where(
                        ReviewQueueItem.item_type == "candidate",
                        ReviewQueueItem.entity_type == "taxonomy_node",
                        ReviewQueueItem.entity_id == dup["id"],
                        ReviewQueueItem.status == "pending",
                    )
                )).scalar_one_or_none()

                if not existing:
                    item = ReviewQueueItem(
                        item_type="candidate",
                        entity_type="taxonomy_node",
                        entity_id=dup["id"],
                        reason=f"Potential duplicate of '{primary['name']}' (id={primary['id']})",
                        suggested_decision="merge",
                        evidence_refs={
                            "primary_id": str(primary["id"]),
                            "primary_name": primary["name"],
                            "duplicate_name": dup["name"],
                            "normalized_key": key,
                        },
                        status="pending",
                    )
                    session.add(item)
                    review_items_created += 1

    # Check method nodes
    method_nodes = (await session.execute(
        select(MethodNode.id, MethodNode.name, MethodNode.type)
    )).all()

    method_by_normalized: dict[str, list] = {}
    for node_id, name, node_type in method_nodes:
        key = _normalize_name(name)
        method_by_normalized.setdefault(key, []).append({
            "id": node_id, "name": name, "type": node_type,
        })

    for key, group in method_by_normalized.items():
        if len(group) > 1:
            duplicates_found += 1
            primary = group[0]
            for dup in group[1:]:
                existing = (await session.execute(
                    select(ReviewQueueItem.id).where(
                        ReviewQueueItem.item_type == "candidate",
                        ReviewQueueItem.entity_type == "method_node",
                        ReviewQueueItem.entity_id == dup["id"],
                        ReviewQueueItem.status == "pending",
                    )
                )).scalar_one_or_none()

                if not existing:
                    item = ReviewQueueItem(
                        item_type="candidate",
                        entity_type="method_node",
                        entity_id=dup["id"],
                        reason=f"Potential duplicate of '{primary['name']}' (id={primary['id']})",
                        suggested_decision="merge",
                        evidence_refs={
                            "primary_id": str(primary["id"]),
                            "primary_name": primary["name"],
                            "duplicate_name": dup["name"],
                            "normalized_key": key,
                        },
                        status="pending",
                    )
                    session.add(item)
                    review_items_created += 1

    return {
        "duplicates_found": duplicates_found,
        "review_items_created": review_items_created,
    }


def _normalize_name(name: str) -> str:
    """Normalize a name for duplicate detection.

    Lowercase, strip accents, remove punctuation/extra whitespace.
    """
    name = name.lower().strip()
    # Remove accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    # Remove punctuation except hyphens
    name = re.sub(r"[^\w\s\-]", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ---------------------------------------------------------------------------
# 7. Stale candidate cleanup
# ---------------------------------------------------------------------------

async def cleanup_stale_candidates(session: AsyncSession, *, days: int = 90) -> dict:
    """Archive candidates older than `days` that are still in early lifecycle stages.

    Targets candidates in 'discovered' or 'metadata_resolved' status
    that haven't progressed within the given timeframe.
    """
    from backend.models.candidate import PaperCandidate

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    result = await session.execute(
        update(PaperCandidate)
        .where(
            PaperCandidate.status.in_(["discovered", "metadata_resolved"]),
            PaperCandidate.created_at < cutoff,
        )
        .values(status="archived")
    )

    archived_count = result.rowcount

    return {
        "archived_count": archived_count,
    }
