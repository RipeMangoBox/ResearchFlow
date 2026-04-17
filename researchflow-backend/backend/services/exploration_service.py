"""Research exploration service — multi-hop cognitive iteration tracking.

Solves the core problem: research exploration is NOT a single query.
It's an evolving process where understanding deepens through iterations:

  "RL有优势消失问题" → 找到了但都是插件型 → 需要本质改进
  → 发现需要可扩展方案 → think with image / agentic → GDPO 分组 reward

This service tracks:
1. The exploration path (sequence of queries + pivots)
2. What was found and what was rejected
3. Why the user pivoted (LLM infers from query evolution)
4. What to explore next (connects dots across sessions)

Uses SearchSession as the DB model, but adds cognitive tracking.
"""

import json
import logging
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.research import ProjectBottleneck, SearchSession
from backend.models.paper import Paper
from backend.models.delta_card import DeltaCard
from backend.models.graph import IdeaDelta

logger = logging.getLogger(__name__)


async def start_exploration(
    session: AsyncSession,
    initial_query: str,
    context: str | None = None,
) -> SearchSession:
    """Start a new research exploration session.

    The initial_query is the user's first question/need.
    Context provides background (e.g., "working on VLM reward design").
    """
    search = SearchSession(
        symptom_query=initial_query,
        latent_need=context,
        search_branches={
            "queries": [{"query": initial_query, "step": 1, "type": "initial"}],
            "pivots": [],
            "rejected_patterns": [],
            "insights": [],
        },
        rewrite_history={"steps": []},
    )
    session.add(search)
    await session.flush()
    await session.refresh(search)
    return search


async def add_exploration_step(
    session: AsyncSession,
    session_id: UUID,
    query: str,
    step_type: str = "refine",
    found_paper_ids: list[UUID] | None = None,
    rejected_reason: str | None = None,
    insight: str | None = None,
) -> dict:
    """Add a step to an exploration session.

    step_type: "refine" (narrow), "pivot" (change direction), "deepen" (go deeper), "broaden" (expand)
    """
    search = await session.get(SearchSession, session_id)
    if not search:
        return {"error": "Session not found"}

    branches = search.search_branches or {"queries": [], "pivots": [], "rejected_patterns": [], "insights": []}
    step_num = len(branches.get("queries", [])) + 1

    # Add query
    branches["queries"].append({
        "query": query,
        "step": step_num,
        "type": step_type,
    })

    # Track pivot if direction changed
    if step_type == "pivot" and rejected_reason:
        branches["pivots"].append({
            "step": step_num,
            "from_query": branches["queries"][-2]["query"] if len(branches["queries"]) >= 2 else "",
            "to_query": query,
            "reason": rejected_reason,
        })
        # Add to rejected patterns
        if rejected_reason not in (branches.get("rejected_patterns") or []):
            branches.setdefault("rejected_patterns", []).append(rejected_reason)

    # Track insight
    if insight:
        branches.setdefault("insights", []).append({
            "step": step_num,
            "insight": insight,
        })

    # Track found papers
    if found_paper_ids:
        existing = list(search.result_paper_ids or [])
        for pid in found_paper_ids:
            if pid not in existing:
                existing.append(pid)
        search.result_paper_ids = existing

    search.search_branches = branches
    search.rejected_solution_patterns = branches.get("rejected_patterns")
    await session.flush()

    # Generate next step suggestion
    suggestion = await _suggest_next_step(session, search)

    return {
        "session_id": str(session_id),
        "step": step_num,
        "type": step_type,
        "total_queries": len(branches["queries"]),
        "total_pivots": len(branches.get("pivots", [])),
        "total_papers_found": len(search.result_paper_ids or []),
        "rejected_patterns": branches.get("rejected_patterns", []),
        "suggestion": suggestion,
    }


async def get_exploration_summary(
    session: AsyncSession,
    session_id: UUID,
) -> dict:
    """Get the full exploration path summary with insights."""
    search = await session.get(SearchSession, session_id)
    if not search:
        return {"error": "Session not found"}

    branches = search.search_branches or {}

    # Enrich with paper info
    papers = []
    for pid in (search.result_paper_ids or []):
        paper = await session.get(Paper, pid)
        if paper:
            # Get DeltaCard if available
            dc_result = await session.execute(
                select(DeltaCard).where(
                    DeltaCard.paper_id == pid,
                    DeltaCard.status != "deprecated",
                ).order_by(desc(DeltaCard.created_at)).limit(1)
            )
            dc = dc_result.scalar_one_or_none()

            papers.append({
                "id": str(paper.id),
                "title": paper.title,
                "venue": paper.venue,
                "year": paper.year,
                "method_tags": [t for t in (paper.tags or []) if t.startswith("method/") or t.startswith("improvement/")],
                "keep_score": paper.keep_score,
                "structurality_score": paper.structurality_score,
                "delta_statement": dc.delta_statement[:200] if dc else None,
            })

    # Group papers by method category
    by_method = {}
    for p in papers:
        for tag in p.get("method_tags", []):
            by_method.setdefault(tag, []).append(p["title"])

    suggestion = await _suggest_next_step(session, search)

    return {
        "session_id": str(session_id),
        "initial_query": search.symptom_query,
        "latent_need": search.latent_need,
        "exploration_path": branches.get("queries", []),
        "pivots": branches.get("pivots", []),
        "rejected_patterns": branches.get("rejected_patterns", []),
        "insights": branches.get("insights", []),
        "papers_found": len(papers),
        "papers_by_method": by_method,
        "papers": papers,
        "next_suggestion": suggestion,
    }


async def smart_explore(
    session: AsyncSession,
    session_id: UUID,
    query: str,
) -> dict:
    """Smart exploration: search + filter + classify + suggest next.

    This is the main entry point that combines search with exploration tracking.
    It searches the KB, classifies results by method type, identifies gaps,
    and suggests the next exploration direction.
    """
    search = await session.get(SearchSession, session_id)
    if not search:
        return {"error": "Session not found"}

    # Search current KB
    from backend.services.search_service import hybrid_search, idea_search

    paper_results = await hybrid_search(session, query, limit=15)
    idea_results = await idea_search(session, query, limit=10)

    # Classify results by method type
    classified = {"structural": [], "plugin": [], "other": []}
    for r in paper_results:
        tags = r.get("tags", [])
        method_tags = [t for t in tags if t.startswith("method/") or t.startswith("improvement/")]

        if any("structural" in t or "fundamental" in t or "component_replacement" in t for t in method_tags):
            classified["structural"].append(r)
        elif any("plugin" in t or "additive" in t or "trick" in t for t in method_tags):
            classified["plugin"].append(r)
        else:
            classified["other"].append(r)

    # Track this step
    found_ids = [UUID(r["paper_id"]) for r in paper_results if r.get("paper_id")]
    step_type = _infer_step_type(search, query)

    await add_exploration_step(
        session, session_id, query,
        step_type=step_type,
        found_paper_ids=found_ids[:20],
        insight=f"Found {len(paper_results)} papers: {len(classified['structural'])} structural, {len(classified['plugin'])} plugin",
    )

    # Generate gap analysis
    gaps = _analyze_gaps(classified, search.search_branches or {})

    return {
        "query": query,
        "step_type": step_type,
        "results": {
            "total": len(paper_results),
            "structural": len(classified["structural"]),
            "plugin": len(classified["plugin"]),
            "other": len(classified["other"]),
        },
        "classified_papers": {
            "structural": classified["structural"][:5],
            "plugin": classified["plugin"][:5],
        },
        "idea_results": idea_results[:5],
        "gaps": gaps,
    }


# ── Internal helpers ──────────────────────────────────────────────

async def _suggest_next_step(
    session: AsyncSession,
    search: SearchSession,
) -> dict | None:
    """Suggest the next exploration step based on current path."""
    branches = search.search_branches or {}
    queries = branches.get("queries", [])
    rejected = branches.get("rejected_patterns", [])
    insights = branches.get("insights", [])

    if not queries:
        return None

    last_query = queries[-1]["query"]
    num_pivots = len(branches.get("pivots", []))

    # Look for related bottlenecks that haven't been explored
    result = await session.execute(
        select(ProjectBottleneck).where(
            ProjectBottleneck.status == "active",
        ).order_by(desc(ProjectBottleneck.priority)).limit(5)
    )
    bottlenecks = [
        {"id": str(bn.id), "title": bn.title, "domain": bn.domain}
        for bn in result.scalars()
    ]

    # Build suggestion
    if num_pivots >= 2:
        return {
            "type": "converge",
            "message": "You've pivoted multiple times. Consider narrowing: pick the most promising structural approach and go deep.",
            "related_bottlenecks": bottlenecks[:2],
        }
    elif any("plugin" in r.lower() or "incremental" in r.lower() for r in rejected):
        return {
            "type": "seek_fundamental",
            "message": "You've rejected plugin/incremental approaches. Look for papers that redesign the core pipeline, not add modules.",
            "search_hint": f"{last_query} fundamental rethink architecture",
        }
    else:
        return {
            "type": "continue",
            "message": "Continue exploring. Try narrowing by method type or broadening to adjacent domains.",
            "related_bottlenecks": bottlenecks[:3],
        }


def _infer_step_type(search: SearchSession, new_query: str) -> str:
    """Infer the step type from query evolution."""
    branches = search.search_branches or {}
    queries = branches.get("queries", [])

    if not queries:
        return "initial"

    last_query = queries[-1]["query"].lower()
    new_lower = new_query.lower()

    # Check for broadening (more general terms)
    if len(new_lower.split()) < len(last_query.split()) * 0.6:
        return "broaden"

    # Check for keywords suggesting pivot
    pivot_signals = ["instead", "alternative", "different approach", "not plugin", "fundamental", "rethink"]
    if any(s in new_lower for s in pivot_signals):
        return "pivot"

    # Check for deepening (more specific)
    if any(word in new_lower for word in last_query.split() if len(word) > 3):
        return "deepen"

    return "refine"


def _analyze_gaps(classified: dict, branches: dict) -> list[str]:
    """Analyze what's missing from current results."""
    gaps = []

    if not classified["structural"]:
        gaps.append("No structural/fundamental improvements found — try broadening to adjacent fields or looking at earlier foundational work")

    if len(classified["plugin"]) > len(classified["structural"]) * 3:
        gaps.append("Dominated by plugin-type work — the field may lack fundamental rethinking. Consider looking at methodology papers from other domains")

    rejected = branches.get("rejected_patterns", [])
    if rejected:
        gaps.append(f"Rejected patterns: {', '.join(rejected[:3])} — explore approaches that explicitly avoid these")

    pivots = branches.get("pivots", [])
    if len(pivots) >= 2:
        directions = [p["to_query"] for p in pivots[-2:]]
        gaps.append(f"Recent pivots suggest uncertainty. Common thread across '{directions[0]}' and '{directions[1]}'?")

    return gaps
