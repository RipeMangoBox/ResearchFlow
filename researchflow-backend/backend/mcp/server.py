"""ResearchFlow MCP Server — tools + resources + prompts for Claude Code / Codex.

v3 architecture: DeltaCard-centric tools, resource URIs, workflow prompts.

Tools (actions):
  search_research_kb, search_ideas, get_paper_report, compare_papers,
  import_research_sources, get_digest, get_reading_plan, enqueue_analysis,
  refresh_assets, record_user_feedback, get_paper_detail,
  get_graph_stats, review_queue, submit_review_decision

Resources (objects):
  paper://{id}, delta-card://{id}, idea://{id}, evidence://{paper_id},
  project://{bottleneck_id}

Prompts (workflow templates):
  deep-paper-report, reviewer-stress-test, weekly-research-review

Run standalone:  python -m backend.mcp.server
Or mount in FastAPI via SSE transport.
"""

import json
import logging
from uuid import UUID

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger(__name__)

server = Server("researchflow")


# ── Helper: run async DB operations ─────────────────────────────

async def _get_session():
    from backend.database import async_session
    return async_session


# ── Tool definitions ────────────────────────────────────────────

TOOLS = [
    # ── Search tools ──────────────────────────────────────────
    Tool(
        name="search_research_kb",
        description="Search the paper knowledge base with keyword + semantic + structured filters. Returns ranked papers.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "category": {"type": "string", "description": "Filter by category"},
                "venue": {"type": "string", "description": "Filter by venue"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_ideas",
        description="Search across DeltaCards and IdeaDeltas by keyword. Returns structured delta info with paper context.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword to search in delta statements and key ideas"},
                "category": {"type": "string"},
                "min_structurality": {"type": "number"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    ),

    # ── Report tools ──────────────────────────────────────────
    Tool(
        name="get_paper_report",
        description="Get a paper report by type: quick (30s), briefing (5min), or deep_compare. Provide paper IDs.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_ids": {"type": "array", "items": {"type": "string"}, "description": "Paper UUIDs"},
                "report_type": {"type": "string", "enum": ["quick", "briefing", "deep_compare"], "default": "briefing"},
                "topic": {"type": "string", "description": "Optional topic context"},
            },
            "required": ["paper_ids"],
        },
    ),
    Tool(
        name="compare_papers",
        description="Compare 2-5 papers side by side: delta cards, evidence strength, structural vs plugin.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5},
            },
            "required": ["paper_ids"],
        },
    ),

    # ── Ingestion tools ───────────────────────────────────────
    Tool(
        name="import_research_sources",
        description="Import paper links into the knowledge base. Handles dedup and normalization.",
        inputSchema={
            "type": "object",
            "properties": {
                "urls": {"type": "array", "items": {"type": "string"}, "description": "Paper URLs (arxiv, etc.)"},
                "category": {"type": "string", "default": "Uncategorized"},
            },
            "required": ["urls"],
        },
    ),

    # ── Digest & planning tools ───────────────────────────────
    Tool(
        name="get_digest",
        description="Get the latest research digest: day (what's new), week (trends), or month (strategy).",
        inputSchema={
            "type": "object",
            "properties": {
                "period_type": {"type": "string", "enum": ["day", "week", "month"]},
            },
            "required": ["period_type"],
        },
    ),
    Tool(
        name="get_reading_plan",
        description="Generate a tiered reading plan: canonical baselines → structural → follow-ups → patches.",
        inputSchema={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "max_papers": {"type": "integer", "default": 15},
            },
        },
    ),
    Tool(
        name="propose_directions",
        description="Propose 1-3 research directions for a topic based on the knowledge base.",
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Research topic or question"},
                "category": {"type": "string"},
            },
            "required": ["topic"],
        },
    ),

    # ── Analysis tools ────────────────────────────────────────
    Tool(
        name="enqueue_analysis",
        description="Queue a paper for L3 skim or L4 deep analysis.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper UUID"},
                "level": {"type": "string", "enum": ["skim", "deep"]},
            },
            "required": ["paper_id", "level"],
        },
    ),
    Tool(
        name="refresh_assets",
        description="Enrich papers missing metadata from arXiv/Crossref. Fills abstract, authors, DOI.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 10},
            },
        },
    ),

    # ── Feedback & review tools ───────────────────────────────
    Tool(
        name="record_user_feedback",
        description="Record a correction, confirmation, or tag edit on a paper or analysis.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string"},
                "feedback_type": {"type": "string", "enum": ["correction", "confirmation", "rejection", "tag_edit"]},
                "comment": {"type": "string"},
            },
            "required": ["paper_id", "feedback_type", "comment"],
        },
    ),
    Tool(
        name="get_paper_detail",
        description="Get full detail of a paper including latest analysis, DeltaCard, and graph data.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper UUID"},
            },
            "required": ["paper_id"],
        },
    ),

    # ── v3 Graph & review tools ───────────────────────────────
    Tool(
        name="get_graph_stats",
        description="Get knowledge graph statistics: idea_deltas, delta_cards, assertions, review queue.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="review_queue",
        description="List pending review tasks (assertions, delta_cards needing human verification).",
        inputSchema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["pending", "in_progress", "approved", "rejected"], "default": "pending"},
                "target_type": {"type": "string", "enum": ["assertion", "delta_card", "idea_delta"]},
                "limit": {"type": "integer", "default": 20},
            },
        },
    ),
    Tool(
        name="submit_review_decision",
        description="Approve or reject a review task. Cascades to the target object.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Review task UUID"},
                "decision": {"type": "string", "enum": ["approve", "reject"]},
                "reviewer": {"type": "string", "default": "claude"},
                "notes": {"type": "string"},
            },
            "required": ["task_id", "decision"],
        },
    ),
]


@server.list_tools()
async def list_tools():
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    session_maker = await _get_session()

    async with session_maker() as session:
        try:
            result = await _dispatch(name, arguments, session)
            await session.commit()
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]
        except Exception as e:
            logger.error(f"MCP tool {name} error: {e}")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _dispatch(name: str, args: dict, session) -> dict:
    """Route tool calls to the appropriate service."""

    if name == "search_research_kb":
        from backend.services import search_service
        results = await search_service.hybrid_search(
            session,
            query=args["query"],
            category=args.get("category"),
            venue=args.get("venue"),
            limit=args.get("limit", 10),
        )
        return {"results": results[:args.get("limit", 10)]}

    elif name == "search_ideas":
        from backend.services import search_service
        results = await search_service.idea_search(
            session,
            query=args["query"],
            category=args.get("category"),
            min_structurality=args.get("min_structurality"),
            limit=args.get("limit", 10),
        )
        return {"results": results}

    elif name == "get_paper_report":
        from backend.services import report_service
        paper_ids = [UUID(pid) for pid in args["paper_ids"]]
        return await report_service.generate_report(
            session, paper_ids, args.get("report_type", "briefing"), args.get("topic"),
        )

    elif name == "compare_papers":
        from backend.services import report_service
        paper_ids = [UUID(pid) for pid in args["paper_ids"]]
        return await report_service.generate_report(session, paper_ids, "deep_compare")

    elif name == "import_research_sources":
        from backend.schemas.import_ import LinkImportItem
        from backend.services import ingestion_service
        items = [LinkImportItem(url=u) for u in args["urls"]]
        results = await ingestion_service.ingest_links(
            session, items, args.get("category", "Uncategorized"), False, 30,
        )
        return {"items": [r.model_dump() for r in results]}

    elif name == "get_digest":
        from backend.services import digest_service
        digest = await digest_service.get_latest_digest(session, args["period_type"])
        if digest:
            return {"period": f"{digest.period_start} to {digest.period_end}", "content": digest.rendered_text}
        digest = await digest_service.generate_digest(session, args["period_type"])
        return {"period": f"{digest.period_start} to {digest.period_end}", "content": digest.rendered_text}

    elif name == "get_reading_plan":
        from backend.services import reading_planner
        return await reading_planner.generate_reading_plan(
            session, category=args.get("category"), max_papers=args.get("max_papers", 15),
        )

    elif name == "propose_directions":
        from backend.services import direction_service
        cards = await direction_service.propose_directions(
            session, topic=args["topic"], category=args.get("category"),
        )
        return {"directions": [
            {"id": str(c.id), "title": c.title, "rationale": c.rationale,
             "is_structural": c.is_structural, "confidence": c.confidence}
            for c in cards
        ]}

    elif name == "enqueue_analysis":
        from backend.services import analysis_service
        pid = UUID(args["paper_id"])
        if args["level"] == "skim":
            analysis = await analysis_service.skim_paper(session, pid)
        else:
            analysis = await analysis_service.deep_analyze_paper(session, pid)
        if analysis:
            return {"analysis_id": str(analysis.id), "level": analysis.level.value}
        return {"error": "Paper not found"}

    elif name == "refresh_assets":
        from backend.services import enrich_service
        results = await enrich_service.enrich_batch(session, limit=args.get("limit", 10))
        return {"processed": len(results), "results": results}

    elif name == "record_user_feedback":
        from backend.models.system import UserFeedback
        from backend.models.enums import FeedbackType
        fb = UserFeedback(
            target_type="paper",
            target_id=UUID(args["paper_id"]),
            feedback_type=FeedbackType(args["feedback_type"]),
            comment=args["comment"],
        )
        session.add(fb)
        return {"status": "recorded"}

    elif name == "get_paper_detail":
        from backend.services import paper_service
        paper, analysis = await paper_service.get_paper_with_analysis(session, UUID(args["paper_id"]))
        if not paper:
            return {"error": "Paper not found"}
        result = {
            "title": paper.title, "venue": paper.venue, "year": paper.year,
            "category": paper.category, "state": paper.state.value,
            "tags": list(paper.tags) if paper.tags else [],
            "core_operator": paper.core_operator,
            "keep_score": paper.keep_score, "structurality_score": paper.structurality_score,
        }
        if analysis:
            result["analysis"] = {
                "level": analysis.level.value,
                "problem_summary": analysis.problem_summary,
                "method_summary": analysis.method_summary,
                "core_intuition": analysis.core_intuition,
            }
        # v3: include DeltaCard if available
        from backend.models.delta_card import DeltaCard
        from sqlalchemy import select, desc
        dc_result = await session.execute(
            select(DeltaCard).where(
                DeltaCard.paper_id == UUID(args["paper_id"]),
                DeltaCard.status != "deprecated",
            ).order_by(desc(DeltaCard.created_at)).limit(1)
        )
        dc = dc_result.scalar_one_or_none()
        if dc:
            result["delta_card"] = {
                "id": str(dc.id),
                "delta_statement": dc.delta_statement,
                "structurality_score": dc.structurality_score,
                "transferability_score": dc.transferability_score,
                "key_ideas_ranked": dc.key_ideas_ranked,
                "assumptions": dc.assumptions,
                "failure_modes": dc.failure_modes,
                "status": dc.status,
            }
        return result

    elif name == "get_graph_stats":
        from backend.services import graph_query_service
        return await graph_query_service.graph_stats(session)

    elif name == "review_queue":
        from backend.services import review_service
        return await review_service.list_reviews(
            session,
            status=args.get("status", "pending"),
            target_type=args.get("target_type"),
            limit=args.get("limit", 20),
        )

    elif name == "submit_review_decision":
        from backend.services import review_service
        task_id = UUID(args["task_id"])
        reviewer = args.get("reviewer", "claude")
        if args["decision"] == "approve":
            return await review_service.approve_review(session, task_id, reviewer, args.get("notes"))
        else:
            return await review_service.reject_review(session, task_id, reviewer, args.get("notes"))

    return {"error": f"Unknown tool: {name}"}


# ── Entry point ─────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
