"""ResearchFlow MCP Server — 10 high-level tools for Claude Code / Codex.

Only exposes high-level research operations.
Does NOT expose raw SQL, object storage, or low-level CRUD.

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
        description="Get full detail of a paper including latest analysis, scores, and metadata.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper UUID"},
            },
            "required": ["paper_id"],
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

    elif name == "get_paper_report":
        from backend.services import report_service
        paper_ids = [UUID(pid) for pid in args["paper_ids"]]
        result = await report_service.generate_report(
            session, paper_ids, args.get("report_type", "briefing"), args.get("topic"),
        )
        return result

    elif name == "compare_papers":
        from backend.services import report_service
        paper_ids = [UUID(pid) for pid in args["paper_ids"]]
        result = await report_service.generate_report(
            session, paper_ids, "deep_compare",
        )
        return result

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
            return {
                "period": f"{digest.period_start} to {digest.period_end}",
                "content": digest.rendered_text,
            }
        # Generate if none exists
        digest = await digest_service.generate_digest(session, args["period_type"])
        return {"period": f"{digest.period_start} to {digest.period_end}", "content": digest.rendered_text}

    elif name == "get_reading_plan":
        from backend.services import reading_planner
        plan = await reading_planner.generate_reading_plan(
            session, category=args.get("category"), max_papers=args.get("max_papers", 15),
        )
        return plan

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
            "tags": list(paper.tags), "core_operator": paper.core_operator,
            "keep_score": paper.keep_score, "structurality_score": paper.structurality_score,
        }
        if analysis:
            result["analysis"] = {
                "level": analysis.level.value,
                "problem_summary": analysis.problem_summary,
                "method_summary": analysis.method_summary,
                "core_intuition": analysis.core_intuition,
            }
        return result

    return {"error": f"Unknown tool: {name}"}


# ── Entry point ─────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
