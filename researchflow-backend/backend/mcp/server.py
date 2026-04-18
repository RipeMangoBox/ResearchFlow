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

from pydantic import AnyUrl

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    Resource,
    TextContent,
    Tool,
)

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

    # ── Pipeline & discovery tools ───────────────────────────
    Tool(
        name="run_full_pipeline",
        description="Run complete pipeline on a paper: download PDF → enrich metadata → parse → L3 skim → L4 deep → build knowledge graph. One call does everything.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper UUID"},
            },
            "required": ["paper_id"],
        },
    ),
    Tool(
        name="discover_related_papers",
        description="Find related papers via Semantic Scholar (references, citations, recommendations) and auto-ingest them. Build domain KB from a seed paper.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Seed paper UUID"},
                "max_references": {"type": "integer", "default": 10},
                "max_citations": {"type": "integer", "default": 10},
                "auto_ingest": {"type": "boolean", "default": True},
            },
            "required": ["paper_id"],
        },
    ),
    Tool(
        name="build_domain",
        description="Build a domain knowledge graph from a single seed paper. Discovers related work, ingests papers, and optionally runs full analysis pipeline on all.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Seed paper UUID"},
                "depth": {"type": "integer", "default": 1, "description": "How many hops to follow (1-3)"},
                "run_pipeline": {"type": "boolean", "default": False, "description": "Run full analysis on discovered papers"},
            },
            "required": ["paper_id"],
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

    elif name == "run_full_pipeline":
        from backend.services import pipeline_service
        return await pipeline_service.run_full_pipeline(session, UUID(args["paper_id"]))

    elif name == "discover_related_papers":
        from backend.services import discovery_service
        return await discovery_service.discover_related_papers(
            session, UUID(args["paper_id"]),
            max_references=args.get("max_references", 10),
            max_citations=args.get("max_citations", 10),
            auto_ingest=args.get("auto_ingest", True),
        )

    elif name == "build_domain":
        from backend.services import discovery_service
        return await discovery_service.build_domain_from_seed(
            session, UUID(args["paper_id"]),
            depth=args.get("depth", 1),
            run_pipeline=args.get("run_pipeline", False),
        )

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


# ── Resource definitions ───────────────────────────────────────

RESOURCES = [
    Resource(
        uri=AnyUrl("paper://example"),
        name="paper",
        description="Paper detail with DeltaCard. Use paper://{paper_id} to access.",
        mimeType="application/json",
    ),
    Resource(
        uri=AnyUrl("delta-card://example"),
        name="delta-card",
        description="DeltaCard for a paper. Use delta-card://{paper_id} to access.",
        mimeType="application/json",
    ),
    Resource(
        uri=AnyUrl("graph://stats"),
        name="graph-stats",
        description="Knowledge graph statistics: counts of ideas, cards, assertions, edges.",
        mimeType="application/json",
    ),
    Resource(
        uri=AnyUrl("canonical-idea://example"),
        name="canonical-idea",
        description="Canonical idea detail with contribution count. Use canonical-idea://{id}.",
        mimeType="application/json",
    ),
    Resource(
        uri=AnyUrl("review-task://example"),
        name="review-task",
        description="Review task with target object detail. Use review-task://{task_id}.",
        mimeType="application/json",
    ),
    Resource(
        uri=AnyUrl("lineage://example"),
        name="lineage",
        description="Method lineage DAG for a paper. Use lineage://{paper_id}.",
        mimeType="application/json",
    ),
]


@server.list_resources()
async def list_resources() -> list[Resource]:
    return RESOURCES


@server.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """Read an MCP resource by URI scheme."""
    uri_str = str(uri)
    session_maker = await _get_session()

    async with session_maker() as session:
        # ── graph://stats ──────────────────────────────────────
        if uri_str.startswith("graph://stats"):
            from backend.services import graph_query_service
            stats = await graph_query_service.graph_stats(session)
            return json.dumps(stats, ensure_ascii=False, default=str)

        # ── paper://{paper_id} ─────────────────────────────────
        if uri_str.startswith("paper://"):
            paper_id_str = uri_str.replace("paper://", "")
            paper_id = UUID(paper_id_str)
            from backend.services import paper_service
            paper, analysis = await paper_service.get_paper_with_analysis(session, paper_id)
            if not paper:
                return json.dumps({"error": "Paper not found"})
            result = {
                "id": str(paper.id),
                "title": paper.title,
                "venue": paper.venue,
                "year": paper.year,
                "category": paper.category,
                "state": paper.state.value,
                "tags": list(paper.tags) if paper.tags else [],
                "core_operator": paper.core_operator,
                "keep_score": paper.keep_score,
                "structurality_score": paper.structurality_score,
            }
            if analysis:
                result["analysis"] = {
                    "level": analysis.level.value,
                    "problem_summary": analysis.problem_summary,
                    "method_summary": analysis.method_summary,
                    "core_intuition": analysis.core_intuition,
                }
            # Attach DeltaCard if available
            from backend.models.delta_card import DeltaCard
            from sqlalchemy import select, desc
            dc_result = await session.execute(
                select(DeltaCard).where(
                    DeltaCard.paper_id == paper_id,
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
            return json.dumps(result, ensure_ascii=False, default=str)

        # ── delta-card://{paper_id} ────────────────────────────
        if uri_str.startswith("delta-card://"):
            paper_id_str = uri_str.replace("delta-card://", "")
            paper_id = UUID(paper_id_str)
            from backend.models.delta_card import DeltaCard
            from sqlalchemy import select, desc
            dc_result = await session.execute(
                select(DeltaCard).where(
                    DeltaCard.paper_id == paper_id,
                    DeltaCard.status != "deprecated",
                ).order_by(desc(DeltaCard.created_at)).limit(1)
            )
            dc = dc_result.scalar_one_or_none()
            if not dc:
                return json.dumps({"error": "DeltaCard not found for this paper"})
            return json.dumps({
                "id": str(dc.id),
                "paper_id": str(dc.paper_id),
                "delta_statement": dc.delta_statement,
                "baseline_paradigm": dc.baseline_paradigm,
                "structurality_score": dc.structurality_score,
                "extensionability_score": dc.extensionability_score,
                "transferability_score": dc.transferability_score,
                "key_ideas_ranked": dc.key_ideas_ranked,
                "assumptions": dc.assumptions,
                "failure_modes": dc.failure_modes,
                "evidence_refs": [str(r) for r in dc.evidence_refs] if dc.evidence_refs else [],
                "status": dc.status,
            }, ensure_ascii=False, default=str)

        # ── canonical-idea://{id} ──────────────────────────────
        if uri_str.startswith("canonical-idea://"):
            idea_id = uri_str.replace("canonical-idea://", "")
            from backend.models.canonical_idea import CanonicalIdea
            ci = await session.get(CanonicalIdea, UUID(idea_id))
            if ci:
                return json.dumps({
                    "id": str(ci.id), "title": ci.title, "description": ci.description,
                    "domain": ci.domain, "status": ci.status,
                    "contribution_count": ci.contribution_count,
                    "aliases": ci.aliases, "tags": ci.tags,
                }, ensure_ascii=False)

        # ── review-task://{id} ────────────────────────────────
        if uri_str.startswith("review-task://"):
            task_id = uri_str.replace("review-task://", "")
            from backend.services.review_service import get_review, get_review_detail
            task = await get_review(session, UUID(task_id))
            if task:
                detail = await get_review_detail(session, task)
                return json.dumps(detail, ensure_ascii=False, default=str)

        # ── lineage://{paper_id} ──────────────────────────────
        if uri_str.startswith("lineage://"):
            paper_id_str = uri_str.replace("lineage://", "")
            from backend.services.evolution_service import get_lineage_tree
            from backend.models.delta_card import DeltaCard
            dc_result = await session.execute(
                select(DeltaCard).where(
                    DeltaCard.paper_id == UUID(paper_id_str),
                    DeltaCard.status != "deprecated",
                ).order_by(DeltaCard.created_at.desc()).limit(1)
            )
            dc = dc_result.scalar_one_or_none()
            if dc:
                tree = await get_lineage_tree(session, dc.id)
                return json.dumps(tree, ensure_ascii=False, default=str)

    return json.dumps({"error": f"Unknown resource URI: {uri_str}"})


# ── Prompt definitions ─────────────────────────────────────────

PROMPTS = [
    Prompt(
        name="deep-paper-report",
        description="Generate a deep analysis report for a paper, covering delta card, evidence strength, structural vs plugin classification, and cross-paper connections.",
        arguments=[
            PromptArgument(
                name="paper_id",
                description="UUID of the paper to analyze",
                required=True,
            ),
            PromptArgument(
                name="focus_area",
                description="Optional focus area (e.g., 'methodology', 'evidence', 'novelty')",
                required=False,
            ),
        ],
    ),
    Prompt(
        name="weekly-research-review",
        description="Generate a weekly research digest covering new papers, key deltas, emerging trends, and recommended reading priorities.",
        arguments=[
            PromptArgument(
                name="category",
                description="Optional category filter (e.g., 'diffusion', 'multimodal')",
                required=False,
            ),
            PromptArgument(
                name="top_n",
                description="Number of top papers to highlight (default: 10)",
                required=False,
            ),
        ],
    ),
    Prompt(
        name="lineage-review",
        description="Review the method lineage DAG for a paper — trace how the method evolved from its ancestors and identify downstream impact.",
        arguments=[
            PromptArgument(name="paper_id", description="UUID of the paper to trace", required=True),
            PromptArgument(name="depth", description="Max depth to traverse (default: 5)", required=False),
        ],
    ),
    Prompt(
        name="direction-gap-analysis",
        description="Analyze gaps in a research direction: which bottlenecks lack structural solutions, which mechanisms are under-explored, and where are the best opportunities.",
        arguments=[
            PromptArgument(name="category", description="Research category to analyze", required=True),
            PromptArgument(name="focus_bottleneck", description="Optional specific bottleneck to focus on", required=False),
        ],
    ),
]


@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    return PROMPTS


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    """Return a filled prompt template."""
    args = arguments or {}

    if name == "deep-paper-report":
        paper_id = args.get("paper_id", "<PAPER_ID>")
        focus_area = args.get("focus_area", "all aspects")
        return GetPromptResult(
            description=f"Deep analysis report for paper {paper_id}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Generate a deep analysis report for paper {paper_id}.\n\n"
                            f"Focus area: {focus_area}\n\n"
                            "Please cover the following sections:\n"
                            "1. **Delta Card Summary** - What did this paper change relative to the canonical paradigm?\n"
                            "2. **Slot Analysis** - Which paradigm slots were modified (structural vs plugin vs tweak)?\n"
                            "3. **Mechanism Family** - What mechanism family does this approach belong to?\n"
                            "4. **Evidence Strength** - How well-grounded are the claims? List evidence units with confidence.\n"
                            "5. **Transferability Assessment** - Can this idea transfer to other domains? Which ones?\n"
                            "6. **Assumptions & Failure Modes** - What are the key assumptions and where might this break?\n"
                            "7. **Cross-Paper Connections** - How does this relate to other ideas in the knowledge base?\n"
                            "8. **Research Impact Score** - Rate structurality, transferability, and field keyness.\n\n"
                            "Use the get_paper_detail, search_ideas, and get_graph_stats tools to gather data."
                        ),
                    ),
                ),
            ],
        )

    elif name == "weekly-research-review":
        category = args.get("category", "all categories")
        top_n = args.get("top_n", "10")
        return GetPromptResult(
            description=f"Weekly research review for {category}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Generate a weekly research digest for: {category}\n\n"
                            f"Highlight top {top_n} papers.\n\n"
                            "Please structure the digest as:\n"
                            "1. **Executive Summary** - 3-5 bullet points on the week's key developments\n"
                            "2. **New Papers** - List newly ingested papers with their delta statements\n"
                            "3. **Key Deltas This Week** - Most impactful IdeaDeltas ranked by structurality score\n"
                            "4. **Emerging Trends** - Patterns across papers (common bottlenecks, mechanism families)\n"
                            "5. **Evidence Highlights** - Strongest new evidence units and what they support\n"
                            "6. **Cross-Domain Transfers** - Any transferable insights discovered\n"
                            "7. **Reading Priorities** - Recommended reading order: canonical > structural > follow-up > patch\n"
                            "8. **Knowledge Base Health** - Current graph stats and quality metrics\n\n"
                            "Use search_research_kb, get_digest, get_graph_stats, and search_ideas tools to gather data."
                        ),
                    ),
                ),
            ],
        )

    raise ValueError(f"Unknown prompt: {name}")


# ── Entry point ─────────────────────────────────────────────────

async def main_stdio():
    """Run MCP server via stdio (local Claude Code / Codex)."""
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


async def main_sse(host: str = "0.0.0.0", port: int = 8001):
    """Run MCP server via SSE transport (remote connection).

    Connect from Claude Code with:
      .mcp.json: {"url": "http://your-server:8001/sse"}
    """
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route
    import uvicorn

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as (read, write):
            await server.run(read, write, server.create_initialization_options())

    async def handle_messages(request):
        await sse.handle_post_message(request.scope, request.receive, request._send)

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/messages/", endpoint=handle_messages, methods=["POST"]),
        ],
    )

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    srv = uvicorn.Server(config)
    await srv.serve()


if __name__ == "__main__":
    import asyncio
    import sys

    if "--sse" in sys.argv:
        port = 8001
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        print(f"Starting MCP SSE server on port {port}...")
        asyncio.run(main_sse(port=port))
    else:
        asyncio.run(main_stdio())
