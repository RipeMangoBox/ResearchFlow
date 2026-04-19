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

    # ── v4 Metadata extraction tools ─────────────────────────
    Tool(
        name="resolve_venue",
        description="Detect conference acceptance status for a paper. Checks OpenReview, DBLP, arXiv comments. Records multi-source observations with authority ranking.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper UUID"},
            },
            "required": ["paper_id"],
        },
    ),
    Tool(
        name="get_paper_citations",
        description="Get structured references (papers cited by this paper) and citations (papers citing this paper) from GROBID parse + Semantic Scholar API.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper UUID"},
            },
            "required": ["paper_id"],
        },
    ),
    Tool(
        name="get_metadata_conflicts",
        description="Show unresolved metadata conflicts for a paper across sources (arXiv vs OpenReview vs DBLP etc).",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper UUID"},
            },
            "required": ["paper_id"],
        },
    ),
    Tool(
        name="get_paper_figures",
        description="Get all extracted figures and tables for a paper with object storage URLs. Includes captions, page numbers, dimensions.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper UUID"},
            },
            "required": ["paper_id"],
        },
    ),

    # ── v6 Domain / Candidate / Graph tools ──────────────────
    Tool(
        name="rf_domain_cold_start",
        description="Cold-start a research domain. Creates skeleton nodes, harvests papers from arXiv/S2, scores candidates, and promotes top anchors.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Domain name"},
                "scope_tasks": {"type": "array", "items": {"type": "string"}, "description": "Task scope keywords"},
                "scope_methods": {"type": "array", "items": {"type": "string"}, "description": "Method scope keywords"},
                "scope_modalities": {"type": "array", "items": {"type": "string"}, "description": "Modality scope keywords"},
                "scope_datasets": {"type": "array", "items": {"type": "string"}, "description": "Dataset scope keywords"},
                "negative_scope": {"type": "array", "items": {"type": "string"}, "description": "Negative scope keywords to exclude"},
            },
            "required": ["name", "scope_tasks", "scope_methods"],
        },
    ),
    Tool(
        name="rf_candidate_list",
        description="List paper candidates with optional filters. Returns candidates with their discovery scores.",
        inputSchema={
            "type": "object",
            "properties": {
                "domain_id": {"type": "string", "description": "Filter by domain UUID"},
                "status": {"type": "string", "description": "Filter by candidate status"},
                "min_score": {"type": "number", "description": "Minimum discovery score"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    ),
    Tool(
        name="rf_candidate_promote",
        description="Promote a paper candidate to a full Paper at the specified absorption level (1=shallow, 2=visible, 3=full).",
        inputSchema={
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string", "description": "Candidate UUID"},
                "absorption_level": {"type": "integer", "default": 1, "description": "1=shallow, 2=visible, 3=full"},
            },
            "required": ["candidate_id"],
        },
    ),
    Tool(
        name="rf_candidate_reject",
        description="Reject a paper candidate with a reason.",
        inputSchema={
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string", "description": "Candidate UUID"},
                "reason": {"type": "string", "description": "Rejection reason"},
            },
            "required": ["candidate_id", "reason"],
        },
    ),
    Tool(
        name="rf_paper_build_neighborhood",
        description="Discover related papers via Semantic Scholar and create candidates (not direct ingest).",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper UUID"},
                "max_references": {"type": "integer", "default": 30},
                "max_citations": {"type": "integer", "default": 50},
            },
            "required": ["paper_id"],
        },
    ),
    Tool(
        name="rf_node_profile_get",
        description="Get the profile (introduction, structured data) for a knowledge graph node.",
        inputSchema={
            "type": "object",
            "properties": {
                "entity_type": {"type": "string", "description": "Node entity type (e.g. 'task', 'method', 'dataset')"},
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "lang": {"type": "string", "default": "zh", "description": "Language for profile (zh or en)"},
            },
            "required": ["entity_type", "entity_id"],
        },
    ),
    Tool(
        name="rf_node_profile_refresh",
        description="Regenerate the profile for a knowledge graph node using the latest data.",
        inputSchema={
            "type": "object",
            "properties": {
                "entity_type": {"type": "string", "description": "Node entity type"},
                "entity_id": {"type": "string", "description": "Entity UUID"},
            },
            "required": ["entity_type", "entity_id"],
        },
    ),
    Tool(
        name="rf_edge_profile_get",
        description="Get the contextual description of a connection between two knowledge graph nodes.",
        inputSchema={
            "type": "object",
            "properties": {
                "source_entity_type": {"type": "string", "description": "Source node entity type"},
                "source_entity_id": {"type": "string", "description": "Source entity UUID"},
                "target_entity_type": {"type": "string", "description": "Target node entity type"},
                "target_entity_id": {"type": "string", "description": "Target entity UUID"},
            },
            "required": ["source_entity_type", "source_entity_id", "target_entity_type", "target_entity_id"],
        },
    ),
    Tool(
        name="rf_graph_get_subgraph",
        description="Get a subgraph centered on a node, including neighbors, edges, and their profiles.",
        inputSchema={
            "type": "object",
            "properties": {
                "center_entity_type": {"type": "string", "description": "Center node entity type"},
                "center_entity_id": {"type": "string", "description": "Center entity UUID"},
                "depth": {"type": "integer", "default": 1, "description": "Traversal depth (1-3)"},
            },
            "required": ["center_entity_type", "center_entity_id"],
        },
    ),
    Tool(
        name="rf_review_queue",
        description="View the review queue for items needing human review (promotions, edges, conflicts).",
        inputSchema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "default": "pending", "description": "Filter by status"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    ),
    Tool(
        name="rf_score_explain",
        description="Explain the scoring breakdown for a paper candidate — shows all sub-scores, signals, caps, and boosts.",
        inputSchema={
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string", "description": "Candidate UUID"},
            },
            "required": ["candidate_id"],
        },
    ),
    Tool(
        name="rf_run_v6_pipeline",
        description="Run the full V6 pipeline: import → score → shallow ingest → deep ingest → profiles → report.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_url": {"type": "string", "description": "Paper URL or arXiv ID"},
                "domain_id": {"type": "string", "description": "Optional domain UUID"},
            },
            "required": ["paper_url"],
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
    from backend.models.paper import Paper
    from sqlalchemy import select

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

    # ── v4 Metadata extraction tools ─────────────────────────
    elif name == "resolve_venue":
        from backend.services.venue_resolver_service import resolve_venue
        pid = UUID(args["paper_id"])
        paper = await session.get(Paper, pid)
        if not paper:
            return {"error": "Paper not found"}
        authors_list = None
        if paper.authors and isinstance(paper.authors, list):
            authors_list = [a.get("name", "") for a in paper.authors if isinstance(a, dict)]
        return await resolve_venue(
            session, pid,
            title=paper.title,
            authors=authors_list,
            arxiv_id=paper.arxiv_id or "",
            current_venue=paper.venue or "",
            current_year=paper.year or 0,
        )

    elif name == "get_paper_citations":
        from backend.models.analysis import PaperAnalysis
        from backend.models.enums import AnalysisLevel
        pid = UUID(args["paper_id"])
        paper = await session.get(Paper, pid)
        if not paper:
            return {"error": "Paper not found"}

        result = {"paper_id": args["paper_id"], "title": paper.title}

        # 1. GROBID-extracted references from L2 parse
        l2 = (await session.execute(
            select(PaperAnalysis).where(
                PaperAnalysis.paper_id == pid,
                PaperAnalysis.level == AnalysisLevel.L2_PARSE,
                PaperAnalysis.is_current.is_(True),
            )
        )).scalar_one_or_none()

        if l2 and l2.evidence_spans:
            grobid_refs = l2.evidence_spans.get("grobid_references", [])
            result["references_from_pdf"] = grobid_refs
            result["reference_count"] = len(grobid_refs)
        else:
            result["references_from_pdf"] = []
            result["reference_count"] = 0

        # 2. S2 citations (papers citing THIS paper)
        try:
            from backend.services import discovery_service
            disc = await discovery_service.discover_related_papers(
                session, pid, max_refs=0, max_citations=20, auto_ingest=False
            )
            result["cited_by"] = disc.get("citations", []) if isinstance(disc, dict) else []
            result["cited_by_count"] = paper.cited_by_count or 0
        except Exception as e:
            result["cited_by"] = []
            result["cited_by_error"] = str(e)[:100]

        return result

    elif name == "get_metadata_conflicts":
        from backend.services.metadata_resolver_service import get_conflicts
        pid = UUID(args["paper_id"])
        conflicts = await get_conflicts(session, pid)
        return {"paper_id": args["paper_id"], "conflicts": conflicts}

    elif name == "get_paper_figures":
        from backend.models.analysis import PaperAnalysis
        from backend.models.enums import AnalysisLevel
        pid = UUID(args["paper_id"])

        l2 = (await session.execute(
            select(PaperAnalysis).where(
                PaperAnalysis.paper_id == pid,
                PaperAnalysis.level == AnalysisLevel.L2_PARSE,
                PaperAnalysis.is_current.is_(True),
            )
        )).scalar_one_or_none()

        result = {"paper_id": args["paper_id"]}
        if l2:
            result["figures"] = l2.extracted_figure_images or []
            result["figure_captions"] = l2.figure_captions or []
            result["tables"] = l2.extracted_tables or []
            result["formulas"] = l2.extracted_formulas or []
        else:
            result["figures"] = []
            result["figure_captions"] = []
            result["tables"] = []
            result["formulas"] = []
            result["note"] = "No L2 parse available. Run the pipeline first."

        return result

    # ── v6 Domain / Candidate / Graph tools ─────────────────────
    elif name == "rf_domain_cold_start":
        from backend.services import cold_start_service
        result = await cold_start_service.cold_start_domain(
            session,
            name=args["name"],
            scope_tasks=args["scope_tasks"],
            scope_methods=args["scope_methods"],
            scope_modalities=args.get("scope_modalities"),
            scope_datasets=args.get("scope_datasets"),
            negative_scope=args.get("negative_scope"),
        )
        return result

    elif name == "rf_candidate_list":
        from backend.services import candidate_service
        result = await candidate_service.list_candidates(
            session,
            domain_id=UUID(args["domain_id"]) if args.get("domain_id") else None,
            status=args.get("status"),
            min_score=args.get("min_score"),
            limit=args.get("limit", 20),
        )
        return result

    elif name == "rf_candidate_promote":
        from backend.services import candidate_service
        result = await candidate_service.promote_candidate(
            session,
            candidate_id=UUID(args["candidate_id"]),
            absorption_level=args.get("absorption_level", 1),
        )
        return result

    elif name == "rf_candidate_reject":
        from backend.services import candidate_service
        result = await candidate_service.reject_candidate(
            session,
            candidate_id=UUID(args["candidate_id"]),
            reason=args["reason"],
        )
        return result

    elif name == "rf_paper_build_neighborhood":
        from backend.services import neighborhood_service
        result = await neighborhood_service.build_neighborhood(
            session,
            paper_id=UUID(args["paper_id"]),
            max_references=args.get("max_references", 30),
            max_citations=args.get("max_citations", 50),
        )
        return result

    elif name == "rf_node_profile_get":
        from backend.services import profile_service
        result = await profile_service.get_node_profile(
            session,
            entity_type=args["entity_type"],
            entity_id=UUID(args["entity_id"]),
            lang=args.get("lang", "zh"),
        )
        return result

    elif name == "rf_node_profile_refresh":
        from backend.services import profile_service
        result = await profile_service.refresh_node_profile(
            session,
            entity_type=args["entity_type"],
            entity_id=UUID(args["entity_id"]),
        )
        return result

    elif name == "rf_edge_profile_get":
        from backend.services import profile_service
        result = await profile_service.get_edge_profile(
            session,
            source_entity_type=args["source_entity_type"],
            source_entity_id=UUID(args["source_entity_id"]),
            target_entity_type=args["target_entity_type"],
            target_entity_id=UUID(args["target_entity_id"]),
        )
        return result

    elif name == "rf_graph_get_subgraph":
        from backend.services import graph_query_service
        result = await graph_query_service.get_subgraph(
            session,
            center_entity_type=args["center_entity_type"],
            center_entity_id=UUID(args["center_entity_id"]),
            depth=args.get("depth", 1),
        )
        return result

    elif name == "rf_review_queue":
        from backend.services import review_service
        result = await review_service.list_reviews(
            session,
            status=args.get("status", "pending"),
            limit=args.get("limit", 20),
        )
        return result

    elif name == "rf_score_explain":
        from backend.services import candidate_service
        result = await candidate_service.explain_score(
            session,
            candidate_id=UUID(args["candidate_id"]),
        )
        return result

    elif name == "rf_run_v6_pipeline":
        from backend.services.ingest_workflow import IngestWorkflow
        workflow = IngestWorkflow(session)
        result = await workflow.run_full_v6_pipeline(
            args["paper_url"],
            domain_id=UUID(args["domain_id"]) if args.get("domain_id") else None,
        )
        return result

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
    from starlette.responses import Response
    import uvicorn

    from backend.config import settings

    sse = SseServerTransport("/messages/")

    def _check_auth(request) -> bool:
        """Verify Bearer token if MCP_AUTH_TOKEN is configured."""
        if not settings.mcp_auth_token:
            return True  # No auth configured (dev mode)
        auth_header = request.headers.get("authorization", "")
        return auth_header == f"Bearer {settings.mcp_auth_token}"

    async def handle_sse(request):
        if not _check_auth(request):
            return Response("Unauthorized", status_code=401)
        async with sse.connect_sse(request.scope, request.receive, request._send) as (read, write):
            await server.run(read, write, server.create_initialization_options())

    async def handle_messages(request):
        if not _check_auth(request):
            return Response("Unauthorized", status_code=401)
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
