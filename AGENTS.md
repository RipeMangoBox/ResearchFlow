# Agent Guide

> Full architecture details: [ARCHITECTURE.md](researchflow-backend/ARCHITECTURE.md)

## Source of truth

**PostgreSQL is the only write target.** `paperAnalysis/`, `paperCollection/`, `paperIDEAs/` are read-only exports. For queries, use `/api/v1/search/*`, not local files.

## MCP tools (18)

| Tool | What it does |
|------|-------------|
| `run_full_pipeline` | One call: download PDF → enrich → parse → L3 → L4 → build graph |
| `discover_related_papers` | Semantic Scholar: refs + citations + related → auto-ingest |
| `build_domain` | Multi-hop discovery from seed paper → domain KB |
| `search_research_kb` | Hybrid search: keyword + semantic + structured |
| `search_ideas` | Search DeltaCards / IdeaDeltas by keyword |
| `get_paper_report` | Generate report: quick / briefing / deep_compare |
| `compare_papers` | Side-by-side comparison of 2–5 papers |
| `import_research_sources` | Ingest paper URLs |
| `get_digest` | Daily / weekly / monthly research digest |
| `get_reading_plan` | Tiered reading: baseline → structural → plugin |
| `propose_directions` | Research direction proposals |
| `enqueue_analysis` | Queue L3 skim or L4 deep analysis |
| `refresh_assets` | Enrich metadata from arXiv / Crossref / OpenAlex / GitHub |
| `record_user_feedback` | Corrections / confirmations / tag edits |
| `get_paper_detail` | Full paper detail with DeltaCard |
| `get_graph_stats` | Knowledge graph statistics |
| `review_queue` | Pending review tasks |
| `submit_review_decision` | Approve / reject reviews (cascades) |

## MCP resources (6)

| URI scheme | What it returns |
|------------|----------------|
| `paper://{id}` | Paper detail with analysis + DeltaCard |
| `delta-card://{id}` | DeltaCard structured snapshot |
| `graph://stats` | Knowledge graph statistics |
| `canonical-idea://{id}` | Cross-paper canonical idea detail |
| `review-task://{id}` | Review task with target object |
| `lineage://{paper_id}` | Method lineage DAG (ancestors + descendants) |

## MCP prompts (4)

| Prompt | Use when |
|--------|----------|
| `deep-paper-report` | Deep analysis report for a paper |
| `weekly-research-review` | Weekly research digest |
| `lineage-review` | Trace method evolution for a paper |
| `direction-gap-analysis` | Find gaps in a research direction |

## Skill routing (Claude Code / Codex)

| Task | Skill |
|------|-------|
| Unsure what to do | `research-workflow` |
| Collect from GitHub | `papers-collect-from-github-awesome` |
| Analyze PDFs | `papers-analyze-pdf` |
| Query KB | `papers-query-knowledge-base` |
| Generate ideas | `research-brainstorm-from-kb` |
| Reviewer stress test | `reviewer-stress-test` |

Full list: [`.claude/skills/User_README.md`](.claude/skills/User_README.md)

## Remote MCP connection

```json
// .mcp.json — connect to remote ResearchFlow server
{
  "mcpServers": {
    "researchflow": {
      "url": "https://researchflow.xyz/sse"
    }
  }
}
```

## Rules

1. All writes go to backend API, never edit Markdown files as source
2. For queries, prefer `/api/v1/search/*` over reading local files
3. Analysis language default: `zh` (override per request)
4. Keep ResearchFlow as active workspace; link external repos under `linkedCodebases/`
