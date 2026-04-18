# Agent Guide

> Full architecture details: [ARCHITECTURE.md](researchflow-backend/ARCHITECTURE.md)

## Source of truth

**PostgreSQL is the only write target.** `paperAnalysis/`, `paperCollection/`, `obsidian-vault/` are read-only exports. For queries, use `/api/v1/search/*`, not local files.

## MCP tools (22)

| Tool | What it does |
|------|-------------|
| `run_full_pipeline` | One call: download PDF → enrich → parse → L3 → L4 (6-step) → build graph |
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

## Connect Claude Code via MCP

ResearchFlow auto-discovers `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "researchflow-remote": {
      "url": "https://researchflow.xyz/sse"
    },
    "researchflow-local": {
      "command": "python",
      "args": ["-m", "backend.mcp.server"],
      "cwd": "researchflow-backend",
      "env": {"PYTHONPATH": "."}
    }
  }
}
```

Use either remote (connect to deployed server) or local (run MCP server alongside backend), or both.

## Skill routing (Claude Code)

| Task | Skill |
|------|-------|
| Unsure what to do | `research-workflow` |
| Collect from GitHub | `papers-collect-from-github-awesome` |
| Collect from web | `papers-collect-from-web` |
| Analyze PDFs | `papers-analyze-pdf` |
| Query KB | `papers-query-knowledge-base` |
| Generate ideas | `research-brainstorm-from-kb` |
| Focus ideas | `idea-focus-coach` |
| Reviewer stress test | `reviewer-stress-test` |
| Export to Obsidian | `notes-export-share-version` |
| Sync from Zotero | `papers-sync-from-zotero` |
| Write daily log | `write-daily-log` |

Full list: [`.claude/skills/User_README.md`](.claude/skills/User_README.md)

## v4.0 Analysis Pipeline

L4 deep analysis now uses 6 independent steps instead of one monolithic LLM call:

```
Step 1: extract_evidence   ← focused on method/experiments, not abstract
Step 2: build_delta_card   ← grounded by Step 1 evidence
Step 3: build_compare_set  ← DB-augmented comparison, not self-reported
Step 4: propose_lineage    ← builds_on/extends/replaces DAG edges
Step 5: synthesize_concept ← cross-paper CanonicalIdea update
Step 6: reconcile_neighbors ← reverse-update old papers
```

Each step retries independently. Step failure doesn't block subsequent steps.

## v4.0 Obsidian Export

5 note types with controlled wikilinks:

| Type | Prefix | Body wikilinks |
|------|--------|---------------|
| Paper | `P__` | 6-8 max (no Domain Overview / Paradigm links) |
| Concept | `C__` | Merged Mechanism + CanonicalIdea |
| Bottleneck | `B__` | Cross-paper synthesis (structural vs plugin solutions) |
| Lineage | `L__` | Method evolution chain with ASCII tree |
| Overview | — | Navigation only |

## Rules

1. All writes go to backend API, never edit Markdown files as source
2. For queries, prefer `/api/v1/search/*` over reading local files
3. Analysis language default: `zh` (override per request)
4. Keep ResearchFlow as active workspace; link external repos under `linkedCodebases/`
5. Obsidian vault is auto-generated — do not edit files in `obsidian-vault/` directly
