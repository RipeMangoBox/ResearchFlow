# Agent Guide

> Full architecture details: [ARCHITECTURE.md](researchflow-backend/ARCHITECTURE.md)

## Source of truth

**PostgreSQL is the only write target.** `paperAnalysis/`, `paperCollection/`, `obsidian-vault/` are read-only exports. For queries, use `/api/v1/search/*`, not local files.

## MCP tools (35)

| Tool | What it does |
|------|-------------|
| `run_full_pipeline` | One call: triage → download → enrich (8 APIs) → L2 parse (GROBID+PyMuPDF) → L3 → L4 (6-step) → post-L4 → discovery |
| `discover_related_papers` | Semantic Scholar: refs + citations + related → auto-ingest |
| `build_domain` | Multi-hop discovery from seed paper → domain KB |
| `search_research_kb` | Hybrid search: keyword + semantic + structured filters |
| `search_ideas` | Search DeltaCards / IdeaDeltas by keyword |
| `get_paper_report` | Generate report: quick / briefing / deep_compare |
| `compare_papers` | Side-by-side comparison of 2–5 papers |
| `import_research_sources` | Ingest paper URLs with dedup |
| `get_digest` | Daily / weekly / monthly research digest |
| `get_reading_plan` | Tiered reading: baseline → structural → plugin |
| `propose_directions` | Research direction proposals |
| `enqueue_analysis` | Queue L3 skim or L4 deep analysis |
| `refresh_assets` | Enrich metadata from arXiv / Crossref / OpenAlex / GitHub / HuggingFace |
| `record_user_feedback` | Corrections / confirmations / tag edits |
| `get_paper_detail` | Full paper detail with DeltaCard + analysis |
| `get_graph_stats` | Knowledge graph statistics |
| `review_queue` | Pending review tasks |
| `submit_review_decision` | Approve / reject reviews (cascades to graph) |
| `resolve_venue` | Conference acceptance detection (OpenReview + DBLP + arXiv) |
| `get_paper_citations` | GROBID refs + S2 citing papers |
| `get_paper_figures` | Extracted figures/tables + OSS URLs + VLM descriptions |
| `get_metadata_conflicts` | View unresolved multi-source metadata conflicts |
| `rf_domain_cold_start` | V6: Cold-start a domain from manifest |
| `rf_candidate_list` | V6: List candidates with filters/scores |
| `rf_candidate_promote` | V6: Promote candidate to Paper at specified level |
| `rf_candidate_reject` | V6: Reject candidate with reason |
| `rf_paper_build_neighborhood` | V6: S2 neighborhood discovery → candidates |
| `rf_node_profile_get` | V6: Get node profile (T/M/C/D/L/Lab) |
| `rf_node_profile_refresh` | V6: Regenerate node profile |
| `rf_edge_profile_get` | V6: Get edge contextual description |
| `rf_graph_get_subgraph` | V6: Get subgraph with profiles |
| `rf_review_queue` | V6: View review queue items |
| `rf_score_explain` | V6: Explain scoring breakdown for candidate |
| `rf_run_v6_pipeline` | V6: Full pipeline (import→score→shallow→deep→report) |

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
| `weekly-research-review` | Weekly research digest synthesis |
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

21 skills available. Key routing:

| Task | Skill |
|------|-------|
| Unsure what to do | `research-workflow` |
| Collect from GitHub | `papers-collect-from-github-awesome` |
| Collect from web | `papers-collect-from-web` |
| Analyze PDFs locally | `papers-analyze-pdf` |
| Download from triage list | `papers-download-from-list` |
| Sync from Zotero | `papers-sync-from-zotero` |
| Query knowledge base | `papers-query-knowledge-base` |
| Build collection index | `papers-build-collection-index` |
| Audit metadata | `papers-audit-metadata-consistency` |
| Generate ideas | `research-brainstorm-from-kb` |
| Focus/narrow ideas | `idea-focus-coach` |
| Deep paper report (V2) | `paper-report-v2` |
| Reviewer stress test | `reviewer-stress-test` |
| Code-context paper retrieval | `code-context-paper-retrieval` |
| Export shareable notes | `notes-export-share-version` |
| Write daily log | `write-daily-log` |
| Obsidian Markdown conventions | `rf-obsidian-markdown` |
| Migrate to new domain | `domain-fork` |
| Detect skill mismatch | `skill-fit-guard` |

Full list: [`.claude/skills/`](.claude/skills/)

## Analysis Pipeline

L4 deep analysis uses 6 independent steps:

```
Step 1: extract_evidence   ← reads method/experiments FIRST, cross-checks abstract
Step 2: build_delta_card   ← grounded by Step 1 evidence
Step 3: build_compare_set  ← DB-augmented comparison (4 sources, not self-reported)
Step 4: propose_lineage    ← builds_on/extends/replaces DAG edges
Step 5: synthesize_concept ← cross-paper CanonicalIdea + MechanismFamily
Step 6: reconcile_neighbors ← reverse-update old papers for consistency
```

Each step retries independently. Step failure doesn't block subsequent steps.

Post-L4: backfill core_operator/primary_logic/claims → infer ring → taxonomy assignment.

## Metadata enrichment (10 steps, 8 APIs)

Enrich uses a metadata observation ledger — raw values from each API are stored with authority_rank, then a canonical resolver picks the best value per field. Placeholder titles (bare arxiv IDs) are protected from being used as search queries.

Sources: arXiv → Crossref → OpenAlex → Semantic Scholar → DBLP → OpenReview → GitHub → HuggingFace

## Obsidian Export

Vault structure with controlled wikilinks:

| Type | Prefix | Directory | Body wikilinks |
|------|--------|-----------|---------------|
| Paper | `P__` | `40_Papers/{A,B,C,D}__*/` | 6-10 max (T+M+C+D+P) |
| Concept | `C__` | `20_Concepts/` | Mechanism + CanonicalIdea merged |
| Bottleneck | `B__` | `30_Bottlenecks/` | Cross-paper synthesis |
| Lineage | `L__` | `10_Lineages/` | Method evolution with ASCII tree |
| Task | `T__` | `10_Tasks/` | Task nodes with common problems |
| Method | `M__` | `20_Methods/` | Method nodes with Mermaid evolution graph |
| Overview | — | `00_Home/` | Navigation only |

Paper levels: A (baseline, struct≥0.7), B (structural, ≥0.5), C (plugin, ≥0.3), D (peripheral).

## Rules

1. All writes go to backend API, never edit Markdown files as source
2. For queries, prefer `/api/v1/search/*` over reading local files
3. Analysis language default: `zh` (override per request)
4. Keep ResearchFlow as active workspace; link external repos under `linkedCodebases/`
5. Obsidian vault is auto-generated — do not edit files in `obsidian-vault/` directly
6. Pipeline steps are idempotent — already-completed steps are auto-skipped
7. Metadata observations are append-only — canonical resolver picks best value
