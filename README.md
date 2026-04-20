

# ResearchFlow

**One paper in, a domain knowledge graph out.**

[English](README.md) · [中文](README_CN.md)

---

ResearchFlow is a research operating system that transforms academic papers into a structured, evolving knowledge graph. Give it a seed paper or a research topic — it discovers related work, analyzes each paper through a 6-step LLM pipeline with built-in skepticism, tracks how methods improve upon each other as a DAG (not a flat list), and exports everything as a navigable Obsidian vault.

**What makes it different:** ResearchFlow doesn't just summarize papers. It reads methods and experiments *before* trusting abstracts, auto-builds comparison sets from its database (not from what papers claim), requires evidence anchors for high-value conclusions, and maintains consistency across the entire knowledge graph when new papers arrive.

## What's Built

### Domain Cold Start

Give a research topic → GitHub awesome-list discovery → auto-import 50-100 papers → triage + score → batch analyze → full knowledge graph. One API call to start.

### 6-Step Analysis Pipeline

Each paper goes through 6 independently retryable analysis steps, not one monolithic LLM call:


| Step                    | What it does                                  | Why it matters                                                    |
| ----------------------- | --------------------------------------------- | ----------------------------------------------------------------- |
| **extract_evidence**    | Equations, figures, evidence spans            | Reads method/experiments FIRST, then cross-checks abstract claims |
| **build_delta_card**    | Baseline comparison, changed slots, mechanism | Grounded by Step 1 evidence — can't hallucinate                   |
| **build_compare_set**   | Auto-fill comparison papers from DB           | 4 sources, not just paper's self-reported baselines               |
| **propose_lineage**     | builds_on / extends / replaces edges          | Methods form a DAG with multi-inheritance                         |
| **synthesize_concept**  | Update cross-paper CanonicalIdea              | Concepts accumulate across papers, not isolated                   |
| **reconcile_neighbors** | Reverse-update related papers                 | Knowledge graph stays globally consistent                         |


### 10-Step Metadata Enrichment (8 APIs)

arXiv → Crossref → OpenAlex → Semantic Scholar → DBLP → OpenReview → GitHub → HuggingFace. Results stored in an observation ledger with authority ranking — conflicts resolved automatically, not overwritten blindly.

### Parser Ensemble (L2)

GROBID (authors, affiliations, references, formula coordinates) + PyMuPDF (sections, figures, captions) + VLM (figure classification, formula OCR → LaTeX). Deterministic extraction first, LLM only for what machines can't parse.

### Method Evolution DAG

Papers aren't a flat list — they form a directed acyclic graph tracking how methods build on each other:

```
GRPO (baseline, depth=0, 7 downstream)
├── GRPO+LP (plugin, depth=1)
│   └── GRPO-LP+sampling (depth=2)
├── GDPO (structural, depth=1, parents=[GRPO, DPO])  ← multi-inheritance
│   └── GDPO+image_thinking (depth=2)
```

When 3+ papers use a method as baseline → auto-promoted to established baseline → can evolve into new paradigm version. All promotions go through a review gate.

### Faceted Taxonomy (15 dimensions, 75 seed nodes)

Papers tagged across domain, modality, task, learning paradigm, mechanism, method baseline, model family, dataset, benchmark, metric, lab, venue — not just one category. DAG structure with `is_a`, `part_of`, `uses`, `optimizes` relations.

### Obsidian Vault Export

One-click export to a structured Obsidian vault with controlled wikilinks (6-10 per paper, not a hairball):

```
00_Home/           Navigation + reading order guide
10_Lineages/       L__ Method evolution chains with ASCII trees
20_Concepts/       C__ Mechanism + CanonicalIdea merged
30_Bottlenecks/    B__ Cross-paper synthesis (symptom → root cause → solutions)
40_Papers/
  A__Baselines/    Must-read foundational papers (struct ≥ 0.7)
  B__Structural/   Structural improvements (struct ≥ 0.5)
  C__Plugins/      Plugin-type changes (struct ≥ 0.3)
  D__Peripheral/   Peripheral references
80_Assets/         Extracted figures from PDFs
90_Views/          Static Markdown tables (no Dataview dependency)
```

### Candidate Queue + Multi-Agent Pipeline (V6)

5-level absorption: `new → shallow → reference_done → deep → graph_ready`. 16 specialized LLM agents (12 prompt files) with Context Pack Builder. 4-tier scoring engine (DiscoveryScore → DeepIngestScore → GraphPromotionScore → AnchorScore). Node/edge profiles for knowledge graph entities. Cold start workflow and incremental sync (7 functions).

### MCP Integration (35 tools)

Full MCP server with 35 tools + 6 resources + 4 prompt templates. Claude Code auto-discovers it — just talk naturally:

```
> Search for papers about "reward hacking in RLHF"
> Analyze the top 3 results with full pipeline
> Compare methods across these papers
> Export the vault to Obsidian
```

### 21 Claude Code Skills

Research workflow automation: collect papers from GitHub/web/Zotero, analyze PDFs, query knowledge base, brainstorm ideas, focus hypotheses, run reviewer stress tests, generate deep reports with formula derivation, write daily logs.

### Interactive Research Exploration

Multi-hop cognitive iteration: search → classify results (structural vs plugin) → gap analysis → pivot → broaden. System remembers rejection patterns and suggests new directions.

### Web Dashboard

Next.js 15 frontend with paper management, search, graph visualization, lineage viewer, review queue, digest reader, bottleneck explorer, and import tools.

## System Scale


| Component          | Count                                                           |
| ------------------ | --------------------------------------------------------------- |
| Database tables    | 58 + 4 materialized views                                       |
| API endpoints      | 130 across 16 routers                                           |
| MCP                | 35 tools + 6 resources + 4 prompts                              |
| Services           | 55 modules                                                      |
| Worker tasks       | 22                                                              |
| ORM model files    | 24 (15 V6 classes)                                              |
| Agent prompts      | 12                                                              |
| Claude Code skills | 21                                                              |
| Metadata APIs      | 8 (arXiv, Crossref, OpenAlex, S2, DBLP, OpenReview, GitHub, HF) |
| Built-in paradigms | 4 (RL, VLM, Agent, MotionGen) + LLM dynamic discovery           |
| DB migrations      | 16 versions                                                     |
| Enums              | 9 types (PaperState with 15 states, Tier, Importance, etc.)     |


## Quick Start

```bash
# 1. Start the system
cd researchflow-backend && cp .env.example .env  # set ANTHROPIC_API_KEY
docker compose up -d
docker compose exec api alembic upgrade head

# 2. Bootstrap a domain from scratch
curl -X POST localhost:8000/api/v1/pipeline/init-domain \
  -H "Content-Type: application/json" \
  -d '{"domain": "RLHF for VLM"}'

# 3. Batch analyze top-priority papers
curl -X POST localhost:8000/api/v1/pipeline/batch?limit=10

# 4. Export Obsidian vault
curl -X POST localhost:8000/api/v1/pipeline/export/obsidian-vault
```

API docs at `http://localhost:8000/api/v1/docs`

## Connect Claude Code via MCP

ResearchFlow exposes an MCP server. Claude Code auto-discovers `.mcp.json` in the project root.

### Remote (connect to deployed server)

```json
{
  "mcpServers": {
    "researchflow": {
      "url": "https://your-domain/sse"
    }
  }
}
```

### Local (run alongside backend)

```json
{
  "mcpServers": {
    "researchflow-local": {
      "command": "python",
      "args": ["-m", "backend.mcp.server"],
      "cwd": "researchflow-backend",
      "env": {"PYTHONPATH": "."}
    }
  }
}
```

## Sync Obsidian Vault

```bash
# Export on server
curl -X POST localhost:8000/api/v1/pipeline/export/obsidian-vault

# Sync to local machine
rsync -avz --delete -e ssh \
  root@your-server:/opt/researchflow/obsidian-vault/ \
  ./obsidian-vault/

# Open in Obsidian → Graph View (Cmd+G)
# Recommended colors: Papers=blue, Concepts=green, Bottlenecks=red, Lineages=orange
```

## Repository Layout

```
researchflow-backend/            # Core backend (single source of truth)
  backend/
    api/                         #   16 routers (130 endpoints)
    models/                      #   24 ORM model files (58 tables)
    services/                    #   55 service modules
    mcp/                         #   MCP server (35 tools + 6 resources + 4 prompts)
    workers/                     #   ARQ background task queue
    utils/                       #   PDF extraction, GROBID client, frontmatter
  alembic/                       #   16 database migrations
  frontend/                      #   Next.js 15 + Tailwind web dashboard
  ARCHITECTURE.md                #   Complete technical reference (v6)
  DEPLOY.md                      #   Production deployment guide
obsidian-vault/                  # Auto-generated Obsidian vault (read-only)
paperAnalysis/                   # Exported analysis Markdown (read-only)
paperCollection/                 # Collection index + navigation (read-only)
paperIDEAs/                      # Research idea notes (read-only)
scripts/                         # Maintenance & utility scripts
.claude/skills/                  # 21 Claude Code skill definitions
.mcp.json                       # MCP server configuration
AGENTS.md                       # Agent/MCP integration guide
```

## Tech Stack


| Layer       | Technology                                                                  |
| ----------- | --------------------------------------------------------------------------- |
| Backend     | FastAPI (async) + SQLAlchemy 2.0 (async)                                    |
| Database    | PostgreSQL 16 + pgvector (1536d embeddings)                                 |
| Task Queue  | ARQ + Redis 7                                                               |
| Frontend    | Next.js 15 + Tailwind CSS                                                   |
| PDF Parsing | PyMuPDF + GROBID 0.8.1 (ensemble)                                           |
| VLM         | Claude Vision (figure classification + formula OCR)                         |
| LLM         | Anthropic Claude / OpenAI (streaming)                                       |
| Metadata    | arXiv + Crossref + OpenAlex + S2 + DBLP + OpenReview + GitHub + HuggingFace |
| MCP         | Python MCP SDK (stdio + SSE transports)                                     |
| Storage     | Tencent COS / Alibaba OSS / Local                                           |
| Deployment  | Docker Compose + Caddy (auto HTTPS)                                         |


## Documentation


| Document                                                | Audience       | Content                                                                           |
| ------------------------------------------------------- | -------------- | --------------------------------------------------------------------------------- |
| [ARCHITECTURE.md](researchflow-backend/ARCHITECTURE.md) | Developers     | Data model, 4-layer extraction, 6-step pipeline, DB schema, all APIs, 55 services |
| [AGENTS.md](AGENTS.md)                                  | Agent builders | 35 MCP tools, 6 resources, 4 prompts, 21 skills, working rules                    |
| [DEPLOY.md](researchflow-backend/DEPLOY.md)             | Ops            | Docker setup, container architecture, daily deployment, proxy, troubleshooting    |


## License

MIT