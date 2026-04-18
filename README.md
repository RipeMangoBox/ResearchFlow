<p align="center">
  <img src="./assets/LOGO.png" alt="ResearchFlow" width="260"/>
</p>
<h1 align="center">ResearchFlow</h1>
<p align="center"><a href="README.md">English</a> · <a href="README_CN.md">中文</a></p>

---

**Give one paper → auto-build a domain knowledge graph with method evolution tracking.**

ResearchFlow is a research operating system. It ingests papers, analyzes them with LLMs through a 6-step pipeline, builds a structured knowledge graph that captures how methods improve upon each other, and exports a clean Obsidian vault for human navigation.

## Features

### 6-Step Analysis Pipeline (v4.0)

Instead of one monolithic LLM call, each paper goes through 6 independently retryable steps:

| Step | What | Defense line |
|------|------|-------------|
| 1. extract_evidence | Equations, figures, evidence spans | Reads method/experiments FIRST, then cross-checks abstract |
| 2. build_delta_card | Baseline, changed slots, mechanism, bottleneck | Grounded by Step 1 evidence |
| 3. build_compare_set | Auto-fill comparison papers from DB | Not just paper's self-reported baselines |
| 4. propose_lineage | builds_on / extends / replaces edges | DAG structure, not flat list |
| 5. synthesize_concept | Update cross-paper CanonicalIdea | Concept accumulation, not isolation |
| 6. reconcile_neighbors | Reverse-update old papers | Knowledge graph stays consistent |

### Obsidian Vault Export (v4.0)

Only 5 note types, each with a clear prefix:

```
00_Home/
  00_方向总览.md          # Navigation, not in main graph
  01_阅读顺序.md          # Layered reading guide
10_Lineages/
  L__DPO_Family.md        # Method evolution chain: A→B→C
20_Concepts/
  C__Direct_Preference.md # Mechanism + CanonicalIdea merged
30_Bottlenecks/
  B__Credit_Assignment.md # Cross-paper synthesized insight
40_Papers/
  A__Baselines/           # Must-read foundational papers
  B__Structural/          # Structural improvements
  C__Plugins/             # Plugin-type changes
  D__Peripheral/          # Peripheral references
80_Assets/figures/        # Extracted PDF figures
90_Views/                 # Dataview queries
```

**Paper Note wikilink budget: 6-8 max.** No links to Domain Overview or Paradigm — those go in frontmatter properties only. Graph View shows clean clusters instead of a hairball.

### Method Evolution DAG

Methods form a DAG, not a flat list:

```
GRPO (baseline, depth=0, 7 downstream)
├── GRPO+LP (plugin, depth=1)
│   └── GRPO-LP+sampling (depth=2)
├── GDPO (structural, depth=1, parent=[GRPO, DPO])  ← multi-inheritance
│   └── GDPO+image_thinking (depth=2)
```

When 3+ papers use a method as baseline → auto-promoted to established baseline → can evolve into new paradigm version.

## Quick Start

```bash
# 1. Start the system
cd researchflow-backend && cp .env.example .env  # set ANTHROPIC_API_KEY
make db && make migrate && make up

# 2. Bootstrap a domain
curl -X POST localhost:8000/api/v1/pipeline/init-domain \
  -H "Content-Type: application/json" \
  -d '{"domain": "RLHF for VLM"}'

# 3. Analyze top papers
curl -X POST localhost:8000/api/v1/pipeline/batch?limit=10

# 4. Export Obsidian vault
curl -X POST localhost:8000/api/v1/pipeline/export/obsidian-vault
```

## Connect Claude Code via MCP

ResearchFlow exposes an MCP server with 22 tools, 6 resources, and 4 prompts. Claude Code auto-discovers `.mcp.json`.

### Option A: Remote (connect to deployed server)

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "researchflow": {
      "url": "https://researchflow.xyz/sse"
    }
  }
}
```

### Option B: Local (run MCP server alongside backend)

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

### Option C: Both (local + remote)

```json
{
  "mcpServers": {
    "researchflow-local": {
      "command": "python",
      "args": ["-m", "backend.mcp.server"],
      "cwd": "researchflow-backend",
      "env": {"PYTHONPATH": "."}
    },
    "researchflow-remote": {
      "url": "http://47.101.167.55:8001/sse"
    }
  }
}
```

### MCP Tools (22)

| Tool | What it does |
|------|-------------|
| `run_full_pipeline` | One call: download → enrich → parse → L3 → L4 (6-step) → graph |
| `discover_related_papers` | Semantic Scholar: refs + citations → auto-ingest |
| `build_domain` | Multi-hop discovery from seed paper |
| `search_research_kb` | Hybrid search: keyword + semantic + structured |
| `search_ideas` | Search IdeaDeltas by keyword |
| `get_paper_report` | Generate report: quick / briefing / deep_compare |
| `compare_papers` | Side-by-side comparison of 2-5 papers |
| `import_research_sources` | Ingest paper URLs |
| `get_digest` | Daily / weekly / monthly digest |
| `get_reading_plan` | Tiered: baseline → structural → plugin |
| `propose_directions` | Research direction proposals |
| `enqueue_analysis` | Queue L3 skim or L4 deep |
| `refresh_assets` | Enrich metadata from arXiv / Crossref |
| `record_user_feedback` | Corrections / confirmations |
| `get_paper_detail` | Full detail with DeltaCard |
| `get_graph_stats` | Knowledge graph statistics |
| `review_queue` | Pending review tasks |
| `submit_review_decision` | Approve / reject reviews |

### Example: Use with Claude Code

```
> Use researchflow to search for papers about "reward hacking in RLHF"
> Analyze the top 3 results with full pipeline
> Export the vault and sync to my Obsidian
```

Claude Code will auto-call the MCP tools to execute these operations.

## Sync Obsidian Knowledge Graph

### Automatic Export

```bash
# Export vault from DB to local
curl -X POST localhost:8000/api/v1/pipeline/export/obsidian-vault
# Output: ../obsidian-vault/ (relative to paperAnalysis/)
```

### Sync from Remote Server

```bash
# 1. Export on server
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55 \
  "cd /opt/researchflow && curl -s -X POST localhost:8000/api/v1/pipeline/export/obsidian-vault"

# 2. Sync to local
rsync -avz --delete \
  -e "ssh -i ~/.ssh/autoresearch.pem" \
  root@47.101.167.55:/opt/researchflow/obsidian-vault/ \
  ./obsidian-vault/

# 3. Open in Obsidian
# Point Obsidian vault to ./obsidian-vault/
```

### Obsidian Setup

1. **Open vault**: File → Open Vault → select `obsidian-vault/`
2. **Install Dataview plugin**: Settings → Community Plugins → Dataview (for 90_Views/ queries)
3. **Graph View**: Cmd+G to see the knowledge graph
4. **Recommended Graph Groups**:
   - `path:40_Papers` → Blue (papers)
   - `path:20_Concepts` → Green (concepts)
   - `path:30_Bottlenecks` → Red (bottlenecks)
   - `path:10_Lineages` → Orange (lineage chains)

### Vault Note Types

| Type | Prefix | Content |
|------|--------|---------|
| Paper | `P__` | One-line summary + baseline comparison + equations + figures + reading advice + full analysis |
| Concept | `C__` | Mechanism + CanonicalIdea merged, with comparison table of all papers using this concept |
| Bottleneck | `B__` | Cross-paper synthesis: symptom → root cause → structural vs plugin solutions |
| Lineage | `L__` | ASCII evolution tree + per-step diff + fork points |
| Overview | — | Navigation only (方向总览 + 阅读顺序) |

## System Scale

| Component | Count |
|-----------|-------|
| DB tables | 42 (+4 materialized views) |
| API routes | 99 |
| MCP tools | 22 tools + 6 resources + 4 prompts |
| Services | 34 |
| Frontend pages | 13 |
| Claude Code skills | 18 |
| Built-in paradigms | 4 (RL, VLM, Agent, MotionGen) + LLM discovery |

## Repository Layout

```
researchflow-backend/            # Core backend (single source of truth)
  backend/                       #   FastAPI + ORM + 34 services + MCP
    api/                         #   14 API routers (99 endpoints)
    mcp/                         #   MCP server (22 tools + 6 resources)
    models/                      #   42 SQLAlchemy models
    services/                    #   34 service modules
      analysis_steps.py          #   v4.0: Step 1+2 focused LLM prompts
      baseline_comparator_service.py  # v4.0: Step 3 compare set
      concept_synthesizer_service.py  # v4.0: Step 5 concept synthesis
      incremental_reconciler_service.py # v4.0: Step 6 neighbor reconciliation
      export_service.py          #   v4.0: 5-note-type Obsidian export
  alembic/                       #   DB migrations (001-011)
  frontend/                      #   Next.js 15 web UI
  ARCHITECTURE.md                #   Full technical reference
  DEPLOY_GUIDE.md                #   Cloud deployment guide
obsidian-vault/                  # Exported Obsidian vault (auto-generated)
paperAnalysis/                   # Read-only export: analysis Markdown
paperCollection/                 # Read-only export: index + navigation
paperIDEAs/                      # Read-only export: research ideas
.claude/skills/                  # Claude Code skill definitions (18 skills)
.mcp.json                       # MCP server configuration
AGENTS.md                       # Agent integration guide
```

## Deployment

### Local Development

```bash
cd researchflow-backend
cp .env.example .env             # Set API keys
make db                          # Start PostgreSQL + Redis
make migrate                     # Run migrations
make up                          # Start API + Worker + Frontend + MCP
# API: localhost:8000 | Frontend: localhost:3000 | MCP: localhost:8001
```

### Production (Docker)

```bash
ssh root@your-server
git clone https://github.com/RipeMangoBox/ResearchFlow.git
cd ResearchFlow/researchflow-backend
cp .env.example .env             # Configure production settings
bash deploy.sh                   # One-click: build + migrate + start
# Caddy handles HTTPS at researchflow.xyz
```

See [DEPLOY_GUIDE.md](researchflow-backend/DEPLOY_GUIDE.md) for detailed production setup.

## Docs

| Document | Audience | Content |
|----------|----------|---------|
| [ARCHITECTURE.md](researchflow-backend/ARCHITECTURE.md) | Developers | Knowledge graph, 6-step pipeline, API reference, DB schema |
| [AGENTS.md](AGENTS.md) | Agent builders | MCP tools, skill routing, working rules |
| [DEPLOY_GUIDE.md](researchflow-backend/DEPLOY_GUIDE.md) | Ops | Cloud setup, Docker, server config |

## License

MIT
