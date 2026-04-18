<p align="center">
  <img src="./assets/LOGO.png" alt="ResearchFlow" width="260"/>
</p>
<h1 align="center">ResearchFlow</h1>
<p align="center"><a href="README.md">English</a> · <a href="README_CN.md">中文</a></p>

---

**Give one paper → auto-build a domain knowledge graph with method evolution tracking.**

ResearchFlow is a research operating system. It ingests papers, analyzes them with LLMs, builds a structured knowledge graph that captures how methods improve upon each other, and supports iterative research exploration.

## Complete example: from zero to knowledge graph

```bash
# 1. Start the system
cd researchflow-backend && cp .env.example .env  # set ANTHROPIC_API_KEY
make db && make migrate && make up

# 2. Bootstrap a domain from its awesome-list
curl -X POST localhost:8000/api/v1/pipeline/init-domain \
  -H "Content-Type: application/json" \
  -d '{"domain": "RLHF for VLM"}'
# → Finds awesome repo → imports 72 papers → scores & prioritizes

# 3. Analyze the top 10 papers
curl -X POST localhost:8000/api/v1/pipeline/batch?limit=10
# → Downloads PDFs → L2 parse → L3 skim → L4 deep analysis
# → Builds DeltaCards → IdeaDeltas → method evolution DAG

# 4. Explore iteratively
curl -X POST localhost:8000/api/v1/explore/start \
  -d '{"query": "RL advantage disappearance in VLM"}'
# → System classifies results: 3 structural, 5 plugin, 2 reward
# → Suggests: "dominated by plugins, try adjacent fields"
```

Or use **Web UI** at `localhost:3000`, or connect **Claude Code / Codex** via MCP (auto-discovers `.mcp.json`).

## What makes it different

**Methods are a DAG, not a flat list.** The system tracks that GRPO → GRPO+LP → GDPO is a chain, and that GDPO inherits from both GRPO and DPO. When 3+ papers use a method as baseline, it's automatically marked as an established baseline and can be promoted to a new paradigm version.

```
GRPO (baseline, depth=0, 7 downstream papers)
├── GRPO+LP (plugin, depth=1)
│   └── GRPO-LP+sampling (depth=2)
├── GDPO (structural, depth=1, parent=[GRPO, DPO])  ← multi-inheritance
│   └── GDPO+image_thinking (depth=2)
```

**Papers are filtered by evidence quality:**

| Priority | Criteria | Score weight |
|----------|----------|-------------|
| Highest | Open data | 0.40 |
| High | Open code | 0.30 |
| Medium | Accepted, no code | 0.20 |
| Low | Preprint | 0.10 |
| Bonus | Top venue + recency + team quality | +0.05–0.25 |

**L4 analysis auto-extracts method classification:**
- `method/structural_architecture` vs `method/plugin_module` vs `method/reward_design` ...
- `improvement/fundamental_rethink` vs `improvement/additive_plugin` ...
- Research bottleneck addressed (auto-creates ProjectBottleneck)

## System at a glance

| | Count |
|-|-------|
| DB tables | 42 (+4 materialized views) |
| API routes | 96 |
| MCP tools | 18 tools + 6 resources + 4 prompts |
| Services | 30 |
| Migrations | 11 (001–011) |
| Frontend pages | 13 |
| Built-in paradigms | 4 (RL, VLM, Agent, MotionGen) + LLM candidate discovery |

## Repository layout

```
researchflow-backend/          # Core backend (single source of truth)
  backend/                     #   FastAPI + ORM + 30 services + MCP server
  alembic/                     #   DB migrations (001–011)
  frontend/                    #   Next.js 15 web UI (papers/search/reviews/reports)
  compatibility/               #   DB → Markdown/CSV export tools
  tests/                       #   pytest suite
  ARCHITECTURE.md              #   ← Full technical reference (v3.2)
  DEPLOY_GUIDE.md              #   Cloud deployment guide
paperAnalysis/                 # Read-only export: analysis Markdown (by domain/venue)
paperCollection/               # Read-only export: index + Obsidian navigation
paperIDEAs/                    # Read-only export: research outputs
.claude/skills/                # Claude Code skill definitions (18 skills)
scripts/                       # Local utility & maintenance scripts
assets/                        # Logo, banner images
linkedCodebases/               # Symlinks to external repos for cross-referencing
AGENTS.md                      # Agent integration guide
```

## Docs

| Document | For whom | Content |
|----------|----------|---------|
| **[ARCHITECTURE.md](researchflow-backend/ARCHITECTURE.md)** | Developers | Knowledge graph structure, method DAG, pipeline, API reference, DB schema |
| **[AGENTS.md](AGENTS.md)** | Agent builders | MCP tools list, skill routing, working rules |
| **[DEPLOY_GUIDE.md](researchflow-backend/DEPLOY_GUIDE.md)** | Ops | Cloud setup, Docker, costs |

## License

MIT
