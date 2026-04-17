<p align="center">
  <img src="./assets/LOGO.png" alt="ResearchFlow logo" width="280"/>
</p>

<h1 align="center">ResearchFlow</h1>

<p align="center"><strong>Web-first research operating system.<br/>Import papers → auto-analyze → generate reports → recommend reading order → daily/weekly/monthly digests.<br/>Normal users use Web UI. Power users connect via Claude Code / Codex (MCP).</strong></p>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README_CN.md">中文</a>
</p>

<p align="center">
  <img alt="Web-first" src="https://img.shields.io/badge/Web--first-Research%20OS-1f6feb?style=flat-square"/>
  <img alt="FastAPI backend" src="https://img.shields.io/badge/FastAPI-backend-0891b2?style=flat-square"/>
  <img alt="PostgreSQL+pgvector" src="https://img.shields.io/badge/PostgreSQL-pgvector-0f766e?style=flat-square"/>
  <img alt="MCP compatible" src="https://img.shields.io/badge/MCP-compatible-d97706?style=flat-square"/>
  <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-111827?style=flat-square"/>
</p>

---

> **Single-core multi-projection architecture (v3.1)**
>
> PostgreSQL is the single source of truth. DeltaCard is the intermediate truth layer (extracted once, projected many times). Web UI, Claude Code, Codex, and Obsidian are read-only projections of the core backend.

---

## Architecture

```text
              ┌─────────────────────────────────────────┐
              │           User Interfaces                │
              │  Web (Next.js)  │  Claude/Codex (MCP)    │
              └────────┬────────────────┬────────────────┘
                       │                │
              ┌────────▼────────────────▼────────────────┐
              │          Core Backend (FastAPI)           │
              │                                          │
              │  67 API routes  │  15 MCP tools           │
              │  3 MCP resources │  2 MCP prompts         │
              ├──────────────────────────────────────────┤
              │  Services (21 modules)                   │
              │  DeltaCard pipeline │ Assertion lifecycle │
              │  6-route search    │ Quality scoring      │
              ├──────────────────────────────────────────┤
              │  PostgreSQL (31 tables) + pgvector        │
              │  Redis (task queue) + Object Storage      │
              └──────────────────────────────────────────┘
                       │
              ┌────────▼────────────────────────────────┐
              │  Read-only Exports (projections)         │
              │  paperAnalysis/ │ paperCollection/       │
              │  paperIDEAs/   │ Obsidian vault          │
              └─────────────────────────────────────────┘
```

### Key concepts

| Concept | Role |
|---------|------|
| **DeltaCard** | Intermediate truth layer — what a paper changed relative to canonical paradigm |
| **IdeaDelta** | Reusable knowledge atom derived from DeltaCard. Published only when evidence >= 2 |
| **GraphAssertion** | Directed edge with lifecycle: candidate → published → deprecated. High-value edges require review |
| **GraphNode** | Unified node registry. Every entity gets a node for graph queries |

### Analysis pipeline (16 steps)

```
ingest → canonicalize → enrich → fetch_assets → parse (L2)
→ skim (L3) → deep (L4) → delta_card_build → entity_resolution
→ assertion_propose → evidence_audit → review → publish
→ index → export → digest
```

---

## System scale

| Component | Count |
|-----------|-------|
| DB tables | 31 (6 Alembic migrations) |
| API routes | 67 (11 routers) |
| MCP tools | 15 |
| MCP resources | 3 |
| Backend services | 21 |
| Test cases | 29 |
| Frontend pages | 7 + layout |
| Paradigm frames | 4 (RL / VLM / Agent / MotionGen) |

---

## Quick start

```bash
git clone https://github.com/RipeMangoBox/ResearchFlow.git
cd ResearchFlow/researchflow-backend

cp .env.example .env    # Set passwords and API keys
make db                 # Start Postgres + Redis
make migrate            # Run 6 Alembic migrations
make up                 # Start all services
```

Open `http://localhost:3000` for Web UI. Claude Code / Codex auto-discover `.mcp.json`.

---

## Repository structure

```text
ResearchFlow/
├── researchflow-backend/           # Core backend (single source of truth)
│   ├── backend/                    # FastAPI + ORM + Services + MCP
│   ├── frontend/                   # Next.js (7 pages)
│   ├── alembic/                    # DB migrations (001-006)
│   ├── compatibility/              # DB → Markdown export
│   ├── tests/                      # pytest (29 tests)
│   ├── ARCHITECTURE.md             # v3.1 architecture spec
│   └── DEPLOY_GUIDE.md
├── paperAnalysis/                  # Export: analysis notes
├── paperCollection/                # Export: index + navigation
├── paperIDEAs/                     # Export: research outputs
├── .claude/skills/                 # Claude Code skills
├── scripts/                        # Utilities
├── AGENTS.md                       # Agent guide
└── README_CN.md                    # Chinese docs
```

## Documentation

| Document | Content |
|----------|---------|
| [ARCHITECTURE.md](researchflow-backend/ARCHITECTURE.md) | v3.1 architecture (7-layer graph, 16-step pipeline, 8 constraints) |
| [DEVELOPMENT_PLAN.md](researchflow-backend/DEVELOPMENT_PLAN.md) | Phased development roadmap |
| [DEPLOY_GUIDE.md](researchflow-backend/DEPLOY_GUIDE.md) | Cloud deployment guide |

## License

MIT
