# researchflow-backend

Core backend for ResearchFlow. See [root README](../README.md) for project overview.

## Setup

```bash
cp .env.example .env     # Set ANTHROPIC_API_KEY (or OPENAI_API_KEY)
make db                   # Start PostgreSQL + Redis
make migrate              # Run 11 Alembic migrations (42 tables + 4 materialized views)
make up                   # Start API + worker + frontend
```

Verify: `curl localhost:8000/api/v1/health` → `{"status": "ok"}`

## Architecture

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the complete technical reference:
- Knowledge graph structure (DeltaCard → IdeaDelta → DAG)
- 16-step analysis pipeline
- Paper filtering & method classification
- Research exploration sessions
- All 96 API routes & 42 DB tables + 4 materialized views

## Key directories

```
backend/
  api/          13 routers (96 routes)
  services/     30 service modules
  models/       ORM models (42 tables)
  mcp/          MCP server (18 tools, 6 resources, 4 prompts)
alembic/        Migrations 001–011
frontend/       Next.js 15 web UI
tests/          pytest suite
compatibility/  DB → Markdown export pipeline
```

## Deployment

See **[DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)** for cloud setup (Docker Compose + Caddy).
