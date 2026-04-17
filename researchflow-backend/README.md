# researchflow-backend

Core backend for ResearchFlow. See [root README](../README.md) for project overview.

## Setup

```bash
cp .env.example .env     # Set ANTHROPIC_API_KEY (or OPENAI_API_KEY)
make db                   # Start PostgreSQL + Redis
make migrate              # Run 7 Alembic migrations (31 tables)
make up                   # Start API + worker + frontend
```

Verify: `curl localhost:8000/api/v1/health` → `{"status": "ok"}`

## Architecture

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the complete technical reference:
- Knowledge graph structure (DeltaCard → IdeaDelta → DAG)
- 16-step analysis pipeline
- Paper filtering & method classification
- Research exploration sessions
- All 81 API routes & 31 DB tables

## Key directories

```
backend/
  api/          13 routers (81 routes)
  services/     25 service modules
  models/       ORM models (31 tables)
  mcp/          MCP server (18 tools, 3 resources, 2 prompts)
alembic/        Migrations 001–007
tests/          29 pytest tests
compatibility/  DB → Markdown export pipeline
```

## Deployment

See **[DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)** for cloud setup (Docker Compose + Caddy).
