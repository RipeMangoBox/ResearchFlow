# researchflow-backend

Core backend for ResearchFlow. See [root README](../README.md) for project overview.

## Setup

```bash
cp .env.example .env     # Set ANTHROPIC_API_KEY (or OPENAI_API_KEY)
docker compose up -d     # Start all services
docker compose exec api alembic upgrade head  # Run 15 migrations
```

Verify: `curl localhost:8000/api/v1/health` → `{"status": "ok"}`

## Architecture

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the complete technical reference:
- 4-layer extraction architecture (CPU → API → VLM → Agent)
- 6-step L4 analysis pipeline with 3 defense lines
- 10-step metadata enrichment (8 APIs)
- Method evolution DAG + faceted taxonomy
- All 100+ API routes, 42 DB tables + 4 materialized views, 45 services

## Key directories

```
backend/
  api/          16 routers (100+ routes)
  services/     45 service modules
  models/       20 ORM models (42 tables + 4 materialized views)
  mcp/          MCP server (23 tools, 6 resources, 4 prompts)
  workers/      ARQ background task queue
  utils/        PDF extraction, GROBID client, frontmatter
alembic/        Migrations 001–015
frontend/       Next.js 15 web UI
tests/          pytest suite
compatibility/  DB → Markdown export pipeline
```

## Deployment

See **[DEPLOY.md](DEPLOY.md)** for production setup (Docker Compose + Caddy).
