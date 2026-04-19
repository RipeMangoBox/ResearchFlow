# researchflow-backend

Core backend for ResearchFlow. See [root README](../README.md) for project overview.

## Setup

```bash
cp .env.example .env     # Set ANTHROPIC_API_KEY (or OPENAI_API_KEY)
docker compose up -d     # Start all services
docker compose exec api alembic upgrade head  # Run 16 migrations
```

Verify: `curl localhost:8000/api/v1/health` → `{"status": "ok"}`

## Architecture

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the complete technical reference:
- 4-layer extraction architecture (CPU → API → VLM → Agent)
- 6-step L4 analysis pipeline with 3 defense lines
- 10-step metadata enrichment (8 APIs)
- Method evolution DAG + faceted taxonomy
- All 130 API routes, 58 DB tables + 4 materialized views, 55 services

## Key directories

```
backend/
  api/          16 routers (130 routes)
  services/     55 service modules
  models/       24 ORM model files (58 tables + 4 materialized views)
  mcp/          MCP server (35 tools, 6 resources, 4 prompts)
  workers/      ARQ background task queue (22 tasks)
  utils/        PDF extraction, GROBID client, frontmatter
alembic/        Migrations 001–016
frontend/       Next.js 15 web UI
tests/          pytest suite
compatibility/  DB → Markdown export pipeline
```

## Deployment

See **[DEPLOY.md](DEPLOY.md)** for production setup (Docker Compose + Caddy).
