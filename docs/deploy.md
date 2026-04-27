# ResearchFlow Deployment Guide

> Authoritative reference for deploying the ResearchFlow backend stack on a
> single host. Code-level details live in `researchflow-backend/`; this
> document is the operator's view.

---

## 1. Service topology

`researchflow-backend/docker-compose.yml` orchestrates one Docker network with:

| Service     | Image                       | Purpose                              | Memory cap |
|-------------|-----------------------------|--------------------------------------|------------|
| `postgres`  | `pgvector/pgvector:pg16`    | Primary store (papers, agents, kb)   | 1280 M     |
| `redis`     | `redis:7-alpine`            | Job queue + cache                    | 256 M      |
| `api`       | (built from repo Dockerfile)| FastAPI HTTP API + worker entrypoint | 2048 M     |
| `worker`    | (same image)                | Background ingestion + reports       | 2048 M     |
| `frontend`  | (built from `frontend/`)    | Next.js UI                           | 512 M     |
| `mcp`       | (same backend image)        | MCP server for Claude / IDE clients  | 512 M     |
| `caddy`     | `caddy:2`                   | TLS terminator / reverse proxy       | 128 M     |

GROBID is intentionally **disabled** — VLM page scanning replaces GROBID
formula extraction (see the commented `grobid:` block in `docker-compose.yml`).

Storage volumes mounted into `api` / `worker`:
- `./backend` (source — supports hot reload in dev)
- `./exports`, `./storage`, `./obsidian-vault`, `./paperAnalysis`

---

## 2. Environment variables

Every variable listed in `researchflow-backend/.env.example` is required.
Copy that file to `.env` (dev) or `.env.deploy` (prod) and fill in the
secrets. Below are the must-set knobs grouped by purpose.

### Database / queue

| Variable | Example | Notes |
|----------|---------|-------|
| `POSTGRES_PASSWORD` | `s3cret`  | Used by `postgres` service AND embedded in `DATABASE_URL`. Keep them in sync. |
| `DATABASE_URL` | `postgresql+asyncpg://rf:s3cret@postgres:5432/researchflow` | When running outside Docker, swap `postgres` → `localhost`. |
| `REDIS_URL` | `redis://redis:6379/0` | DB index 0 is the worker queue. |

### Object storage (figures, PDFs)

| Variable | Example | Notes |
|----------|---------|-------|
| `OBJECT_STORAGE_PROVIDER` | `oss` (Aliyun) / `cos` (Tencent) / `local` | `local` falls back to filesystem under `./storage`. |
| `OBJECT_STORAGE_BUCKET` | `researchflow` | Must already exist with public-read permission for figure URLs. |
| `OBJECT_STORAGE_SECRET_ID` / `_SECRET_KEY` | _(secret)_ | AccessKey pair for the configured provider. |
| `OBJECT_STORAGE_REGION` | `oss-cn-shanghai` / `ap-shanghai` | Region matches the bucket's home region. |
| `OBJECT_STORAGE_CDN_DOMAIN` | _(optional)_ | If set, `get_public_url` returns CDN URLs; otherwise direct bucket URL. |

> Aliyun OSS auto-detects whether the deployment is inside Alibaba Cloud and
> uses the free `*-internal.aliyuncs.com` endpoint when reachable.

### LLM access

| Variable | Example | Notes |
|----------|---------|-------|
| `OPENAI_API_KEY` | _(Kimi K2.6 key)_ | Production setup uses an OpenAI-compatible endpoint. |
| `OPENAI_BASE_URL` | `https://api.kimi.com/coding/v1` | Empty value → default OpenAI endpoint. |
| `OPENAI_MODEL` | `kimi-k2.6` | Used by figure / formula / agent extractors. |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` | _(optional)_ | Fallback if OpenAI-compatible vars are unset. |
| `VLM_MAX_TOKENS_HEAVY/MEDIUM/LIGHT/TINY` | `4096 / 2048 / 1024 / 512` | Per-tier cap to control cost. |

### External enrichment

| Variable | Example | Notes |
|----------|---------|-------|
| `S2_API_KEY` | _(Semantic Scholar key)_ | Used by discovery & enrichment phases. |
| `GITHUB_TOKEN` | _(PAT)_ | Bumps GitHub API rate limit when fetching code repo metadata. |
| `GROBID_URL` | _(unset)_ | Leave empty unless you re-enable GROBID. |

### Networking / surface

| Variable | Example | Notes |
|----------|---------|-------|
| `DOMAIN` | `researchflow.example.com` | Caddy uses this for ACME TLS. Use `localhost` for local-only deploy. |
| `MCP_AUTH_TOKEN` | _(opaque)_ | Required header for `mcp` service auth. |

---

## 3. First-time deploy

```bash
git clone <repo> /opt/researchflow
cd /opt/researchflow/researchflow-backend
cp .env.example .env
$EDITOR .env                # fill in every secret listed above

docker compose pull
docker compose build api worker frontend mcp
docker compose up -d postgres redis
docker compose run --rm api alembic upgrade head        # apply schema (≥024)
docker compose up -d api worker frontend mcp caddy
```

Sanity checks:

```bash
curl http://127.0.0.1:8000/healthz                       # API
docker compose exec postgres psql -U rf -d researchflow \
  -c "SELECT count(*) FROM papers;"                      # DB
docker compose exec api python -m scripts.audit_kb_quality \
  --out /paperAnalysis/quality_report_$(date +%F).md     # G audit
```

## 4. Schema upgrades

Migrations live in `researchflow-backend/alembic/versions/`. Apply:

```bash
docker compose run --rm api alembic upgrade head
```

When introducing a new migration, list it in commit notes — `024_…` is the
last additive change adding `papers.source_quality` and the `paper_figures`
table. To migrate historical figure JSONB into the new table, run the
optional backfill once:

```bash
docker compose run --rm api python -m scripts.backfill_paper_figures --commit
```

## 5. Disaster recovery

- Postgres data: `pgdata` Docker volume — back up with `pg_dump` daily.
- Object storage: bucket lifecycle handled by the cloud provider; no local
  copy.
- `obsidian-vault/` and `paperAnalysis/` are derivable from the database
  via `vault_export_v6.export_vault` and `papers-build-collection-index`.

If the database is healthy but the vault is missing, regenerate it:

```bash
docker compose exec api python -c \
  "import asyncio; from backend.database import async_session; \
   from backend.services.vault_export_v6 import export_vault; \
   asyncio.run(export_vault(next(async_session().__aiter__()).__anext__()))"
```

(Or use the dedicated CLI exposed by the API once it ships.)

## 6. Routine operations

| Task | Command |
|------|---------|
| Apply migrations | `docker compose run --rm api alembic upgrade head` |
| Rebuild Obsidian vault | run `vault_export_v6.export_vault` (above) |
| Rebuild paperCollection index | `python .claude/skills/papers-build-collection-index/scripts/build_paper_collection.py` |
| Audit KB quality | `python -m scripts.audit_kb_quality` |
| Backfill `paper_figures` | `python -m scripts.backfill_paper_figures --commit` |
| Tail API logs | `docker compose logs -f api` |
