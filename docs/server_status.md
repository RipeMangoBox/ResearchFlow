# Production Server Status — `47.101.167.55`

> Read-only audit performed 2026-04-26 from a developer laptop using
> `~/.ssh/autoresearch.pem`. Numbers in this document are point-in-time;
> re-run the commands at the top of each section to refresh.

---

## 1. Access & topology

| Field | Value |
|-------|-------|
| Public IP | `47.101.167.55` |
| Hostname | `iZuf61ycyu59jngazy1e3nZ` (Alibaba Cloud ECS) |
| OS | Ubuntu 22.04.5 LTS (jammy) |
| Kernel | `5.15.0-142-generic` x86_64 |
| Hardware | 4 vCPU / 7.1 GiB RAM / 0 swap |
| Disk (`/`) | 69 GB total, 12 GB used (19%), 54 GB free |
| Docker | 29.4.0 + Compose v5.1.3 |
| Uptime | ~8 days at audit time |
| SSH | `ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55` |
| Deploy root | `/opt/researchflow/researchflow-backend/` |
| Branch on server | `feat/paper-filter-and-report-generation`, last commit `bc66a67` |
| ⚠ Local HEAD | `6ca5b91` — **the new code from this session is NOT yet deployed** |

`docker compose ps` snapshot:

```
api       Up 11h (healthy)   127.0.0.1:8000->8000/tcp
worker    Up 11h (healthy)   8000/tcp
mcp       Up 23h (healthy)   127.0.0.1:8001->8001/tcp
caddy     Up 6d              0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
postgres  Up 24h (healthy)   127.0.0.1:5432->5432/tcp
redis     Up 24h (healthy)   127.0.0.1:6379->6379/tcp
frontend  Up 6d              127.0.0.1:3000->3000/tcp
```

Public surface (Caddy `:80`, `auto_https off`):

| Path | Backend |
|------|---------|
| `/api/*`, `/docs`, `/openapi.json` | `api:8000` |
| `/sse`, `/messages/*` | `mcp:8001` |
| anything else | `frontend:3000` |

> No HTTPS configured. Domain is bound to `:80` only.

Disk usage of bind-mounts:

| Path | Size |
|------|------|
| `storage/` (figure cache) | 1.4 M |
| `obsidian-vault/` | 4.1 M |
| `paperAnalysis/` (server-side) | 4 K (essentially empty — exports come from DB on demand) |
| `paperPDFs/` | absent on server (PDFs live in OSS / are streamed in) |
| `exports/` | 4 K |

---

## 2. Environment variables (live `/opt/researchflow/researchflow-backend/.env`)

Secrets redacted. Only fields differing from `.env.example` are listed.

```
# ── DB / queue ─────────────────────────────────
POSTGRES_PASSWORD=<redacted>           # matches `DATABASE_URL`
DATABASE_URL=postgresql+asyncpg://rf:<redacted>@postgres:5432/researchflow
REDIS_URL=redis://redis:6379/0

# ── Object storage ─────────────────────────────
OBJECT_STORAGE_PROVIDER=local          # NOT oss/cos — figures land in storage/
OBJECT_STORAGE_BUCKET=researchflow
OBJECT_STORAGE_REGION=oss-cn-shanghai

# ── LLM (Anthropic-protocol bridge to Kimi) ────
ANTHROPIC_API_KEY=<redacted>
ANTHROPIC_BASE_URL=https://api.kimi.com/coding
# OPENAI_* are NOT set on the live server — the OpenAI-compatible
# path documented in .env.deploy is currently unused.

# ── External enrichment ────────────────────────
S2_API_KEY=<redacted>
GITHUB_TOKEN=<redacted>
OPENREVIEW_USERNAME=51275901129@stu.ecnu.edu.cn
OPENREVIEW_PASSWORD=<redacted>

# ── Outbound proxies (for venue scraping & HTTPS egress) ──
HTTPS_PROXY=http://172.17.0.1:7890
VENUE_PROXY=http://172.18.0.1:7890
NO_PROXY=localhost,postgres,redis,127.0.0.1,grobid,api.semanticscholar.org,api.crossref.org,api.openalex.org,dblp.org

# ── PWC dump path ──────────────────────────────
PWC_DUMP_PATH=/app/storage/pwc/links-between-papers-and-code.json.gz
```

> ⚠ **Mismatch with `.env.deploy` in the repo**: the repo's deploy template
> uses `OPENAI_API_KEY` + `OPENAI_BASE_URL=https://api.kimi.com/coding/v1`
> + `OPENAI_MODEL=kimi-k2.6` (OpenAI-compatible). The live server uses the
> Anthropic-protocol path. Both routes terminate at Kimi, but
> `figure_extraction_service.py` and `_classify_and_detect_missed` build an
> `openai.AsyncOpenAI` client from `settings.openai_api_key` /
> `settings.openai_base_url`. On the live server those are empty, so the
> OpenAI SDK would fall back to `api.openai.com` with the Anthropic key
> — which 401s. **L2 figures are nevertheless 100% covered (185/185)**, so
> some fallback path is producing them; investigate before turning on the
> new figure-table writes from migration 024.

---

## 3. Database schema state

`alembic_version`: **`023`** (next migration, **`024_source_quality_and_paper_figures.py`**, exists in this repo but is NOT yet applied).

63 tables in `public`. Key counts (point-in-time):

| Table | Rows |
|-------|------|
| `papers` | 193 |
| `paper_analyses` | 524 |
| `paper_reports` | 14 |
| `paper_report_sections` | 140 |
| `delta_cards` | 153 |
| `taxonomy_nodes` | 112 |
| `method_nodes` | 0 |
| `venue_papers` | 42 795 |

Coverage for the deep-report pipeline:

| Signal | Count |
|--------|-------|
| `paper_analyses` rows at `l2_parse` (current) | 185 |
| … of which `extracted_figure_images IS NOT NULL` | 185 |
| `paper_analyses` rows at `l4_deep` with `full_report_md` | 134 |
| `paper_reports` with `title_zh` | **14 / 14 — 100%** |

`title_zh` already follows the "中文核心词 + 论文 acronym" pattern that
`vault_export_v6._paper_slug` now consumes:

```
Co-Evolving LLM Decision and …       → COS-PLAY：共进化双代理框架实现长程任务中的技能发现与复用
WebGen-R1: Incentivizing LLMs to …    → WebGen-R1：基于强化学习与模板约束的网站生成框架
UDM-GRPO: Stable and Efficient …      → UDM-GRPO：面向均匀离散扩散模型的稳定高效群组相对策略优化
```

Pre-existing data anomalies (matter for the audit script):

| Anomaly | Count | Action |
|---------|-------|--------|
| `venue` empty string | 79 | Run audit, then triage |
| `venue = 'arXiv (Cornell University)'` | 71 | Already normalized at export time by `VENUE_NORMALIZE` |
| Title is a bare arXiv id (`^[0-9]{4}\.[0-9]{4,5}`) | 14 | Soft-flag via `source_quality='low'` after migration 024 |
| Title length > 140 | 2 | Same |
| `extracted_figure_images` stored as JSON scalar (not array) | ≥1 | **Patched today**: `audit_kb_quality.py` and `backfill_paper_figures.py` now guard with `jsonb_typeof(...) = 'array'` |

⚠ **None of the 134 existing reports contain `{{FIG:xxx}}` markers** —
the live data was generated by the older prompt. The new
`paper_report` agent in `agent_runner.py` will only produce markers for
papers re-run after deploy. Legacy reports will be inlined by
`_autoinject_figures_by_role` (semantic_role → section heading).

---

## 4. External dependencies — connectivity

Tests run from inside the `api` container.

| Endpoint | Status | Notes |
|----------|--------|-------|
| `https://api.kimi.com/coding/v1` | HTTP 404 (~480 ms) | 404 on the bare path is expected; host reachable. |
| `https://oss-cn-shanghai.aliyuncs.com` | HTTP 403 | Bucket root rejects unauth GET; DNS + TCP path OK. |
| `http://127.0.0.1:8000/healthz` | HTTP 404 | The healthz path differs from `docs/deploy.md` — investigate. |
| `http://127.0.0.1/` (Caddy 80) | HTTP 308 | Normal redirect to frontend. |

⚠ **Worker error in tail logs**: `sqlalchemy.exc.MissingGreenlet` —
unrelated to the changes in this session, but indicates an existing
async-session-misuse bug somewhere in the ARQ task graph. Surfaces as a
Sunday night background job; not blocking the new pipeline but worth
filing.

---

## 5. Pre-rollout checklist for this session's changes

The 13 code changes (A through F) must be **deployed and migrated**
before E2E. Sequenced steps:

```bash
# 1) Sync the new code to the server (matches existing deploy.sh pattern)
rsync -avz --delete --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='storage/' --exclude='.env' --exclude='obsidian-vault/' \
  -e "ssh -i ~/.ssh/autoresearch.pem" \
  ./researchflow-backend/ root@47.101.167.55:/opt/researchflow/researchflow-backend/

# 2) Apply migration 024
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55 \
  'cd /opt/researchflow/researchflow-backend && \
   docker compose exec -T api bash -c "PYTHONPATH=/app alembic -c alembic/alembic.ini upgrade head"'

# 3) Verify the new schema landed
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55 \
  'docker compose -f /opt/researchflow/researchflow-backend/docker-compose.yml \
   exec -T postgres psql -U rf -d researchflow -c "\d paper_figures" \
   -c "SELECT column_name FROM information_schema.columns WHERE table_name='\''papers'\'' AND column_name='\''source_quality'\'';"'

# 4) Restart api+worker so the new agent prompt and figure dual-write take effect
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55 \
  'docker compose -f /opt/researchflow/researchflow-backend/docker-compose.yml restart api worker'

# 5) Run a baseline audit (now that paper_figures exists)
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55 \
  'docker compose -f /opt/researchflow/researchflow-backend/docker-compose.yml \
   exec -T api python -m scripts.audit_kb_quality \
     --out /paperAnalysis/quality_report_$(date +%F).md'

# 6) Backfill legacy figures into paper_figures (dry-run first)
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55 \
  'docker compose -f /opt/researchflow/researchflow-backend/docker-compose.yml \
   exec -T api python -m scripts.backfill_paper_figures'
# then re-run with --commit when satisfied

# 7) Pick ONE known-good paper, regenerate its report and vault page
#    (use whichever CLI/RPC the project exposes; or call the agent
#    directly through ingest_workflow on a single paper_id).
#    Verify on the server:
#      paper_figures           — has rows for that paper_id
#      paper_reports.title_zh  — non-null
#      full_report_md tail     — contains <!-- figure_placements: ... -->
#      obsidian-vault/paper/<venue_year>/P__<chinese>_<acronym>.md
#                              — has metadata table at top, inline figures, no "## 论文图表"

# 8) Only after step 7 passes for one paper, repeat for a small batch (~10),
#    then a full re-export.
```

---

## 6. Open questions / follow-ups

1. **Live `.env` uses Anthropic-protocol bridge but figure extraction
   builds an OpenAI client.** Needs a small reconciliation: either set
   `OPENAI_API_KEY=<kimi-key>` and `OPENAI_BASE_URL=https://api.kimi.com/coding/v1`
   in the live `.env` (matching `.env.deploy` in the repo) **or** patch
   `figure_extraction_service.py` to also accept the Anthropic env vars.
   Leaving this until we have a concrete failure during step 7 of the
   rollout.
2. **Worker `MissingGreenlet`** — existing bug, surface in next session.
3. **`/healthz` path returns 404** — `docs/deploy.md` references it as a
   sanity check. Either the route was renamed or the doc is stale; verify
   with `curl http://127.0.0.1:8000/api/v1/healthz` before claiming it
   broken.
4. **No HTTPS** — `Caddyfile` runs `auto_https off` and serves `:80`. If
   the team wants `https://researchflow.example.com`, set `DOMAIN` and
   re-enable Caddy ACME.
