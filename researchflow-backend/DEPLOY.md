# ResearchFlow 部署指南

> 经过生产验证 (2026-04-19)，适用于阿里云 ECS 4C8G 或同等配置。

---

## 1. 服务器要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 2 核 | 4 核 |
| 内存 | 4 GB | 8 GB |
| 磁盘 | 40 GB SSD | 70 GB SSD (大文件走 OSS) |
| 系统 | Ubuntu 22.04+ | Ubuntu 24.04 |
| 网络 | 开放 80/443 | 有域名 + SSL |

---

## 2. 一键部署

```bash
# 1. 上传代码
rsync -avz --delete --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='storage/' --exclude='.env' --exclude='obsidian-vault/' \
  -e "ssh -i ~/.ssh/autoresearch.pem" \
  ./researchflow-backend/ root@47.101.167.55:/opt/researchflow/researchflow-backend/

# 2. SSH 登录
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55
cd /opt/researchflow/researchflow-backend

# 3. 配置环境变量 (首次部署)
cp .env.example .env
vim .env   # 必填项见下方 §5

# 4. 启动所有容器
docker compose up -d

# 5. 运行数据库迁移
docker compose exec api bash -c 'PYTHONPATH=/app alembic -c alembic/alembic.ini upgrade head'
```

部署完成后访问:
- Web UI: `http://47.101.167.55/`
- API Docs: `http://47.101.167.55/api/v1/docs`
- MCP SSE: `http://47.101.167.55/sse`

---

## 3. 容器架构 (7 容器)

| 容器 | 镜像 | 内存限制 | 端口 | 说明 |
|------|------|---------|------|------|
| **postgres** | pgvector/pgvector:pg16 | 1280 MB | 5432 | PostgreSQL 16 + pgvector |
| **redis** | redis:7-alpine | 256 MB | 6379 | 任务队列 (arq) |
| **api** | 自构建 (Python 3.12) | 2048 MB | 8000 | FastAPI + uvicorn (prod --workers 2) |
| **worker** | 同 api | 1536 MB | - | ARQ 42 个任务 (含 V6) |
| **mcp** | 同 api | 256 MB | 8001 | MCP SSE 35 个工具 |
| **frontend** | Node.js (Next.js) | 256 MB | 3000 | Web UI |
| **caddy** | caddy:2-alpine | ~50 MB | 80/443 | 反向代理 + HTTPS |

**总内存峰值**: ~5.7 GB (含系统)，8 GB 服务器余量充裕

> **GROBID 已移除 (2026-04-20)**: 长 PDF 频繁 OOM 且占 2-3 GB 内存。公式提取改为 VLM page scan（Claude Sonnet + PyMuPDF 渲染），引用/作者由 S2 API fallback 补全。如需恢复，取消 `docker-compose.yml` 中 grobid 注释即可。

### 公式提取方案 (VLM page scan)

| 步骤 | 实现 | 说明 |
|------|------|------|
| 页面选择 | PyMuPDF 文本扫描 | 按数学符号密度排序，选 top 9 页 |
| 图片渲染 | PyMuPDF 1.5x zoom | 每批 3 页，最多 3 批 |
| LaTeX 提取 | Claude VLM (1 call/batch) | 返回 LaTeX + label + context |
| 引用/作者 | S2 API | GROBID 不可用时自动 fallback |
| 章节/文本 | PyMuPDF | 不依赖 GROBID |

成本: ~$0.02-0.05/篇 (2-3 次 VLM API 调用)

### LLM 调用路由逻辑

`llm_service.py` 按以下优先级选择 LLM 调用方式:

1. **有 `OPENAI_BASE_URL` + `OPENAI_API_KEY`** → 走 OpenAI 兼容代理 (推荐，国内必选)
2. **有 `ANTHROPIC_API_KEY`** → 直连 Anthropic API (国内无法直连)
3. **都没有** → Mock 模式 (返回占位响应)

> **⚠️ 关键经验**: 如果同时设了 `ANTHROPIC_API_KEY` 和 `OPENAI_BASE_URL`，LLM 调用优先走代理。
> 之前的 bug: 代码先检查 `ANTHROPIC_API_KEY`，导致国内服务器直连 Anthropic 超时，L4 分析反复失败。

### L4 分析截断问题

代理 API 可能截断长 JSON 响应，导致 L4 pipeline 缺少字段 (`evidence_units`, `changed_slots` 等)。已有的防御机制:

| 防御层 | 实现 | 位置 |
|--------|------|------|
| JSON 暴力修复 | `_repair_truncated_json()` 补全括号 | `analysis_steps.py` |
| 字段缺失重试 | 最多 2 次重试，缺失字段提示 LLM 补全 | `analysis_steps.py` |
| 垃圾响应检测 | `_is_garbage_response()` 检测空响应 | `llm_service.py` |
| 降级继续 | 字段仍缺时标记 partial，继续后续步骤 | `analysis_steps.py` |
| 自动恢复 cron | `task_pipeline_recover` 每 30 分钟扫描卡住论文 | `arq_app.py` |

**L4 截断根因 (2026-04-20 修复)**:
Step 1 和 Step 2 的 prompt 要求返回大量中文摘要 + 结构化 JSON，中文每字 ~1.5-2 tokens。
旧配置 `max_tokens=4096` 只够输出 JSON 前半段（`key_equations`, `key_figures`），后面的字段被截断。
修复: Step 1 → 8192, Step 2 → 16000。修改位置: `analysis_steps.py` 的 `run_step1/2` 函数。

**其他排查方向**:
- 确认 `OPENAI_MODEL=kimi-k2.6`
- Kimi Coding API 要求 `User-Agent: claude-code/1.0`（代码已自动设置）
- 使用 streaming 模式减少截断风险

### 代码挂载 (volume mount)

api/worker/mcp 通过 `./backend:/app/backend` 挂载源码，rsync 后自动生效无需 rebuild:

```yaml
# docker-compose.yml 中 api/worker/mcp 共有
volumes:
  - ./backend:/app/backend
```

---

## 4. 日常部署

### 场景 A: Python 代码变更 (最常见，~8 秒)

```bash
# 本地
rsync -avz --delete --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='storage/' --exclude='.env' --exclude='obsidian-vault/' \
  -e "ssh -i ~/.ssh/autoresearch.pem" \
  ./researchflow-backend/ root@47.101.167.55:/opt/researchflow/researchflow-backend/
# API 会自动 reload (--reload)，worker/mcp 需要手动重启
ssh ... "cd /opt/researchflow/researchflow-backend && docker compose restart worker mcp"
```

### 场景 B: docker-compose.yml 或 .env 变更

```bash
# ⚠️ 必须用 up --force-recreate，restart 不会重新读取 .env 和内存限制
docker compose up -d --force-recreate api worker mcp
```

### 场景 C: requirements.txt 或 Dockerfile 变更

```bash
docker compose build api && docker compose up -d api worker mcp
```

### 场景 D: 数据库迁移

```bash
docker compose exec api bash -c 'PYTHONPATH=/app alembic -c alembic/alembic.ini upgrade head'
```

---

## 5. 环境变量 (.env)

```bash
# === 数据库 (必填) ===
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql+asyncpg://rf:${POSTGRES_PASSWORD}@postgres:5432/researchflow
DATABASE_URL_SYNC=postgresql://rf:${POSTGRES_PASSWORD}@postgres:5432/researchflow
REDIS_URL=redis://redis:6379/0

# === LLM (必填) ===
# Kimi K2.6 — 原生多模态，OpenAI 兼容接口，需要 User-Agent: claude-code/1.0
OPENAI_API_KEY=sk-kimi-xxx
OPENAI_BASE_URL=https://api.kimi.com/coding/v1
OPENAI_MODEL=kimi-k2.6
ANTHROPIC_API_KEY=                           # 留空，全部走 Kimi

# === 对象存储 OSS (推荐) ===
OBJECT_STORAGE_PROVIDER=oss                 # oss | cos | local
OBJECT_STORAGE_BUCKET=researchflow
OBJECT_STORAGE_SECRET_ID=your_access_key_id
OBJECT_STORAGE_SECRET_KEY=your_access_key_secret
OBJECT_STORAGE_REGION=oss-cn-shanghai       # 含 oss- 前缀

# === Semantic Scholar (推荐，提升速率) ===
S2_API_KEY=your_s2_api_key                  # https://www.semanticscholar.org/product/api

# === GROBID (已禁用，worker 中设为空字符串) ===
# GROBID_URL=http://grobid:8070             # 如需恢复取消注释

# === OpenReview (可选) ===
OPENREVIEW_USERNAME=your_email
OPENREVIEW_PASSWORD=your_password

# === 代理 (阿里云环境) ===
HTTP_PROXY=http://172.17.0.1:7890           # mihomo on host
HTTPS_PROXY=http://172.17.0.1:7890
NO_PROXY=localhost,postgres,redis,127.0.0.1```

---

## 6. 存储规划

### 磁盘 vs OSS 分工

| 数据类型 | 存储位置 | 大小/篇 | 说明 |
|---------|---------|---------|------|
| PDF 原文 | **OSS** | 5-10 MB | 自动上传，本地只做临时缓存 |
| 论文图片 | **OSS** | 1-4 MB | 提取后直接存 OSS |
| PostgreSQL | 本地 volume | 慢增长 | 结构化数据 |
| Docker 镜像 | 本地 | ~6.8 GB 固定 | 不可缩减 |
| 代码 + 前端 | 本地 | ~530 MB | rsync 同步 |
| Markdown 导出 | 本地 | <1 MB | exports/, obsidian-vault/ |

### 容量预估 (70G 系统盘)

| 论文数量 | OSS 用量 | 本地用量 | 本地剩余 |
|---------|---------|---------|---------|
| 100 篇 | ~1.5 GB | ~10 GB | 47 GB |
| 1,000 篇 | ~15 GB | ~12 GB | 45 GB |
| 5,000 篇 | ~75 GB | ~15 GB | 42 GB |
| 10,000 篇 | ~150 GB | ~18 GB | 39 GB |

### OSS 配置要点

- **Internal endpoint**: 同区域自动使用 `oss-cn-shanghai-internal.aliyuncs.com`，免流量费
- **权限**: 最小权限即可 — PUT + GET + HEAD (ListObject/DeleteObject 非必需)
- **缓存**: PDF 解析时临时下载到 `/tmp/rf_oss_cache/`
- **Public URL**: `https://{bucket}.oss-cn-shanghai.aliyuncs.com/{key}`

---

## 7. 外部 API 可达性

### 全量 API 清单 (38 个端点，11 个服务)

| API | 基础 URL | 用途 | 认证 | 直连 | 速率限制 |
|-----|---------|------|------|------|---------|
| **Semantic Scholar** | api.semanticscholar.org | 元数据、引用、推荐 | API Key (x-api-key) | ✅ | 1 req/s (有 key) |
| **arXiv API** | export.arxiv.org/api | 元数据、分类 | 无需 | ✅ | 3 req/s |
| **arXiv PDF** | arxiv.org/pdf | 下载 PDF | 无需 | ✅ | 无明确限制 |
| **Crossref** | api.crossref.org | DOI、作者、会议 | User-Agent (polite) | ✅ | 50 req/s (polite) |
| **OpenAlex** | api.openalex.org | 会议、引用、类型 | User-Agent | ✅ | 10 req/s |
| **GitHub** | api.github.com | 代码仓库、README | 无需 (有限) | ✅ | 10 req/min (无 key) |
| **HuggingFace** | huggingface.co/api | 模型/数据集发现 | 无需 | ✅ | 无明确限制 |
| **DBLP** | dblp.org/search | 会议元数据 | 无需 | ✅ | 无明确限制 |
| **OpenReview** | api2.openreview.net | 审稿/接收状态 | Bearer token (login) | ✅ | 无明确限制 |
| ~~GROBID~~ | ~~grobid:8070~~ | ~~PDF 解析~~ | - | ❌ 已移除 | 改用 VLM page scan |
| **LLM (Kimi K2.6)** | api.kimi.com/coding/v1 | Agent/分析/VLM | API Key + User-Agent | ✅ | 按 plan |

### 不需要 IP 池/分流/隔离的原因

1. **请求量低**: 单篇论文入库触发 ~10 个 API 调用，每日新增 30-50 篇 ≈ 300-500 次/天
2. **已有限速保护**: discovery_service 有 429 重试 + sleep 机制
3. **API Key 提权**: S2 有 key 后从 100 req/5min 提升到 1 req/s
4. **无并发爆发**: Worker 串行处理 (max_jobs=2)，天然限流
5. **优雅降级**: 每个 enrichment 步骤独立 try/except，单 API 失败不阻塞管线

> **结论**: 当前规模下不需要 IP 池或分流。如果日处理量超过 500 篇/天，优先考虑:
> - GitHub: 申请 Personal Access Token (5000 req/hour)
> - S2: 已有 API Key，足够
> - OpenAlex: 注册 polite pool email (更高速率)

---

## 8. 元数据获取完整指南

### 8.1 提取架构总览

```
┌─────────────────────────────────────────────────────┐
│  L0: arXiv TeX 源码 (最高保真，需要 arxiv_id)        │
│  ├─ 精确 LaTeX 公式 (零 OCR 误差)                   │
│  ├─ \cite{} 引用 bibkey + 上下文                     │
│  ├─ \includegraphics 图 + caption                   │
│  └─ \url{} / \href{} GitHub/项目链接                 │
│                    ↓ fallback                        │
│  L1: PyMuPDF (确定性，CPU，始终可用)                  │
│  ├─ 全文 + section 切分 (含子 section 层级)          │
│  ├─ 公式 regex (fallback)                           │
│  ├─ 图表 caption regex                              │
│  ├─ 图片区域检测 + 嵌入图提取                        │
│  ├─ 行内引用上下文 (±120 chars)                     │
│  └─ Dataset 检测 (已知名 + URL 模式 + 发布动词)      │
│                    ↓ fallback                        │
│  L2: VLM + 外部 API (Kimi K2.6 + S2/Crossref/…)     │
│  ├─ 公式 VLM 页扫描 (按数学密度选页 → Kimi Vision)  │
│  ├─ 图表 VLM 分类 + 遗漏恢复 → 高清裁切 → OSS      │
│  ├─ 表格内容 VLM OCR → Markdown table               │
│  ├─ 引用/作者: S2 API (DOI / title search fallback) │
│  └─ GitHub + HuggingFace: 搜索 + README 解析        │
└─────────────────────────────────────────────────────┘
```

### 8.2 各元数据获取方案 & 准确率

| 元数据 | 主方案 | fallback | 预估准确率 | 限速 | 成本/paper |
|--------|--------|----------|-----------|------|-----------|
| **title / abstract** | arXiv API | S2 > OpenAlex | ~95% | 2 req/s | 免费 |
| **authors / 机构** | arXiv + S2 API | Crossref | ~90% | 1 req/s (S2) | 免费 |
| **venue / year** | OpenReview > Crossref > OpenAlex | arXiv comment 解析 | ~85% | 各 API | 免费 |
| **citation_count** | S2 API | OpenAlex | ~90% | 1 req/s | 免费 |
| **section 文字** | PyMuPDF regex (27 模式) | — | ~85% 含子 section | 无 | 免费 |
| **子 section 层级** | PyMuPDF heading regex | — | ~80% | 无 | 免费 |
| **公式** | arXiv TeX > VLM page scan > regex | — | 90% / 80% / 40% | Kimi RPM | ~¥0.15 |
| **图表 caption** | PyMuPDF regex + VLM 分类 | — | ~80% | Kimi RPM | ~¥0.08 |
| **图表图片** | PyMuPDF 区域检测 + VLM 遗漏恢复 | xref fallback | ~80% | Kimi RPM | ~¥0.08 |
| **表格内容** | VLM OCR → Markdown → 结构化 | — | ~75% | Kimi RPM | ~¥0.2 |
| **引用文献列表** | S2 API (100 条) | title search | ~85% | 1 req/s | 免费 |
| **行内引用上下文** | PDF regex [N] ±120 chars | TeX \cite{} | ~80% | 无 | 免费 |
| **GitHub 开源链接** | GitHub search + README 匹配 | — | ~75% | 10 req/min (无 token) | 免费 |
| **是否发布数据集** | PDF 文本检测 (120 已知名 + URL) | GitHub + HuggingFace | ~70% | 无 + API | 免费 |
| **acceptance_status** | OpenReview > arXiv comment | GitHub README > VLM 判断 | ~80% | 各 API | ≤¥0.1 |

### 8.3 API 限速配置

系统使用 `TokenBucketLimiter` 令牌桶实现统一限速 (`backend/utils/rate_limiter.py`)。

| API | rate (req/s) | burst | 依据 | 配置项 |
|-----|-------------|-------|------|--------|
| arXiv | 2.0 | 3 | 官方 3 req/s 上限 | 内置 |
| Semantic Scholar | 0.9 (有 key) / 0.3 (无 key) | 3 | 有 key 1 req/s | `S2_API_KEY` |
| GitHub | 1.0 (有 token) / 0.15 (无 token) | 3/2 | 无 token 10 req/min | `GITHUB_TOKEN` |
| Crossref | 5.0 | 10 | polite pool 50/s | 内置 User-Agent |
| OpenAlex | 5.0 | 10 | 10 req/s | 内置 |
| HuggingFace | 1.5 | 3 | 100 req/min | 内置 |
| Kimi VLM | 0.5 | 2 | 按套餐 | 内置 |

**环境变量:**
```bash
S2_API_KEY=your_key_here           # 推荐: 1 req/s vs 无 key 100/5min
GITHUB_TOKEN=ghp_xxxxxxxxxxxx     # 推荐: 5000 req/hr vs 无 token 10 req/min
```

### 8.4 VLM (Kimi K2.6) 配置

```bash
OPENAI_API_KEY=sk-kimi-xxx        # Kimi API key
OPENAI_BASE_URL=https://api.kimi.com/coding/v1
OPENAI_MODEL=kimi-k2.6
```

**max_tokens 分级** (config.py，可通过环境变量覆盖):

| 级别 | 值 | 用途 | 环境变量 |
|------|-----|------|---------|
| heavy | 16384 | 公式页扫描 (多页多公式)、大表格 OCR | `VLM_MAX_TOKENS_HEAVY` |
| medium | 8192 | 图表分类 + 遗漏恢复 | `VLM_MAX_TOKENS_MEDIUM` |
| light | 4096 | 单图描述、单公式 OCR | `VLM_MAX_TOKENS_LIGHT` |
| tiny | 2048 | acceptance 判断等短回复 | `VLM_MAX_TOKENS_TINY` |

**为什么必须开大 max_tokens:**
Kimi K2.6 在 max_tokens 不足时会**直接截断 JSON 输出**，导致下游 JSON parse 失败。
公式页扫描一次处理 3 页、每页可能有 5+ 公式，JSON array 很长，4000 tokens 远远不够。

### 8.5 代理配置 (大陆服务器)

**需要代理 (延迟或被墙):**
- `arxiv.org` — 直连 10KB/s，代理 0.8s/PDF
- `huggingface.co` — 直连被墙
- `github.com` / `api.github.com` — 直连不稳定
- `api.kimi.com` — 需要代理

**直连更快:**
- `api.semanticscholar.org` — 直连 400ms
- `api.openalex.org` — 直连 1700ms (代理 2700ms)
- `dblp.org` — 直连稳定
- `api2.openreview.net` — 直连稳定

**mihomo 规则示例:**
```yaml
rules:
  - DOMAIN-SUFFIX,arxiv.org,Proxy
  - DOMAIN-KEYWORD,huggingface,Proxy
  - DOMAIN-KEYWORD,github,Proxy
  - DOMAIN-KEYWORD,kimi,Proxy
  - MATCH,DIRECT
```

**NO_PROXY:**
```
localhost,postgres,redis,127.0.0.1,api.semanticscholar.org,api.openalex.org,dblp.org
```

### 8.6 非 arXiv 论文处理

当论文没有 arxiv_id 时，系统使用 fallback 链：

| 能力 | 有 arxiv_id | 无 arxiv_id |
|------|------------|-------------|
| TeX 公式提取 | 精确 LaTeX | ❌ 不可用，退化到 VLM/regex |
| S2 引用列表 | `ARXIV:{id}` 直查 | DOI → OpenReview URL → title search |
| S2 作者信息 | 同上 | 同上 |
| PDF 下载 | `arxiv.org/pdf/{id}` | 需要 paper_link 或手动上传 |
| 其他元数据 | 正常 | Crossref + OpenAlex + S2 title search |

**fallback 链:** `S2(ARXIV:id)` → `S2(DOI:doi)` → `S2(URL:openreview)` → `S2 title search` → Crossref → OpenAlex

### 8.7 批量处理建议

- Worker `max_jobs=2` 串行，各步骤间自动限速
- **安全阈值:** ~100 篇/天 (含所有 API 调用)
- **成本估算:** ~¥15-35/100 篇 (主要是 VLM 公式+图表+表格)

| 步骤 | 耗时/篇 | API 调用数 | 可并行 |
|------|---------|-----------|--------|
| PDF 下载 | 0.8-110s | 1 (arXiv) | ✅ |
| Enrich (10 API) | 6s | 5-10 | ✅ (按 paper) |
| L2 Parse | 30-60s | 2-3 (VLM) | ✅ |
| L3 Skim | 20s | 1 (LLM) | ✅ |
| L4 Deep | 40s | 6-12 (LLM) | ❌ |
| **串行总计** | **~4-6 min** | **15-30** | |

**arXiv OAI-PMH 批量元数据同步:**

对于全顶会知识库建设 (CVPR/ICLR/KDD/ACL 等数千篇)，不要逐篇调 arXiv API。
使用 OAI-PMH (`https://oaipmh.arxiv.org/oai`) 按 category + date 批量拉取元数据:
- 限速: 3 秒/请求 (全局)
- 支持 resumptionToken 断点续爬
- 每次返回 1000 条记录
- 配合 venue matcher (Crossref/DBLP/OpenReview) 筛选顶会论文

### ~~GROBID 增强~~ (已移除，保留备查)

> GROBID 容器已于 2026-04-20 移除。VLM page scan + S2 API 替代。
> 长 PDF 频繁 OOM 且占 2-3 GB 内存。

---

## 9. Alembic 迁移

```bash
# 查看当前版本
docker compose exec api bash -c 'PYTHONPATH=/app alembic -c alembic/alembic.ini current'

# 应用所有迁移 (当前链: 001..015 → 005_v2 → 016)
docker compose exec api bash -c 'PYTHONPATH=/app alembic -c alembic/alembic.ini upgrade head'

# 回滚一步
docker compose exec api bash -c 'PYTHONPATH=/app alembic -c alembic/alembic.ini downgrade -1'
```

**当前版本**: v016 (72 张表 + 物化视图)

**迁移链注意事项**:
- `005_v2` 的 `down_revision` 是 `015` (v2 refactor tables)
- `016` 的 `down_revision` 是 `005_v2` (V6 candidate queue + KB profiles)
- 不是简单的 001→002→...→016 线性链

---

## 10. 网络与代理

### 直连可用 (阿里云上海)

arXiv、DBLP、GitHub、OpenAlex、Crossref、HuggingFace、Semantic Scholar — 均可直连。

### 代理配置 (mihomo，用于 Docker Hub 拉镜像)

```yaml
# /opt/clash/config.yaml
allow-lan: true
bind-address: "*"
mixed-port: 7890
```

容器内代理 (`.env`):
```bash
HTTP_PROXY=http://172.17.0.1:7890
HTTPS_PROXY=http://172.17.0.1:7890
# ⚠️ 重要: 直连可用的 API 必须加入 NO_PROXY，否则走代理绕路更慢
# ⚠️ 注意: huggingface.co 不能加 NO_PROXY（直连被墙，必须走代理）
NO_PROXY=localhost,postgres,redis,127.0.0.1,grobid,arxiv.org,export.arxiv.org,api.semanticscholar.org,api.crossref.org,api.openalex.org,api.github.com,raw.githubusercontent.com,dblp.org
```

### 代理加速实测 (关键！)

arXiv PDF 下载必须走代理才快：

| 方式 | arXiv PDF 1.5MB | 原因 |
|------|-----------------|------|
| 直连 (NO_PROXY) | **150s** (10 KB/s) | 阿里云→arXiv CDN 带宽极低 |
| **走代理 (mihomo)** | **0.8s** (2 MB/s) | 代理节点→arXiv 快 200 倍 |

**mihomo 配置要点**: 必须在 `config.yaml` 的 rules 中添加 `DOMAIN-SUFFIX,arxiv.org,Proxy`，否则 arXiv 走 `MATCH,DIRECT` 直连。

httpx 0.28.x 默认 `trust_env=True`，自动读取环境变量。`NO_PROXY` 中的域名绕过代理直连。

### 各 API 最优路径 (实测)

| API | proxy (ms) | direct (ms) | 应走 | NO_PROXY |
|-----|-----------|-------------|------|----------|
| arXiv API/PDF | **600** | 9700 | **代理** | 不加 |
| S2 | - | **400** | **直连** | ✅ 加 |
| Crossref | **700** | 800 | 代理 | 可选 |
| OpenAlex | 2700 | **1700** | **直连** | ✅ 加 |
| GitHub | **300** | 300 | 均可 | 不加 |
| HuggingFace | **400** | FAIL | **代理** | 不加 |
| DBLP | 700 | 700 | 均可 | ✅ 加 |

### mihomo 规则 (/opt/clash/config.yaml)

```yaml
rules:
- DOMAIN-SUFFIX,arxiv.org,Proxy        # 必须！直连极慢
- DOMAIN-SUFFIX,docker.io,Proxy
- DOMAIN-SUFFIX,openreview.net,Proxy
- DOMAIN-KEYWORD,huggingface,Proxy     # 必须！直连被墙
- DOMAIN-KEYWORD,github,Proxy
- DOMAIN-KEYWORD,openai,Proxy
- DOMAIN-KEYWORD,anthropic,Proxy
- MATCH,DIRECT
```

### NO_PROXY 配置

```bash
# 这些 API 直连更快或代理无意义
NO_PROXY=localhost,postgres,redis,127.0.0.1,grobid,api.semanticscholar.org,api.crossref.org,api.openalex.org,dblp.org
```

---

## 11. 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| ~~GROBID 循环重启~~ | ~~内存不足~~ | 已移除 GROBID，改用 VLM page scan |
| ~~GROBID 解析返回空~~ | ~~retry 代码缩进 bug~~ | 已移除 GROBID |
| `docker compose restart` 后 .env 未生效 | restart 不重新读取 .env | 用 `up -d --force-recreate` |
| Alembic `Multiple head revisions` | 分叉 migration | 检查 `alembic heads`，确保单一链 |
| Alembic `No module named backend` | PYTHONPATH 未设置 | 加 `PYTHONPATH=/app` |
| Alembic `DuplicateTable` | migration 已手动执行过 | `alembic stamp <version>` 标记 |
| Worker 只有 20 个函数 | 代码未挂载到 worker 容器 | 检查 docker-compose volumes |
| arq `month_day` TypeError | arq.cron 不支持月日调度 | 改为 weekday 调度 |
| S2 API 429 | 未配置 API Key | 设置 S2_API_KEY |
| OpenAlex DOI 404 | arXiv DOI 格式不被直接支持 | 代码已 fallback 到 title search |
| OpenReview 403 | 未认证 | 配置 OPENREVIEW_USERNAME/PASSWORD |
| OSS `AccessDenied` on delete | Key 无 DeleteObject 权限 | 不影响正常使用，delete 会静默失败 |
| ~~API 容器等 GROBID 3 分钟~~ | ~~依赖 service_healthy~~ | 已移除 GROBID 依赖 |

---

## 12. 运维命令

```bash
# 查看所有容器状态
docker compose ps

# 查看日志
docker compose logs -f api --tail=50
docker compose logs -f worker --tail=50

# 重启服务 (代码变更后)
docker compose restart api        # API 有 --reload 会自动检测文件变更
docker compose restart worker mcp # 需手动重启

# 清理磁盘 (回收 Docker build cache)
docker builder prune -f
docker image prune -f

# 进入容器
docker compose exec api bash
docker compose exec api python3

# 手动跑 pipeline
docker compose exec api python3 -c "
import asyncio
from backend.services.pipeline_service import run_full_pipeline
from backend.database import async_session_factory
async def run():
    async with async_session_factory() as session:
        result = await run_full_pipeline(session, 'paper-uuid')
        print(result)
asyncio.run(run())
"

# 数据库备份
docker compose exec postgres pg_dump -U rf researchflow > backup_$(date +%Y%m%d).sql
```

---

## 13. Pipeline 全链路验证

> 测试日期: 2026-04-19 | 测试论文: FlowMDM (CVPR 2024, arXiv:2402.15509)

### 单篇论文处理耗时

| 步骤 | 耗时 | 质量 | 可并行 | 限流风险 |
|------|------|------|--------|---------|
| 1. Import | <1s | 10/10 | - | 无 |
| 2. PDF Download | **110s** | 6/10 | 可与 3 并行 | 无 (带宽瓶颈) |
| 3. Enrich (6 个 API) | **6s** | 8/10 | 可与 2 并行 | S2 1req/s |
| 4. Parse (PyMuPDF+VLM) | **30-60s** | 9/10 | 依赖步骤 2 | VLM ~$0.03 |
| 5. Figure Extraction | ~30s | 9/10 | 依赖步骤 4 | 无 |
| 6. S2 Discovery | ~8s | 8/10 | 可与 4-5 并行 | S2 1req/s |
| 7. L3 Skim (LLM) | ~20s | 9/10 | 依赖步骤 4 | LLM API |
| 8. L4 Deep (LLM) | ~40s | 9/10 | 依赖步骤 7 | LLM API |
| **总计 (串行)** | **~350s (~6分钟)** | | | |
| **总计 (优化并行)** | **~250s (~4分钟)** | | 2+3, 5+6 | |

### Enrich 元数据获取明细

| 字段 | 值 | 来源 | 质量 |
|------|-----|------|------|
| title | Seamless Human Motion Composition... | arXiv | ✅ |
| abstract | 完整 | arXiv | ✅ |
| authors | 3 人 (含 affiliation via S2) | arXiv + S2 | ✅ |
| year | 2024 | arXiv | ✅ |
| venue | (空) | - | ❌ 见下方分析 |
| doi | 10.1109/cvpr52733.2024.00051 | OpenAlex | ✅ |
| cited_by_count | 24 (OpenAlex) / 71 (S2) | OpenAlex, S2 | ⚠️ 不一致 |
| code_url | None | GitHub | ⚠️ 未找到 |
| formulas | 4+ 个 LaTeX | VLM page scan | ✅ |
| figures | 多个 (OSS public_url) | PyMuPDF | ✅ |
| structured refs | 完整 | S2 API | ✅ |

### 已知质量问题

| 问题 | 原因 | 优先级 |
|------|------|--------|
| venue 缺失 | OpenAlex/S2 未返回，arXiv comment 未解析 | P1 |
| cited_by_count 不一致 | S2 值 (71) vs OpenAlex (24)，应优先用 S2 | P1 |
| VLM 公式 OCR 失败 | `response` 变量未定义 bug | P0 |
| code_url 未匹配 | GitHub search 未索引该仓库 | P2 |
| OpenReview 搜索降级 | `x-search-degraded: true`，CVPR 不在 OpenReview | P2 |
| S2 recommendations 404 | 非所有论文都有推荐数据 | P3 |

### 对比: Chain-of-Thought 论文 (之前处理)

完整 L4 Deep 分析后的数据:

| 字段 | 质量 |
|------|------|
| title, abstract, authors (9人), year, venue (NeurIPS) | ✅ 全部获取 |
| figures (5个 OSS), formulas (13个) | ✅ |
| L3 skim (问题/方法/证据摘要) | ✅ |
| L4 deep (核心洞察 + 证据链 + 完整报告) | ✅ |
| cited_by_count: 217 (实际 ~23,000) | ⚠️ 值偏小 |

---

## 14. 部署问题与修复记录

| # | 问题 | 原因 | 修复方案 |
|---|------|------|---------|
| 1 | Alembic 多头冲突 | 005_v2 和 016 都从 015 分叉 | 016.down_revision 改为 005_v2 |
| 2 | Alembic DuplicateTable | 005_v2 表已手动创建 | `alembic stamp 005_v2` 标记 |
| 3 | arq `month_day` TypeError | arq.cron 不支持月日调度 | 改为 weekday 调度 |
| 4 | Worker 只有 20 个函数 | 代码未挂载到 worker 容器 | 添加 `./backend:/app/backend` volume |
| 5 | `.env` 不生效 | `restart` 不重读 .env | 改用 `up -d --force-recreate` |
| 6 | GROBID OOM 循环重启 | 2048MB 不够 | 调到 3072MB |
| 7 | API 启动等 GROBID 3 分钟 | depends_on: service_healthy | 改为 service_started |
| 8 | OpenAlex arXiv DOI 404 | filter 格式错误 | 修复 filter 查询 |
| 9 | S2 API 429 | 无 API Key | 4 个文件加 x-api-key header |
| 10 | PDF 下载超时 | arXiv 带宽低 (~96 KB/s) | NO_PROXY 加直连域名 |
| 11 | VLM 公式 OCR 失败 | `response` 变量未定义 | 待修复 |
| 12 | OpenReview 搜索降级 | 服务端 x-search-degraded | API 限制，非代码问题 |
| 13 | cited_by_count 不一致 | S2 和 OpenAlex 返回不同值 | 应优先用 S2 值 |
| 14 | venue 缺失 | arXiv comment 未解析 | 需增加 comment→venue 提取 |
| 15 | GROBID 从 compose 管理后 OOM | 之前独立运行 3GB，compose 内 2GB | compose 内调到 3GB |

### 修复记录 (2026-04-19)

| # | 问题 | 状态 | 修复内容 |
|---|------|------|---------|
| P0 | VLM 公式 OCR `response` 未定义 | ✅ 已修复 | `formula_extraction_service.py`: 改用 `in_tokens`/`out_tokens` 局部变量 |
| P0 | PDF 下载超时 60s | ✅ 已修复 | `pipeline_service.py`: timeout 60→180s |
| P1 | venue 未从 arXiv comment 回写 | ✅ 已修复 | `enrich_service.py`: acceptance venue + S2 venue 写入 paper.venue |
| P1 | cited_by_count S2 值未覆盖 | ✅ 已修复 | `enrich_service.py`: S2 count > 旧值时覆盖 |
| P1 | enrich S2 报 `settings` 未定义 | ✅ 已修复 | `enrich_service.py`: 添加 `from backend.config import settings` |
| P1 | enrich client timeout=5s (默认) | ✅ 已修复 | `pipeline_service.py` L139: 5s→60s; `discovery_service.py`: 加 timeout=30 |
| P1 | HuggingFace 直连被墙 | ✅ 已修复 | 从 NO_PROXY 移除 `huggingface.co`（必须走代理） |
| P0 | GROBID retry 缩进 bug 导致永远返回空 | ✅ 已修复 | `grobid_client.py`: 修复缩进，成功路径不再是死代码 |
| P1 | GROBID GC 暂停 | ✅ 已修复 | JAVA_OPTS 加 G1GC + retry 机制 |
| P1 | GROBID 失败时无引用/作者 | ✅ 已修复 | `parse_service.py`: 加 S2 API fallback 自动补全 |
| P0 | arXiv PDF 直连 150s | ✅ 已修复 | mihomo rules 加 `arxiv.org→Proxy`，0.8s (提速 187 倍) |
| P1 | mihomo 双进程冲突 | ✅ 已修复 | 杀旧进程 + systemctl restart |
| P2 | GitHub code_url 搜索不到 | ⚠️ 部分修复 | 加了 arXiv ID fallback 搜索，但 GitHub 索引覆盖有限 |
| P2 | OpenReview 搜索降级 | ⏳ 待观察 | API 端问题，非代码问题 |

### 修复记录 (2026-04-20 — 自主运行改造)

| # | 问题 | 状态 | 修复内容 |
|---|------|------|---------|
| P0 | GROBID 长 PDF OOM 占 2-3GB | ✅ 已移除 | 改用 VLM page scan (PyMuPDF+Claude Vision) |
| P0 | API 容器 pipeline/run 同步执行 OOM | ✅ 已修复 | 改为 worker 异步执行 (`task_pipeline_run`) |
| P0 | 代理 API 模型名不匹配 (`claude-sonnet-4-20250514`) | ✅ 已修复 | 改为 `so-4.6` (代理短名) |
| P0 | 代理 API 返回自我介绍而非 completion | ✅ 已修复 | `_is_garbage_response` 检测 + 自动重试 |
| P0 | L4 报告为空 (JSON 截断) | ✅ 大幅改善 | 暴力 JSON repair + `rfind` 阈值 ≥3 keys |
| P0 | PyMuPDF 文本含 `\x00` → PG 拒绝 | ✅ 已修复 | `_clean_text()` 过滤 null byte |
| P0 | `prompt_version` varchar(20) 溢出 | ✅ 已修复 | ALTER TABLE → varchar(50) |
| P1 | `not data.get(f)` 误判空列表 [] | ✅ 已修复 | 改为 `data.get(f) is None` |
| P1 | pipeline/batch 500 错误 (greenlet) | ✅ 已修复 | 改为 arq enqueue，不在 API 进程跑 |
| P1 | 失败论文无人恢复 | ✅ 已修复 | `task_pipeline_recover` cron 每 30 分钟扫描 |
| P1 | VLM figure classify 400 (payload 太大) | ✅ 已修复 | 限制 candidates=15, pages=5 |
| P1 | venue_resolve 结果不写入 paper.venue | ✅ 已修复 | accepted 时覆盖 venue |
| P1 | paper_facets 唯一键冲突 | ✅ 已修复 | DELETE 旧 auto-facets + seen_node_roles 去重 |

---

## 15. 端到端冷启动验证 (2026-04-20)

### 测试设计

- **领域**: Video Question Answering (2025-2026)
- **论文**: 5 篇 arXiv 论文，通过 `/api/v1/import/links` 导入
- **操作**: 一次 `enrich` + 一次 `pipeline/batch` 触发，**零人工干预**
- **环境**: 阿里云 ECS 4C8G，Kimi K2.6 API

### 结果

| 指标 | 结果 |
|------|------|
| 导入到 l4_deep | **5/5 (100%)** |
| 自动发现引用论文 | **90 篇** (自动 ingest 到 wait) |
| L2 Parse 公式提取 | 平均 **11.8 个/篇**，LaTeX 质量高 |
| L3 Skim problem_summary | **4/5 (80%)** |
| L4 Deep 完整报告 | **3/5 (60%)** |
| Venue 检测 | CVPR 2026 ✅ (OpenReview) |
| Worker 自主处理 | ✅ 2 job 并行，每篇 ~6 分钟 |
| 自动恢复 cron | ✅ `task_pipeline_recover` 已触发 |

### 自主运行操作方式

```bash
# 1. 导入论文
curl -X POST /api/v1/import/links -d '{"items": [{"url": "https://arxiv.org/abs/..."}], "default_category": "VideoQA"}'

# 2. 补全元数据 (可选，pipeline/run 也会自动 enrich)
curl -X POST /api/v1/papers/enrich

# 3. 一键触发全部处理 (异步返回，worker 后台执行)
curl -X POST /api/v1/pipeline/batch?limit=50

# 4. 无需干预 — worker 逐篇处理，cron 每 30 分钟自动恢复卡住的论文
```

### 已知限制

| 限制 | 原因 | 影响 |
|------|------|------|
| L4 报告偶尔为空 | API 截断 JSON (5000-7000 chars) | 自动重试 + JSON 修复 |
| arXiv API 429 限流 | 批量 enrich 时过快 | 部分论文 title/authors 延迟填充 |

### LLM 配置

```env
OPENAI_API_KEY=sk-kimi-xxx
OPENAI_BASE_URL=https://api.kimi.com/coding/v1
OPENAI_MODEL=kimi-k2.6
```

| 代理模型名 | 对应模型 | 用途 |
|-----------|---------|------|
| `so-4.6` | Claude Sonnet 4.6 | LLM 分析 + VLM 公式/图片 |
| `op-4.6` | Claude Opus 4.6 | 复杂任务 (公式推导等) |

> **不要使用 `claude-sonnet-4-20250514` 等完整 ID** — 代理可能路由到错误后端导致截断或垃圾响应。

---

## 16. 资源与成本

| 项目 | 费用 |
|------|------|
| 阿里云 ECS 4C8G | ~2000-3000 CNY/年 |
| 域名 (.xyz) | ~25 CNY/年 |
| LLM API | ~$5-20/月 (按分析量) |
| OSS 存储 | ~0.12 元/GB/月 (标准型) |
| OSS 流量 | 同区域 internal 免费 |
