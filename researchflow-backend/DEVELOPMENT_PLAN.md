# ResearchFlow 开发方案

> 总工期约 36 周，分 8 个阶段。每阶段有明确验收标准。
> Phase 0 已完成。

---

## Phase 0: 基础骨架 [已完成]

**产出**：
- [x] Docker Compose (Postgres+pgvector, Redis, API, Worker)
- [x] 14 张表 ORM 模型 + Alembic 001 迁移
- [x] 迁移脚本 (CSV→DB, MD→DB, validate)
- [x] 导出兼容脚本 (DB→Markdown, DB→CSV)
- [x] FastAPI 入口 + config + database
- [x] 工具模块 (frontmatter parser, sanitize)
- [x] Alembic 002: 库外输入状态机 (ephemeral_received/canonicalized/enriched/archived_or_expired) + 置信度分级 (EvidenceBasis enum, confidence_notes JSONB, per-claim confidence+basis+source_page)

---

## Phase 1: API 核心 + Ingestion [第 3-5 周]

### 目标
FastAPI 提供论文 CRUD，导入流程可用，自动补全+评分。

### 任务分解

| # | 任务 | 文件 | 天数 |
|---|------|------|------|
| 1.1 | 新增 5 张表 ORM + Alembic 003 (repo_analyses, report_cache, user_bookmarks, user_events, direction_cards) | `models/repo.py`, `models/report.py`, `models/user_action.py`, `alembic/003` | 2 |
| 1.2 | Pydantic schemas (paper, analysis, import) | `schemas/paper.py`, `schemas/import_.py` | 2 |
| 1.3 | Papers CRUD router + 多维过滤 | `api/papers.py`, `services/paper_service.py` | 3 |
| 1.4 | Import endpoints (links/pdfs/awesome/repo/zotero/batch) | `api/import_.py`, `services/import_service.py` | 4 |
| 1.5 | Enrich worker (Crossref/arXiv/Semantic Scholar 补全) | `workers/enrich_worker.py`, `services/enrich_service.py` | 3 |
| 1.6 | Triage service (4 维评分) | `services/triage_service.py`, `workers/triage_worker.py` | 3 |
| 1.7 | arq worker 配置 | `workers/arq_app.py` | 1 |
| 1.8 | 库外输入 ingestion 流 (ephemeral→canonicalized→enriched) | `services/ingestion_service.py` | 3 |
| 1.9 | 临时对象过期清理 worker | `workers/cleanup_worker.py` | 1 |
| 1.10 | API 测试 | `tests/test_papers.py`, `tests/test_import.py` | 2 |

### 验收标准
- `POST /api/v1/import/links` 导入 3 篇论文 → 自动补全元数据 → `GET /api/v1/papers` 返回含 4 维评分的结果
- `GET /api/v1/papers?category=Human_Object_Interaction&year=2025` 返回正确过滤结果
- enrich worker 成功从 Crossref 补全 abstract/authors/doi
- 库外 PDF 上传 → state 为 `ephemeral_received` → 自动走到 `enriched` → 30 天后自动归档
- 用户点"加入知识库"→ state 变为 `wait`，`is_ephemeral` 变为 false

---

## Phase 2: 对象存储 + PDF 管线 [第 6-8 周]

### 目标
PDF 进对象存储，L1/L2 管线可用，资产自动补全。

### 任务分解

| # | 任务 | 文件 | 天数 |
|---|------|------|------|
| 2.1 | COS/OSS 抽象层 | `services/object_storage.py` | 3 |
| 2.2 | PDF 上传 API + 迁移脚本 | `api/import_.py` 扩展, `migration/migrate_pdfs_to_cos.py` | 2 |
| 2.3 | Parse worker (L2): pymupdf 章节提取 | `workers/parse_worker.py`, `utils/pdf_extract.py` | 4 |
| 2.4 | Asset refresh worker (查找 repo/project page/data) | `workers/asset_refresh_worker.py` | 3 |
| 2.5 | Asset refresh 定时任务 (arq cron) | `workers/arq_app.py` 扩展 | 1 |
| 2.6 | paper_assets CRUD | `api/papers.py` 扩展 | 1 |
| 2.7 | 测试 | `tests/test_object_storage.py`, `tests/test_parse.py` | 2 |

### 验收标准
- 上传 PDF → 存入 COS → `paper_assets` 有记录
- L1: Crossref 补全 abstract/authors
- L2: pymupdf 提取出 intro/method/experiments/conclusion 段落
- state 从 `downloaded` → `l2_parsed`
- 资产补全 worker 找到至少 1 个 repo/project page

---

## Phase 3: LLM 分析管线 + 报告生成 [第 9-13 周]

### 目标
L3 skim + L4 deep 端到端可用。汇报报告 (30s/5min/deep) 可生成。

### 任务分解

| # | 任务 | 文件 | 天数 |
|---|------|------|------|
| 3.1 | LLM service (Agent SDK 封装 + model_runs 追踪) | `services/llm_service.py` | 3 |
| 3.2 | Skim worker (L3): 轻量卡片 + changed_slots | `workers/skim_worker.py` | 4 |
| 3.3 | Deep report worker (L4): 全文分析 | `workers/deep_report_worker.py` | 5 |
| 3.4 | Method delta 生成 (canonical delta card) | `services/delta_service.py` | 3 |
| 3.5 | Evidence units 提取 | `services/evidence_service.py` | 3 |
| 3.6 | Paradigm templates 种子数据 | `migration/seed_paradigms.py` | 1 |
| 3.7 | Report generate worker (30s/5min/deep 三级) | `workers/report_generate_worker.py`, `services/report_service.py` | 5 |
| 3.8 | Report API endpoints | `api/reports.py` | 2 |
| 3.9 | 重分析触发器 (模型升级/taxonomy 变化/用户纠错) | `services/reanalyze_service.py` | 2 |
| 3.10 | 置信度分级: L3/L4 输出 confidence_notes + evidence_units 带 basis | `workers/skim_worker.py`, `workers/deep_report_worker.py` 扩展 | 2 |
| 3.11 | 三角色复核管线 (extractor→auditor→taxonomy_reviewer) | `workers/audit_worker.py`, `services/audit_service.py` | 3 |
| 3.12 | 测试 | `tests/test_analysis.py`, `tests/test_reports.py`, `tests/test_quality.py` | 3 |

### 验收标准
- 排队 L3 → 生成 skim card + delta card，state → `l3_skimmed`
- 晋升 L4 → 完整 report，质量匹配现有 20 篇手动分析
- evidence_units 从 L4 报告中提取出 3+ 个证据原子，**每个带 confidence + basis**
- L4 报告含 `confidence_notes`，区分 `code_verified`/`experiment_backed`/`text_stated`/`inferred`/`speculative`
- 高价值论文 (importance=S) 自动触发 auditor + taxonomy_reviewer 复核
- 导入 10 篇论文列表 → 自动 triage → L3 → 生成 5 分钟汇报版报告
- model_runs 记录每次 LLM 调用的 tokens + cost

---

## Phase 4: 语义搜索 + 阅读推荐 [第 14-16 周]

### 目标
向量搜索可用，分层阅读推荐可用。

### 任务分解

| # | 任务 | 文件 | 天数 |
|---|------|------|------|
| 4.1 | Embedding worker (batch) | `workers/embedding_worker.py` | 3 |
| 4.2 | pgvector HNSW 索引 + Alembic 003 | `alembic/003` | 1 |
| 4.3 | 混合搜索 service (tsvector + cosine + 结构化) | `services/search_service.py` | 4 |
| 4.4 | Search API | `api/papers.py` 扩展 | 2 |
| 4.5 | Reading planner (canonical→structural→follow-up→patch→negative) | `services/reading_planner.py` | 3 |
| 4.6 | Reading plan API | `api/reading_plans.py` | 1 |
| 4.7 | 搜索会话 + latent_need 重写 | `services/search_session_service.py` | 3 |
| 4.8 | 测试 | `tests/test_search.py` | 2 |

### 验收标准
- 搜索 "diffusion model for human-object interaction with physics" 返回相关论文 top-5
- 阅读计划输出 4 层推荐，顺序合理
- 搜索会话记录 latent_need 重写历史

---

## Phase 5: 总结系统 + MCP [第 17-20 周]

### 目标
日/周/月总结自动生成。MCP 接通 Claude Code 和 Codex。

### 任务分解

| # | 任务 | 文件 | 天数 |
|---|------|------|------|
| 5.1 | Digest generate worker (日/周/月模板) | `workers/digest_worker.py`, `services/digest_service.py` | 4 |
| 5.2 | Digest 定时任务 (arq cron: 23:00/周日/月末) | `workers/arq_app.py` 扩展 | 1 |
| 5.3 | Digest API | `api/digests.py` | 2 |
| 5.4 | MCP server (FastMCP, 10 工具) | `mcp/server.py`, `mcp/tools.py` | 5 |
| 5.5 | .mcp.json (Claude Code) | 项目根目录 `.mcp.json` | 1 |
| 5.6 | .codex/config.toml (Codex MCP 配置) | `.codex/config.toml` | 1 |
| 5.7 | 更新 .claude/skills 调用 MCP | `.claude/skills/*` | 3 |
| 5.8 | 测试 | `tests/test_digest.py`, `tests/test_mcp.py` | 3 |

### 验收标准
- 系统 23:00 自动生成日总结，含"今日新增/推荐深读/后台建议"
- 周日自动生成周总结
- Claude Code 中 `search_research_kb("diffusion motion generation")` 返回结果
- `get_paper_report(paper_ids, type="briefing")` 返回 5 分钟版报告
- `enqueue_analysis(paper_id, "l4_deep")` 创建 job

---

## Phase 6: 前端 [第 21-28 周]

### 目标
Web UI 可用，非技术用户可完成全流程。

### 任务分解

| # | 任务 | 页面 | 天数 |
|---|------|------|------|
| 6.1 | Next.js 项目初始化 + Tailwind + shadcn/ui | `frontend/` | 2 |
| 6.2 | API client + types | `frontend/src/lib/api.ts` | 2 |
| 6.3 | 公共组件 (PaperCard, SearchBar, FilterPanel, ScoreRadar) | `frontend/src/components/` | 4 |
| 6.4 | Dashboard 首页 | `frontend/src/app/page.tsx` | 3 |
| 6.5 | 导入中心 | `frontend/src/app/import/` | 4 |
| 6.6 | 论文库/检索页 | `frontend/src/app/papers/` | 5 |
| 6.7 | 论文详情页 (概览/报告/证据/delta card/历史) | `frontend/src/app/papers/[id]/` | 5 |
| 6.8 | 报告页 (快速/汇报/深度对比) | `frontend/src/app/reports/` | 4 |
| 6.9 | 总结与复盘页 | `frontend/src/app/digests/` | 3 |
| 6.10 | 方向推荐页 | `frontend/src/app/directions/` | 3 |
| 6.11 | 用户反馈 UI (收藏/批注/纠错) | 各页面内嵌 | 3 |
| 6.12 | Caddy 配置 + Docker 集成 | `caddy/Caddyfile`, `docker-compose.yml` | 1 |
| 6.13 | 响应式适配 (tablet/desktop) | 全局 | 2 |

### 验收标准
- 非技术用户在浏览器中完成：导入 awesome 列表 → 查看自动生成的汇报报告 → 看阅读推荐 → 查看日总结
- 论文库页面支持按任务/方法槽位/开源状态/结构性分数过滤
- 报告页面支持 3 种视图切换

---

## Phase 7: 高级功能 [第 29-36 周]

### 目标
Repo 深剖、方向推荐、反馈闭环、评测集。

### 任务分解

| # | 任务 | 天数 |
|---|------|------|
| 7.1 | Repo × Paper 联合分析 worker | 5 |
| 7.2 | Repo 分析页面 (流程图/公式映射/shape trace) | 5 |
| 7.3 | Direction propose worker + expand feasibility | 4 |
| 7.4 | 方向推荐页面 (卡片 + 可行性方案展开) | 3 |
| 7.5 | 用户行为事件采集 | 2 |
| 7.6 | 反馈 → 重分析触发闭环 | 3 |
| 7.7 | 评测集: 50 篇标杆 + 10 查询 + 10 重构场景 | 3 |
| 7.8 | 认证 (API key / JWT) | 2 |
| 7.9 | 2C4G 压力测试 + Postgres 调优 | 3 |
| 7.10 | Admin 审核台 (taxonomy/prompt 变更审批) | 3 |

### 验收标准
- 输入 GitHub repo URL → 生成 repo×paper 对齐报告，含 formula_code_mapping
- 输入子方向描述 → 生成 1-3 方向卡片 → 展开可行性方案
- 用户纠错 3 次 "这个判断不对" → 触发相关 taxonomy 重分析
- 2C4G 上 3 并发用户 + 2 后台任务不 OOM

---

## 里程碑总览

```
Week  1-2    Phase 0  ████████ 基础骨架 [已完成]
Week  3-5    Phase 1  ████████████ API + Ingestion
Week  6-8    Phase 2  ████████████ 对象存储 + PDF
Week  9-13   Phase 3  ████████████████████ LLM 分析 + 报告
Week 14-16   Phase 4  ████████████ 语义搜索 + 阅读推荐
Week 17-20   Phase 5  ████████████████ 总结 + MCP
Week 21-28   Phase 6  ████████████████████████████████ 前端
Week 29-36   Phase 7  ████████████████████████████████ 高级功能
```

---

## 技术风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 2C4G 内存不够 | 服务 OOM | MCP 合入 API 进程省 200MB；前端静态导出省 200MB |
| LLM token 成本超预算 | 分析频率受限 | 四级管线严控晋升；L1/L2 零 token；批量用小模型 |
| Repo 分析准确度低 | 用户体验差 | 必须带置信度标注，分"证据支持"和"推测性" |
| pgvector 在大规模下变慢 | 搜索延迟 | 5000 篇以下够用；超过后拆 Qdrant |
| 论文叙事欺骗 | 报告质量差 | 强制 canonical delta card 对照，不允许顺叙事走 |

---

## 依赖与前置条件

- 腾讯云 Lighthouse 2C4G 实例（已确认价格 ~1020 元/年）
- 腾讯云 COS 桶（或阿里云 OSS）
- 域名 + 备案（如需公网访问）
- Anthropic API key（token 费用另算）
- OpenAI API key（embedding 用，可选本地模型替代）
