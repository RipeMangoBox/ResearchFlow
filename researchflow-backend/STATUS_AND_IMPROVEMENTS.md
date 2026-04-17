# ResearchFlow 系统全局现状分析与改进路线

> 截至 2026-04-18 的完整状态评估
> 覆盖外层（Skills + Markdown KB）和内层（Backend + 知识图谱）

---

## 一、系统双层架构总览

ResearchFlow 是**双层系统**：

```
┌─── 外层：Local Knowledge Base (Skills + Markdown + CSV) ──────────┐
│                                                                    │
│  18 个 Skills (.claude/skills/)                                   │
│  paperAnalysis/ (analysis_log.csv + 结构化分析笔记)               │
│  paperCollection/ (index.jsonl + Obsidian 导航页)                 │
│  paperIDEAs/ (研究产出)                                           │
│  paperPDFs/ (原始 PDF)                                            │
│  23 个 Python 维护脚本                                            │
│  STATE_CONVENTION.md + AGENTS.md                                  │
│                                                                    │
├─── 内层：Backend System (DB + API + 知识图谱) ────────────────────┤
│                                                                    │
│  23 张 DB 表 (PostgreSQL + pgvector)                              │
│  42 个 API 端点 (FastAPI, 10 router)                              │
│  10 个 MCP 工具                                                   │
│  5 层知识图谱 (IdeaDelta 为核心)                                  │
│  17 个 Service 模块 (58 函数)                                     │
│  7 个前端页面 (Next.js)                                           │
│  6 个 Docker 服务                                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**核心关系**：外层独立可用（纯本地 Claude Code 工作流）；内层增强外层（Web UI + DB + 图谱 + API）。两层通过 MCP + 兼容层导出连接。

---

## 二、外层组件详细盘点

### 2.1 Skills 系统 (18 个)

| # | Skill | 类别 | 是否仍需要 |
|---|-------|------|-----------|
| 1 | research-workflow | 路由入口 | 必须保留 |
| 2 | papers-sync-from-zotero | 收集 | 保留（后端暂无 Zotero） |
| 3 | papers-collect-from-web | 收集 | 保留（后端可替代但更重） |
| 4 | papers-collect-from-github-awesome | 收集 | 保留 |
| 5 | papers-download-from-list | 下载 | 保留（后端 import/pdf 可部分替代） |
| 6 | papers-analyze-pdf | 分析 | 保留（后端 L3/L4 可替代） |
| 7 | papers-audit-metadata-consistency | 审计 | 保留 |
| 8 | papers-build-collection-index | 索引 | 保留（后端 API 可替代） |
| 9 | papers-query-knowledge-base | 检索 | 保留，建议优先走后端 API |
| 10 | code-context-paper-retrieval | 检索 | 保留 |
| 11 | research-brainstorm-from-kb | Idea | 保留 |
| 12 | idea-focus-coach | Idea | 保留（独立） |
| 13 | reviewer-stress-test | 评审 | 保留（独立） |
| 14 | notes-export-share-version | 导出 | 保留 |
| 15 | skill-fit-guard | 工具 | 保留 |
| 16 | write-daily-log | 日志 | 保留 |
| 17 | domain-fork | 迁移 | 保留 |
| 18 | rf-obsidian-markdown | 规范 | 保留 |

**结论**：18 个 skills 全部保留。Skills 是 Claude Code 专家模式的工作流抽象，与后端 API 互补不冲突。后续可升级 skills 优先调后端 API。

### 2.2 数据层

| 目录 | 内容 | 保留？ | 与后端关系 |
|------|------|--------|-----------|
| `paperAnalysis/` | 35 行 CSV + 20 篇分析笔记 | **必须** | 后端镜像到 DB，Markdown 是更丰富的格式 |
| `paperCollection/` | index.jsonl + Obsidian 导航 | **条件保留** | 后端 API 提供同等过滤能力 |
| `paperPDFs/` | 原始 PDF | **必须** | 后端对象存储是额外一层 |
| `paperIDEAs/` | 研究产出 | **必须** | 后端 IdeaDelta 是结构化版本 |

### 2.3 脚本 (23 个)

| 类别 | 脚本数 | 保留？ | 说明 |
|------|--------|--------|------|
| 下载 (auto_download, playwright) | 2 | 保留 | 后端 import 服务可替代但不完全 |
| 维护 (fix_analysis, merge_tags 等) | 12 | **归档** | 一次性数据清洗，后续不常用 |
| 审计 (audit_knowledge_batch) | 1 | 保留 | 与后端审计互补 |
| 设置 (setup_shared_skills, link_codebase) | 2 | **必须** | 基础设施 |
| 转换 (sync_xlsx, format_titles 等) | 6 | **归档** | 偶尔使用 |

### 2.4 配置文件

| 文件 | 保留？ | 说明 |
|------|--------|------|
| `AGENTS.md` | **必须，需更新** | 加入双层模型描述 |
| `STATE_CONVENTION.md` | **必须** | 状态机通用规范 |
| `.mcp.json` | **必须** | 后端 MCP 注册 |
| `.claude/settings.local.json` | **必须** | 权限配置 |
| `.claude/skills-config.json` | **必须** | Skill 路由 |

---

## 三、内层（后端）已实现功能

### 3.1 核心管线

| 功能 | 实现 | 状态 |
|------|------|------|
| 论文导入 (链接/PDF/批量) | ingestion_service + import API | 生产可用 |
| 去重 (arxiv_id/doi/title) | ingestion_service._find_duplicate | 生产可用 |
| 元数据补全 (arXiv/Crossref) | enrich_service | 生产可用 |
| 4 维评分 | triage_service | 生产可用 |
| L2 PDF 解析 | parse_service + pymupdf | 生产可用 |
| L3 轻量分析 | analysis_service + LLM | 需 API key |
| L4 深度分析 | analysis_service + LLM | 需 API key |
| IdeaDelta 自动生成 | graph_service + analysis hook | 需 API key |
| GraphEdge 自动创建 | graph_service.create_edges_for_idea | 生产可用 |
| 发布审核 (evidence gate) | graph_service.check_publish | 生产可用 |
| 混合搜索 | search_service | 生产可用 |
| 5 路图查询 | graph_query_service | 生产可用 |
| 汇报报告 (3 级) | report_service | 需 API key |
| 分层阅读推荐 | reading_planner | 生产可用 |
| 方向推荐 | direction_service | 需 API key |
| 日/周/月总结 | digest_service | 需 API key |
| 用户反馈/收藏 | feedback_service | 生产可用 |
| 库外输入状态机 | ingestion_service | 生产可用 |

### 3.2 知识图谱

| 组件 | 数量 |
|------|------|
| ParadigmFrame | 4 (RL/VLM/Agent/MotionGen) |
| Slot | 25 (类型化: architecture/objective/data/inference) |
| MechanismFamily | 19 (层级: Generative/RL/Architecture/Alignment/Agent) |
| 边类型 | 12 (cites, changes_slot, supported_by 等) |
| 硬约束 | 6 条 (evidence gate, assertion_source 等) |

---

## 四、数据流冲突分析

### 冲突 1：论文元数据双写

| 操作 | 外层 | 内层 | 问题 |
|------|------|------|------|
| 新增论文 | CSV 追加行 + 创建 .md | DB INSERT | 两边不同步 |
| 编辑元数据 | 改 .md frontmatter | DB UPDATE | 互不感知 |
| 补全 | scripts/ 改 CSV | enrich_service 改 DB | 结果不同 |

**现状**：migration 脚本可 CSV→DB 单向同步。反向（DB→CSV）有 export 脚本。
**建议**：Phase 2 让后端成为元数据 canonical source，Markdown 作为导出物。

### 冲突 2：PDF 存储分裂

| 来源 | 外层 | 内层 |
|------|------|------|
| 路径 | paperPDFs/{cat}/{venue}/ | storage/papers/raw-pdf/ 或 COS |
| 下载 | auto_download_papers.py | import/pdf API |

**建议**：后端对象存储为 canonical，paperPDFs/ 可选同步用于离线 Obsidian。

### 冲突 3：搜索结果不一致

| 检索 | 外层 | 内层 |
|------|------|------|
| 方式 | regex/grep on .md + CSV | tsvector + pgvector + structured SQL |
| 结果 | 依赖本地文件内容 | 依赖 DB + embedding |

**建议**：升级 papers-query-knowledge-base skill 检测后端是否可用，优先调 API。

### 冲突 4：Idea 管理

| 输出 | 外层 | 内层 |
|------|------|------|
| 格式 | paperIDEAs/*.md | DB idea_deltas + direction_cards |

**建议**：后端 export → paperIDEAs/ 自动同步。

---

## 五、待实现 / 需改进的功能

### P0 — 高优先级

| # | 改进项 | 工时 | 说明 |
|---|--------|------|------|
| 1 | **真实 LLM 端到端验证** | 3h | 配 API key，跑 5 篇论文 L3→L4→IdeaDelta→Evidence→Edge |
| 2 | **EvidenceUnit 持久化完善** | 2h | 确保 L4 prompt 总返回 evidence_units 数组 |
| 3 | **Entity Resolution 服务** | 4h | MechanismFamily 别名归一 + Slot 名称标准化 |
| 4 | **AGENTS.md 更新** | 1h | 加入双层模型描述，声明后端优先规则 |

### P1 — 中优先级

| # | 改进项 | 工时 | 说明 |
|---|--------|------|------|
| 5 | **OpenAlex 对接** | 4h | 填充 cited_by_count、authors、topic hierarchy |
| 6 | **Citation 边自动创建** | 4h | 从 references 或 OpenAlex 导入 cites 边 |
| 7 | **前端图查询页面** | 4h | /graph 页面：paradigm/mechanism/bottleneck 浏览 |
| 8 | **IdeaDelta-to-IdeaDelta 边** | 6h | same_mechanism / patch_of / contradicts 自动推断 |
| 9 | **CSV↔DB 双向同步工具** | 3h | 解决冲突 1 |
| 10 | **Skills 升级为 API-first** | 4h | papers-query-knowledge-base 优先调后端 |

### P2 — 低优先级（长期增强）

| # | 改进项 | 工时 | 说明 |
|---|--------|------|------|
| 11 | **评测集 (50 篇 Gold Set)** | 8h | 人工标注 bottleneck/slots/ideas |
| 12 | **多模态证据增强** | 4h | L2 公式/表格/图注 → 自动 EvidenceUnit |
| 13 | **图可视化 (D3.js 或 Neo4j)** | 8h | 前端可视化边关系 |
| 14 | **增量图更新** | 6h | 保留 human_verified 边，只更新变化 |
| 15 | **跨领域迁移检测** | 6h | 自动生成 transferable_to 边 |
| 16 | **三角色复核管线** | 8h | extractor → auditor → taxonomy_reviewer |
| 17 | **阿里云 OSS 后端** | 2h | object_storage.py 增加 OSSStorage |
| 18 | **维护脚本归档** | 2h | 12 个一次性脚本移入 scripts/legacy/ |
| 19 | **paperCollection 从后端重建** | 3h | 不再从 CSV 生成，改从 DB 查询 |
| 20 | **IdeaDelta → paperIDEAs/ 导出** | 2h | 后端图谱产出 → Markdown 同步 |

---

## 六、系统风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| **双层数据不一致** | 搜索/报告结果矛盾 | CSV→DB 迁移脚本已有；需加定期同步 |
| **LLM JSON 输出不稳定** | IdeaDelta/evidence 解析失败 | fallback 解析已有；需真实测试 |
| **Mock 模式掩盖问题** | 上线后发现 prompt 不对 | 优先配 key 跑真实论文 |
| **graph_edges 无 FK 约束** | 孤儿边 | 定期审计 + 软删除 |
| **Slot 匹配硬编码** | 新领域需手动添加 | 后续加 LLM-based 匹配 |
| **4C8G 内存紧张** | 高并发 OOM | 监控 + 减 workers |
| **Skill 和后端同时改同一数据** | 状态冲突 | 约定：有后端时 API 优先 |

---

## 七、建议的下一步优先序

```
阶段 1：上线前（本周）
  1. 配 Anthropic API key → 5 篇真实论文验证     [3h]
  2. 部署到服务器 (deploy.sh)                    [2h]
  3. 更新 AGENTS.md 加入双层说明                  [1h]

阶段 2：上线后一周
  4. Entity resolution 基础版                     [4h]
  5. CSV↔DB 同步工具                              [3h]
  6. OpenAlex 对接                                [4h]
  7. 前端图查询页面                                [4h]

阶段 3：上线后一个月
  8. Citation 边 + IdeaDelta 边                   [10h]
  9. Skills 升级为 API-first                       [4h]
  10. 评测集 (50 篇)                               [8h]
  11. 维护脚本归档                                  [2h]
```

---

## 八、代码库完整结构

```
ResearchFlow/
├── .claude/
│   ├── skills/                    # 18 个 skill 定义
│   │   ├── research-workflow/
│   │   ├── papers-{sync,collect,download,analyze,build,query}/
│   │   ├── {brainstorm,focus,review,export,guard,log,fork}/
│   │   ├── rf-obsidian-markdown/
│   │   ├── STATE_CONVENTION.md
│   │   └── User_README_CN.md
│   ├── skills-config.json         # skill 路由注册
│   └── settings.local.json        # 权限配置
├── paperAnalysis/                 # 外层：结构化分析笔记 (35行CSV + 20篇MD)
├── paperCollection/               # 外层：索引 + Obsidian 导航 (生成)
├── paperPDFs/                     # 外层：原始 PDF
├── paperIDEAs/                    # 外层：研究产出
├── scripts/                       # 外层：23 个维护脚本
├── linkedCodebases/               # 外层：外部代码库符号链接
├── assets/                        # Logo
├── AGENTS.md                      # Agent 指南
├── .mcp.json                      # MCP 注册 → 后端
├── README_CN.md                   # 项目说明
│
└── researchflow-backend/          # 内层：后端系统
    ├── backend/
    │   ├── models/                # 12 files, 23 model classes
    │   ├── services/              # 17 files, 58 functions
    │   ├── api/                   # 9 files, 42 endpoints
    │   ├── workers/               # 5 tasks + 3 cron
    │   ├── mcp/                   # 10 MCP tools
    │   └── utils/                 # frontmatter, pdf_extract, sanitize
    ├── frontend/                  # Next.js: 7 pages + layout
    ├── alembic/                   # 4 migrations (23 tables)
    ├── migration/                 # CSV→DB, MD→DB, seed, validate
    ├── compatibility/             # DB→Markdown export
    ├── docker-compose.prod.yml    # 6 services
    ├── deploy.sh                  # 一键部署
    ├── ARCHITECTURE.md            # 架构 v3
    ├── DEPLOY_GUIDE.md            # 部署指南
    └── STATUS_AND_IMPROVEMENTS.md # 本文档
```
