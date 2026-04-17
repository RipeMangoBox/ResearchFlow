# ResearchFlow 重构设计文档 v2

## Context

ResearchFlow 当前是纯本地系统（Markdown + CSV + JSONL），没有数据库、API 或持久服务。需要重构为 **Web-first 的科研操作系统**：

**产品定位（一句话）**：
> ResearchFlow = Web 产品 + 自有后端编排 + MCP 兼容层 + 专家模式（Claude/Codex）

**核心原则**：
- 普通用户只用网页，不碰 CLI
- 系统本体是自有后端（DB + 对象存储 + 任务队列 + 检索层），不是 Claude Code 会话
- Claude/Codex 是专家驾驶舱，不是唯一入口也不是后端本体
- 需要 LLM 能力时通过 Agent SDK 调入，同时通过 MCP 暴露给 Claude Code/Codex
- 架构是 "workflow-first"，不是 "agent-first"

**预算约束**：2000 元/年基础设施（不含 token），2C4G + 对象存储。

---

## Phase 0 已完成

后端骨架已搭建在 `researchflow-backend/`，包含：
- Docker Compose (Postgres+pgvector, Redis, API, Worker)
- 14 张表的 ORM 模型 + Alembic 001 迁移
- 迁移脚本 (CSV→DB, MD→DB, validate)
- 导出兼容脚本 (DB→Markdown, DB→CSV)
- FastAPI 入口 + config + database 模块

---

## 用户角色定义

| 角色 | 入口 | 能做什么 |
|------|------|----------|
| 普通研究生/老师 | Web 前端 | 导入论文、看报告、看总结、选方向、给反馈 |
| 高级研究用户 | Web + Claude Code/Codex | 深度交互、自定义检索、跨论文对比、idea 收敛 |
| 维护者/开发者 | Claude Code/Codex + 后台 | 开发 skill、修规则、审结果、管理 taxonomy |
| 后端服务 | 自有 API + Worker | 自动收集、分析、生成总结，必要时调 Agent SDK |

---

## 8 个核心用户功能

### 1. 导入任意研究输入

用户可提交：awesome 列表、论文链接、PDF/HTML/arXiv、GitHub 仓库、子方向描述、Zotero 集合。

系统统一做：规范化 → 去重 → canonical identity 对齐 → 补全缺失资产 → 放进知识库。

**关键原则**：输入形式不决定后续流程，所有输入落到同一个对象模型。

### 2. 一键生成可汇报的论文报告

三个层级：
- **30 秒版**：快速扫一眼
- **5 分钟汇报版**：组会口头汇报
- **深度对比版**：课题推进和立项

汇报版固定结构：
1. 这个列表在讲什么问题
2. 该方向的 canonical baseline
3. 这些论文各改了哪些槽位
4. 哪些是插件 patch，哪些是结构性改进
5. 开源情况、复现价值、证据强度
6. 建议先读哪几篇
7. 当前最值得追的 1-3 条线

**核心**：报告先对照领域标准范式（canonical delta card），再说改了什么，避免被论文叙事欺骗。

### 3. 论文 + GitHub 仓库联合深度解析

输入一个 repo，系统补到对应论文/项目页/数据/训练配置，输出"论文-公式-代码-例子"对齐报告：

- 整体流程图（模块级）
- 核心公式 → 代码位置映射
- 核心代码路径（只解释关键路径）
- Shape trace / tensor trace（维度变化链）
- 参数说明（改什么影响什么）
- 设计解释（与证据绑定，分"代码证据支持"和"推测性解释"）
- 可修改性分析（改 reward/层/planner 的连锁影响）

**必须带置信度和证据锚点**。

### 4. 增量更新与资产补全

平台默认行为：
- 任意资产入库后做 canonical identity 对齐
- 自动补全：论文页、repo、project page、数据说明、补充材料、训练配置、checkpoint 线索
- 所有发现记录来源与置信度
- 最近 1 个月内可能变更的对象自动重检
- 超过 1 个月默认不刷新，管理员可强制覆盖

### 5. 给定子方向，推荐可落地改进方向

两段式输出：

**第一段 — 方向卡片**（1-3 个）：
- 为什么值得做 / 解决哪个瓶颈
- 结构性路线还是 patch
- 依赖哪些开源数据/代码
- 大致实验成本 / 最大风险

**第二段 — 用户选定后生成详细方案**：
- 相关论文链 + 可迁移的跨领域机制
- baseline 选择 + 第一版最小实验
- 数据/代码/算力条件
- 可能失败点 + 成功判据 + 退出条件

**关键**：系统主动重写用户需求 → 症状是什么 → 真正瓶颈 → 哪些方向已排除 → 哪些跨领域机制值得迁移。

### 6. 阅读顺序与阅读深度推荐

默认推荐顺序：canonical baseline → 结构性改进 → 最新强工作 → patch paper → 负例/边界论文。

每篇标注建议阅读深度：30 秒版 / 5 分钟版 / 20 分钟精读。读完一篇后推荐下一篇。

### 7. 日/周/月总结（产品一等公民）

**日总结** — 今天最值得看什么：
- 新入库高优先论文 / 优先级变化最大的主题
- 推荐深读 3 篇 / 需要人确认的关键判断
- 今晚后台建议跑什么

**周总结** — 方向有没有偏：
- 哪些方向热度上升 / 哪些只是 patch 堆积
- 哪些结构性路线开始成形 / 哪些旧判断被推翻
- 下周值得跟进 3 条线

**月总结** — 研究策略要不要变：
- 方向地图变化 / 最可信瓶颈排序
- 值得立项的方向 / 可以停追的方向
- 哪些旧论文值得用更强模型重分析

### 8. 反馈、批注、收藏、导出

用户可以：收藏论文/方向、标注"判断不对"、标注"更像 patch"、标注"真正关心 xxx"、导出组会版/分享版/Obsidian 版。

---

## 多维评分体系

| 分数 | 含义 | 影响 |
|------|------|------|
| keep_score | 进主库还是观察池 | 开数据最高 > 开代码 > 已中稿不开源 > 预印本 |
| analysis_priority | 先分析谁 | 相关性 + 开源完整度 + 团队可信度 + 新鲜度 + 证据强度 |
| structurality_score | 插件 patch / 局部技巧 / 中层机制 / 结构性改动 | 决定是否值得迁移 |
| extensionability_score | 可扩展到其他任务/模态/训练设定/数据规模 | idea 生成时的权重 |

---

## 方法分类三层设计

### 第一层：硬结构（固定字段）
domain, task, changed_slot, mechanism_family, supervision_type, inference_pattern

`changed_slot` 示例：reward, objective, sampling, planner, memory, encoder, projector, intermediate_layer, optimizer

### 第二层：软标签（社区术语 + 自有词汇）
plugin, reward_patch, grouped_reward, agentic_planner, think_with_image, self_correction, bottleneck_transfer

### 第三层：检索分支树（最关键）
- symptom_query：用户最初描述
- latent_need：系统推测真实需求
- candidate_bottlenecks：当前候选瓶颈
- rejected_solution_patterns：已排除的伪方向
- search_branches：后来开出的新搜索支路

---

## 提取质量保障六层机制

高质量提取靠"程序化证据管线 + 多角色复核 + 可回放评测"，不靠单纯提示词工程。

| 层 | 机制 | 说明 |
|----|------|------|
| 1 | 四级分析管线 | L1→L2→L3→L4 分步晋升，"是否花 token"是明确决策点 |
| 2 | 先规则解析再 LLM | 程序抽章节/公式/表格/图注/repo 文件；LLM 只做槽位归因+机制抽象+证据解释 |
| 3 | 标准范式对齐 | 强制 Canonical Delta Card 对照 paradigm_template，不顺论文叙事走 |
| 4 | 三角色复核 | extractor→auditor→taxonomy_reviewer，高价值论文不由单角色决定 |
| 5 | 置信度分级 | 每条结论标注 confidence(0-1) + basis(code_verified/experiment_backed/text_stated/inferred/speculative) |
| 6 | 评测集回放 | 50 篇标杆 + 10 查询 + 10 重构场景，每次升级 prompt/taxonomy/模型先回放 |

### EvidenceBasis 枚举

| basis | 含义 |
|-------|------|
| `code_verified` | 已在源代码中确认 |
| `experiment_backed` | 有 ablation/表/图支持 |
| `text_stated` | 作者正文明确写出 |
| `inferred` | 系统逻辑推断 |
| `speculative` | 系统猜测，无直接证据 |

---

## 库外输入状态机

普通用户可随时丢入"不在库中的内容"。系统支持"先输入，再入库"。

### 状态流转

```
ephemeral_received → canonicalized → enriched → l3_skimmed
                                                    │
                                    用户点"加入知识库" 或 系统判定高价值
                                                    ▼
                                            accepted_to_kb (= wait)
                                                    │
                                                    ▼ 正式管线
                                            L1→L2→L3→L4→checked

                        30 天未操作 → archived_or_expired (清理)
```

### 三类库外输入

| 类型 | 场景 | 保留期 |
|------|------|--------|
| 一次性分析 | 随手丢 PDF | 默认 30 天 |
| 候选入库 | awesome list / repo | 入库后永久 |
| 强制入库 | 管理员指定 | 永久 |

### 新增数据表字段

- `papers.is_ephemeral` (bool) — 是否临时对象
- `papers.expires_at` (timestamptz) — 过期时间
- `papers.retention_days` (smallint, default 30) — 保留天数
- `paper_analyses.confidence_notes` (JSONB) — 每条结论的置信度和依据
- `evidence_units.confidence` (float) — 该证据的置信度
- `evidence_units.basis` (evidence_basis enum) — 证据依据类型
- `evidence_units.source_page` (smallint) — 来源页码

---

## 后端六层架构

```
┌────────────────────────────────────────────────────────────┐
│  Presentation   渲染论文卡片/汇报报告/深度对比/repo 解析/总结 │
├────────────────────────────────────────────────────────────┤
│  Feedback/Eval  用户纠错/偏好记录/重分析触发/离线评测/canary │
├────────────────────────────────────────────────────────────┤
│  Workflow/Job   collect→enrich→triage→skim→deep_report→    │
│                 taxonomy_review→repo_alignment→digest→      │
│                 direction_propose→feasibility→reanalyze     │
├────────────────────────────────────────────────────────────┤
│  Retrieval      Postgres + pgvector (结构化+全文+语义)      │
├────────────────────────────────────────────────────────────┤
│  Parse/Extract  本地 PDF 解析 → delta card + evidence spans │
│                 + method slots + limitations + transfer atoms│
├────────────────────────────────────────────────────────────┤
│  Ingestion      规范化 → 去重 → canonical identity 对齐 →  │
│                 资产补全 → 入库                             │
└────────────────────────────────────────────────────────────┘
```

---

## 前端 7 个页面

### 1. Dashboard（首页）
- 今日新增 / 今日推荐深读
- 当前项目瓶颈 / 本周趋势
- 近期未读高优先论文
- 最新日/周/月总结入口

### 2. 导入中心
- 粘贴链接 / 上传 PDF / 导入 awesome 列表 / 导入 repo / 批量导入 / 同步 Zotero
- 导入后显示：已识别对象、自动补全资产、去重结果、缺失项、是否进主分析队列

### 3. 论文库 / 检索页
- 过滤维度：任务、方法槽位、机制家族、是否开源、是否已中稿、结构性分数、可扩展性分数、相关性
- 卡片显示：改了什么、patch 还是结构性、值不值得看、开源情况、建议阅读深度

### 4. 报告页
- 输入论文列表或主题 → 生成汇报报告
- 三种视图切换：快速版 / 汇报版 / 深度对比版

### 5. Repo × Paper 深剖页
- 代码流程 / 公式映射 / 核心模块
- 输入输出 shape / 改动建议 / 风险提示

### 6. 方向推荐页
- 输入子方向描述 → 1-3 个方向卡片（成本/风险/复现条件）
- 点开后详细可行性方案

### 7. 总结与复盘页
- 日/周/月总结聚合
- 历史方向变化 / 用户反馈纠偏记录

---

## MCP Server 工具定义（修订版，10 个）

只暴露高层工具，不暴露底层 CRUD/SQL/对象存储读写：

| 工具名 | 功能 |
|--------|------|
| `import_research_sources` | 导入任意研究输入（链接/PDF/repo/awesome list） |
| `search_research_kb` | 关键词+语义+结构化过滤搜索知识库 |
| `get_paper_report` | 获取论文报告（30s/5min/deep 三级） |
| `get_repo_paper_report` | 获取 repo×paper 联合深度分析报告 |
| `compare_papers` | 2-5 篇论文对比（delta card + 证据强度 + 槽位表） |
| `refresh_assets` | 触发资产补全与增量更新 |
| `propose_directions` | 给定子方向，生成 1-3 个改进方向卡片 |
| `expand_feasibility_plan` | 展开选定方向为详细可行性方案 |
| `get_digest` | 获取日/周/月总结 |
| `record_user_feedback` | 记录纠错/确认/收藏/标签修改 |

---

## Claude Code / Codex 适配策略

### Claude Code 侧
- **skills** 封装固定研究工作流（按需加载 SKILL.md）
- **subagents** 做证据核查 / taxonomy review（隔离上下文）
- **hooks** 做确定性动作（写回 DB、导出报告、跑校验）
- **MCP** 连接 ResearchFlow 后端
- 需要程序化执行时用 **Agent SDK**
- Routines 可做辅助自动化，但不作为唯一生产后端

### Codex 侧
- **AGENTS.md** 写仓库级规则（简洁地图，不是百科）
- **.codex/config.toml** 配 MCP server
- **skills** 复用同一套中立 workflow 描述
- 自动化深度集成：用 Codex SDK
- 富交互嵌入自有前端：用 Codex app-server

### 中立适配层
- 维护一套中立 skill 源 → 生成 Claude 版 / Codex 版包装
- 坚持两层中立边界：MCP 工具边界 + open agent skills 文档边界
- 不被某一家锁死

---

## 用户行为监控（结构化事件，最小化采集）

采集事件：
- 导入类型 / 论文打开 / 停留时长 / 报告导出
- 方向卡点击 / 推荐否决 / 结构性→patch 改标
- 手工补标签 / 瓶颈描述改写 / 最终方向选择

驱动更新：
1. **排序** — 哪些结果更该排前面
2. **标签** — 同义词映射学习
3. **报告风格** — 结论先行 vs 证据先行
4. **重分析触发** — 同类判断频繁被纠正 → 触发该 taxonomy/prompt 重跑

**红线**：线上行为不直接改生产 prompt → 先写候选规则 → 跑离线评测 → 小流量验证 → 管理员确认上线。

---

## 数据库 Schema（已实现，14+ 表）

### 已在 Phase 0 创建的表
papers, paper_assets, paper_versions, paper_analyses, method_deltas, paradigm_templates, evidence_units, transfer_atoms, project_bottlenecks, search_sessions, reading_plans, digests, execution_memories, jobs, model_runs, user_feedback

### 需要新增的表（Phase 1+）

#### repo_analyses — Repo 深度分析记录

| 列名 | 类型 | 说明 |
|------|------|------|
| id | UUID PK | |
| paper_id | UUID FK→papers | 关联论文（可空） |
| repo_url | TEXT NOT NULL | GitHub/GitLab URL |
| repo_snapshot_key | TEXT | 对象存储中的 code snapshot |
| flow_diagram_md | TEXT | 整体流程图 (Markdown/Mermaid) |
| formula_code_mapping | JSONB | [{formula, code_file, code_line, variable_mapping}] |
| critical_path | JSONB | [{module, file, function, description}] |
| shape_trace | JSONB | [{layer, input_shape, output_shape, notes}] |
| modifiability | JSONB | [{target, impact_chain, risk_level}] |
| confidence_notes | JSONB | [{claim, confidence, evidence_type}] |
| model_provider | VARCHAR(50) | |
| model_name | VARCHAR(100) | |
| generated_at | TIMESTAMPTZ | |

#### report_cache — 汇报报告缓存

| 列名 | 类型 | 说明 |
|------|------|------|
| id | UUID PK | |
| report_type | VARCHAR(20) | quick/briefing/deep_compare |
| input_paper_ids | UUID[] | 输入论文 ID 列表 |
| input_topic | TEXT | 或输入主题 |
| rendered_md | TEXT | 渲染后的报告 |
| render_version | VARCHAR(20) | |
| expires_at | TIMESTAMPTZ | 缓存过期时间 |
| created_at | TIMESTAMPTZ | |

#### user_bookmarks — 用户收藏

| 列名 | 类型 | 说明 |
|------|------|------|
| id | UUID PK | |
| target_type | VARCHAR(30) | paper/direction/bottleneck/report |
| target_id | UUID | |
| note | TEXT | 用户批注 |
| created_at | TIMESTAMPTZ | |

#### user_events — 结构化行为事件

| 列名 | 类型 | 说明 |
|------|------|------|
| id | UUID PK | |
| event_type | VARCHAR(50) | import/view_paper/export_report/click_direction/override_label/... |
| target_type | VARCHAR(30) | |
| target_id | UUID | |
| payload | JSONB | 事件详情 |
| created_at | TIMESTAMPTZ | |

#### direction_cards — 方向推荐卡片

| 列名 | 类型 | 说明 |
|------|------|------|
| id | UUID PK | |
| bottleneck_id | UUID FK | 关联瓶颈 |
| title | TEXT | 方向标题 |
| rationale | TEXT | 为什么值得做 |
| is_structural | BOOLEAN | 结构性 vs patch |
| required_assets | JSONB | {data, code, compute} |
| estimated_cost | TEXT | 实验成本估算 |
| max_risk | TEXT | 最大风险 |
| confidence | REAL | |
| related_paper_ids | UUID[] | |
| feasibility_plan_md | TEXT | 详细可行性方案（展开后填入） |
| created_at | TIMESTAMPTZ | |

---

## API 端点设计（修订版）

### Ingestion（导入）
- `POST /api/v1/import/links` — 导入论文链接列表
- `POST /api/v1/import/pdfs` — 上传 PDF 文件
- `POST /api/v1/import/awesome` — 导入 awesome 列表 URL
- `POST /api/v1/import/repo` — 导入 GitHub 仓库
- `POST /api/v1/import/zotero` — 同步 Zotero 集合
- `POST /api/v1/import/batch` — 批量导入（CSV/JSONL）

### Papers（论文）
- `GET /api/v1/papers` — 列表+多维过滤
- `GET /api/v1/papers/{id}` — 详情+最新分析
- `POST /api/v1/papers/search` — 高级搜索（关键词+语义+结构化）
- `POST /api/v1/papers/compare` — 2-5 篇对比

### Reports（报告）
- `POST /api/v1/reports/generate` — 生成汇报报告（type: quick/briefing/deep_compare）
- `GET /api/v1/reports/{id}` — 获取已生成的报告
- `GET /api/v1/reports/latest` — 最近生成的报告

### Repo Analysis（代码分析）
- `POST /api/v1/repos/analyze` — 提交 repo 分析任务
- `GET /api/v1/repos/{id}` — 获取 repo×paper 分析结果

### Directions（方向推荐）
- `POST /api/v1/directions/propose` — 生成方向卡片
- `POST /api/v1/directions/{id}/expand` — 展开为详细可行性方案
- `GET /api/v1/directions` — 列表

### Analysis Pipeline
- `POST /api/v1/analyses/enqueue` — 排队分析（L2/L3/L4）
- `GET /api/v1/analyses/{id}` — 获取分析
- `POST /api/v1/analyses/re-analyze` — 触发重分析

### Research Workflow
- `GET/POST /api/v1/bottlenecks` — 瓶颈 CRUD
- `GET/POST /api/v1/reading-plans` — 阅读计划

### Digests（总结）
- `GET /api/v1/digests` — 列表
- `POST /api/v1/digests/generate` — 生成
- `GET /api/v1/digests/latest` — 最新各类型

### User Actions（用户操作）
- `POST /api/v1/bookmarks` — 收藏
- `POST /api/v1/feedback` — 纠错/确认
- `GET /api/v1/events` — 行为事件查询（管理员）
- `POST /api/v1/export/markdown` — 导出 Markdown
- `POST /api/v1/export/briefing` — 导出汇报版

### System
- `GET /api/v1/jobs` — 任务队列
- `GET /api/v1/health` — 健康检查

---

## Workflow/Job 层任务类型

| 任务类型 | 说明 | 触发方式 |
|----------|------|----------|
| collect | 收集候选论文 | 用户导入 |
| enrich | Crossref/arXiv/Semantic Scholar 补全元数据 | 自动（导入后） |
| triage | 多维评分 + 分层 | 自动（补全后） |
| parse | L2 本地 PDF 解析 | 自动（下载后） |
| skim | L3 轻量卡片生成 | 自动（解析后，高优先） |
| deep_report | L4 全文深度分析 | 手动或高优先自动 |
| taxonomy_review | 分类与槽位映射校验 | 定时 |
| repo_alignment | Repo×Paper 联合分析 | 手动 |
| report_generate | 汇报报告生成 | 用户请求 |
| direction_propose | 方向推荐生成 | 用户请求 |
| feasibility_expand | 可行性方案展开 | 用户请求 |
| digest_generate | 日/周/月总结 | 定时 (23:00/周日/月末) |
| asset_refresh | 资产增量更新 | 定时（每日低峰） |
| embedding_batch | 批量 embedding 生成 | 定时（低峰） |
| reanalyze | 重分析（模型升级/taxonomy变化/用户纠错） | 触发器 |

---

## Docker Compose 服务布局（2C4G）

| 服务 | 镜像 | 内存限制 | 说明 |
|------|------|----------|------|
| postgres | pgvector/pgvector:pg16 | 1280 MB | 数据库 + 向量检索 |
| redis | redis:7-alpine | 256 MB | 缓存 + 任务队列 |
| api | 自建 | 400 MB | FastAPI（含 MCP endpoint） |
| worker | 自建 | 1024 MB | arq worker, 2 并发 |
| frontend | Next.js | 300 MB | Web 前端 |
| caddy | caddy:2-alpine | - | 反向代理 + HTTPS |

并发配额：parser 1 / deep_report 2 / repo_alignment 1 / digest 1（低优先）。

---

## 三层产品路线图

### 第一层：必须先做（产品能用的底座）

| 功能 | 对应后端 | 优先级 |
|------|----------|--------|
| 任意输入导入 | Ingestion 层 | P0 |
| 资产补全与去重 | enrich + triage workers | P0 |
| 初筛排序（4 维评分） | triage_service | P0 |
| 论文列表汇报报告（30s/5min/deep） | report_generate worker | P0 |
| 阅读顺序推荐 | reading_planner | P0 |
| 日/周/月总结 | digest_generate worker | P0 |
| Web 前端 7 页 | frontend/ | P0 |
| Postgres + COS + arq | 基础设施 | P0 |
| MCP server（10 个工具） | mcp/ | P0 |

### 第二层：很值钱，放后面

| 功能 | 说明 |
|------|------|
| Repo × Paper 深剖 | 代码-公式-论文对齐 |
| 方向推荐 + 可行性方案 | propose_directions + expand_feasibility |
| 用户反馈闭环 | feedback → 重分析触发 |
| Admin 审核台 | taxonomy/prompt 变更审批 |

### 第三层：最难，长期回报最大

| 功能 | 说明 |
|------|------|
| 用户行为驱动自更新 | 事件 → 候选规则 → 评测 → 上线 |
| 领域评测集与回放 | 50 篇标杆 + 10 查询 + 10 重构场景 |
| 自动重分析策略 | 模型升级/taxonomy 变化/新论文冲突 |
| Cross-domain transfer graph | 跨领域机制迁移图谱 |
| Execution memory | 环境指纹 + 踩坑记录 + 修复方案检索 |

---

## 开发阶段详细路线

### Phase 1: API 核心 + Ingestion（第 3-5 周）
**目标**：FastAPI 提供论文数据，导入流程可用

- 完善 ORM 模型（新增 repo_analyses, report_cache, user_bookmarks, user_events, direction_cards）
- Alembic 002 迁移
- Papers CRUD + 多维过滤 + 全文搜索
- Import endpoints（links/pdfs/awesome/repo）
- paper_service + import_service
- enrich worker（Crossref/arXiv 自动补全）
- triage_service（4 维评分）

**验收**：POST /api/v1/import/links 导入 3 篇论文 → 自动补全元数据 → GET /api/v1/papers 返回含评分的结果。

### Phase 2: 对象存储 + PDF 管线（第 6-8 周）
**目标**：PDF 进对象存储，L1/L2 管线可用

- COS/OSS 抽象层
- PDF 上传迁移
- parse_worker (L2): pymupdf 提取章节
- 资产补全 worker（自动查找 repo/project page/data）
- asset_refresh 定时任务

**验收**：上传 PDF → COS → L1 补全 → L2 解析 → state 更新。

### Phase 3: LLM 分析管线 + 报告生成（第 9-13 周）
**目标**：L3 skim + L4 deep + 汇报报告

- llm_service（Agent SDK 调用 + model_runs 追踪）
- skim_worker (L3) + deep_report_worker (L4)
- method_deltas 生成（canonical delta card）
- evidence_units 提取
- paradigm_templates 种子数据
- **report_generate worker**（30s/5min/deep 三级报告）
- 报告 API endpoints

**验收**：导入 10 篇论文列表 → 自动 triage → L3 skim → 生成 5 分钟汇报版报告。

### Phase 4: 语义搜索 + 阅读推荐（第 14-16 周）
**目标**：向量搜索 + 分层阅读推荐

- embedding_worker + pgvector HNSW
- 混合搜索：tsvector + cosine + 结构化
- reading_planner（canonical → structural → follow-up → patch → negative）
- 搜索会话 + latent_need 重写

### Phase 5: 总结系统 + MCP（第 17-20 周）
**目标**：日/周/月总结 + MCP 接通 Claude/Codex

- digest_generate worker（arq cron 定时）
- 日/周/月三种模板
- MCP server 实现 10 个工具
- .mcp.json (Claude Code) + .codex/config.toml
- Claude skills 更新为 MCP 调用版

### Phase 6: 前端（第 21-28 周）
**目标**：Web UI 可用

- Next.js + Tailwind + shadcn/ui
- 7 页按序：Dashboard → 导入中心 → 论文库 → 报告页 → 总结页 → 方向推荐 → Repo 深剖
- 用户反馈 UI（收藏/批注/纠错）

### Phase 7: 高级功能（第 29-36 周）
- Repo × Paper 联合分析 worker
- 方向推荐 + 可行性方案
- 用户行为事件采集 + 分析
- 反馈 → 重分析闭环
- 评测集 + canary 发布
- 2C4G 压力测试

---

## 验证方式

1. **迁移验证**：validate_migration.py 检查 DB vs 原始文件
2. **导出往返**：导出 Markdown diff 为空
3. **API 测试**：pytest 覆盖核心 endpoints
4. **报告质量**：人工评审 10 篇论文的 3 级报告
5. **搜索质量**：10 个查询的 recall@10 和 precision@10
6. **MCP 集成**：Claude Code 中 10 个工具全部可用
7. **前端验收**：非技术用户完成导入→看报告→选方向→读总结全流程
8. **压力测试**：2C4G 上 3 并发用户 + 2 后台任务不 OOM

---

## 关键架构决策

1. **Web-first**：普通用户只用网页，Claude/Codex 是专家模式
2. **自有后端为本体**：不把 Claude Code 会话当后端，需要 LLM 时调 Agent SDK
3. **Workflow-first, not Agent-first**：自建轻量编排层，不造通用 agent 平台
4. **arq over Celery**：asyncio 原生，2C4G 更轻量
5. **单 Postgres 承载一切**：到 5000+ 篇再考虑拆 Qdrant
6. **MCP 高层工具**：不暴露 SQL/对象存储/底层 CRUD
7. **导出作为兼容桥梁**：Markdown 导出保持旧 skill 可用
8. **行为不直接改 prompt**：候选规则 → 评测 → 验证 → 管理员上线

---

## 关键文件清单

### 已创建（Phase 0）
- `researchflow-backend/backend/models/*.py` — 14 张表 ORM
- `researchflow-backend/alembic/versions/001_initial_schema.py` — 初始迁移
- `researchflow-backend/migration/migrate_csv_to_db.py` — CSV 迁移
- `researchflow-backend/migration/migrate_md_to_db.py` — MD 迁移
- `researchflow-backend/compatibility/export_*.py` — 导出脚本
- `researchflow-backend/docker-compose.yml` — 服务编排
- `researchflow-backend/backend/main.py` — FastAPI 入口

### 待创建（Phase 1+）
- `backend/api/papers.py` — Papers CRUD router
- `backend/api/import_.py` — Import endpoints
- `backend/api/reports.py` — Report generation
- `backend/services/import_service.py` — 导入+规范化+去重
- `backend/services/triage_service.py` — 4 维评分
- `backend/services/report_service.py` — 3 级报告生成
- `backend/services/llm_service.py` — Agent SDK 封装
- `backend/workers/skim_worker.py` — L3 轻量分析
- `backend/workers/deep_report_worker.py` — L4 深度分析
- `backend/workers/report_generate_worker.py` — 汇报报告
- `backend/workers/digest_worker.py` — 日/周/月总结
- `backend/mcp/server.py` — MCP server
- `backend/mcp/tools.py` — 10 个 MCP 工具
- `frontend/src/app/**` — 7 个页面
