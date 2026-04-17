# ResearchFlow 系统架构设计文档 v3.1

> **产品定位**：ResearchFlow = Web 产品 + 自有后端编排 + MCP 兼容层 + 专家模式（Claude/Codex）
>
> **核心理念**：Paper 不是主对象。**DeltaCard 是中间真相层，IdeaDelta 是可复用知识原子**。Paper 是容器，Evidence 是锚点，GraphAssertion 是检索与推理加速器。

---

## 1. 系统全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                          用 户 层                                   │
│  Web 前端 (Next.js 7页)  │  Claude Code (MCP)  │  Codex CLI (MCP)  │
└──────────┬───────────────────────┬───────────────────────┬──────────┘
           │ HTTP/REST             │ MCP (JSON-RPC)        │
           ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Caddy 反向代理 (HTTPS + Let's Encrypt)                             │
│  / → frontend:3000  │  /api/* → api:8000  │  /mcp/* → api:8000     │
└─────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  后端七层 + 知识图谱六层                                             │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Serving Views (Layer 6)                                      │  │
│  │ ReadingPath / Digest / ContrastView / Report / DirectionCard │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │ Quality Control (Layer 5) — NEW                              │  │
│  │ ReviewTask (审核队列) / HumanOverride (人工覆盖)              │  │
│  │ GraphAssertion 生命周期: candidate → published → deprecated   │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │ Evidence & Implementation (Layer 4)                          │  │
│  │ EvidenceUnit (独立行, FK→IdeaDelta, FK→DeltaCard)            │  │
│  │ GraphAssertionEvidence (supports/contradicts/qualifies)      │  │
│  │ ImplementationUnit (file/class/function/shape_trace)         │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │ Idea Layer (Layer 3) — 核心主对象                             │  │
│  │ DeltaCard: 中间真相层 (从L4一次构建，多次渲染)                │  │
│  │ IdeaDelta: 从DeltaCard派生的可复用知识原子                    │  │
│  │ GraphAssertion: 替代GraphEdge，支持生命周期管理               │  │
│  │ GraphNode: 统一节点注册表                                     │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │ Canonical Domain (Layer 2)                                   │  │
│  │ ParadigmFrame (4域: RL/VLM/Agent/MotionGen)                  │  │
│  │ Slot (25个独立类型化槽位)                                     │  │
│  │ MechanismFamily (19个, 层级结构)                              │  │
│  │ Bottleneck (研究瓶颈, 带 embedding)                          │  │
│  │ Alias (实体别名归一)                                          │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │ Scholarly Backbone (Layer 1)                                 │  │
│  │ Paper / Author / Venue / Topic + cites 边                    │  │
│  │ OpenAlex ID 对接准备 (openalex_id 字段)                      │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │ Asset Layer (Layer 0)                                        │  │
│  │ PDF / HTML / Repo / Dataset / Supplementary                  │  │
│  │ 对象存储: LocalStorage (dev) / COS (prod)                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
           │                    │
           ▼                    ▼
┌──────────────────┐  ┌─────────────────────┐
│ PostgreSQL 16    │  │ 对象存储 (COS/本地)  │
│ + pgvector       │  │                     │
│ 31 张表          │  │ papers/raw-pdf/     │
│ (事实源)         │  │ reports/            │
└──────────────────┘  └─────────────────────┘
```

---

## 2. 数量统计

| 组件 | 数量 | 明细 |
|------|------|------|
| **数据库表** | 31 | 5次 Alembic 迁移 (含 7 张 v3.1 新表) |
| **API 端点** | 63 | 11 个 router (含 assertions) |
| **MCP 工具** | 15 | 高层研究操作 (含 graph/review) |
| **前端页面** | 7+layout | Dashboard/Papers/Import/Search/Reports/Digests/Directions |
| **后端 Service** | 21 | 含 4 个 v3.1 新增 service |
| **后台任务** | 5 task + 3 cron | arq worker |
| **Docker 服务** | 6 | postgres/redis/api/worker/frontend/caddy |
| **枚举类型** | 9 | 45 个枚举值 |
| **ParadigmFrame** | 4 | RL/VLM/Agent/MotionGen |
| **Slot** | 25 | 按领域类型化 |
| **MechanismFamily** | 19 | 层级分类 |

---

## 3. v3.1 新增核心概念

### DeltaCard — 中间真相层

DeltaCard 位于 PaperAnalysis（原始LLM输出）和 IdeaDelta（可复用知识原子）之间：

```
L4 analysis_data → DeltaCard (一次构建)
                       ├─→ IdeaDelta (派生知识原子)
                       ├─→ EvidenceUnits (结构化证据)
                       └─→ GraphAssertions (图谱断言)
```

核心字段：delta_statement, baseline_paradigm, changed_slot_ids, mechanism_family_ids, key_ideas_ranked, structurality/extensionability/transferability scores, assumptions, failure_modes, evidence_refs。

发布门槛：frame + bottleneck + changed_slots + evidence_refs >= 2。

### GraphAssertion — 替代 GraphEdge

GraphAssertion 支持完整生命周期：

```
candidate → published → deprecated/superseded
         ↘ rejected
```

- 结构性边 (supported_by, changes_slot 等) 自动发布
- 高价值语义边 (contradicts, transferable_to, patch_of) 默认 candidate，需审核后发布
- 每个 assertion 可链接多条 evidence (GraphAssertionEvidence)，角色支持 supports/contradicts/qualifies

### GraphNode — 统一节点注册

所有参与图谱的实体注册为 GraphNode (node_type + ref_table + ref_id)，消除了旧 graph_edges 的 type+id 字符串拼接。

### ReviewTask — 审核队列

自动或手动创建，支持 pending → in_progress → approved/rejected。审核通过级联更新目标对象状态。

---

## 4. 分析管线（16 步状态机）

```
ingest → canonicalize → enrich → fetch_assets → parse
→ skim_extract (L3) → deep_extract (L4)
→ delta_card_build    ← NEW
→ entity_resolution   ← NEW
→ assertion_propose   ← NEW (替代 edge_create)
→ evidence_audit      ← NEW
→ review              ← NEW
→ publish → index → export → digest
```

### 与 Service 的映射

| 步骤 | Service | 状态 |
|------|---------|------|
| ingest | ingestion_service | 保留 |
| canonicalize | ingestion_service | 保留 |
| enrich | enrich_service | 保留 |
| parse | parse_service (L2) | 保留 |
| skim_extract | analysis_service.skim_paper | 保留 |
| deep_extract | analysis_service.deep_analyze_paper | 保留 |
| **delta_card_build** | **delta_card_service.build_delta_card** | **v3.1 新增** |
| **entity_resolution** | **entity_resolution_service** | **v3.1 新增** |
| **assertion_propose** | **delta_card_service.propose_assertions** | **v3.1 新增** |
| **evidence_audit** | **assertion_service.audit_assertion** | **v3.1 新增** |
| **review** | **review_service** | **v3.1 新增** |
| publish | delta_card_service.check_and_publish | 重构 |
| index | embedding_service | 保留 |
| export | compatibility/export_*.py | 增强 (DeltaCard 段落) |
| digest | digest_service | 增强 (DeltaCard 统计) |

---

## 5. 检索架构（6 路 Query Router）

| 路由 | 查询类型 | 数据路径 |
|------|---------|----------|
| 1. 事实/引用 | "谁提的""谁引了谁" | GraphAssertions(cites) + 回退 graph_edges |
| 2. 瓶颈 | "本质瓶颈有哪些" | Bottleneck → IdeaDelta → structurality |
| 3. 机制 | "diffusion 有哪些路线" | MechanismFamily → IdeaDelta (entity_resolution 支持别名) |
| 4. 迁移 | "RL insight 能否迁到 VLM" | GraphAssertions(transferable_to) |
| 5. 综述 | "最近三个月结构性进展" | IdeaDelta + DeltaCard 聚合 + evidence |
| 6. **Idea 搜索** | "关键词搜索改动" | DeltaCard.delta_statement + IdeaDelta ilike |

基础层保留：keyword(tsvector) + semantic(pgvector) + structured SQL。

---

## 6. API 端点总览 (63 routes)

| Router | 前缀 | 端点数 | 说明 |
|--------|------|--------|------|
| papers | /papers | 8 | Paper CRUD + 列表 |
| import | /import | 4 | 导入链接/PDF/Zotero |
| analyses | /analyses | 5 | L3/L4 分析触发 |
| reports | /reports | 2 | 报告生成 |
| search | /search | 4 | hybrid + idea + embedding + reading plan |
| digests | /digests | 3 | 日/周/月摘要 |
| directions | /directions | 3 | 方向提议 + 展开 |
| feedback | /feedback | 4 | 用户反馈 |
| graph | /graph | 11 | 图谱查询 + 统计 |
| **assertions** | **/assertions** | **15** | **断言/审核/覆盖/别名** |
| health | / | 1 | 健康检查 |

### v3.1 新增 API (assertions router)

```
POST   /assertions                        # 提议断言
GET    /assertions/{id}                   # 断言详情 (含证据+节点)
GET    /assertions/{id}/audit             # 证据审计
POST   /assertions/{id}/publish           # 发布断言
POST   /assertions/{id}/reject            # 拒绝断言
GET    /assertions/node/{node_id}         # 节点关联断言
GET    /assertions/reviews/queue          # 审核队列
GET    /assertions/reviews/stats          # 队列统计
POST   /assertions/reviews/{id}/assign    # 分配审核
POST   /assertions/reviews/{id}/approve   # 批准 (级联)
POST   /assertions/reviews/{id}/reject    # 拒绝 (级联)
POST   /assertions/overrides              # 人工覆盖
GET    /assertions/overrides              # 覆盖列表
POST   /assertions/aliases                # 注册别名
GET    /assertions/aliases                # 别名列表
```

---

## 7. MCP 工具 (15 tools)

| 工具 | 说明 | v3.1 |
|------|------|------|
| search_research_kb | Hybrid 论文搜索 | |
| **search_ideas** | **DeltaCard/IdeaDelta 关键词搜索** | **新增** |
| get_paper_report | 报告生成 (quick/briefing/deep) | |
| compare_papers | 论文对比 | |
| import_research_sources | 导入论文链接 | |
| get_digest | 日/周/月摘要 | 增强 (DeltaCard 统计) |
| get_reading_plan | 分层阅读计划 | 增强 (DeltaCard 评分) |
| **propose_directions** | **研究方向提议** | **新增** |
| enqueue_analysis | L3/L4 分析排队 | |
| refresh_assets | 元数据补全 | |
| record_user_feedback | 用户反馈 | |
| get_paper_detail | 论文详情 | 增强 (含 DeltaCard) |
| **get_graph_stats** | **图谱统计** | **新增** |
| **review_queue** | **审核队列查询** | **新增** |
| **submit_review_decision** | **审核决策 (级联)** | **新增** |

---

## 8. 数据库 Schema (31 张表)

### v3.1 新增表 (7 张)

| 表 | 说明 | 关键字段 |
|----|------|----------|
| delta_cards | 中间真相层 | paper_id, frame_id, delta_statement, scores, evidence_refs, status |
| graph_nodes | 统一节点注册 | node_type, ref_table, ref_id, status |
| graph_assertions | 断言 (替代 graph_edges) | from_node_id, to_node_id, edge_type, status, assertion_source |
| graph_assertion_evidence | 断言-证据关联 | assertion_id, evidence_unit_id, role, weight |
| review_tasks | 审核队列 | target_type, target_id, task_type, status, priority |
| human_overrides | 人工覆盖记录 | target_type, field_name, old_value, new_value, reason |
| aliases | 实体别名归一 | entity_type, entity_id, alias, confidence |

### v3.1 修改的表

| 表 | 新增列 |
|----|--------|
| idea_deltas | delta_card_id (FK → delta_cards) |
| evidence_units | delta_card_id (FK → delta_cards) |

### 保留但废弃的表

| 表 | 状态 |
|----|------|
| graph_edges | 数据已迁移到 graph_assertions，保留作回退查询 |
| method_deltas | 数据已迁移到 delta_cards，保留不删 |

---

## 9. 硬约束 (8 条)

| # | 约束 | 实现 |
|---|------|------|
| 1 | DeltaCard 无充分结构不发布 | frame + slots + evidence >= 2 |
| 2 | IdeaDelta 无证据不发布 | publish_status gate: evidence_count >= 2 |
| 3 | 高价值边需审核 | contradicts/transferable_to/patch_of → candidate + ReviewTask |
| 4 | 边必须区分来源 | assertion_source: paper_asserted/system_inferred/human_verified |
| 5 | paper→paper 只有 cites | edge_type 约束 |
| 6 | 强做 entity resolution | Alias 表 + 3 级解析 (exact → alias → fuzzy) |
| 7 | 多模态证据进图 | EvidenceUnit.atom_type 含 formula/table/figure |
| 8 | 人工覆盖可追溯 | HumanOverride 记录 old/new value + reason |

---

## 10. 质量系统 (4 指标)

| 指标 | 度量 |
|------|------|
| idea correctness | primary_bottleneck_F1, changed_slot_F1, mechanism_family_F1 |
| idea keyness | Top1 hit, Recall@3, nDCG@3 |
| link accuracy | 按边类型分开算 (same_mechanism, patch_of, contradicts, transferable_to) |
| evidence grounding | 每个 IdeaDelta 是否 ≥2 证据, 强语义边是否有 evidence |

---

## 11. 技术选型

| 组件 | 选型 |
|------|------|
| Web 框架 | FastAPI (async) |
| 前端 | Next.js 15 + Tailwind |
| ORM | SQLAlchemy 2.0 (async) |
| 数据库 | PostgreSQL 16 + pgvector |
| 任务队列 | arq (Redis) |
| PDF 解析 | pymupdf |
| 对象存储 | LocalStorage / Tencent COS |
| LLM | Anthropic Claude + OpenAI + mock |
| MCP | FastMCP (Python) |
| 部署 | Docker Compose + Caddy |
