# ResearchFlow 系统架构设计文档 v3

> **产品定位**：ResearchFlow = Web 产品 + 自有后端编排 + MCP 兼容层 + 专家模式（Claude/Codex）
>
> **核心理念**：Paper 不是主对象，**IdeaDelta 才是主对象**。Paper 是容器，Evidence 是锚点，Graph 是检索与推理加速器。

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
│  后端六层 + 知识图谱五层                                             │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Serving Views (Layer 5)                                      │  │
│  │ ReadingPath / Digest / ContrastView / Report / DirectionCard │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │ Evidence & Implementation (Layer 4)                          │  │
│  │ EvidenceUnit (独立行, FK→IdeaDelta)                          │  │
│  │ ImplementationUnit (file/class/function/shape_trace)         │  │
│  │ 置信度分级: code_verified > experiment_backed > text_stated  │  │
│  │              > inferred > speculative                        │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │ Idea Layer (Layer 3) — 核心主对象                             │  │
│  │ IdeaDelta: delta_statement + changed_slots + 4维评分          │  │
│  │ GraphEdge: 12种边 + assertion_source + confidence             │  │
│  │ 硬约束: evidence_count >= 2 才能发布                          │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │ Canonical Domain (Layer 2)                                   │  │
│  │ ParadigmFrame (4域: RL/VLM/Agent/MotionGen)                  │  │
│  │ Slot (25个独立类型化槽位)                                     │  │
│  │ MechanismFamily (19个, 层级结构)                              │  │
│  │ Bottleneck (研究瓶颈, 带 embedding)                          │  │
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
│ 23 张表          │  │ papers/raw-pdf/     │
│ (事实源)         │  │ reports/            │
└──────────────────┘  └─────────────────────┘
```

---

## 2. 数量统计

| 组件 | 数量 | 明细 |
|------|------|------|
| **数据库表** | 23 | 4次 Alembic 迁移 |
| **API 端点** | 42 | 10 个 router |
| **MCP 工具** | 10 | 高层研究操作 |
| **前端页面** | 7+layout | Dashboard/Papers/Import/Search/Reports/Digests/Directions |
| **后端 Service** | 17 | 58 个公开函数 |
| **后台任务** | 5 task + 3 cron | arq worker |
| **Docker 服务** | 6 | postgres/redis/api/worker/frontend/caddy |
| **枚举类型** | 9 | 45 个枚举值 |
| **ParadigmFrame** | 4 | RL/VLM/Agent/MotionGen |
| **Slot** | 25 | 按领域类型化 |
| **MechanismFamily** | 19 | 层级分类 |

---

## 3. 知识图谱五层架构

### Layer 0: Asset Layer
存原件，不做结论。对象：Paper/PDF/HTML/Repo/Dataset/Supplementary。
已有 `paper_assets` 表 (8 种 asset_type)。

### Layer 1: Scholarly Backbone
回答"这是什么东西，它和谁相连"。
- 节点：Paper (60+列), Author, Venue, Topic
- 边：cites (via graph_edges)
- OpenAlex 对接准备：`papers.openalex_id` 字段
- `papers.role_in_kb`: foundational/baseline/extension/negative/patch

### Layer 2: Canonical Domain Layer
**知识库质量的核心**。不从论文自由生长实体，先定义 ParadigmFrame。

4 个领域范式：
```
RL:            rollout → reward → credit_assignment → policy_update → value_baseline → exploration → planner_memory
VLM:           vision_encoder → projector → language_core → objective → data_mixture → inference_planner
Agent:         perception → planning → action → memory → tool_use → reflection
MotionGen:     motion_tokenizer → denoiser → conditioning → objective → sampling → physics_sim
```

每个 Slot 有：name, description, slot_type (architecture/objective/data/inference), is_required。

19 个 MechanismFamily 按域分组：Generative(6) / RL(4) / Architecture(3) / Alignment(2) / Agent(4)。

### Layer 3: Idea Layer — 核心主对象

**IdeaDelta** 字段：
- delta_statement — 一句话描述改了什么
- changed_slots — [{slot_name, from, to, change_type}]
- 4 维评分：structurality / transferability / local_keyness / field_keyness
- evidence_count + publish_status (draft → auto_published → human_verified)

**GraphEdge** — 统一边表，12 种核心边：
```
cites, has_code, targets_bottleneck, changes_slot,
instance_of_mechanism, supported_by, implemented_by,
same_mechanism_as, patch_of, structural_variant_of,
contradicts, transferable_to
```

每条边必须带 `assertion_source`：asserted_by_paper / inferred_by_system / verified_by_human。

### Layer 4: Evidence & Implementation Layer

**EvidenceUnit** — 独立 DB 行（不再是 JSONB），FK→IdeaDelta：
- atom_type: mechanism/evidence/boundary/transfer/formula/table/figure
- confidence (0-1) + basis (EvidenceBasis 5级)
- source_section + source_page + source_quote

**ImplementationUnit** — 代码锚点：
- repo_url / file_path / class_or_function
- config_snippet / shape_trace [{layer, input_shape, output_shape}]

### Layer 5: Serving Views
不是新知识，是投影视图：ReadingPath, Digest, ContrastView, DirectionCard, Report。

---

## 4. 分析管线（含图谱构建）

```
paper → L1 (metadata, API补全, 0T)
     → L2 (parse, pymupdf, 0T)
     → L3 (skim, LLM ~2K tokens)
     → L4 (deep, LLM ~10-20K tokens)
           ↓
     → frame_assign (匹配 ParadigmFrame)
     → idea_extract (生成 IdeaDelta)
     → evidence_persist (EvidenceUnit 独立存储)
     → edge_create (GraphEdge: supported_by, changes_slot, ...)
     → publish_check (evidence_count >= 2 → auto_published)
```

---

## 5. 检索架构（5 路 Query Router）

| 路由 | 查询类型 | 数据路径 |
|------|---------|----------|
| 1. 事实/引用 | "谁提的""谁引了谁" | graph_edges(cites) + SQL |
| 2. 瓶颈 | "本质瓶颈有哪些" | Bottleneck → IdeaDelta → structurality |
| 3. 机制 | "diffusion 有哪些路线" | MechanismFamily → IdeaDelta → Evidence |
| 4. 迁移 | "RL insight 能否迁到 VLM" | graph_edges(transferable_to) |
| 5. 综述 | "最近三个月结构性进展" | IdeaDelta 聚合 + evidence → LLM synthesis |

基础层保留：keyword(tsvector) + semantic(pgvector) + structured SQL。

---

## 6. 6 条硬约束

| # | 约束 | 实现 |
|---|------|------|
| 1 | IdeaDelta 无证据不发布 | publish_status gate: evidence_count >= 2 |
| 2 | 边必须区分来源 | graph_edges.assertion_source 字段 |
| 3 | paper→paper 只有 cites | edge_type 约束 |
| 4 | 强做 entity resolution | 别名归一 + 近义归并 (待完善) |
| 5 | 多模态证据进图 | EvidenceUnit.atom_type 含 formula/table/figure |
| 6 | 高价值节点 HITL | publish_status + confidence 阈值 |

---

## 7. 提取质量保障（6 层）

```
第6层: 评测集回放 + 人工反馈闭环
第5层: 置信度分级 (EvidenceBasis 5级)
第4层: 三角色复核 (extractor → auditor → taxonomy_reviewer)
第3层: 标准范式对齐 (Canonical Delta Card)
第2层: 先规则解析再 LLM
第1层: 四级分析管线 L1→L4 分步晋升
```

---

## 8. 库外输入状态机

```
ephemeral_received → canonicalized → enriched → l3_skimmed
    │                                               │
    │ 30天未操作                    用户点"加入知识库"
    ▼                                               ▼
archived_or_expired                          wait (正式管线)
```

---

## 9. 部署架构（阿里云轻量 4C8G）

| 服务 | 内存 | 说明 |
|------|------|------|
| PostgreSQL + pgvector | 1280 MB | 23 张表 + 向量索引 |
| Redis | 256 MB | 任务队列 + 缓存 |
| FastAPI (2 workers) | 400 MB | 42 个 API 端点 |
| arq Worker | 1024 MB | 5 task + 3 cron |
| Next.js Frontend | 256 MB | 7 页 standalone |
| Caddy | ~50 MB | HTTPS 反向代理 |

---

## 10. 技术选型

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
