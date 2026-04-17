# ResearchFlow Architecture v3.1

## 1. 一句话定义

**PostgreSQL 是唯一真相源。DeltaCard 是中间真相层。一切 UI/导出/Agent 都是投影。**

```
                        ┌─────────────────────┐
                        │    用户界面 (投影)     │
                        │ Web │ MCP │ Obsidian  │
                        └──────────┬──────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│                        Core Backend                                 │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐ │
│  │ 81 API 路由  │  │ 18 MCP 工具  │  │ 25 Service 模块            │ │
│  └─────────────┘  └──────────────┘  └────────────────────────────┘ │
│                                                                     │
│  PostgreSQL (31 表) + pgvector + Redis + 对象存储                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 知识库结构：不是论文列表，是方法演化图谱

### 2.1 核心数据模型

```
Paper (容器)
  └─→ DeltaCard (中��真相层: 这篇论文改了��么)
        ├─→ IdeaDelta (知识原子: 可复用的改进描述)
        │     ├─→ EvidenceUnit (证据: 实验结果/代码验证/推理)
        │     └─→ GraphAssertion (图谱边: supported_by/changes_slot/...)
        ├─→ parent_delta_card_ids (DAG 继承: 基于哪些方法改进的)
        └─→ ProjectBottleneck (解决了什么瓶颈)

ParadigmTemplate (领域标准范式: RL/VLM/Agent/MotionGen/...)
  └─→ Slot (范式中的可替换组件: denoiser/reward/encoder/...)

MechanismFamily (机制族: diffusion/flow_matching/RL/...)
  └─→ Alias (别名: DDPM → diffusion, FM → flow_matching)
```

### 2.2 方法演化 DAG (核心创新)

论文之间不是扁平列表，而是 **有向无环图 (DAG)**:

```
GRPO (depth=0, baseline=true, downstream=7)
├── GRPO+LP (depth=1, parent=[GRPO])
│   ├── GRPO-LP+sampling (depth=2, parent=[GRPO+LP])
│   └── GRPO-LP+KL (depth=2, parent=[GRPO+LP])
├── GRPO+tree (depth=1, parent=[GRPO])
└── GDPO (depth=1, parent=[GRPO, DPO])  ← 多继承: 两个范式组合
    └─�� GDPO+image_thinking (depth=2, parent=[GDPO])
```

**自动升级规则**:
- 当一个改进被 ≥3 篇论文用作 baseline → `is_established_baseline = true`
- 当它还具有结构性 (`structurality_score ≥ 0.6`) → 候选新范式版本
- `promote_to_paradigm()` 创建 `rl_standard_v2` 替代 `rl_standard_v1`

### 2.3 论文过滤评分

每篇论文入库后自动计算 4 个分数:

| 分数 | 权重因素 | 用途 |
|------|---------|------|
| **keep_score** | Tier(开数据>开代码>中稿>预印本) + 顶会 + 重要度 + 时间衰减 | 是否值得入库 |
| **analysis_priority** | 重要度 + 开源资产 + 有PDF + 顶会 + 新鲜度 | 分析优先级排序 |
| **structurality_score** | 关键词信号 + L4分析的 method_category | 结构性改进 vs 插件 |
| **extensionability_score** | 跨域关键词 + 多任务标签 + 开源资产 | 可扩展性 |

**方法分类标签** (L4 自动提取):
```
method/structural_architecture    — 改了核心架构
method/plugin_module             — 加了一个模块
method/reward_design             — 改了奖励函数
method/training_recipe           — 改了训练方法
method/representation_change     — 改了表示方式
improvement/fundamental_rethink  — 根本性重新思考
improvement/additive_plugin      — 加插件
improvement/component_replacement — 替换核心组件
```

---

## 3. 完整流程: 从零到知识图谱

### 3.1 冷启动: 给一个领域，从零构建

```
用户: "我要研究 RLHF for VLM"
         │
         ▼
[Step 1] POST /pipeline/init-domain {"domain": "RLHF VLM"}
         │  GitHub API 搜 awesome 仓库 → 解析 README → 提取论文链接
         │  → 导入 72 篇 → triage 评分 → 按优先级排队
         ▼
[Step 2] POST /pipeline/batch?limit=10
         │  对最高优先��的 10 篇依次:
         │  下载PDF → 元数据补全 → L2解析 → L3速读 → L4深度分析
         │  → DeltaCard构建 → IdeaDelta派生 → GraphAssertion提议
         ▼
[Step 3] 自动产出:
         ├── 10 张 DeltaCard (每篇论文的结构化改动)
         ├── 10 个 IdeaDelta (可复用知识原子)
         ├── 30+ 条 GraphAssertion (图谱边)
         ├── 方法分类: 3 structural + 5 plugin + 2 reward
         ├── 3 个 ProjectBottleneck (自动从 L4 提取)
         └── 范式: 如果领域没有现成��式，LLM 动态发现并创建
```

### 3.2 单篇论文完整管线 (16 步)

```
ingest → triage → download_pdf → enrich (arXiv/Crossref)
→ parse_L2 (pymupdf) → skim_L3 (LLM) → deep_L4 (LLM)
→ delta_card_build → link_to_parent_baselines
→ entity_resolution → assertion_propose → evidence_audit
→ review → publish → index → export
```

### 3.3 研究探索: 多跳认知迭代

```
POST /explore/start {"query": "RL advantage disappearance"}
         │
POST /explore/{id}/search {"query": "not plugin, fundamental"}
         │  → 搜索 + 自动分类: structural=1, plugin=6
         │  → gap分析: "缺少根本性改进，试试相邻领域"
         │
POST /explore/{id}/step {"step_type": "pivot",
         │                "rejected_reason": "都是插件型"}
         │  → 记录pivot + 建议: "seek_fundamental"
         │
POST /explore/{id}/search {"query": "think with image agentic GDPO"}
         │  → 继续探索，系统记住拒绝模式
         │
GET /explore/{id}
         └→ 完整路径: initial → refine → pivot → broaden
            + 论文按 method/ 分类 + 下一步建议
```

---

## 4. 图谱断言模型

### 4.1 GraphAssertion 生命周期

```
candidate ──→ published ──→ deprecated/superseded
    │              │
    └→ rejected    └→ (被新版本替代)
```

| 边类型 | 自动发布? | 说明 |
|--------|----------|------|
| supported_by | 是 | IdeaDelta ← Evidence |
| changes_slot | 是 | IdeaDelta → Slot |
| instance_of_mechanism | 是 | IdeaDelta → MechanismFamily |
| targets_bottleneck | 是 | IdeaDelta → Bottleneck |
| builds_on | 是 | DeltaCard → parent DeltaCard |
| contradicts | **否 (需审核)** | 两个方法矛盾 |
| transferable_to | **否** | 跨域迁移 |
| patch_of | **否** | 一个方法是另一个的插件 |

### 4.2 发布门控

| 对象 | 发布条件 |
|------|---------|
| DeltaCard | frame_id + changed_slots + evidence_refs ≥ 2 |
| IdeaDelta | evidence_count ≥ 2 + min(confidence) ≥ 0.85 |
| 高价值断言 | 需要 ReviewTask 审核通��� |

---

## 5. 领域范式动态发现

```python
assign_paradigm(category, tags, title, abstract)
  1. 静态映射: category/tags → 已知范式 (4 个内置)
  2. 模糊匹配: DB 中已有范式的 domain 相似度
  3. LLM 动态发现: 给 abstract → LLM 识别领域 + 生成 slots + bottleneck
     → 自动创建 ParadigmTemplate + Slot 行
     → 后续同领域论文复用
```

内置范式:
```
RL:            rollout → reward → credit_assignment → policy_update → exploration → planner
VLM:           vision_encoder → projector → language_core → objective → data_mixture
Agent:         perception → planning → action → memory → tool_use → reflection
MotionGen:     motion_tokenizer → denoiser → conditioning → objective → sampling
```

---

## 6. 数据库 Schema 总览 (31 表, 7 次迁移)

### 核心表

| 表 | 行数概念 | 说明 |
|----|---------|------|
| papers | 每篇论文 1 行 | 60+ 列: 元数据 + 评分 + 状态 |
| delta_cards | 每篇论文 1 行 | 中间真相层 + DAG 继承字段 |
| idea_deltas | 每篇论文 1 行 | 可复用知识原子 |
| evidence_units | 每篇 2-5 行 | 证据单元 (实验/代码/推理) |
| graph_assertions | 每篇 4-8 行 | 图谱边 + 生命周期 |
| graph_nodes | 每个实体 1 行 | 统一节点注册 |
| paradigm_templates | 每个领域 1-3 行 | 范式模板 + 版本演化 |
| slots | 每范式 4-8 行 | 可替换组件 |
| mechanism_families | ~20 行 | 机制族 (层级结构) |
| project_bottlenecks | 每瓶颈 1 行 | 研究瓶颈 (L4 自动提取) |
| review_tasks | 待审核项 | 审核队列 |
| aliases | 别名映射 | 实体归一 |

### 支��表

paper_analyses, paper_assets, paper_versions, method_deltas (legacy),
graph_edges (legacy), implementation_units, transfer_atoms, search_sessions,
reading_plans, direction_cards, digests, jobs, model_runs, execution_memories,
user_feedback, user_bookmarks, user_events, human_overrides, graph_assertion_evidence

---

## 7. API 路由总览 (81 路由, 13 Router)

| Router | 前缀 | 关键端点 |
|--------|------|---------|
| pipeline | /pipeline | `run`, `batch`, `init-domain`, `discover`, `build-domain`, `lineage`, `evolution` |
| explore | /explore | `start`, `step`, `search`, `summary` |
| assertions | /assertions | CRUD + `reviews/*` + `overrides` + `aliases` |
| graph | /graph | `stats`, `quality`, `ideas`, `paradigms`, `mechanisms` |
| search | /search | `hybrid`, `ideas`, `bottlenecks`, `mechanisms`, `transfers` |
| papers | /papers | CRUD + 列表 + 过滤 |
| import | /import | `links`, `pdf`, `parse` |
| analyses | /analyses | `skim`, `deep`, `batch` |
| reports | /reports | `generate` (quick/briefing/deep) |
| digests | /digests | day/week/month 摘要 |
| directions | /directions | 方向提议 + 展开 |
| feedback | /feedback | 纠错/确认/标签修改 |
| health | / | 健康检查 |

---

## 8. 技术栈

| 组件 | 选型 |
|------|------|
| Web框架 | FastAPI (async) |
| 前端 | Next.js 15 + Tailwind |
| ORM | SQLAlchemy 2.0 (async) |
| 数据库 | PostgreSQL 16 + pgvector |
| 任务队列 | arq (Redis) |
| PDF解析 | pymupdf |
| LLM | Anthropic Claude / OpenAI / mock |
| MCP | Python MCP SDK |
| 论文发现 | Semantic Scholar API (免费) |
| 领域初始化 | GitHub Search API → awesome 仓库解析 |
| 部署 | Docker Compose + Caddy |
