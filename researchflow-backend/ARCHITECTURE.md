# ResearchFlow Architecture v4.0

## 1. 一句话定义

**PostgreSQL 是唯一真相源。DeltaCard 是不可变中间真相层。一切 UI/导出/Agent 都是投影。**

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
│  │ 99 API 路由  │  │ 22 MCP 工具  │  │ 34 Service 模块            │ │
│  └─────────────┘  └──────────────┘  └────────────────────────────┘ │
│                                                                     │
│  PostgreSQL (42 表 + 4 物化视图) + pgvector + Redis + 对象存储       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 知识库结构：不是论文列表，是方法演化图谱

### 2.1 核心数据模型

```
Paper (容器, current_delta_card_id 指向当前发布版)
  └─→ DeltaCard (不可变快照, append-only, 带 analysis_run_id/source_asset_hash)
        ├─→ IdeaDelta (知识原子: 可复用的改进描述)
        │     ├─→ EvidenceUnit (证据: 实验结果/代码验证/推理)
        │     ├─→ GraphAssertion (图谱边: supported_by/changes_slot/...)
        │     └─→ ContributionToCanonicalIdea (映射到跨论文归一概念)
        ├─→ DeltaCardLineage (独立 DAG 继承表: builds_on/extends/replaces)
        └─→ PaperBottleneckClaim (论文声称解决的瓶颈)

ProjectBottleneck (全局瓶颈本体)
  ├─→ PaperBottleneckClaim (论文级事实)
  └─→ ProjectFocusBottleneck (项目级决策, 带负约束)

CanonicalIdea (跨论文归一概念层)
  └─→ ContributionToCanonicalIdea (N 个贡献 → M 个概念)

ParadigmTemplate / Slot / MechanismFamily (领域本体)
  └─→ ParadigmCandidate / SlotCandidate / MechanismCandidate (候选层, 需审核)

TaxonomyVersion (所有 ontology 变更的版本快照)
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

**v3.2 升级规则 (candidate → review → publish)**:
- 当一个改进被 ≥3 篇论文用作 baseline → `is_established_baseline = true`
- 当它还具有结构性 (`structurality_score ≥ 0.6`) → 候选新范式版本
- `builds_on` 边默认为 `candidate` 状态，需审核后发布
- 自动发现的范式创建 `ParadigmCandidate`，不直接创建 `ParadigmTemplate`
- 通过 `POST /reviews/candidates/paradigms/{id}/promote` 审核后升级
- 所有 ontology 变更记录在 `taxonomy_versions` 表中

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
[Step 2] POST /pipeline/batch?limit=15
         │  对所有待处理论文依次:
         │  triage → download → enrich → L2 → L3 → L4 (6步) → graph
         ▼
[Step 3] POST /pipeline/export/obsidian-vault
         │  → 生成 5 类笔记的 Obsidian vault
         ▼
[Step 4] rsync 同步到本地 Obsidian
```

### 3.2 外层管线: 单篇论文从入库到图谱

入口: `pipeline_service.run_full_pipeline(paper_id)`

每步已完成则自动跳过，任何一步失败不阻断后续步骤:

```
POST /pipeline/{paper_id}/run  (或 /pipeline/batch?limit=N)
 │
 ├─ Step 0: triage_paper()                          [triage_service]
 │    计算 keep_score / analysis_priority / tier
 │    基于: venue + open_code + open_data + 时间衰减
 │
 ├─ Step 1: download_arxiv_pdf()                    [pipeline_service]
 │    有 arxiv_id 且没下过 → 下载 PDF 到本地 + 对象存储
 │
 ├─ Step 2: enrich_paper()                          [enrich_service]
 │    缺 abstract/authors → 查 arXiv API / Crossref 补全
 │
 ├─ Step 3: parse_paper_pdf()                       [parse_service]
 │    pymupdf 提取 PDF 章节文本 → paper_analyses (level=l2_parse)
 │    产出: extracted_sections {intro, method, results, conclusion, ...}
 │
 ├─ Step 4: skim_paper()                            [analysis_service]
 │    1 次 LLM 调用 (abstract + 关键章节, ≤8K tokens)
 │    产出: problem_summary, changed_slots, is_plugin_patch, worth_deep_read
 │    → paper.state = l3_skimmed
 │
 └─ Step 5: deep_analyze_paper()                    [analysis_service]
      L4 深度分析 — 6 步管线 (见下方详解)
      → paper.state = l4_deep
```

### 3.3 L4 深度分析: 6 步管线 (核心)

入口: `analysis_service.deep_analyze_paper(paper_id)`

旧版用 1 次 LLM 调用输出 23 个字段；v4.0 拆成 6 个独立步骤，各自可重试:

```
deep_analyze_paper(paper_id)
 │
 │  从 L2 提取的章节文本拼接全文 (≤25K tokens)
 │
 ▼
┌─ Step 1: extract_evidence ──────────────────────────────────────┐
│  文件: analysis_steps.py → run_step1_extract_evidence()          │
│  LLM 调用 #1 (独立 prompt, ≤3000 output tokens)                  │
│                                                                  │
│  ★ 防线 #1: 强制阅读顺序                                         │
│    Prompt: "FIRST read Method → THEN Experiments → ONLY THEN     │
│    Abstract. Do NOT take the abstract at face value."             │
│                                                                  │
│  输出 JSON:                                                      │
│    key_equations[]:    核心公式 (≤4), 含 slot_affected            │
│    key_figures[]:      关键图表 (≤4), 含 evidence_for             │
│    evidence_units[]:   证据锚点 (3-8个), 每个含:                   │
│      atom_type, claim, confidence, basis, source_section          │
│    narrative_vs_substance: abstract 是否与实验吻合                  │
│    baseline_fairness:  baseline 是否最强? 是否故意漏比?             │
│    paper_type:         method/survey/benchmark/position/...       │
│                                                                  │
│  失败 → 自动重试 (最多 2 次), 检查 required_fields                │
└──────────────────────────────────────────────────────────────────┘
 │ step1_data
 ▼
┌─ Step 2: build_delta_card ──────────────────────────────────────┐
│  文件: analysis_steps.py → run_step2_build_delta()               │
│  LLM 调用 #2 (独立 prompt, ≤4096 output tokens)                  │
│  输入: 全文 + Step 1 证据 JSON 作为 grounding                      │
│                                                                  │
│  ★ 关键: Step 1 的证据防止 Step 2 hallucinate                     │
│                                                                  │
│  输出 JSON:                                                      │
│    problem_summary:    问题与挑战 (200-400 字, 中文)               │
│    method_summary:     方法与洞察 (400-600 字, 中文)               │
│    evidence_summary:   证据与局限 (200-400 字, 中文)               │
│    core_intuition:     核心直觉 (100-200 字)                      │
│    changed_slots[]:    改了哪些 slot                              │
│    unchanged_slots[]:  没改的 slot                                │
│    structurality_score: 0.0-1.0  ← ★ 决定 A/B/C/D 分级           │
│    delta_card:         {paradigm, slots, is_structural}           │
│    bottleneck_addressed: {title, description, is_fundamental}     │
│    same_family_method: 方法族名称                                  │
│    confidence_notes[]: 逐条置信度分析                               │
│                                                                  │
│  失败 → 自动重试 (最多 2 次)                                       │
└──────────────────────────────────────────────────────────────────┘
 │ merge_step_outputs(step1, step2) → analysis_data
 ▼
┌─ 持久化 ────────────────────────────────────────────────────────┐
│  保存 PaperAnalysis (level=l4_deep, schema_version=v2)           │
│  保存 MethodDelta (legacy 兼容)                                   │
│  更新 paper.state → l4_deep                                      │
│  更新 paper.structurality_score ← LLM 输出 (决定 A/B/C/D)        │
│  更新 paper.tags ← method/*, improvement/*                       │
│                                                                  │
│  _maybe_create_bottleneck()                                      │
│    → 查 DB 是否已有同名瓶颈 → 有则关联, 无则创建                    │
│    → 创建 PaperBottleneckClaim (论文声称解决此瓶颈)                  │
│                                                                  │
│  _build_idea_graph()                                             │
│    → assign_paradigm(): 匹配范式 (静态→模糊→LLM发现)               │
│    → run_delta_card_pipeline():                                  │
│       build_delta_card → persist_evidence → derive_idea_delta     │
│       → propose_assertions → check_and_publish                   │
│       门控: evidence_refs ≥ 2 → DeltaCard 发布                    │
│       门控: min(confidence) ≥ 0.85 → IdeaDelta 自动发布            │
│    → 更新 paper.current_delta_card_id                             │
└──────────────────────────────────────────────────────────────────┘
 │ 以下每步独立 try/except + session.rollback(), 互不阻断
 ▼
┌─ Step 3: build_compare_set ─────────────────────────────────────┐
│  文件: baseline_comparator_service.py                             │
│  纯 DB 查询, 无 LLM 调用                                          │
│                                                                  │
│  ★ 防线 #2: "比较集不是论文自己说了算"                               │
│  4 个来源:                                                        │
│    ① domain_baseline:  同范式已确立基线 (is_established_baseline)   │
│    ② same_mechanism:   同 mechanism_family 的论文                  │
│    ③ strong_peer:      同 category 同时期高 structurality 论文      │
│    ④ self_reported:    论文自述的 baseline_paper_titles             │
│                                                                  │
│  → 更新 DeltaCard.baseline_paper_ids                               │
└──────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─ Step 4: propose_lineage ───────────────────────────────────────┐
│  文件: evolution_service.py → link_to_parent_baselines()          │
│  纯 DB 查询 + 写入                                                │
│                                                                  │
│  查找父节点策略:                                                    │
│    ① 同 paradigm name 已发布的 DeltaCard                           │
│    ② 同 mechanism_family 的 DeltaCard                              │
│    ③ 同 frame 已确立基线                                            │
│                                                                  │
│  → 创建 DeltaCardLineage (status=candidate, relation=builds_on)   │
│  → 创建 builds_on GraphAssertion (candidate, 需审核)                │
│  → 更新 parent.downstream_count += 1                               │
│  → downstream ≥ 3 → parent.is_established_baseline = true          │
│  → 创建 ReviewTask (自动审核任务)                                    │
└──────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─ Step 5: synthesize_concept ────────────────────────────────────┐
│  文件: concept_synthesizer_service.py                             │
│  纯 DB 查询 + 写入                                                │
│                                                                  │
│  ① same_family_method → 找/建 MechanismFamily                     │
│     精确匹配 → 别名匹配 → 新建                                      │
│  ② core_intuition → 找/建 CanonicalIdea                           │
│     精确匹配 → 同 mechanism 匹配 → 新建                              │
│  ③ 创建 ContributionToCanonicalIdea 链接                           │
│     contribution_type: origin(≥0.7) / extension(≥0.4) / instance  │
│  → paper.mechanism_family = 方法族名称                               │
└──────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─ Step 6: reconcile_neighbors ───────────────────────────────────┐
│  文件: incremental_reconciler_service.py                          │
│                                                                  │
│  ① refresh_connections(): 更新 same_family_paper_ids               │
│  ② 将新论文加入邻居 DeltaCard 的 same_family 列表                    │
│  ③ structurality ≥ 0.6 → 检查 baseline 候选                       │
└──────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─ 自动导出 ──────────────────────────────────────────────────────┐
│  export_paper_analysis() → paperAnalysis/{category}/{venue}/X.md  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.4 三道防线

| # | 防线 | 机制 | 对应步骤 |
|---|------|------|---------|
| 1 | **先看改动不先听故事** | Step 1 prompt 强制: FIRST Method → THEN Experiments → ONLY THEN Abstract | Step 1 |
| 2 | **比较集不是论文自己说了算** | 从 DB 查 4 个来源自动补齐比较集, 不只用论文自述 | Step 3 |
| 3 | **高价值结论必须有证据锚点** | DeltaCard 发布门控: evidence_refs ≥ 2 | 持久化阶段 |

### 3.5 容错机制

- **Step 1+2 (LLM)**: 缺 required_fields → 自动重试 (最多 2 次), 仍缺则用 partial data 继续
- **Step 3-6 (DB)**: 每步独立 `try/except + session.rollback()`, 单步失败不阻断后续步骤
- **LLM 返回非 JSON**: 自动去 markdown fence → 找 `{...}` → retry
- **Session 级容错**: DB 错误后 rollback 恢复 session, 后续步骤可继续

### 3.6 Service 文件对照

| 文件 | 职责 | 调用方 |
|------|------|--------|
| `pipeline_service.py` | 外层编排 (triage→download→enrich→L2→L3→L4) | API `/pipeline/*` |
| `analysis_service.py` | L3 skim + L4 deep 6步编排器 | pipeline_service |
| `analysis_steps.py` | Step 1+2 的 prompt + JSON 解析 + retry + merge | analysis_service |
| `baseline_comparator_service.py` | Step 3: 4源比较集 | analysis_service |
| `evolution_service.py` | Step 4: 方法演化 DAG + baseline 晋升 | analysis_service |
| `concept_synthesizer_service.py` | Step 5: MechanismFamily + CanonicalIdea | analysis_service |
| `incremental_reconciler_service.py` | Step 6: 反向更新邻居 | analysis_service |
| `delta_card_service.py` | DeltaCard → Evidence → IdeaDelta → Assertions | analysis_service |
| `frame_assign_service.py` | 范式匹配 (静态→模糊→LLM发现) | delta_card_service |
| `export_service.py` | Obsidian vault 导出 (5类笔记) | API `/pipeline/export/*` |

### 3.7 研究探索: 多跳认知迭代

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

## 6. 数据库 Schema 总览 (42 表 + 4 物化视图, 11 次迁移)

### 核心表

| 表 | 行数概念 | 说明 |
|----|---------|------|
| papers | 每篇论文 1 行 | 60+ 列: 元数据 + 评分 + 状态 + current_delta_card_id |
| delta_cards | 每篇论文 N 行 (append-only) | 不可变快照 + analysis_run_id/source_asset_hash |
| idea_deltas | 每篇论文 1+ 行 | 可复用知识原子 |
| evidence_units | 每篇 2-5 行 | 证据单元 (实验/代码/推理) |
| graph_assertions | 每篇 4-8 行 | 图谱边 + 生命周期 |
| graph_nodes | 每个实体 1 行 | 统一节点注册 |
| paradigm_templates | 每个领域 1-3 行 | 范式模板 + 版本演化 |
| slots | 每范式 4-8 行 | 可替换组件 |
| mechanism_families | ~20 行 | 机制族 (层级结构) |
| project_bottlenecks | 每瓶颈 1 行 | 全局瓶颈本体 |
| review_tasks | 待审核项 | 审核队列 (自动 + 人工) |
| aliases | 别名映射 | 实体归一 |
| **paper_bottleneck_claims** | 每篇 0-3 行 | **v3.2** 论文级瓶颈声称 |
| **project_focus_bottlenecks** | 每项目 N 行 | **v3.2** 项目级关注瓶颈 (带负约束) |
| **canonical_ideas** | 跨论文概念 | **v3.2** 归一概念层 |
| **contribution_to_canonical_idea** | 1:N 映射 | **v3.2** 论文贡献→概念 |
| **delta_card_lineage** | 每对 1 行 | **v3.2** 独立演化 DAG (candidate 默认) |
| **paradigm_candidates** | 候选项 | **v3.2** 候选范式 (需审核) |
| **slot_candidates** | 候选项 | **v3.2** 候选槽位 |
| **mechanism_candidates** | 候选项 | **v3.2** 候选机制族 |
| **taxonomy_versions** | 每次变更 1 行 | **v3.2** ontology 变更快照 |
| **search_branches** | 探索分支 | **v3.2** 搜索会话中的分支决策 |
| **render_artifacts** | 输出制品 | **v3.2** 报告/摘要/导出追踪 |

### 支撑表

paper_analyses, paper_assets, paper_versions, method_deltas (legacy),
graph_edges (legacy), implementation_units, transfer_atoms, search_sessions,
reading_plans, direction_cards, digests, jobs, model_runs, execution_memories,
user_feedback, user_bookmarks, user_events, human_overrides, graph_assertion_evidence

### 物化视图 (CQRS-lite)

| 视图 | 说明 | 刷新 |
|------|------|------|
| paper_search_docs | 论文 + DeltaCard + 证据数 去规范化 | `POST /search/refresh-views` |
| idea_search_docs | IdeaDelta + 论文 + DeltaCard 去规范化 | 同上 |
| lineage_view | 方法演化 DAG + 论文标题展平 | 同上 |
| review_queue_view | 待审核项 + 目标摘要 | 同上 |

---

## 7. API 路由总览 (99 路由, 14 Router)

| Router | 前缀 | 关键端点 |
|--------|------|---------|
| pipeline | /pipeline | `run`, `batch`, `init-domain`, `discover`, `build-domain`, `lineage`, `evolution` |
| explore | /explore | `start`, `step`, `search`, `summary` |
| assertions | /assertions | CRUD + `aliases` |
| graph | /graph | `stats`, `quality`, `ideas`, `paradigms`, `mechanisms` |
| search | /search | `hybrid`, `ideas`, `bottlenecks`, `mechanisms`, `transfers`, **`query`** (意图路由), **`refresh-views`** |
| **reviews** | /reviews | **v3.2** 队列CRUD, `approve`, `reject`, `assign`, `override`, `candidates/paradigms`, `candidates/lineage` |
| papers | /papers | CRUD + 列表 + 过滤 |
| import | /import | `links`, `pdf`, `parse` |
| analyses | /analyses | `skim`, `deep`, `batch` |
| reports | /reports | `generate` (quick/briefing/deep) |
| digests | /digests | day/week/month 摘要 |
| directions | /directions | 方向提议 + 展开 |
| feedback | /feedback | 纠错/确认/标签修改 |
| health | / | 健康检查 |

---

## 8. Obsidian Vault 导出 (v4.0)

触发: `POST /pipeline/export/obsidian-vault`

导出前自动清理旧文件 (清理子项, 不删根目录 — 兼容 Docker 挂载)。

### 5 类笔记

| 类型 | 前缀 | 目录 | 正文 wikilinks |
|------|------|------|---------------|
| Paper | `P__` | `40_Papers/{A/B/C/D}__*/` | 6-8 个 (1-2 baseline + 1 concept + 1 bottleneck + 1 lineage + 2 same-family) |
| Concept | `C__` | `20_Concepts/` | 不限 (代表论文对比表) |
| Bottleneck | `B__` | `30_Bottlenecks/` | 不限 (结构性/插件型解法分层) |
| Lineage | `L__` | `10_Lineages/` | 不限 (ASCII 演化树) |
| Overview | — | `00_Home/` | 纯导航 |

### 论文分级 A/B/C/D

分级由 `_paper_level(p)` 函数决定，优先级: **ring 字段 → dc.structurality_score → paper.structurality_score**

| 等级 | 目录 | 分数条件 | 含义 | 示例 |
|------|------|---------|------|------|
| **A** | `A__Baselines` | ring=baseline 或 score ≥ 0.7 | 必读 baseline，建立标准框架 | DPO, InstructGPT, KTO |
| **B** | `B__Structural` | ring=structural 或 score ≥ 0.5 | 结构性改进，改了核心 slot | GRPO, ORPO, SPIN |
| **C** | `C__Plugins` | ring=plugin 或 score ≥ 0.3 | 插件型改进，加模块/改 loss | SimPO, RAFT, RRHF |
| **D** | `D__Peripheral` | 其余 (score < 0.3 或无分析) | 外围参考 | — |

**分数来源**: `structurality_score` 由 L4 Step 2 的 LLM 输出。Prompt 明确要求:
> "structurality_score: 0.0 (pure plugin/trick) to 1.0 (fundamental rethink).
> Most papers should be 0.2-0.5. Reserve 0.7+ for truly structural work."

**在 Obsidian 中查看分级**:
1. **目录结构** — `40_Papers/A__Baselines/`、`B__Structural/`、`C__Plugins/`
2. **Paper frontmatter** — `paper_level: A`
3. **90_Views/papers_by_structurality.md** — 按结构性降序的完整表格
4. **00_Home/01_阅读顺序.md** — 按 A→B→C 分层推荐阅读
5. **00_Home/00_方向总览.md** — 论文分布统计 (A: 5, B: 5, C: 5)

### Paper Note 模板

每篇论文的 Obsidian 页面结构:

```yaml
---
title: "Direct Preference Optimization..."
type: paper
paper_id: P__DPO
paper_level: A                    # ← A/B/C/D 分级
frame: rl_standard                # ← 范式 (仅属性, 不做 wikilink)
changed_slots: [loss_function, training_pipeline]
structurality_score: 0.8
concepts: ["[[C__direct_preference_optimization]]"]
bottleneck: ["[[B__RLHF训练的复杂性...]]"]
lineage: ["[[L__DPO_Family]]"]    # ← 如有 lineage
same_family_papers: ["[[P__SimPO]]", "[[P__KTO]]"]
---

# 一眼看懂
> 基于 [[P__InstructGPT]]，改了 `loss_function`, `training_pipeline`，
> 属于 [[C__direct_preference_optimization]]，
> 目标是缓解 [[B__RLHF训练的复杂性...]]

## 相对 baseline 改了什么
| 相比 | 改动 slot | 收益 | 代价 |
|------|---------|------|------|
| [[P__InstructGPT]] | loss_function | ... | ... |

## 关键公式
## 关键图表
## 同类型工作 (max 2 links)
## 在主线中的位置 (1 lineage link)
## 阅读建议           ← 按 A/B/C/D 等级给不同建议
## 详细分析
```

**正文 wikilink 预算**: 1-2 baseline + 1 concept + 1 bottleneck + 1 lineage + 2 same-family = **6-8 个**

**不链接**: Domain Overview, Paradigm, System Index → 这些只放 frontmatter 属性

### 关键规则

- Paper Note **不链接** Domain Overview / Paradigm → 只放 frontmatter `frame` 属性
- Concept = MechanismFamily + CanonicalIdea **合并** → 单一信息密集页
- Bottleneck = **跨论文综合** insight → 不是每篇论文各建一个
- Lineage = **人类可读**的演化链 → 含 ASCII 树 + 每步 diff + 分叉点
- 90_Views/ 是**静态 Markdown 表格** → 不依赖 Dataview 插件

### 目录结构

```
00_Home/
  00_方向总览.md        # 方法主线 + 核心概念 + 研究瓶颈 + 论文分布
  01_阅读顺序.md        # 分层: 框架 → baseline → 结构性 → 按需
10_Lineages/            # L__ 方法演化链 (需 delta_card_lineage 数据)
20_Concepts/            # C__ 概念 = Mechanism + CanonicalIdea
30_Bottlenecks/         # B__ 跨论文瓶颈 (症状 + 根因 + 解法分层)
40_Papers/
  A__Baselines/         # struct ≥ 0.7 (必读)
  B__Structural/        # struct ≥ 0.5 (结构性改进)
  C__Plugins/           # struct ≥ 0.3 (插件型)
  D__Peripheral/        # < 0.3 或无数据 (外围)
80_Assets/figures/      # PDF 提取的图表 (按 paper_sanitized 分目录)
90_Views/               # 静态 Markdown 表格 (按结构性/年份/概念/瓶颈)
```

### Obsidian 同步

```bash
# 1. 服务器上导出
curl -X POST localhost:8000/api/v1/pipeline/export/obsidian-vault

# 2. rsync 到本地
rsync -avz --delete \
  -e "ssh -i ~/.ssh/autoresearch.pem" \
  root@47.101.167.55:/opt/researchflow/researchflow-backend/obsidian-vault/ \
  ./obsidian-vault/

# 3. Obsidian 打开 → Graph View (Cmd+G)
#    推荐 Graph 颜色:
#    path:40_Papers → 蓝    path:20_Concepts → 绿
#    path:30_Bottlenecks → 红   path:10_Lineages → 橙
```

---

## 9. 技术栈

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
