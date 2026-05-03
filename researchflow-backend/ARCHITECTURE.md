# ResearchFlow Architecture v7.0

## 1. 设计原则

**PostgreSQL 是唯一真相源。DeltaCard 是不可变中间真相层。一切 UI/导出/Agent 都是投影。**

```
                        ┌─────────────────────┐
                        │    用户界面 (投影)     │
                        │ Web │ MCP │ Obsidian  │
                        └──────────┬──────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│                        Core Backend                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │  API Routers  │  │  MCP 工具    │  │  Service 模块             │ │
│  └──────────────┘  └──────────────┘  └───────────────────────────┘ │
│                                                                     │
│  PostgreSQL 16 (pgvector) + Redis 7 + 对象存储 (COS/OSS)           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 四层知识图谱架构

```
Layer A: Faceted Taxonomy DAG (6 表)
  taxonomy_nodes → taxonomy_edges → paper_facets
  problem_nodes → problem_claims
  paradigm_templates

Layer B: Method Evolution DAG (6 表)
  method_nodes → method_edges → method_applications
  paradigm_candidates, slot_candidates, method_candidates

Layer C: Paper Layer (15 表)
  papers → delta_cards → evidence_units → graph_assertions
  paper_analyses, paper_assets, paper_versions
  delta_card_lineage, graph_nodes, graph_assertion_evidence
  paper_reports, paper_report_sections
  project_bottlenecks, paper_bottleneck_claims

Layer D: Cross-paper Abstraction (2 表, Phase 2)
  canonical_ideas → contribution_to_canonical_idea
```

### 支撑层

```
Discovery:  paper_candidates, candidate_scores, score_signals
Agent:      agent_runs, agent_blackboard_items, paper_extractions, reference_role_maps
Metadata:   metadata_observations, canonical_paper_metadata
Domain:     domain_specs, domain_source_registry, incremental_checkpoints
Review:     review_queue, human_overrides, aliases, taxonomy_versions
KB:         kb_node_profiles, kb_edge_profiles, graph_node_candidates, graph_edge_candidates
System:     jobs, model_runs, digests
```

**总计: 40 表**

---

## 3. 核心数据模型

### 3.1 实体关系

```
Paper (容器, current_delta_card_id 指向当前发布版)
  └─→ DeltaCard (不可变快照, 含 publish_status/evidence_count/changed_slots_json)
        ├─→ EvidenceUnit (证据: 实验结果/代码验证/推理)
        ├─→ GraphAssertion (图谱边: supported_by/changes_slot/instance_of_method/...)
        ├─→ DeltaCardLineage (独立 DAG: builds_on/extends/replaces)
        └─→ ContributionToCanonicalIdea (映射到跨论文归一概念)

MethodNode (统一方法/机制实体, 含 parent_method_id 层级)
  └─→ MethodEdge (演化边: extends/modifies_slot/replaces)
  └─→ MethodApplication (论文使用方法: baseline/proposed/component)

TaxonomyNode (分面分类: domain/task/dataset/modality/paradigm/...)
  └─→ TaxonomyEdge (层级: is_a/part_of)
  └─→ PaperFacet (论文多维标签: primary_task/dataset/paradigm/...)
```

### 3.2 方法演化 DAG

```
RL (type=mechanism_family, maturity=established_baseline)
├── RLHF (parent=RL)
│   ├── PPO (parent=RLHF)
│   └── GRPO (parent=RLHF, downstream=7)
│       ├── GRPO+LP (parent=GRPO)
│       └── GRPO+tree (parent=GRPO)
├── Preference Optimization (parent=RL)
│   ├── DPO (parent=Pref Opt)
│   └── ORPO (parent=Pref Opt)
└── GDPO (parent=[GRPO, DPO])  ← 多继承: method_edges
```

### 3.3 DeltaCard 发布门控

| 对象 | 发布条件 |
|------|---------|
| DeltaCard.status | evidence_refs ≥ 2 AND (frame_id + slots 或 delta_statement) |
| DeltaCard.publish_status | evidence_count ≥ 2 AND min(confidence) ≥ 0.85 → auto_published |
| GraphAssertion (高价值) | contradicts/transferable_to/patch_of → 需 review |

---

## 4. 6-Agent Pipeline

### 4.1 完整流程

```
POST /pipeline/{paper_id}/run
 │
 ├─ Step 0: triage_paper()                [triage_service]
 ├─ Step 1: download_arxiv_pdf()          [pipeline_service]
 ├─ Step 2: enrich_paper() (10 API)       [enrich_service]
 ├─ Step 2.5: venue_resolution            [venue_resolver_service]
 ├─ Step 3: parse_paper_pdf() (L2)        [parse_service]
 ├─ Step 4: skim_paper() (L3)             [analysis_service]
 │
 └─ Step 5: deep_ingest() (V6 pipeline)   [ingest_workflow]
      │
      ├─ Agent 1: shallow_extractor (18K tokens)
      │   输入: abstract + method/experiment excerpts
      │   输出: paper_essence + method_delta
      │
      ├─ Agent 2: reference_role (30K tokens)
      │   输入: reference_list + citation_contexts
      │   输出: classifications + anchor_baselines
      │
      │   ↓ 确定性评分 (scoring_engine → DeepIngestScore)
      │   ↓ ≥88 auto_deep / 80-87 review_deep / <80 stay L1
      │
      ├─ Agent 3: deep_analyzer (40K tokens)
      │   输入: 全文 + shallow results
      │   输出: method/experiment/formulas
      │
      ├─ Agent 4: graph_candidate (20K tokens)
      │   输入: all prior + graph summary
      │   输出: node/edge/lineage candidates
      │
      ├─ Agent 5: kb_profiler (20K tokens, batched)
      │   输入: qualifying candidates (node ≥75, edge ≥70)
      │   输出: wiki profiles (Chinese)
      │
      ├─ Agent 6: paper_report (80K tokens)
      │   输入: ALL verified blackboard items
      │   输出: 10-section structured report
      │
      └─ _materialize_to_graph() (纯 DB)
           ├─ run_delta_card_pipeline()
           │   build_delta_card → persist_evidence → finalize → propose_assertions → publish
           ├─ link_to_parent_baselines() → DeltaCardLineage
           ├─ synthesize_concepts() → MethodNode + CanonicalIdea
           ├─ reconcile_neighbors() → same_family updates
           └─ _write_taxonomy_facets() → TaxonomyNode + PaperFacet
```

### 4.2 Agent 共享记忆

```
Blackboard 模式 (agent_blackboard_items 表):
  Agent N 写入 → Agent N+1 通过 ContextPackBuilder 读取

4 层上下文:
  Global:  schema 定义 (node_types, relation_types, slot_types)
  Domain:  DomainSpec 范围 + 已有方法/任务
  Paper:   L2 提取结果 (sections, formulas, tables, figures)
  Run:     前序 Agent 的 blackboard items
```

### 4.3 三道防线

| # | 防线 | 机制 |
|---|------|------|
| 1 | 先看改动不先听故事 | deep_analyzer prompt: FIRST Method → THEN Experiments → ONLY THEN Abstract |
| 2 | 比较集不是论文自己说了算 | baseline_comparator_service: 从 DB 查 4 个来源自动补齐 |
| 3 | 高价值结论必须有证据 | DeltaCard 发布门控: evidence_refs ≥ 2 |

### 4.4 分析计划契约

批量分析的总览和 agent 执行约束维护在
[`docs/analysis_plan.md`](../docs/analysis_plan.md)。架构层只定义能力边界；
实际执行时，每个批次必须先声明 goal、source、selection rule、budget 和
output target，再按 6-Agent Pipeline 推进。

执行约束:

- 只通过 API/service 写入 PostgreSQL；Markdown/Obsidian 是导出投影。
- 每个 agent 只消费声明过的上下文，并把证据 anchor 写入 blackboard/DB。
- DeepIngestScore、node score、edge score 和 DeltaCard evidence gate 是硬门槛。
- paper_report 与 kb_profile 只能使用已验证 blackboard items，不新增无证据结论。
- 生成导出、快照、备份、本地 storage 和软链接不得作为源码提交。

---

## 5. Obsidian Vault 导出

```
obsidian-vault/
├── 00_Home/00_方向总览.md        ← 全局概览
├── domain/                       ← Task 按 domain 分目录
│   ├── Video_Understanding/
│   │   ├── _overview.md
│   │   ├── T__Video_QA.md
│   │   └── T__Long_Video_QA.md
│   └── Image_Generation/
│       └── T__Text_to_Image.md
├── method/                       ← Method 平铺
│   ├── M__GRPO.md
│   └── M__DPO.md
├── dataset/                      ← Dataset 平铺
│   └── D__MMMU.md
├── paper/                        ← Paper 按 venue_year
│   ├── CVPR_2025/P__xxx.md
│   └── ICLR_2025/P__yyy.md
└── views/                        ← 聚合视图
    ├── papers_by_year.md
    └── method_evolution.md
```

---

## 6. 数据库 Schema (40 表)

### Layer A: Faceted Taxonomy

| 表 | 说明 |
|----|------|
| taxonomy_nodes | 分面节点 (domain/task/dataset/modality/paradigm/...) |
| taxonomy_edges | 层级关系 (is_a/part_of) |
| paper_facets | 论文多维标签 (primary_task/dataset/paradigm) |
| problem_nodes | 任务下的常见问题 |
| problem_claims | 论文对问题的声称 |
| paradigm_templates | 4 个内置范式 (RL/VLM/Agent/MotionGen) |

### Layer B: Method Evolution

| 表 | 说明 |
|----|------|
| method_nodes | 统一方法/机制实体 (含 parent_method_id 层级 + aliases) |
| method_edges | 演化边 (extends/modifies_slot/replaces/combines_with) |
| method_applications | 论文使用方法 (baseline/proposed/component) |
| paradigm_candidates | 候选范式 |
| slot_candidates | 候选槽位 |
| method_candidates | 候选方法/机制 |

### Layer C: Paper

| 表 | 说明 |
|----|------|
| papers | ~42 列: 元数据 + 状态 + current_delta_card_id |
| paper_assets | PDF, HTML, figures |
| paper_versions | arXiv 版本追踪 |
| paper_analyses | L2/L3/L4 分析结果 |
| delta_cards | 不可变快照 + publish_status + evidence_count |
| evidence_units | 原子证据 (confidence + basis + source_anchor) |
| delta_card_lineage | 方法演化 DAG 边 |
| graph_nodes | 统一节点注册 |
| graph_assertions | 有向类型边 + 生命周期 |
| graph_assertion_evidence | 边↔证据关联 |
| evidence_items | 细粒度证据索引 |
| paper_reports | 结构化报告 |
| paper_report_sections | 报告章节 |
| project_bottlenecks | 全局瓶颈本体 |
| paper_bottleneck_claims | 论文级瓶颈声称 |

### 已删除的表 (v6→v7)

IdeaDelta (→DeltaCard), graph_edges (→graph_assertions), slots (→paradigm_templates.slots),
method_slots, MethodDelta, MechanismFamily (→method_nodes), implementation_units,
transfer_atoms, search_sessions, search_branches, reading_plans, direction_cards,
render_artifacts, user_bookmarks, user_events, execution_memories, user_feedback,
project_focus_bottlenecks, review_tasks (→review_queue)
