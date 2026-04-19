# ResearchFlow Architecture v6.0

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
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │ 16 API Router │  │ 35 MCP 工具  │  │ 55 Service 模块           │ │
│  │ (130 端点)    │  │ 6 资源 4 提示 │  │                           │ │
│  └──────────────┘  └──────────────┘  └───────────────────────────┘ │
│                                                                     │
│  PostgreSQL 16 (pgvector) + Redis 7 + 对象存储 (COS/OSS)           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 四层提取架构

```
Layer 1: 确定性后端 (CPU, 免费)
├── PyMuPDF: 文本/section/图片/caption 提取
├── GROBID (Docker): 结构化 authors/affiliations/references/formulas
├── 公式区域检测: 数学符号聚类 + 区域截图
└── Figure 区域检测: caption 锚定 + 向上扫描

Layer 2: 来源适配器 (8 个 API)
├── arXiv API: title/abstract/authors/year/keywords/comments
├── Crossref: DOI/venue/year
├── OpenAlex: venue/citations/open_access
├── Semantic Scholar: citations/recommendations
├── DBLP: 会议 proceedings 验证
├── OpenReview (SDK): decisions/reviews/scores
├── GitHub: code repo search + README 分析
└── HuggingFace: models + datasets discovery

Layer 3: Claude VLM (API, 按需)
├── Figure 分类 + 补漏: 1 次调用/篇 (~$0.02)
├── Formula OCR → LaTeX: 1 次调用/篇
└── Acceptance 冲突判断: 按需

Layer 4: Agent 编排
├── Skills: 21 个领域知识 + 工作流技能
├── MCP Tools: 35 个数据操作接口
└── Review Gates: 低置信度 → 人工审核
```

---

## 3. 核心数据模型

### 3.1 实体关系

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

### 3.2 方法演化 DAG (核心创新)

论文之间不是扁平列表，而是 **有向无环图 (DAG)**:

```
GRPO (depth=0, baseline=true, downstream=7)
├── GRPO+LP (depth=1, parent=[GRPO])
│   ├── GRPO-LP+sampling (depth=2, parent=[GRPO+LP])
│   └── GRPO-LP+KL (depth=2, parent=[GRPO+LP])
├── GRPO+tree (depth=1, parent=[GRPO])
└── GDPO (depth=1, parent=[GRPO, DPO])  ← 多继承: 两个范式组合
    └── GDPO+image_thinking (depth=2, parent=[GDPO])
```

**升级规则 (candidate → review → publish)**:
- 当一个改进被 ≥3 篇论文用作 baseline → `is_established_baseline = true`
- 当它还具有结构性 (`structurality_score ≥ 0.6`) → 候选新范式版本
- `builds_on` 边默认为 `candidate` 状态，需审核后发布
- 自动发现的范式创建 `ParadigmCandidate`，通过 `POST /reviews/candidates/paradigms/{id}/promote` 审核升级
- 所有 ontology 变更记录在 `taxonomy_versions` 表中

### 3.3 Faceted Taxonomy DAG

```
taxonomy_nodes (75 个种子节点)
├── dimension: domain/modality/task/subtask/learning_paradigm/scenario/
│              constraint/mechanism/method_baseline/model_family/
│              dataset/benchmark/metric/lab/venue
├── name + name_zh + aliases
└── status: candidate / reviewed / canonical

taxonomy_edges (14 条种子边)
├── parent_id → child_id
├── relation_type: is_a / part_of / uses / optimizes / applies_to
└── 例: RL → RLHF (is_a), Video Understanding → Long Video QA (part_of)

paper_facets (论文的多维标签)
├── paper_id + node_id + facet_role
├── facet_role: primary_task / modality / paradigm / mechanism / baseline
└── 一篇论文可同时是: Video + VQA + RL + GRPO-derived + Reward Design
```

### 3.4 Method Evolution 模型

```
method_nodes
├── name, type (algorithm/recipe/model_family/system)
├── maturity: seed → emerging → established_baseline
├── downstream_count (≥3 → promote)
└── 从 Paper 中抽象出来的"方法"概念

method_slots
├── method_id + slot_name
└── 例 GRPO: reward_function, advantage_estimator, policy_update

method_edges
├── parent → child + relation_type
├── applies_to_domain / modifies_slot / combines_with / replaces
├── changed_slot_ids + delta_description
└── 区分三种关系: 应用(A) / 改进(B) / 组合(C)
```

### 3.5 Metadata Observation Ledger

多源元数据不直接覆盖 Paper 字段，走观察账本 + canonical resolver：

```
metadata_observations
├── entity_type: paper / author / venue
├── field_name: venue / status / authors / citation_count / code_url
├── value_json: JSONB (原始值)
├── source: arxiv / crossref / openalex / semantic_scholar / dblp / openreview / github
├── authority_rank: 1=最高权威, 10=最低

canonical_paper_metadata
├── 从 observations 中 resolve 的 canonical 值
├── unresolved_conflicts: [{field, sources, values}]
└── resolver_version

权威优先级:
  会议中稿: official_conf > openreview > dblp > crossref > arxiv > s2
  引用数: s2 > openalex > crossref > google_scholar
  作者机构: pdf_grobid > openalex > crossref > s2
```

### 3.6 论文过滤评分

每篇论文入库后自动计算 4 个分数:

| 分数 | 权重因素 | 用途 |
|------|---------|------|
| **keep_score** | Tier + 顶会 + 重要度 + 时间衰减 | 是否值得入库 |
| **analysis_priority** | 重要度 + 开源资产 + 有PDF + 顶会 + 新鲜度 | 分析优先级排序 |
| **structurality_score** | 关键词信号 + L4 LLM 输出 | 结构性 vs 插件 (A/B/C/D 分级) |
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

## 4. 完整流程: 从零到知识图谱

### 4.1 冷启动: 给一个领域，从零构建

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
         │  → 生成 Obsidian vault
         ▼
[Step 4] rsync 同步到本地 Obsidian
```

### 4.2 单篇论文完整 Pipeline

入口: `pipeline_service.run_full_pipeline(paper_id)`

每步已完成则自动跳过，任何一步失败不阻断后续步骤:

```
POST /pipeline/{paper_id}/run  (或 /pipeline/batch?limit=N)
 │
 ├─ Step 0: triage_paper()                          [triage_service]
 │    计算 keep_score / analysis_priority / tier
 │
 ├─ Step 1: download_arxiv_pdf()                    [pipeline_service]
 │    有 arxiv_id 且没下过 → 下载 PDF 到本地 + 对象存储
 │
 ├─ Step 2: enrich_paper() — 10 步元数据补全         [enrich_service]
 │    2.1 arXiv API → title/abstract/authors/year/keywords/comments
 │    2.2 arXiv comments → 会议中稿解析 ("Accepted at ICLR 2025")
 │    2.3 Crossref → DOI/venue/year (跳过 placeholder title)
 │    2.4 OpenAlex → venue/citations/open_access
 │    2.5 Semantic Scholar → citation_count/venue
 │    2.6 GitHub → code_url + README 中稿/数据集提取
 │    2.7 HuggingFace → models + datasets
 │    2.8 GitHub README → acceptance + dataset links
 │    2.9 Project page → acceptance check
 │    2.10 PDF 首页文本 → acceptance detection
 │    * 所有结果写 metadata_observations (观察账本)
 │    * Placeholder title 保护：不用 arxiv ID 搜其他 API
 │
 ├─ Step 2.5: venue_resolution                      [venue_resolver_service]
 │    → OpenReview SDK (ICLR/NeurIPS decisions + review scores)
 │    → DBLP proceedings lookup
 │    → LLM 冲突判断 (多源不一致时)
 │    → canonical resolver (按 authority_rank 选最优)
 │
 ├─ Step 3: parse_paper_pdf() — L2 Parser Ensemble  [parse_service]
 │    → GROBID: authors(含机构) + references(结构化) + formulas(坐标)
 │    → PyMuPDF: sections + figure images + captions
 │    → Figure extraction: caption 锚定 + VLM 分类补漏
 │    → Formula extraction: 区域检测 + 合并 + VLM OCR → LaTeX
 │    → 结果合并，冲突标记
 │
 ├─ Step 4: skim_paper() — L3                       [analysis_service]
 │    1 次 LLM 调用 (abstract + 关键章节, ≤8K tokens)
 │    产出: problem_summary, method_summary, worth_deep_read, is_plugin_patch
 │
 ├─ Step 5: deep_analyze_paper() — L4 (6 步)        [analysis_service]
 │    详见 §4.3
 │
 ├─ Step 5.5: Post-L4 回填                           [pipeline_service]
 │    → 回填 paper 字段: core_operator, primary_logic, claims
 │    → 推断 ring (baseline/structural/plugin)
 │    → 设置 role_in_kb
 │    → Taxonomy assignment: tags/category/keywords → paper_facets
 │
 └─ Step 6: citation_discovery                       [discovery_service]
      → S2 references + citations → 自动 ingest
```

### 4.3 L4 深度分析: 6 步管线 (核心)

入口: `analysis_service.deep_analyze_paper(paper_id)`

```
deep_analyze_paper(paper_id)
 │  从 L2 提取的章节文本拼接全文 (≤25K tokens)
 ▼
┌─ Step 1: extract_evidence ──────────────────────────────────────┐
│  文件: analysis_steps.py → run_step1_extract_evidence()          │
│  LLM 调用 #1                                                    │
│                                                                  │
│  ★ 防线 #1: 强制阅读顺序                                         │
│    Prompt: "FIRST read Method → THEN Experiments → ONLY THEN     │
│    Abstract. Do NOT take the abstract at face value."             │
│                                                                  │
│  输出:                                                           │
│    key_equations[]:    核心公式 (≤4), 含 slot_affected            │
│    key_figures[]:      关键图表 (≤4), 含 evidence_for             │
│    evidence_units[]:   证据锚点 (3-8), 每个含:                     │
│      atom_type, claim, confidence, basis, source_section          │
│    narrative_vs_substance: abstract 是否与实验吻合                  │
│    baseline_fairness:  baseline 是否最强?                         │
│    paper_type:         method/survey/benchmark/position/...       │
│                                                                  │
│  失败 → 自动重试 (最多 2 次)                                       │
└──────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─ Step 2: build_delta_card ──────────────────────────────────────┐
│  文件: analysis_steps.py → run_step2_build_delta()               │
│  LLM 调用 #2, 输入: 全文 + Step 1 证据作 grounding                │
│                                                                  │
│  输出:                                                           │
│    problem_summary:    问题与挑战 (200-400 字, 中文)               │
│    method_summary:     方法与洞察 (400-600 字, 中文)               │
│    evidence_summary:   证据与局限 (200-400 字, 中文)               │
│    core_intuition:     核心直觉 (100-200 字)                      │
│    changed_slots[]:    改了哪些 slot                              │
│    unchanged_slots[]:  没改的 slot                                │
│    structurality_score: 0.0-1.0  ← 决定 A/B/C/D 分级             │
│    delta_card:         {paradigm, slots, is_structural}           │
│    bottleneck_addressed: {title, description, is_fundamental}     │
│    same_family_method: 方法族名称                                  │
│                                                                  │
│  失败 → 自动重试 (最多 2 次)                                       │
└──────────────────────────────────────────────────────────────────┘
 │ merge → 持久化: PaperAnalysis + MethodDelta + DeltaCard +
 │                  Evidence + IdeaDelta + GraphAssertions
 ▼
┌─ Step 3: build_compare_set ─────────────────────────────────────┐
│  baseline_comparator_service.py (纯 DB, 无 LLM)                  │
│  ★ 防线 #2: "比较集不是论文自己说了算"                               │
│  4 个来源:                                                        │
│    ① domain_baseline:  同范式已确立基线                              │
│    ② same_mechanism:   同 mechanism_family 的论文                  │
│    ③ strong_peer:      同 category 同时期高 structurality 论文      │
│    ④ self_reported:    论文自述的 baseline_paper_titles             │
└──────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─ Step 4: propose_lineage ───────────────────────────────────────┐
│  evolution_service.py (纯 DB)                                     │
│  查找父节点 → 创建 DeltaCardLineage (candidate)                    │
│  → 创建 builds_on GraphAssertion                                  │
│  → downstream ≥ 3 → parent.is_established_baseline = true         │
│  → 创建 ReviewTask                                                │
└──────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─ Step 5: synthesize_concept ────────────────────────────────────┐
│  concept_synthesizer_service.py (纯 DB)                           │
│  ① same_family_method → 找/建 MechanismFamily                     │
│  ② core_intuition → 找/建 CanonicalIdea                           │
│  ③ 创建 ContributionToCanonicalIdea 链接                           │
└──────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─ Step 6: reconcile_neighbors ───────────────────────────────────┐
│  incremental_reconciler_service.py                                │
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

### 4.4 三道防线

| # | 防线 | 机制 | 对应步骤 |
|---|------|------|---------|
| 1 | **先看改动不先听故事** | Prompt 强制: FIRST Method → THEN Experiments → ONLY THEN Abstract | Step 1 |
| 2 | **比较集不是论文自己说了算** | 从 DB 查 4 个来源自动补齐比较集 | Step 3 |
| 3 | **高价值结论必须有证据锚点** | DeltaCard 发布门控: evidence_refs ≥ 2 | 持久化阶段 |

### 4.5 容错机制

- **Step 1+2 (LLM)**: 缺 required_fields → 自动重试 (最多 2 次), 仍缺则用 partial data 继续
- **Step 3-6 (DB)**: 每步独立 `try/except + session.rollback()`, 单步失败不阻断后续
- **LLM 返回非 JSON**: 自动去 markdown fence → 找 `{...}` → retry
- **Session 级容错**: DB 错误后 rollback 恢复 session

---

## 5. 图谱断言模型

### 5.1 GraphAssertion 生命周期

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

### 5.2 发布门控

| 对象 | 发布条件 |
|------|---------|
| DeltaCard | frame_id + changed_slots + evidence_refs ≥ 2 |
| IdeaDelta | evidence_count ≥ 2 + min(confidence) ≥ 0.85 |
| 高价值断言 | 需要 ReviewTask 审核通过 |

---

## 6. 领域范式动态发现

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

## 7. Figure/Formula 提取

### 图表提取流程

```
1. PyMuPDF caption 文本扫描 → 找到 "Figure X." / "Table X."
2. 向上扫描找最近的正文文本 → 确定图的上边界
3. caption 宽度判断: >55% 页宽 = 全宽图, 否则列宽
4. 裁剪区域 → 2.5x 高清 PNG → OSS 上传
5. VLM (1 次调用): 分类 + 补漏漏检的图
6. VLM 返回: label, semantic_role, description (中文)
```

### 公式提取流程

```
1. GROBID 检测 <formula> 坐标 (优先) 或数学符号聚类 (fallback)
2. bbox 横向扩展到文本列宽 (包含公式编号)
3. 相邻公式行合并 (多行方程)
4. 3x 高清截图 → VLM OCR → LaTeX
```

---

## 8. 数据库 Schema (58 表 + 4 物化视图, 16 次迁移)

### 核心表

| 表 | 说明 |
|----|------|
| papers | 60+ 列: 元数据 + 评分 + 状态 + current_delta_card_id + embedding (pgvector 1536d) |
| delta_cards | 不可变快照 (append-only) + analysis_run_id/source_asset_hash |
| idea_deltas | 可复用知识原子 |
| evidence_units | 证据单元 (实验/代码/推理) |
| evidence_links | 证据关联 |
| graph_assertions | 图谱边 + 生命周期 |
| graph_nodes | 统一节点注册 |
| graph_edges | 图谱边 (legacy) |

### 领域本体

| 表 | 说明 |
|----|------|
| paradigm_templates | 范式模板 + 版本演化 |
| slots | 可替换组件 (每范式 4-8 个) |
| mechanism_families | 机制族 (层级结构) |
| domains | 领域定义 |
| project_bottlenecks | 全局瓶颈本体 |
| paper_bottleneck_claims | 论文级瓶颈声称 |
| project_focus_bottlenecks | 项目级关注瓶颈 (带负约束) |

### 跨论文概念

| 表 | 说明 |
|----|------|
| canonical_ideas | 跨论文归一概念 |
| contribution_to_canonical_idea | 论文贡献→概念映射 |
| delta_card_lineage | 独立演化 DAG (candidate 默认) |

### 审核与候选

| 表 | 说明 |
|----|------|
| review_tasks | 审核队列 (自动 + 人工) |
| assertion_overrides | 人工覆写 |
| paradigm_candidates | 候选范式 (需审核) |
| slot_candidates | 候选槽位 |
| mechanism_candidates | 候选机制族 |

### Faceted Taxonomy

| 表 | 说明 |
|----|------|
| taxonomy_nodes | 75 个种子节点, 15 个维度 |
| taxonomy_edges | 层级关系 (is_a/part_of/uses/optimizes) |
| taxonomy_versions | ontology 变更快照 |
| paper_facets | 论文多维标签 |
| problem_nodes | 任务下的共性问题 |
| problem_claims | 论文对问题的声称 |

### Method Evolution

| 表 | 说明 |
|----|------|
| method_nodes | 方法节点 (algorithm/recipe/model_family/system) |
| method_slots | 方法的可替换组件 |
| method_edges | 方法间关系 (applies_to/modifies_slot/combines_with/replaces) |
| method_applications | 论文对方法的使用 (baseline/proposed/component) |

### 支撑表

paper_analyses, paper_assets, paper_versions, method_deltas,
metadata_observations, canonical_paper_metadata,
digests, digest_entries, directions, direction_insights,
research_sessions, exploration_steps, search_branches,
bookmarks, feedback, render_artifacts, aliases

### 物化视图 (CQRS-lite)

| 视图 | 说明 |
|------|------|
| paper_search_docs | 论文 + DeltaCard + 证据数 去规范化 |
| idea_search_docs | IdeaDelta + 论文 + DeltaCard 去规范化 |
| lineage_view | 方法演化 DAG + 论文标题展平 |
| review_queue_view | 待审核项 + 目标摘要 |

刷新: `POST /search/refresh-views`

### PaperState 状态机

```
EPHEMERAL_RECEIVED → CANONICALIZED → ENRICHED → WAIT → DOWNLOADED
  → L1_METADATA → L2_PARSED → L3_SKIMMED → L4_DEEP → CHECKED
  (分支: SKIP | MISSING | TOO_LARGE | ANALYSIS_MISMATCH | ARCHIVED_OR_EXPIRED)
```

### 枚举类型 (9 个)

| 枚举 | 值 |
|------|---|
| PaperState | 15 个状态值 |
| Importance | S, A, B, C, D |
| AnalysisLevel | L1_METADATA, L2_PARSE, L3_SKIM, L4_DEEP |
| AssetType | RAW_PDF, RAW_HTML, EXTRACTED_TEXT, FIGURE, CODE_SNAPSHOT, SKIM_REPORT, DEEP_REPORT, EXPORTED_MD |
| PeriodType | DAY, WEEK, MONTH |
| JobStatus | PENDING, RUNNING, COMPLETED, FAILED, CANCELLED |
| FeedbackType | CORRECTION, CONFIRMATION, REJECTION, TAG_EDIT |
| Tier | A_OPEN_DATA, B_OPEN_CODE, C_ACCEPTED_NO_CODE, D_PREPRINT |
| EvidenceBasis | CODE_VERIFIED, EXPERIMENT_BACKED, TEXT_STATED, INFERRED, SPECULATIVE |

---

## 9. API 路由总览 (16 Router, 130 端点)

| Router | 前缀 | 关键端点 |
|--------|------|---------|
| papers | /papers | CRUD, triage, triage-all, enrich, search, download-pdf |
| import | /import | links, accept, pdf, parse, cleanup-expired |
| analyses | /analyses | skim, deep, skim-batch |
| pipeline | /pipeline | run, batch, init-domain, discover, build-domain, download-pdf, lineage, evolution, export/obsidian-vault, export/build-collection-index, sync-domain, refresh-connections |
| search | /search | hybrid, ideas, bottlenecks, mechanisms, transfers, query, refresh-views, embeddings/generate, reading-plan |
| graph | /graph | stats, ideas, edges, citations, bottleneck, mechanism, transfers, synthesis, paradigms, mechanisms, quality, admin-stats, vis-data |
| reviews | /reviews | CRUD, approve, reject, assign, override, candidates/paradigms (promote/reject), candidates/lineage |
| assertions | /assertions | reviews/queue, reviews/stats, approve, reject, overrides, aliases, node, propose, audit, publish |
| explore | /explore | start, step, search, summary |
| taxonomy | /taxonomy | nodes, tree, dimensions, paper/{id}/facets, problems |
| methods | /methods | nodes, nodes/{id}, lineage/{id} |
| reports | /reports | generate |
| digests | /digests | generate, latest |
| directions | /directions | propose, expand, list |
| feedback | /feedback | feedback CRUD, bookmarks CRUD |
| bottlenecks | /bottlenecks | normalize, merge-duplicates, unlinked-claims, focus |

---

## 10. Service 模块 (55 个)

| 文件 | 职责 |
|------|------|
| **pipeline_service.py** | 外层编排 (triage→download→enrich→L2→L3→L4→post-L4→discovery) |
| **analysis_service.py** | L3 skim + L4 deep 6步编排器 |
| **analysis_steps.py** | Step 1+2 的 prompt + JSON 解析 + retry + merge |
| **baseline_comparator_service.py** | Step 3: 4源比较集 |
| **evolution_service.py** | Step 4: 方法演化 DAG + baseline 晋升 |
| **concept_synthesizer_service.py** | Step 5: MechanismFamily + CanonicalIdea |
| **incremental_reconciler_service.py** | Step 6: 反向更新邻居 |
| **delta_card_service.py** | DeltaCard → Evidence → IdeaDelta → Assertions |
| **frame_assign_service.py** | 范式匹配 (静态→模糊→LLM发现) |
| **parse_service.py** | L2 PDF 解析 (GROBID + PyMuPDF ensemble) |
| **enrich_service.py** | 10 步元数据补全 (8 个 API) |
| **triage_service.py** | 论文评分 (4 维) |
| **venue_resolver_service.py** | 会议中稿检测 (OpenReview + DBLP + arXiv) |
| **metadata_resolver_service.py** | 多源元数据 canonical resolve |
| **figure_extraction_service.py** | 图表提取 (caption 锚定 + VLM) |
| **formula_extraction_service.py** | 公式提取 (GROBID + VLM OCR) |
| **vlm_extraction_service.py** | Vision-Language Model 调用 |
| **search_service.py** | 混合搜索 (keyword + semantic + structured) |
| **embedding_service.py** | 向量嵌入生成管理 |
| **query_router_service.py** | 意图路由 (自然语言→对应搜索) |
| **reading_planner.py** | 分层阅读计划 |
| **graph_service.py** | 图谱构建维护 |
| **graph_query_service.py** | 复杂图谱查询 |
| **discovery_service.py** | 论文发现 (Semantic Scholar) |
| **exploration_service.py** | 交互式探索会话 |
| **domain_init_service.py** | 领域冷启动 (GitHub awesome 仓库) |
| **domain_sync_service.py** | 领域同步 |
| **direction_service.py** | 研究方向管理 |
| **report_service.py** | 报告生成 (quick/briefing/deep) |
| **digest_service.py** | 每日/周/月摘要 |
| **review_service.py** | 审核任务管理 |
| **assertion_service.py** | 断言管理 |
| **quality_service.py** | 知识库质量评估 |
| **feedback_service.py** | 用户反馈收集 |
| **bottleneck_normalization_service.py** | 瓶颈归一化聚类 |
| **entity_resolution_service.py** | 实体去重归一 |
| **ingestion_service.py** | 论文入库 |
| **paper_service.py** | 论文 CRUD |
| **llm_service.py** | LLM API 调用 (Anthropic/OpenAI) |
| **object_storage.py** | 对象存储 (COS/OSS) |
| **vault_export_v5.py** | Obsidian vault 导出 |
| **export_service.py** | 通用导出 |
| **openreview_adapter.py** | OpenReview API 集成 |
| **dblp_adapter.py** | DBLP 元数据查询 |
| **candidate_service.py** | V6 候选队列管理 (5 级吸收 + 去重 + 批量评分) |
| **scoring_engine.py** | V6 4 层评分引擎 (Discovery/DeepIngest/GraphPromotion/Anchor) + hard caps + boosts |
| **agent_runner.py** | V6 12 Agent prompt 模板 + LLM 调用 + Blackboard 写入 + AgentRun 追踪 |
| **context_pack_builder.py** | V6 10 种 Agent 上下文包 (Global→Domain→Paper→Run 4 层, per-agent token 预算) |
| **ingest_workflow.py** | V6 管线编排 (import_and_score → shallow → deep → profile → report) |
| **node_profile_service.py** | V6 节点 Profile 生成/刷新/staleness (kb_node_profiles) |
| **edge_profile_service.py** | V6 边 Profile 生成/批量生成 (kb_edge_profiles) |
| **cold_start_service.py** | V6 冷启动 (DomainManifest → skeleton → arXiv/S2 harvest → anchor) |
| **incremental_sync_service.py** | V6 增量同步 (arXiv daily + citation refresh + awesome diff + lineage detect + dedup) |
| **vault_export_v6.py** | V6 Obsidian 导出 (注入 node/edge profile + Lab 页面) |

---

## 11. MCP Server (35 工具 + 6 资源 + 4 提示)

### Tools

| 工具 | 功能 |
|------|------|
| `search_research_kb` | 混合搜索: keyword + semantic + structured |
| `search_ideas` | 搜索 DeltaCard / IdeaDelta |
| `get_paper_report` | 生成报告: quick / briefing / deep_compare |
| `compare_papers` | 2-5 篇论文并排比较 |
| `import_research_sources` | 导入论文 URL |
| `get_digest` | 每日/周/月摘要 |
| `get_reading_plan` | 分层阅读计划 |
| `propose_directions` | 研究方向提议 |
| `run_full_pipeline` | 完整管线执行 |
| `discover_related_papers` | Semantic Scholar 发现 |
| `build_domain` | 多跳发现构建领域 |
| `enqueue_analysis` | 排队 L3/L4 分析 |
| `refresh_assets` | 元数据刷新 |
| `record_user_feedback` | 记录反馈 |
| `get_paper_detail` | 论文详情 + DeltaCard |
| `get_graph_stats` | 图谱统计 |
| `review_queue` | 待审核队列 |
| `submit_review_decision` | 提交审核决策 |
| `resolve_venue` | 会议中稿检测 (OpenReview + DBLP) |
| `get_paper_citations` | GROBID refs + S2 citing papers |
| `get_paper_figures` | 图表 + OSS URLs + VLM 描述 |
| `get_metadata_conflicts` | 多源元数据冲突查看 |
| `rf_domain_cold_start` | V6: 基于 domain manifest 冷启动 |
| `rf_candidate_list` | V6: 查看候选队列 (筛选/排序) |
| `rf_candidate_promote` | V6: 提升候选到指定 absorption level |
| `rf_candidate_reject` | V6: 拒绝候选 + 记录原因 |
| `rf_paper_build_neighborhood` | V6: S2 邻域检索 → 候选 (不直接 ingest) |
| `rf_node_profile_get` | V6: 获取节点 Profile |
| `rf_node_profile_refresh` | V6: 刷新节点 Profile |
| `rf_edge_profile_get` | V6: 获取边 Profile |
| `rf_graph_get_subgraph` | V6: 获取子图 (节点+边+profiles) |
| `rf_review_queue` | V6: 查看审核队列 |
| `rf_score_explain` | V6: 解释候选评分明细 |
| `rf_run_v6_pipeline` | V6: 完整管线 (import→score→shallow→deep→profile→report) |

### Resources

| URI | 返回内容 |
|-----|---------|
| `paper://{id}` | 论文详情 + 分析 + DeltaCard |
| `delta-card://{id}` | DeltaCard 结构化快照 |
| `graph://stats` | 知识图谱统计 |
| `canonical-idea://{id}` | 跨论文归一概念 |
| `review-task://{id}` | 审核任务 + 目标对象 |
| `lineage://{paper_id}` | 方法演化 DAG |

### Prompts

| 名称 | 用途 |
|------|------|
| `deep-paper-report` | 论文深度分析报告 |
| `weekly-research-review` | 周度研究综述 |
| `lineage-review` | 方法演化追踪 |
| `direction-gap-analysis` | 研究方向 gap 分析 |

---

## 12. Obsidian Vault 导出

触发: `POST /pipeline/export/obsidian-vault`

### 目录结构

```
00_Home/
  00_方向总览.md        # 方法主线 + 核心概念 + 研究瓶颈 + 论文分布
  01_阅读顺序.md        # 分层: 框架 → baseline → 结构性 → 按需
10_Lineages/            # L__ 方法演化链
20_Concepts/            # C__ 概念 = Mechanism + CanonicalIdea
30_Bottlenecks/         # B__ 跨论文瓶颈 (症状 + 根因 + 解法分层)
40_Papers/
  A__Baselines/         # struct ≥ 0.7 (必读)
  B__Structural/        # struct ≥ 0.5 (结构性改进)
  C__Plugins/           # struct ≥ 0.3 (插件型)
  D__Peripheral/        # < 0.3 或无数据 (外围)
80_Assets/figures/      # PDF 提取的图表
90_Views/               # 静态 Markdown 表格
```

### 论文分级 A/B/C/D

分级由 `_paper_level(p)` 函数决定，优先级: **ring 字段 → dc.structurality_score → paper.structurality_score**

| 等级 | 分数条件 | 含义 | 示例 |
|------|---------|------|------|
| **A** | ring=baseline 或 score ≥ 0.7 | 必读 baseline | DPO, InstructGPT, KTO |
| **B** | ring=structural 或 score ≥ 0.5 | 结构性改进 | GRPO, ORPO, SPIN |
| **C** | ring=plugin 或 score ≥ 0.3 | 插件型改进 | SimPO, RAFT, RRHF |
| **D** | 其余 | 外围参考 | — |

### Paper Note wikilink 预算

**6-10 个正文 wikilinks**: T (Task) + M (Method) + C (Concept) + D (Dataset) + P (Papers)

其余 facets 放 frontmatter YAML，不生成 wikilinks。不链接 Domain Overview / Paradigm。

---

## 13. 研究探索: 多跳认知迭代

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
GET /explore/{id}
         └→ 完整路径 + 论文分类 + 下一步建议
```

---

## 14. 技术栈

| 组件 | 选型 |
|------|------|
| Web 框架 | FastAPI (async) |
| 前端 | Next.js 15 + Tailwind |
| ORM | SQLAlchemy 2.0 (async) |
| 数据库 | PostgreSQL 16 + pgvector |
| 任务队列 | ARQ (Redis) |
| PDF 解析 | PyMuPDF + GROBID (ensemble) |
| LLM | Anthropic Claude / OpenAI (streaming) |
| VLM | Claude Vision (图表分类 + 公式 OCR) |
| MCP | Python MCP SDK (stdio + SSE) |
| 元数据 | arXiv + Crossref + OpenAlex + S2 + DBLP + OpenReview + GitHub + HuggingFace |
| 对象存储 | Tencent COS / Alibaba OSS / Local |
| 部署 | Docker Compose + Caddy (自动 HTTPS) |

---

## 15. V6: Candidate Queue + Multi-Agent Pipeline

### 15.1 paper_candidates 表与 5 级吸收

候选论文不再直接进入 papers 表，而是经过分级吸收：

```
new → shallow → reference_done → deep → graph_ready
 │       │           │              │         │
 │   ShallowPaper  ReferenceRole  6 deep    Profile
 │   agent         agent          agents    + promote
 │                                          to graph
```

每级吸收由对应 agent 完成，状态机确保幂等重试。

### 15.2 4 层评分引擎

| 评分 | 计算时机 | 因素 | 用途 |
|------|---------|------|------|
| **DiscoveryScore** | 候选入队时 | 来源可信度 + 引用信号 + 时效性 | 决定是否值得 shallow 分析 |
| **DeepIngestScore** | shallow 完成后 | ShallowPaper 输出 + 方法新颖度 + 领域相关度 | 决定是否投入 deep agent |
| **GraphPromotionScore** | deep 完成后 | 6 deep agent 输出质量 + 证据强度 + 图谱互补性 | 决定是否晋升到 graph |
| **AnchorScore** | 入图后 | 被引用次数 + 下游方法数 + profile 丰富度 | 锚点论文识别 |

### 15.3 12 Agent Prompt (agent_runner.py)

所有 prompt 定义在 `agent_runner.py` 的 `AGENT_PROMPTS` dict 中:

| Agent | Key | 阶段 | 输出 Schema | Token 预算 |
|-------|-----|------|------------|-----------|
| ShallowPaperAgent | `shallow_paper` | shallow | PaperEssence + TaskFacet + MechanismFacet | 6-12K |
| ReferenceRoleAgent | `reference_role` | shallow | ReferenceRoleMap (12 种角色) | 10-30K |
| MethodDeltaAgent-lite | `method_delta_lite` | shallow | 初步 MethodDelta | 8-15K |
| ScoreAgent | `score` | shallow | score_signals (结构化信号) | 4-8K |
| MethodDeltaAgent-full | `method_delta_full` | deep | 完整 MethodDelta + pipeline_modules | 15-30K |
| ExperimentAgent | `experiment` | deep | ExperimentMatrix + ablation + costs | 10-25K |
| FormulaFigureAgent | `formula_figure` | deep | key_formulas + figure_roles + derivation | 15-30K |
| GraphCandidateAgent | `graph_candidate` | deep | node_candidates + edge_candidates + lineage | 10-20K |
| NodeProfileAgent | `node_profile` | profile | one_liner + short_intro + detailed | 8-15K |
| EdgeProfileAgent | `edge_profile` | profile | contextual one_liner + summary | 6-12K |
| PaperReportAgent | `paper_report` | report | 10 section 结构化报告 | 30-80K |
| QualityAuditAgent | `quality_audit` | audit | issues + quality_score + review_items | 8-15K |

Context Pack Builder 为每个 agent 组装不同上下文 (Global→Domain→Paper→Run 4 层)。

### 15.4 kb_node_profiles + kb_edge_profiles

知识图谱节点和边不再只是 ID + label，而是拥有结构化 profile：

- **kb_node_profiles**: 方法节点的能力矩阵、适用场景、局限性、代表论文
- **kb_edge_profiles**: 边的变更描述、证据支持、置信度、影响范围

Profile 由 NodeProfile / EdgeProfile agent 生成，随新论文增量更新。

### 15.5 冷启动工作流 (cold_start_service.py)

```
POST /candidates/domains/cold-start
{
    "name": "video_rl",
    "scope": { "tasks": [...], "seed_methods": [...], "modalities": [...], ... }
}
  → Save DomainSpec + skeleton taxonomy/method nodes
  → Query expansion (tasks × methods × modalities)
  → arXiv API harvest → paper_candidates
  → S2 API harvest → paper_candidates
  → Batch DiscoveryScore (deterministic)
  → Auto-promote top K (budget_deep_ingest)
  → Deep ingest anchors (worker 异步)
```

### 15.6 增量同步 (incremental_sync_service.py, 7 个函数)

| 函数 | 频率 | 功能 |
|------|------|------|
| `sync_arxiv_daily()` | 每日 10:00 | 按 domain scope 搜索 arXiv 近 2 天新论文 → 候选 |
| `refresh_citation_counts()` | 每周三 03:00 | S2 API 刷新 L3+ 论文引用数，检测显著增长 |
| `detect_awesome_repo_changes()` | 每周四 04:00 | GitHub README diff → 新论文候选 |
| `detect_lineage_chains()` | 每周五 05:00 | 从 delta_card_lineage 检测 ≥3 节点演化链 |
| `recompute_node_scores()` | 每周六 04:00 | 重算 GraphNodeCandidate 分数，≥85 自动晋升 |
| `detect_duplicate_nodes()` | 每月 1 日 02:00 | 名称归一化去重 → review_queue |
| `cleanup_stale_candidates()` | 每月 15 日 02:00 | 归档 90 天未处理的候选 |

### 15.7 新 API 端点

| Router | 前缀 | 关键端点 |
|--------|------|---------|
| candidates | /candidates | discover/{paper_id}, list, {id}/score, score-batch, {id}/promote, {id}/reject, auto-promote, stats, domains/cold-start |
| pipeline (v6) | /pipeline/v6 | run, shallow/{candidate_id}, deep/{paper_id} |
| pipeline (export) | /pipeline/export | obsidian-vault-v6 |
