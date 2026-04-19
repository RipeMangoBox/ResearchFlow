# V6 实现方案审计报告

---

## 1. 需求覆盖检查

### 1.1 已覆盖的需求 (25/31)

| # | 需求 | 方案对应 | 状态 |
|---|------|---------|------|
| 1 | 候选队列，不直接 ingest | paper_candidates 表 + 5 级吸收 | OK |
| 2 | 三层/五层吸收策略 | L0→L4 absorption_level | OK |
| 3 | DiscoveryScore | 7 维子分数 + hard caps | OK |
| 4 | DeepIngestScore | 7 维子分数 + hard caps + boosts | OK |
| 5 | GraphPromotionScore (节点) | NodePromotionScore 6 维 + 硬性要求 | OK |
| 6 | EdgeConfidenceScore | 6 维 + 边类型最低要求 | OK |
| 7 | Score signals 可追溯 | score_signals 表 | OK |
| 8 | Hard caps 防膨胀 | 7 条 cap + 3 条 boost | OK |
| 9 | T/M/C/P/D/L 节点设计 | 保留现有表 + profile 层 | OK |
| 10 | Node profiles | kb_node_profiles (多态) | OK |
| 11 | Edge profiles (链接有介绍) | kb_edge_profiles (one_liner) | OK |
| 12 | 结构化 Paper Report (10 section) | paper_reports + paper_report_sections | OK |
| 13 | 引用角色分类 (ReferenceRoleMap) | reference_role_maps 表 + 12 种角色 | OK |
| 14 | 递归发现由角色决定 | ReferenceRoleAgent → recommended_ingest_level | OK |
| 15 | 15 个小 Agent 架构 | Agent 总览 + 分 phase | OK |
| 16 | Context Pack Builder | 7 种 pack 配置 + token 预算 | OK |
| 17 | Evidence Store | evidence_items 表 | OK |
| 18 | Blackboard | agent_blackboard_items 表 | OK |
| 19 | 10 种 Extraction Schema | PaperEssence/MethodDelta/... | OK |
| 20 | MCP 工具升级 | 12 个新工具 + 4 个新资源 | OK |
| 21 | 后端执行/CC 审查分工 | 设计原则明确 | OK |
| 22 | OSS 存储规划 | PDF/图表只放 OSS | OK |
| 23 | 审核队列 | review_queue_items | OK |
| 24 | 增量更新 worker | 15 个新任务 | OK |
| 25 | Domain Manifest (scope/budget) | DomainSpec 扩展字段 | OK |

### 1.2 遗漏的需求 (6 个)

| # | 遗漏需求 | 原始要求出处 | 影响 | 修复建议 |
|---|---------|-------------|------|---------|
| **A** | **冷启动骨架节点** | "冷启动时先生成 T/M/C/D skeleton nodes (status=candidate)" | **高** — 冷启动缺少结构引导 | 冷启动流程需新增 "Build Initial Skeleton" 步骤 |
| **B** | **Lab__Team 提取 Agent** | "Lab__Team 可选节点" | **低** — 可推迟 | GROBID affiliations + OpenAlex institutions → 自动创建 |
| **C** | **Lineage Story Agent** | "L__Lineage 不是每条边一个页面，而是一组演化路径稳定时生成故事线" | **中** — Lineage 页面没有叙事生成逻辑 | 新增 LineageStoryAgent 或在 GraphCandidateAgent 中扩展 |
| **D** | **冷启动完整流程** | "Query Expansion → Wide Harvest → Candidate Clustering → Anchor Selection" | **高** — 方案只提到 "冷启动流程升级" 但没有详细步骤 | 需补充完整冷启动 workflow |
| **E** | **AnchorScore** | 需求明确提了 4 类分数，AnchorScore 决定 L4 anchor 晋升 | **中** — 方案提到但没给公式 | 需补充 AnchorScore 公式和阈值 |
| **F** | **新旧 Pipeline 的关系** | 现有 L3+L4 6 步 pipeline 与新 Agent pipeline 的关系 | **高** — 可能冲突 | 需明确: 替代还是共存 |

### 1.3 潜在冲突 (3 个)

| # | 冲突 | 风险 | 解决方案 |
|---|------|------|---------|
| **X1** | **新 Agent Pipeline vs 现有 L3+L4 Pipeline** — ShallowPaperAgent ≈ L3 skim，MethodDeltaAgent ≈ L4 Step 2，但两套都在方案中 | **高** — 同一篇论文可能跑两遍分析 | V6 Agent Pipeline **替代** L3+L4，不共存。现有 analysis_service/analysis_steps 作为 legacy 保留但新论文走新管线 |
| **X2** | **candidate.absorption_level vs paper.state** — 两套状态机并行 | **中** — 状态不一致 | absorption_level 只在 paper_candidates 上，promote 到 papers 后用 PaperState。两者映射: L1→WAIT, L2→L3_SKIMMED, L3→L4_DEEP, L4→CHECKED |
| **X3** | **review_queue_items vs 现有 review_tasks** — 两张审核表 | **低** — 功能重叠 | review_queue_items 替代 review_tasks，迁移后删旧表 |

---

## 2. 全量检索 + 冷启动 + 增量更新

### 2.1 全量检索 (单篇论文邻域)

```
import_paper(source)
  │
  ├─ IdentityResolver: 去重 (arxiv_id / DOI / normalized_title)
  ├─ SourceEnricher: 8 API 补全 metadata
  │   └─ arXiv → Crossref → OpenAlex → S2 → DBLP → OpenReview → GitHub → HF
  ├─ DiscoveryScore 计算 (deterministic)
  │
  ├─ [如果 ≥65] shallow_ingest → PDFParser → EvidenceIndexer
  │
  └─ ReferenceRoleAgent: 全量引用分类
      │  输入: reference_list + citation_contexts + experiment_table_mentions
      │  输出: 每篇引用的 role + recommended_ingest_level
      │
      ├─ role=direct_baseline → import_paper(ref, level="full")       ← 递归
      ├─ role=method_source → import_paper(ref, level="full")         ← 递归
      ├─ role=dataset_source → import_paper(ref, level="full")        ← 递归
      ├─ role=comparison_baseline → import_paper(ref, level="shallow")
      ├─ role=same_task_prior → import_paper(ref, level="shallow")
      ├─ role=background_citation → metadata_only (不递归)
      └─ role=unimportant → ignore

  同时:
  ├─ S2 API → 被引用论文 metadata (cap=100, 分页)
  │   └─ 全部进 paper_candidates (不递归)
  └─ S2 API → related papers (max=20)
      └─ 全部进 paper_candidates (不递归)
```

**关键区分**: 引用论文 (references) 走 ReferenceRoleAgent 做角色分类后选择性递归；被引用论文 (citations) 和推荐论文只进候选池不递归。

### 2.2 冷启动流程 (方案遗漏，此处补全)

```
POST /api/v1/domains/cold-start
{
    "name": "video_rl",
    "scope": { modalities, tasks, paradigms, seed_methods, ... }
}

Step 1: Save DomainSpec + seed taxonomy nodes
  ├─ 从 scope.tasks → 创建 taxonomy_nodes(dimension=task, status=candidate)
  ├─ 从 scope.seed_methods → 创建 method_nodes(status=seed)
  ├─ 从 scope.seed_datasets → 创建 taxonomy_nodes(dimension=dataset, status=candidate)
  └─ 从 scope.paradigms → 匹配/创建 paradigm_templates

Step 2: Query Expansion
  ├─ 组合 tasks × methods × modalities 生成搜索关键词
  └─ 示例: "video QA GRPO", "long video RL reward", "VLM reinforcement learning"

Step 3: Wide Metadata Harvest (全部进 paper_candidates)
  ├─ GitHub awesome repo 搜索 → 解析 README → 候选
  ├─ arXiv API 搜索 (多个关键词组合) → 候选
  ├─ OpenAlex topic/concept 搜索 → 候选
  ├─ S2 推荐 (从 seed papers) → 候选
  └─ 去重 (normalized_title + arxiv_id + DOI)

Step 4: Batch DiscoveryScore (deterministic)
  └─ 对所有候选算分，按分数排序

Step 5: Anchor Selection
  ├─ Top K (K = budget_deep_ingest, 默认 50) 自动 promote
  ├─ 优先: seed papers > baseline > dataset原论文 > 高 DiscoveryScore
  └─ 其余保留在候选池

Step 6: Deep Ingest Anchors
  └─ 对 anchor 论文跑完整 agent pipeline (shallow → deep → profile → report)

Step 7: Build Initial Graph
  ├─ 从 anchor 论文的 GraphCandidates 创建 T/M/C/D 节点
  ├─ 生成 Node Profiles
  ├─ 生成 Edge Profiles
  └─ 检测 Lineage 候选

Step 8: Selective Recursive Expansion
  └─ 从 anchor 论文的 ReferenceRoleMap 递归发现 → 新候选 → 评分 → 选择性吸收
```

**预算参考**:

| 阶段 | 数量 |
|------|------|
| metadata candidates (L0) | 300-1000 |
| shallow ingested (L1) | 100-200 |
| visible graph nodes (L2) | 50-100 |
| full paper nodes (L3) | 20-50 |
| anchor nodes (L4) | 5-15 |

### 2.3 增量更新

| 频率 | 任务 | 触发方式 | 已有/新增 |
|------|------|---------|----------|
| **每日** | arXiv 新论文同步 | cron | 新增 |
| **每日 2x** | HuggingFace Trending | cron | 已有 |
| **每日** | 候选论文评分 | cron 每 2h | 新增 |
| **每日** | 自动 promote (score≥80) | cron | 新增 |
| **每日** | Stale profile 刷新 | cron | 新增 |
| **每周** | S2 citation 扩展 | cron | 新增 |
| **每周** | GitHub awesome repo diff | cron | 新增 |
| **每周** | OpenReview 会议状态 | cron | 已有 (venue_resolve) |
| **每周** | Node promotion 重算 | cron | 新增 |
| **每周** | Lineage 检测 | cron | 新增 |
| **每周** | DBLP proceedings 同步 | cron | 新增 |
| **每月** | Taxonomy 清理 | cron | 新增 |
| **每月** | Duplicate 合并 | cron | 新增 |
| **每月** | 过期候选清理 | cron | 新增 |
| **事件** | 论文导入 → 邻域检索 | event | 新增 |
| **事件** | L4 完成 → staleness++ | event | 新增 |
| **事件** | Baseline 晋升 → profile 刷新 | event | 新增 |
| **事件** | 被引用检测 → 重要下游判断 | event | 新增 |
| **事件** | Awesome repo 变更 → diff 解析 | event | 新增 |

---

## 3. 评分区分度分析

### 3.1 DiscoveryScore 枚举分析

**公式**: `0.25*DomainMatch + 0.20*SourceSignal + 0.20*GraphProximity + 0.10*ImpactSignal + 0.10*ArtifactSignal + 0.10*NoveltySignal + 0.05*RecencySignal`

**场景枚举 (子分数 0-100 → 加权后 0-100)**:

| 场景 | Domain | Source | Graph | Impact | Artifact | Novelty | Recency | **总分** | **决策** |
|------|--------|--------|-------|--------|----------|---------|---------|---------|---------|
| A. 用户手动 seed 论文 | 95 | 100 | 90 | 80 | 80 | 70 | 80 | **89.8** | 立即 shallow |
| B. method 段引用的直接 baseline | 90 | 95 | 95 | 90 | 80 | 40 | 50 | **82.0** | 立即 shallow |
| C. 实验表 comparison baseline | 85 | 90 | 85 | 70 | 60 | 50 | 60 | **76.3** | 候选池 |
| D. 同领域 dataset 原论文 | 85 | 90 | 80 | 70 | 90 | 60 | 40 | **77.3** | 候选池 |
| E. 高质量 awesome repo 论文 | 80 | 70 | 65 | 60 | 70 | 60 | 70 | **69.0** | 候选池 |
| F. S2 推荐的相关论文 | 70 | 45 | 55 | 50 | 40 | 50 | 60 | **55.5** | metadata only |
| G. 低关键词命中的搜索结果 | 40 | 30 | 30 | 50 | 20 | 30 | 50 | **34.5** | archive |
| H. 高引用但偏离领域 | 30 | 45 | 40 | 95 | 60 | 30 | 60 | **46.0** | metadata only |
| I. 纯背景引用 | 50 | 30 | 25 | 40 | 20 | 20 | 40 | **33.5** | archive |
| J. 近期低引用但填补图谱空白 | 90 | 60 | 70 | 40 | 90 | 90 | 95 | **73.8** | 候选池 |
| J+boost | 同上 + low_citation_high_quality boost +8 | | | | | | | **81.8** | 立即 shallow |

**区分度判断**: A-B (82-90) vs C-E (69-77) vs F-H (34-56) → **三档清晰分离**，最窄间距 5 分 (E→F)。

**问题发现**:
- **场景 C (实验表 baseline): 76.3** — 落在候选池 (65-79)，不会立即 shallow ingest。但实验表中的 baseline 通常很重要。
- **修复**: 对 `discovery_source=baseline_table` 加 boost +5，使其达到 ~81 进入立即 shallow。

### 3.2 DeepIngestScore 枚举分析

**公式**: `0.22*DomainFit + 0.28*RelationRole + 0.18*ReusableKnowledge + 0.12*EvidenceQuality + 0.10*ExperimentValue + 0.06*ArtifactValue + 0.04*NoveltyFreshness`

| 场景 | Domain | Relation | Reuse | Evidence | Experiment | Artifact | Novelty | **总分** | **决策** |
|------|--------|----------|-------|----------|------------|----------|---------|---------|---------|
| A. 直接 baseline (改 slot) | 90 | 100 | 90 | 85 | 85 | 80 | 30 | **88.7** | auto full paper |
| A+boost | +10 (baseline+experiment+same_task) | | | | | | | **98.7** | auto full + anchor review |
| B. dataset/benchmark 原论文 | 85 | 90 | 85 | 85 | 90 | 90 | 40 | **85.6** | full + review |
| C. 下游改进论文 (结构性) | 90 | 85 | 90 | 85 | 80 | 75 | 80 | **86.0** | full + review |
| C+boost | +12 (baseline+changed_slot+ablation) | | | | | | | **98.0** | auto full |
| D. 可复用机制提出论文 | 85 | 80 | 95 | 80 | 75 | 70 | 85 | **83.8** | full + review |
| E. 同任务的增量改进 | 80 | 60 | 60 | 70 | 70 | 60 | 50 | **65.0** | 候选卡片 |
| F. related work 中的方法 | 70 | 50 | 50 | 60 | 60 | 40 | 40 | **54.8** | 候选卡片 |
| G. 背景引用 (hard cap 35) | 50 | 25 | 30 | 40 | 30 | 20 | 20 | **30.7** | 不进入图谱 |
| H. 高引用偏领域 (cap DomainFit<50→总分 cap 60) | 45 | 70 | 60 | 65 | 60 | 50 | 30 | ~~58.1~~ → **60** (capped) | 候选卡片 |

**区分度判断**: A (89-99) vs B-D (84-98) vs E (65) vs F (55) vs G (31) → **五档清晰分离**。

**问题发现**:
- **场景 E (同任务增量改进): 65** — 刚好在 "候选卡片" 下限。如果加上方法改进但 ablation 不足，可能到 60 左右，在 50-67 范围的中间。这是合理的 — 增量改进不需要 full report。
- **场景 B (dataset 原论文): 85.6** — 落在 "full + review" (80-87)，不是 auto full。但 dataset 论文通常应该直接吸收。
- **修复**: 对 dataset/benchmark 原论文加 boost +5 `(dataset_used + benchmark_metrics + multiple_papers_use)`，使其达到 ~91 进入 auto full。

### 3.3 阈值合理性总结

| 分数 | 当前阈值 | 评估 | 调整建议 |
|------|---------|------|---------|
| Discovery ≥80 立即 shallow | **偏高** — 实验表 baseline (76) 不能立即进入 | 降至 **75** 或给 baseline_table boost +5 |
| Discovery 65-79 候选池 | **合理** | 保持 |
| Discovery 45-64 metadata only | **合理** | 保持 |
| Discovery <45 archive | **合理** | 保持 |
| DeepIngest ≥88 auto full | **合理** — 只有直接 baseline + boost 才能达到 | 保持 |
| DeepIngest 80-87 full + review | **合理** | 保持 |
| DeepIngest 68-79 shallow card | **合理** | 保持 |
| DeepIngest 50-67 候选卡片 | **合理** | 保持 |
| NodePromotion ≥85 canonical | **合理** | 保持 |
| EdgeConfidence ≥85 canonical | **合理** | 保持 |
| modifies_slot/extends 最低 80 | **合理** — 防止弱证据的关键边 | 保持 |

**建议调整**:
1. DiscoveryScore 阈值: 立即 shallow 从 80 降至 **75**
2. 新增 boost: `baseline_table_discovered` → DiscoveryScore +5
3. 新增 boost: `dataset_benchmark_with_multiple_users` → DeepIngestScore +5

---

## 4. Agent 详细分析

### 4.1 Agent 职责矩阵

| Agent | 输入 | 输出 | 需要 LLM? | Token 预算 | 全量/增量 | 图谱作用 |
|-------|------|------|----------|-----------|----------|---------|
| IdentityResolver | URL/title/arxiv_id | canonical identity + dedup | 否 | 0 | 全量+增量 | 无 (去重) |
| SourceEnricher | candidate metadata | enriched metadata + observations | 否 | 0 | 全量+增量 | 无 (数据) |
| PDFParser | PDF file | sections + figures + captions + refs | 否 | 0 | 全量+增量 | 无 (提取) |
| EvidenceIndexer | parsed sections | evidence_items (细粒度索引) | 否 | 0 | 全量+增量 | 无 (索引) |
| **ShallowPaperAgent** | abstract + intro + method要点 + experiment要点 | PaperEssence + TaskFacet + MechanismFacet | **是** | 6-12K | 全量+增量 | **T, C 候选** |
| **ReferenceRoleAgent** | reference list + citation contexts | ReferenceRoleMap (每篇引用的角色) | **是** | 10-30K | 全量+增量 | **递归发现控制** |
| **MethodDeltaAgent-lite** | method section 要点 + PaperEssence | 初步 MethodDelta (baseline/slot) | **是** | 8-15K | 全量+增量 | **M 候选** |
| **ScoreAgent** | PaperEssence + RefRoles + MethodDelta | score_signals (结构化信号) | **是** | 4-8K | 全量+增量 | 无 (评分) |
| **MethodDeltaAgent-full** | method全文 + formulas + baselines profiles | 完整 MethodDelta | **是** | 15-30K | 全量 (deep only) | **M 节点 + 边** |
| **ExperimentAgent** | tables + experiment + ablation | ExperimentMatrix | **是** | 10-25K | 全量 (deep only) | **D 关联** |
| **FormulaFigureAgent** | figures + formulas + pipeline | 公式推导 + 图表角色分类 | **是/VLM** | 15-30K | 全量 (deep only) | **C 细节** |
| **GraphCandidateAgent** | 所有前序提取结果 + 图谱上下文 | node_candidates + edge_candidates | **是** | 10-20K | 全量+增量 | **T/M/C/D/L 全部** |
| **NodeProfileAgent** | node metadata + top papers + edges | one_liner + short_intro + detailed | **是** | 8-15K/node | 全量+增量(stale) | **所有节点内容** |
| **EdgeProfileAgent** | source/target node + relation + evidence | one_liner + summary | **是** | 6-12K/batch | 全量+增量(stale) | **所有边内容** |
| **PaperReportAgent** | 所有已验证提取 + profiles | 10 section 完整报告 | **是** | 30-80K | 全量 (deep only) | **P 节点内容** |
| **QualityAuditAgent** | 论文全部产出 | issues + review_queue_items | **是** | 8-15K | 全量+增量 | **审核门控** |

### 4.2 全量 vs 增量 Agent 分工

**全量 (冷启动 + 新论文完整入库)**:
```
Import:  IdentityResolver → SourceEnricher
Shallow: PDFParser → EvidenceIndexer → ShallowPaperAgent → ReferenceRoleAgent → MethodDeltaAgent-lite → ScoreAgent
Deep:    MethodDeltaAgent-full → ExperimentAgent → FormulaFigureAgent → GraphCandidateAgent
Profile: NodeProfileAgent → EdgeProfileAgent
Report:  PaperReportAgent → QualityAuditAgent
```

**增量 (定时/事件触发)**:
```
新论文发现:      IdentityResolver → SourceEnricher → DiscoveryScore → [可能] Shallow
Citation 扩展:   S2 API → paper_candidates (metadata only)
Awesome diff:   GitHub API → paper_candidates (metadata only)
Profile 刷新:    NodeProfileAgent (stale nodes only)
                EdgeProfileAgent (stale edges only)
Lineage 检测:   deterministic (不需要 LLM)
Node 重算:      deterministic scoring engine (不需要 LLM)
```

### 4.3 图谱节点 ← Agent 对应关系

| 节点类型 | 创建来源 Agent | Profile 生成 Agent | 更新触发 |
|----------|-------------|-------------------|---------|
| **T__Task** | ShallowPaperAgent (TaskFacet) → GraphCandidateAgent | NodeProfileAgent | 新论文关联时 staleness++ |
| **M__Method** | MethodDeltaAgent → GraphCandidateAgent | NodeProfileAgent | 新变体出现 / baseline 晋升 |
| **C__Mechanism** | ShallowPaperAgent (MechanismFacet) → GraphCandidateAgent | NodeProfileAgent | 新论文使用此机制 |
| **P__Paper** | promote_to_paper() + PaperReportAgent | PaperReportAgent | 元数据更新 / citation 变化 |
| **D__Dataset** | ExperimentAgent → GraphCandidateAgent | NodeProfileAgent | 新论文在此 dataset 评测 |
| **L__Lineage** | GraphCandidateAgent (lineage_candidates) + deterministic 检测 | **(遗漏: 需 LineageStoryAgent)** | 新论文加入演化链 |
| **Lab__Team** | GROBID affiliations + OpenAlex institutions (deterministic) | NodeProfileAgent | 新论文来自此团队 |
| **边** | GraphCandidateAgent (edge_candidates) | EdgeProfileAgent | 新证据出现 |

---

## 5. API 调用成本分析

### 5.1 每篇论文的调用明细

**外部 HTTP API (元数据, 不计费)**:

| API | 调用次数 | 说明 |
|-----|---------|------|
| arXiv API | 1 | 元数据 |
| Crossref | 0-1 | 有 DOI 时 |
| OpenAlex | 1 | 元数据 + citations |
| Semantic Scholar | 1-2 | metadata + refs/citations |
| DBLP | 0-1 | venue 验证 |
| OpenReview | 0-1 | 有 forum_id 时 |
| GitHub | 0-2 | code search + README |
| HuggingFace | 0-1 | models/datasets |
| GROBID | 1 | PDF 结构化 |
| **小计** | **5-10** | **免费** |

**LLM API 调用 (计费)**:

| 阶段 | Agent | 调用次数 | 输入 tokens | 输出 tokens | 估算成本 |
|------|-------|---------|-----------|-----------|---------|
| **L1 Shallow** | ShallowPaperAgent | 1 | 8K | 2K | $0.04 |
| | ReferenceRoleAgent | 1 | 20K | 3K | $0.09 |
| | MethodDeltaAgent-lite | 1 | 12K | 2K | $0.06 |
| | ScoreAgent | 1 | 6K | 1K | $0.03 |
| **L1 小计** | | **4** | **46K** | **8K** | **$0.22** |
| **L3 Deep** | MethodDeltaAgent-full | 1 | 25K | 4K | $0.12 |
| | ExperimentAgent | 1 | 18K | 3K | $0.09 |
| | FormulaFigureAgent | 1-2 | 20K | 3K | $0.10 |
| | GraphCandidateAgent | 1 | 15K | 3K | $0.08 |
| | NodeProfileAgent | 2-5 | 10K×N | 1K×N | $0.04-0.10 |
| | EdgeProfileAgent | 1 (batch) | 10K | 2K | $0.05 |
| | PaperReportAgent | 1 | 50K | 8K | $0.24 |
| | QualityAuditAgent | 1 | 12K | 2K | $0.06 |
| **L3 小计** | | **9-12** | **160-200K** | **26-34K** | **$0.78-1.08** |
| **L2 VLM** | Figure 分类 | 0-1 | images | 1K | $0.02 |
| | Formula OCR | 0-1 | images | 1K | $0.02 |
| **VLM 小计** | | **0-2** | | | **$0-0.04** |

**总成本 (按论文吸收级别)**:

| 级别 | LLM 调用次数 | HTTP 调用次数 | 估算成本/篇 |
|------|------------|------------|-----------|
| **L0 metadata only** | **0** | 5-10 | **$0** |
| **L1 shallow card** | **4** | 5-10 + GROBID | **$0.22** |
| **L2 visible graph** | **4** | 同 L1 | **$0.22** |
| **L3 full paper** | **13-18** | 同 L1 | **$1.00-1.34** |
| **L4 anchor** | **13-18** | 同 L1 | **$1.00-1.34** + 人工审核 |

**注**: 以 Claude Sonnet 价格估算 ($3/M input, $15/M output)。用 Claude Haiku 或 GPT-4o-mini 可降低 3-5x。

### 5.2 冷启动成本估算

| 阶段 | 论文数 | 级别 | LLM 调用 | 估算成本 |
|------|-------|------|---------|---------|
| Wide harvest | 500 | L0 | 0 | $0 |
| Shallow ingest | 150 | L1 | 600 | $33 |
| Deep ingest | 40 | L3 | 520-720 | $40-54 |
| Anchor | 10 | L4 | (含在 L3) | (含在 L3) |
| Profile 生成 | ~100 nodes, ~200 edges | - | ~100 | $5-10 |
| **总计** | **500** | | **~1220-1420** | **$78-97** |

### 5.3 对比现有系统

| 指标 | V5 现有 | V6 方案 | 变化 |
|------|--------|--------|------|
| 每篇 full pipeline LLM 调用 | 3-5 | 13-18 | 3-4x |
| 每篇 full pipeline 成本 | $0.10-0.30 | $1.00-1.34 | 3-10x |
| 冷启动 50 篇深度 + 100 浅层 | 150-250 calls, $15-50 | 1200-1400 calls, $78-97 | 5-6x |
| 单篇 metadata 检索成本 | $0 (但直接 ingest) | $0 (不 ingest) | — |
| 产出 | analysis + DeltaCard | analysis + 10-section report + profiles + scored graph | 显著更丰富 |

---

## 6. 需要修复的问题清单

| # | 问题 | 严重度 | 修复方案 |
|---|------|--------|---------|
| 1 | **冷启动缺少骨架节点和 Query Expansion 步骤** | 高 | 补充完整冷启动 workflow (本文 §2.2 已补全) |
| 2 | **新旧 Pipeline 关系未明确** | 高 | V6 Agent Pipeline 替代 L3+L4；现有 analysis_service 作为 legacy fallback 保留 |
| 3 | **AnchorScore 没有公式** | 中 | AnchorScore = 权重组合 (downstream_count, is_baseline, structural_importance, graph_centrality) |
| 4 | **Lineage Story 缺少生成 Agent** | 中 | 在 Phase 6 增加 LineageStoryAgent，或扩展 GraphCandidateAgent |
| 5 | **实验表 baseline 的 DiscoveryScore (76) 低于立即 shallow 阈值 (80)** | 中 | DiscoveryScore 阈值降至 75，或给 baseline_table boost +5 |
| 6 | **Dataset 原论文 DeepIngestScore (85.6) 需要 review** | 低 | 给 dataset_with_multiple_users boost +5 |
| 7 | **absorption_level 和 PaperState 映射未定义** | 低 | L0=candidate only, L1→WAIT, L2→L3_SKIMMED, L3→L4_DEEP, L4→CHECKED |
| 8 | **review_queue_items 与 review_tasks 重复** | 低 | Phase 0 迁移 review_tasks → review_queue_items |
| 9 | **Lab__Team 没有专门提取 Agent** | 低 | 从 GROBID affiliations + OpenAlex institutions 确定性提取，不需要 LLM |
| 10 | **DiscoveryScore 的 GraphProximity 在冷启动时图谱为空** | 低 | 冷启动时 GraphProximity 权重降低，DomainMatch 权重提升；或给 seed papers GraphProximity=90 |
