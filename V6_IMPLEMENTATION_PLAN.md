# ResearchFlow V6 实现方案

> 从"单篇论文报告系统"升级为"分层知识图谱 + 候选队列 + 递归发现 + 选择性深度吸收"系统。
>
> 核心原则：**后端负责持续构建知识库，Claude Code 负责触发、审查、解释、修正。**

---

## 第一部分：Gap 分析

### 现有系统 vs 目标设计

| 维度 | 现状 | 目标 | 差距评级 |
|------|------|------|---------|
| 论文发现 | S2 refs/citations (max=10), GitHub awesome, OpenAlex sync, HF trending | 全量轻量邻域检索 + 候选池 + 分层吸收 | **需重构** |
| 论文状态机 | 15 个 PaperState，但所有发现的论文直接 ingest | 5 级吸收 (L0→L4) + 候选队列 + 多阶段评分 | **核心缺失** |
| 评分系统 | 4 维分数 (keep/priority/structurality/extensionability) | 4 类分数 (Discovery/DeepIngest/GraphPromotion/Anchor) + score signals + hard caps | **需重设计** |
| Agent 架构 | 单体 pipeline (triage→download→enrich→parse→L3→L4) | 15 个专用 Agent + Evidence Store + Blackboard + Context Pack | **需重构** |
| 节点 Profile | 不存在 | kb_node_profiles (多态关联) + 结构化 JSON | **核心缺失** |
| 边 Profile | 不存在 | kb_edge_profiles (contextual one-liner) | **核心缺失** |
| 引用角色分类 | 不存在 | ReferenceRoleMap (12 种角色 + 递归策略) | **核心缺失** |
| 证据索引 | evidence_units (paper 级) | evidence_items (细粒度: 段落/表格/公式/图表) | **需增强** |
| Agent 上下文 | 全文 + prompt | Context Pack Builder (分层预算) + Blackboard | **核心缺失** |
| 结构化提取 | L4 Step 1+2 (evidence + delta_card) | 10 种 extraction schema | **需增强** |

### 架构决策

| 决策点 | 选择 | 原因 |
|--------|------|------|
| 统一 kb_nodes vs 保留分散表 | **保留分散表** + Profile 层 | 避免重写 45 个 service |
| 评分是 LLM 还是规则 | **LLM 提取 signals → 规则引擎算分** | 可审计、可调参、成本可控 |
| Agent 间通信 | **Blackboard (共享 JSONB 表)** | 避免上下文漂移 |
| PDF 存储 | **处理完删除本地，只留 OSS** | 70GB 磁盘限制 |
| 候选状态机 vs 现有 PaperState | **新建 paper_candidates + 独立状态机** | 不破坏已有 pipeline |

---

## 第二部分：数据模型设计

### 新增表总览 (16 张)

```
核心候选系统:
  paper_candidates          — 候选论文池
  candidate_scores          — 多维评分记录
  score_signals             — 评分信号追溯

证据与上下文:
  evidence_items            — 细粒度证据索引
  agent_runs                — Agent 执行记录
  agent_blackboard_items    — Agent 间共享数据

结构化提取:
  paper_extractions         — 统一提取结果 (PaperEssence/MethodDelta/...)
  reference_role_maps       — 引用角色分类

图谱候选:
  graph_node_candidates     — 节点候选
  graph_edge_candidates     — 边候选

Profile 内容层:
  kb_node_profiles          — 节点介绍
  kb_edge_profiles          — 边的上下文描述

结构化报告:
  paper_reports             — 论文报告
  paper_report_sections     — 报告分节

审核队列:
  review_queue_items        — 统一审核队列 (合并现有 review_tasks)
```

### 扩展现有表 (2 张)

```
  domain_specs              — 新增 scope/budget 字段
  papers                    — 新增 absorption_level / candidate_id
```

---

### 表 1: paper_candidates

```python
class PaperCandidate(Base):
    __tablename__ = "paper_candidates"

    id = Column(UUID, primary_key=True, default=uuid4)

    # ── Identity ──
    title               = Column(Text, nullable=False)
    normalized_title    = Column(Text, nullable=True)  # 去标点小写，用于去重
    arxiv_id            = Column(String(30), nullable=True, index=True)
    doi                 = Column(String(100), nullable=True, index=True)
    s2_paper_id         = Column(String(50), nullable=True)
    openalex_id         = Column(String(50), nullable=True)
    openreview_id       = Column(String(100), nullable=True)
    dblp_id             = Column(String(100), nullable=True)
    paper_link          = Column(Text, nullable=True)

    # ── Discovery Context ──
    discovered_from_paper_id  = Column(UUID, ForeignKey("papers.id"), nullable=True)
    discovered_from_domain_id = Column(UUID, ForeignKey("domain_specs.id"), nullable=True)
    discovery_source    = Column(String(30), nullable=False)
        # s2_reference / s2_citation / s2_recommendation / arxiv_search
        # openalex_topic / awesome_repo / github_readme / hf_daily / hf_trending
        # openreview_venue / dblp_proceedings / manual_import / same_author
    discovery_reason    = Column(String(50), nullable=True)
        # cited_by_target / cites_target / baseline_table / related_work_section
        # formula_context / experiment_comparison / awesome_list_mention
        # same_repo / same_dataset / semantic_recommendation
    relation_hint       = Column(String(30), nullable=True)
        # direct_baseline / method_source / formula_source / dataset_source
        # benchmark_source / comparison_baseline / same_task_prior
        # survey_or_taxonomy / background / unrelated

    # ── Metadata (轻量) ──
    authors_json        = Column(JSONB, nullable=True)
    venue               = Column(String(100), nullable=True)
    year                = Column(SmallInteger, nullable=True)
    abstract            = Column(Text, nullable=True)
    citation_count      = Column(Integer, nullable=True)
    code_url            = Column(Text, nullable=True)
    metadata_json       = Column(JSONB, nullable=True)  # 原始 API 返回

    # ── Status ──
    status = Column(String(25), default="discovered", index=True)
        # discovered → metadata_resolved → shallow_ingested → scored
        # → queued_for_deep → deep_ingested → full_paper → anchor
        # (分支: rejected / archived / merged)
    absorption_level = Column(SmallInteger, default=0)
        # 0=metadata, 1=shallow_card, 2=visible_graph, 3=full_paper, 4=anchor

    # ── Promotion ──
    ingested_paper_id = Column(UUID, ForeignKey("papers.id"), nullable=True)
    reject_reason     = Column(Text, nullable=True)

    # ── Dedup ──
    duplicate_of_candidate_id = Column(UUID, ForeignKey("paper_candidates.id"), nullable=True)
    duplicate_of_paper_id     = Column(UUID, ForeignKey("papers.id"), nullable=True)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
```

### 表 2: candidate_scores

```python
class CandidateScore(Base):
    __tablename__ = "candidate_scores"

    id = Column(UUID, primary_key=True, default=uuid4)
    candidate_id = Column(UUID, ForeignKey("paper_candidates.id", ondelete="CASCADE"), nullable=False)

    # ── 4 类分数 ──
    discovery_score        = Column(Float, nullable=True)   # 0-100
    deep_ingest_score      = Column(Float, nullable=True)   # 0-100
    graph_promotion_score  = Column(Float, nullable=True)   # 0-100
    anchor_score           = Column(Float, nullable=True)   # 0-100

    # ── 可审计细节 ──
    discovery_breakdown    = Column(JSONB, nullable=True)
        # {domain_match, source_signal, graph_proximity, impact_signal,
        #  artifact_signal, novelty_signal, recency_signal}
    deep_ingest_breakdown  = Column(JSONB, nullable=True)
        # {domain_fit, relation_role, reusable_knowledge, evidence_quality,
        #  experiment_value, artifact_value, novelty_freshness}
    hard_caps_applied      = Column(JSONB, nullable=True)
        # [{cap_name, original_score, capped_score, reason}]
    boosts_applied         = Column(JSONB, nullable=True)
        # [{boost_name, value, evidence}]
    penalties_applied      = Column(JSONB, nullable=True)

    # ── Decision ──
    decision = Column(String(25), nullable=True)
        # metadata_only / shallow_ingest / visible_graph / full_paper / anchor_review
    decision_reason = Column(Text, nullable=True)

    score_version = Column(SmallInteger, default=1)
    created_at = Column(DateTime, default=func.now())
```

### 表 3: score_signals

```python
class ScoreSignal(Base):
    __tablename__ = "score_signals"

    id = Column(UUID, primary_key=True, default=uuid4)

    # ── 关联 (多态) ──
    entity_type     = Column(String(30), nullable=False)
        # paper_candidate / graph_node_candidate / graph_edge_candidate
    entity_id       = Column(UUID, nullable=False)

    # ── Signal ──
    signal_name     = Column(String(80), nullable=False)
        # direct_baseline_in_method_section / appears_in_main_experiment_table
        # same_primary_task / has_official_code / citation_velocity_high
        # fills_graph_gap / ...
    signal_value    = Column(JSONB, nullable=False)  # true/false/number/string
    signal_strength = Column(Float, nullable=True)   # 0-1

    # ── 溯源 ──
    evidence_refs   = Column(JSONB, nullable=True)  # [{source, section, quote}]
    producer        = Column(String(30), nullable=False)
        # deterministic / shallow_agent / method_agent / experiment_agent / reference_role_agent
    confidence      = Column(Float, nullable=True)

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_score_signals_entity", "entity_type", "entity_id"),
    )
```

### 表 4: evidence_items

```python
class EvidenceItem(Base):
    __tablename__ = "evidence_items"

    id = Column(UUID, primary_key=True, default=uuid4)
    paper_id = Column(UUID, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)

    # ── 来源类型 ──
    source_type     = Column(String(30), nullable=False)
        # pdf_text / figure / table / formula / abstract / caption
        # webpage / repo_readme / api_response
    source_id       = Column(String(100), nullable=True)  # fig_ref, table_ref 等

    # ── 位置 ──
    section_name    = Column(String(50), nullable=True)
    page            = Column(SmallInteger, nullable=True)
    bbox            = Column(JSONB, nullable=True)  # {x0, y0, x1, y1}

    # ── 内容 ──
    text            = Column(Text, nullable=True)
    image_object_key = Column(Text, nullable=True)     # OSS key
    table_html      = Column(Text, nullable=True)
    formula_latex   = Column(Text, nullable=True)
    token_count     = Column(Integer, nullable=True)

    # ── 向量 ──
    embedding       = Column(Vector(1536), nullable=True)

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_evidence_items_paper", "paper_id"),
        Index("ix_evidence_items_type", "paper_id", "source_type"),
    )
```

### 表 5-6: agent_runs + agent_blackboard_items

```python
class AgentRun(Base):
    __tablename__ = "agent_runs"

    id = Column(UUID, primary_key=True, default=uuid4)
    paper_id        = Column(UUID, ForeignKey("papers.id"), nullable=True)
    candidate_id    = Column(UUID, ForeignKey("paper_candidates.id"), nullable=True)
    domain_id       = Column(UUID, ForeignKey("domain_specs.id"), nullable=True)

    agent_name      = Column(String(50), nullable=False)
        # identity_resolver / source_enricher / pdf_parser / evidence_indexer
        # shallow_paper / reference_role / method_delta_lite / method_delta_full
        # experiment_lite / experiment_full / formula_figure / graph_candidate
        # score / node_profile / edge_profile / paper_report / quality_audit
    phase           = Column(String(20), nullable=False)
        # import / shallow_ingest / deep_ingest / profile / report / audit
    status          = Column(String(15), default="running")
        # running / completed / failed / skipped
    model_name      = Column(String(50), nullable=True)
    prompt_version  = Column(String(20), nullable=True)
    input_token_count  = Column(Integer, nullable=True)
    output_token_count = Column(Integer, nullable=True)
    cost_usd        = Column(Float, nullable=True)
    duration_ms     = Column(Integer, nullable=True)
    error_message   = Column(Text, nullable=True)

    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)


class AgentBlackboardItem(Base):
    __tablename__ = "agent_blackboard_items"

    id = Column(UUID, primary_key=True, default=uuid4)
    run_id       = Column(UUID, ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False)
    paper_id     = Column(UUID, ForeignKey("papers.id"), nullable=True)
    candidate_id = Column(UUID, ForeignKey("paper_candidates.id"), nullable=True)

    item_type    = Column(String(30), nullable=False)
        # paper_essence / method_delta / experiment_matrix / task_facet
        # mechanism_facet / dataset_use / reference_role_map / graph_candidates
        # pipeline_modules / formula_explanations / figure_roles
    value_json   = Column(JSONB, nullable=False)
    confidence   = Column(Float, nullable=True)
    evidence_refs = Column(JSONB, nullable=True)

    producer_agent = Column(String(50), nullable=False)
    is_verified    = Column(Boolean, default=False)  # QualityAudit 通过后 = True

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_blackboard_paper", "paper_id", "item_type"),
        Index("ix_blackboard_candidate", "candidate_id", "item_type"),
    )
```

### 表 7: paper_extractions (统一提取结果)

```python
class PaperExtraction(Base):
    """统一存储各 Agent 的结构化提取结果。
    value_json 的 schema 由 extraction_type 决定。"""
    __tablename__ = "paper_extractions"

    id = Column(UUID, primary_key=True, default=uuid4)
    paper_id         = Column(UUID, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    extraction_type  = Column(String(30), nullable=False)
        # paper_identity / paper_essence / method_delta / task_facet
        # mechanism_facet / dataset_benchmark_use / experiment_matrix
        # reference_role_map / graph_candidates / pipeline_modules
    value_json       = Column(JSONB, nullable=False)
    evidence_refs    = Column(JSONB, nullable=True)
    producer_run_id  = Column(UUID, ForeignKey("agent_runs.id"), nullable=True)
    extraction_version = Column(SmallInteger, default=1)
    review_status    = Column(String(20), default="auto")

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("paper_id", "extraction_type", "extraction_version"),
        Index("ix_extractions_paper_type", "paper_id", "extraction_type"),
    )
```

### 表 8: reference_role_maps

```python
class ReferenceRoleMap(Base):
    __tablename__ = "reference_role_maps"

    id = Column(UUID, primary_key=True, default=uuid4)
    paper_id         = Column(UUID, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    ref_index        = Column(String(10), nullable=True)  # R12, R23 等

    # ── 引用论文信息 ──
    ref_title        = Column(Text, nullable=True)
    ref_arxiv_id     = Column(String(30), nullable=True)
    ref_candidate_id = Column(UUID, ForeignKey("paper_candidates.id"), nullable=True)
    ref_paper_id     = Column(UUID, ForeignKey("papers.id"), nullable=True)

    # ── 角色分类 ──
    role = Column(String(30), nullable=False)
        # direct_baseline / method_source / formula_source / dataset_source
        # benchmark_source / comparison_baseline / same_task_prior_work
        # survey_or_taxonomy / background_citation / implementation_reference
        # unimportant_related_work
    role_confidence  = Column(Float, nullable=True)

    # ── 出现位置 ──
    where_mentioned  = Column(ARRAY(Text), default=[])
        # method / experiment_table / formula / related_work / introduction / abstract
    mention_count    = Column(SmallInteger, default=1)

    # ── 递归策略 ──
    recommended_ingest_level = Column(String(20), nullable=True)
        # full / shallow / metadata_only / ignore
    recommendation_reason = Column(Text, nullable=True)

    evidence_refs = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_ref_role_paper", "paper_id"),
    )
```

### 表 9-10: graph_node_candidates + graph_edge_candidates

```python
class GraphNodeCandidate(Base):
    __tablename__ = "graph_node_candidates"

    id = Column(UUID, primary_key=True, default=uuid4)
    paper_id         = Column(UUID, ForeignKey("papers.id"), nullable=True)
    candidate_id     = Column(UUID, ForeignKey("paper_candidates.id"), nullable=True)

    node_type        = Column(String(20), nullable=False)
        # task / method / mechanism / dataset / benchmark / lineage / lab
    name             = Column(String(200), nullable=False)
    name_zh          = Column(String(200), nullable=True)
    one_liner        = Column(Text, nullable=True)

    # ── 晋升评分 ──
    promotion_score  = Column(Float, nullable=True)  # NodePromotionScore 0-100
    promotion_breakdown = Column(JSONB, nullable=True)
        # {evidence_count, connected_paper_quality, source_diversity,
        #  structural_importance, name_stability, profile_completeness}

    # ── 状态 ──
    status = Column(String(20), default="candidate")
        # candidate / reviewed / canonical / rejected / merged
    promoted_entity_type = Column(String(30), nullable=True)
        # taxonomy_node / method_node / mechanism_family
    promoted_entity_id   = Column(UUID, nullable=True)

    evidence_refs = Column(JSONB, nullable=True)
    confidence    = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now())


class GraphEdgeCandidate(Base):
    __tablename__ = "graph_edge_candidates"

    id = Column(UUID, primary_key=True, default=uuid4)
    paper_id         = Column(UUID, ForeignKey("papers.id"), nullable=True)

    source_entity_type = Column(String(30), nullable=False)
    source_entity_id   = Column(UUID, nullable=True)     # 已有节点
    source_candidate_id = Column(UUID, nullable=True)     # 或候选节点
    target_entity_type = Column(String(30), nullable=False)
    target_entity_id   = Column(UUID, nullable=True)
    target_candidate_id = Column(UUID, nullable=True)

    relation_type      = Column(String(50), nullable=False)
        # proposes_method / evaluates_on / uses_dataset / compares_against
        # modifies_slot / extends_method / cites_as_baseline / cites_as_related
        # part_of_lineage / produced_by_lab / belongs_to_task
    slot_name          = Column(String(100), nullable=True)  # for modifies_slot

    # ── 评分 ──
    confidence_score   = Column(Float, nullable=True)  # EdgeConfidenceScore 0-100
    confidence_breakdown = Column(JSONB, nullable=True)
        # {evidence_directness, relation_specificity, extractor_agreement,
        #  source_reliability, graph_consistency, description_quality}

    # ── 内容 ──
    one_liner          = Column(Text, nullable=True)
        # "本文保留 GRPO 的 group optimization 框架，但修改 reward 以适配长视频时序定位"

    # ── 状态 ──
    status = Column(String(20), default="candidate")
        # candidate / visible / canonical / rejected
    promoted_edge_id   = Column(UUID, nullable=True)  # 晋升后指向 graph_edges/method_edges

    evidence_refs = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=func.now())
```

### 表 11-12: kb_node_profiles + kb_edge_profiles

```python
class KBNodeProfile(Base):
    __tablename__ = "kb_node_profiles"

    id = Column(UUID, primary_key=True, default=uuid4)

    # ── 多态关联到现有表 ──
    entity_type = Column(String(30), nullable=False)
        # taxonomy_node / method_node / mechanism_family / paper / problem_node / domain_spec
    entity_id   = Column(UUID, nullable=False)

    profile_kind = Column(String(20), default="page")
        # card (50字) / page (完整) / obsidian / frontend / mcp_resource
    lang         = Column(String(5), default="zh")

    # ── 内容层次 ──
    one_liner         = Column(Text, nullable=True)
    short_intro_md    = Column(Text, nullable=True)   # 2-3 段 (卡片用)
    detailed_md       = Column(Text, nullable=True)   # 完整页面
    structured_json   = Column(JSONB, nullable=True)  # 节点类型特定字段

    # ── 溯源 ──
    evidence_refs     = Column(JSONB, nullable=True)
    generated_by_run_id = Column(UUID, ForeignKey("agent_runs.id"), nullable=True)
    model_name        = Column(String(50), nullable=True)
    prompt_version    = Column(String(20), nullable=True)
    profile_version   = Column(SmallInteger, default=1)

    # ── 生命周期 ──
    review_status = Column(String(20), default="auto")
        # auto / human_verified / needs_refresh / stale
    staleness_trigger_count = Column(SmallInteger, default=0)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("entity_type", "entity_id", "profile_kind", "lang"),
        Index("ix_node_profiles_entity", "entity_type", "entity_id"),
    )


class KBEdgeProfile(Base):
    __tablename__ = "kb_edge_profiles"

    id = Column(UUID, primary_key=True, default=uuid4)

    # ── 边的两端 ──
    source_entity_type = Column(String(30), nullable=False)
    source_entity_id   = Column(UUID, nullable=False)
    target_entity_type = Column(String(30), nullable=False)
    target_entity_id   = Column(UUID, nullable=False)
    relation_type      = Column(String(50), nullable=False)

    # ── 可选关联到具体边表 ──
    edge_table      = Column(String(30), nullable=True)
        # graph_edges / method_edges / delta_card_lineage / paper_facets / taxonomy_edges
    edge_id         = Column(UUID, nullable=True)

    # ── 内容 ──
    lang             = Column(String(5), default="zh")
    one_liner        = Column(Text, nullable=True)
    relation_summary = Column(Text, nullable=True)
    source_context   = Column(Text, nullable=True)
    target_context   = Column(Text, nullable=True)

    display_priority = Column(SmallInteger, default=5)

    # ── 溯源 ──
    evidence_refs       = Column(JSONB, nullable=True)
    generated_by_run_id = Column(UUID, ForeignKey("agent_runs.id"), nullable=True)
    review_status       = Column(String(20), default="auto")

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("ix_edge_profiles_source", "source_entity_type", "source_entity_id"),
        Index("ix_edge_profiles_target", "target_entity_type", "target_entity_id"),
    )
```

### 表 13-14: paper_reports + paper_report_sections

```python
class PaperReport(Base):
    __tablename__ = "paper_reports"

    id              = Column(UUID, primary_key=True, default=uuid4)
    paper_id        = Column(UUID, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    report_version  = Column(SmallInteger, default=1)
    title_zh        = Column(Text, nullable=True)
    title_en        = Column(Text, nullable=True)

    generated_by_run_id = Column(UUID, ForeignKey("agent_runs.id"), nullable=True)
    model_name      = Column(String(50), nullable=True)
    prompt_version  = Column(String(20), nullable=True)
    review_status   = Column(String(20), default="auto")

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class PaperReportSection(Base):
    __tablename__ = "paper_report_sections"

    id = Column(UUID, primary_key=True, default=uuid4)
    report_id    = Column(UUID, ForeignKey("paper_reports.id", ondelete="CASCADE"))
    section_type = Column(String(30), nullable=False)
        # metadata / core_claim / motivation / pipeline / formula
        # experiment / related_work / lineage / limitations / knowledge_position
    title       = Column(Text, nullable=True)
    body_md     = Column(Text, nullable=True)
    evidence_refs = Column(JSONB, nullable=True)
    order_index = Column(SmallInteger, nullable=False)
    created_at  = Column(DateTime, default=func.now())
```

### 表 15: review_queue_items

```python
class ReviewQueueItem(Base):
    __tablename__ = "review_queue_items"

    id = Column(UUID, primary_key=True, default=uuid4)
    item_type     = Column(String(30), nullable=False)
        # paper_promotion / node_promotion / edge_promotion
        # lineage_edge / metadata_conflict / baseline_promotion
    entity_type   = Column(String(30), nullable=False)
    entity_id     = Column(UUID, nullable=False)

    priority_score = Column(Float, nullable=True)
    reason         = Column(Text, nullable=True)
    suggested_decision = Column(String(20), nullable=True)
    evidence_refs  = Column(JSONB, nullable=True)

    status = Column(String(25), default="pending")
        # pending / accepted / rejected / needs_more_evidence
    reviewed_by = Column(String(50), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    review_notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_review_queue_status", "status", "priority_score"),
    )
```

### DomainSpec 扩展

```python
# 在现有 DomainSpec 上新增:
    scope_modalities      = Column(ARRAY(Text), default=[])
    scope_tasks           = Column(ARRAY(Text), default=[])
    scope_paradigms       = Column(ARRAY(Text), default=[])
    scope_seed_methods    = Column(ARRAY(Text), default=[])
    scope_seed_models     = Column(ARRAY(Text), default=[])
    scope_seed_datasets   = Column(ARRAY(Text), default=[])
    negative_scope        = Column(ARRAY(Text), default=[])

    budget_metadata_candidates = Column(Integer, default=500)
    budget_shallow_ingest      = Column(Integer, default=200)
    budget_deep_ingest         = Column(Integer, default=50)
    budget_anchor_methods      = Column(Integer, default=20)
```

---

## 第三部分：评分系统设计

### 3.1 DiscoveryScore (元数据阶段, 0-100)

**目的**: 只有 metadata，决定是否进一步解析。

```
DiscoveryScore =
    0.25 * DomainMatch         # title/abstract vs domain scope
  + 0.20 * SourceSignal        # 发现来源可信度
  + 0.20 * GraphProximity      # 与现有图谱的距离
  + 0.10 * ImpactSignal        # citation_velocity 归一化
  + 0.10 * ArtifactSignal      # code/data/model 开源情况
  + 0.10 * NoveltySignal       # 填补图谱缺口 / 新机制
  + 0.05 * RecencySignal       # 时效性
  - Penalty                    # 重复度/领域偏离
```

**SourceSignal 权重表:**

| 来源 | 分数 |
|------|------|
| 用户手动 seed | 100 |
| method section 明确引用 | 95 |
| 实验表格 baseline | 90 |
| dataset/benchmark 原论文 | 90 |
| 多个 anchor 共同引用 | 85 |
| OpenReview accepted | 80 |
| 高质量 awesome repo | 70 |
| GitHub README 提到 | 60 |
| S2/OpenAlex 推荐 | 45 |
| 普通关键词搜索 | 30 |

**冷启动特殊规则**: 图谱为空时 GraphProximity 无意义，权重重分配: DomainMatch 0.35, SourceSignal 0.25, GraphProximity 0.05, 其余不变。seed papers 直接给 GraphProximity=90。

**阈值:**

| DiscoveryScore | 决策 |
|---------------|------|
| ≥ 75 | 立即 shallow ingest |
| 60-74 | 候选池，等待批量处理 |
| 40-59 | metadata only，可搜索但不主动解析 |
| < 40 | archive |

### 3.2 DeepIngestScore (shallow 后, 0-100)

**目的**: shallow extraction 后，决定是否做完整报告。

```
DeepIngestScore =
    0.22 * DomainFit           # 领域匹配 (比 Discovery 更精确)
  + 0.28 * RelationRole        # 在图谱中的角色 (最重要)
  + 0.18 * ReusableKnowledge   # 可提取的结构化知识
  + 0.12 * EvidenceQuality     # method+experiment+ablation 证据强度
  + 0.10 * ExperimentValue     # 实验覆盖度和质量
  + 0.06 * ArtifactValue       # code/data 可用性
  + 0.04 * NoveltyFreshness    # 新颖性
  - Penalty
```

**RelationRole 权重表 (最关键子分数):**

| 角色 | 分数 |
|------|------|
| 直接改进的 baseline | 100 |
| 核心方法来源 | 95 |
| 实验表格主要 comparison | 90 |
| benchmark/dataset 原论文 | 90 |
| 多篇后续论文基于它改进 | 90 |
| 已有方法迁移到当前子任务 | 85 |
| 可复用机制提出者 | 80 |
| survey/taxonomy paper | 75 |
| related work 中的一类方法 | 45-60 |
| 背景引用 | 10-35 |

**阈值:**

| DeepIngestScore | 决策 |
|----------------|------|
| ≥ 88 | 自动 full P__Paper，可进入 anchor review |
| 80-87 | full P__Paper + review_needed |
| 68-79 | shallow card, 进入图谱但不生成完整报告 |
| 50-67 | 候选卡片，搜索可见 |
| < 50 | 不进入主图谱 |

### 3.3 GraphPromotionScore (节点/边, 0-100)

**NodePromotionScore:**

```
NodePromotionScore =
    0.25 * EvidenceCount           # 支撑此节点的证据数
  + 0.20 * ConnectedPaperQuality   # 关联论文的平均质量
  + 0.20 * SourceDiversity         # 来自多少不同论文/来源
  + 0.15 * StructuralImportance    # 在图谱中的结构位置
  + 0.10 * NameStability           # 名称是否收敛
  + 0.10 * ProfileCompleteness    # Profile 完整度
  - DuplicatePenalty
```

**各节点类型硬性要求:**

| 节点 | 成为 canonical 的最低条件 |
|------|------------------------|
| M__Method | canonical paper OR 被 ≥2 篇高分论文使用 OR 被作为 baseline |
| C__Mechanism | 出现在 ≥2 个方法中 OR 对应具体公式/模块 |
| T__Task | 有明确输入输出 + 代表性 benchmark + ≥3 篇相关 paper |
| D__Dataset | 有原论文/官方页面 + 被 ≥1 篇 full paper 使用 |
| L__Lineage | ≥3 个节点构成演化链 + 方向清晰 + 含 baseline→改进 |
| Lab__Team | ≥2 篇相关 paper + 有明确组织信息 |

**EdgeConfidenceScore:**

```
EdgeConfidenceScore =
    0.35 * EvidenceDirectness      # 证据直接性
  + 0.20 * RelationSpecificity     # 关系描述精确度
  + 0.15 * ExtractorAgreement      # 多 agent/来源一致
  + 0.15 * SourceReliability       # 来源可信度
  + 0.10 * GraphConsistency        # 与图谱其他边一致
  + 0.05 * DescriptionQuality      # edge profile 质量
  - Penalty
```

**关键边类型的额外要求:**

| 边类型 | 最低 EdgeConfidenceScore |
|--------|------------------------|
| modifies_slot | 80 或人工审核 |
| extends_method | 80 或人工审核 |
| new_baseline_from | 80 或人工审核 |
| part_of_lineage | 80 或人工审核 |
| proposes_method | 70 |
| evaluates_on / uses_dataset | 70 |
| cites_as_related_work | 55 |

### 3.4 AnchorScore (L4 锚点晋升, 0-100)

**目的**: 决定论文/方法是否成为领域 anchor (baseline, 核心数据集, lineage 转折点)。

```
AnchorScore =
    0.25 * DownstreamCount        # 被多少后续论文用作 baseline
  + 0.20 * StructuralImportance   # 在图谱中的拓扑位置 (hub/bridge)
  + 0.20 * GraphCentrality        # PageRank / betweenness
  + 0.15 * EvidenceConsensus      # 多篇论文对其角色描述一致
  + 0.10 * IsEstablishedBaseline  # 已被 ≥3 篇论文用作 baseline
  + 0.10 * CommunityRecognition   # 高引用 + 顶会 + best paper
```

**阈值:**

| AnchorScore | 决策 |
|------------|------|
| ≥ 85 | canonical anchor — 作为领域骨架节点 |
| 70-84 | anchor candidate — 需人工确认 |
| < 70 | 不是 anchor |

**自动晋升条件** (任意一条):
- downstream_count ≥ 5 且 is_established_baseline = true
- 多个 lineage 分支的共同祖先
- dataset 被 ≥10 篇 full paper 使用

### 3.5 Hard Caps (防止分数膨胀)

```python
HARD_CAPS = {
    # DomainFit 不足时限制 DeepIngestScore
    "low_domain_fit": {
        "condition": lambda s: s["domain_fit"] < 50,
        "cap": {"deep_ingest_score": 60},
        "override": "manual_import"
    },
    # 无证据时限制边可信度
    "no_evidence_refs": {
        "condition": lambda s: not s.get("evidence_refs"),
        "cap": {"edge_confidence_score": 55}
    },
    # 只出现在 related work
    "related_work_only": {
        "condition": lambda s: s["relation_role"] == "related_work_section",
        "cap": {"relation_role_score": 60}
    },
    # 背景引用
    "background_only": {
        "condition": lambda s: s["relation_role"] == "background_citation",
        "cap": {"relation_role_score": 35}
    },
    # 无 PDF 内容时不能自动生成 full report
    "no_pdf": {
        "condition": lambda s: not s.get("has_pdf"),
        "cap": {"auto_full_report": False}
    },
    # 无 method/experiment 证据时关键边不能 canonical
    "no_method_evidence": {
        "condition": lambda s: not s.get("has_method_evidence"),
        "cap": {"modifies_slot_canonical": False, "extends_method_canonical": False}
    },
    # 高重复度
    "high_redundancy": {
        "condition": lambda s: s.get("redundancy_score", 0) > 0.7,
        "cap": {"deep_ingest_score": 70}
    },
}

BOOSTS = {
    "direct_baseline_experiment_same_task": {
        "condition": lambda s: all([
            s.get("is_direct_baseline"),
            s.get("in_experiment_table"),
            s.get("same_primary_task")
        ]),
        "boost": 10
    },
    "baseline_changed_slot_ablation": {
        "condition": lambda s: all([
            s.get("is_baseline"),
            s.get("has_changed_slots"),
            s.get("has_ablation")
        ]),
        "boost": 12
    },
    "low_citation_high_quality": {
        "condition": lambda s: all([
            s.get("citation_count", 0) < 10,
            s.get("has_official_code"),
            s.get("has_strong_ablation"),
            s.get("fills_graph_gap")
        ]),
        "boost": 8
    },
    # DiscoveryScore boosts
    "baseline_table_discovered": {
        "condition": lambda s: s.get("discovery_source") == "baseline_table",
        "boost": 5,
        "applies_to": "discovery_score"
    },
    # DeepIngestScore boosts
    "dataset_with_multiple_users": {
        "condition": lambda s: all([
            s.get("is_dataset_source"),
            s.get("used_by_paper_count", 0) >= 3
        ]),
        "boost": 5,
        "applies_to": "deep_ingest_score"
    },
    "method_transfer_new_domain": {
        "condition": lambda s: all([
            s.get("same_method_family"),
            s.get("is_downstream_citation"),
            s.get("is_new_task_application")
        ]),
        "boost": 8,
        "applies_to": "deep_ingest_score"
    },
}
```

---

## 第三点五部分：新旧 Pipeline 关系 + 冷启动 + 状态映射

### 3.6 新旧 Pipeline 关系

**V6 Agent Pipeline 替代现有 L3+L4，不共存。**

| 现有步骤 | V6 替代 | 说明 |
|---------|--------|------|
| L3 skim (1 次 LLM) | ShallowPaperAgent | 输出更结构化 (PaperEssence + TaskFacet + MechanismFacet) |
| L4 Step 1: extract_evidence | MethodDeltaAgent-lite (shallow) + MethodDeltaAgent-full (deep) | 分两阶段，shallow 阶段就能决定是否深读 |
| L4 Step 2: build_delta_card | ExperimentAgent + FormulaFigureAgent | 拆分为独立 agent，各自可重试 |
| L4 Step 3: build_compare_set | GraphCandidateAgent | 从 DB 查 baseline + agent 生成候选边 |
| L4 Step 4: propose_lineage | GraphCandidateAgent (lineage_candidates) | 合并到图谱候选生成 |
| L4 Step 5: synthesize_concept | GraphCandidateAgent (node_candidates) | 合并到图谱候选生成 |
| L4 Step 6: reconcile_neighbors | 事件触发 (staleness++) | 不再同步执行，改为异步 |

**迁移策略**: 新论文走 V6 Agent Pipeline；已有论文的 analysis/delta_card 数据保留不迁移；现有 `analysis_service`/`analysis_steps` 代码保留为 legacy fallback，但默认不调用。

### 3.7 冷启动完整流程

```
POST /api/v1/domains/cold-start
{
    "name": "video_rl",
    "display_name_zh": "视频强化学习",
    "scope": {
        "modalities": ["video", "multimodal"],
        "tasks": ["video question answering", "long video understanding"],
        "paradigms": ["reinforcement learning"],
        "seed_methods": ["GRPO", "PPO", "DPO"],
        "seed_models": ["QwenVL", "InternVL"],
        "seed_datasets": ["VideoMME", "LongVideoBench"],
        "negative_scope": ["pure robotics RL"]
    }
}
```

**Step 1: Save Domain + Build Skeleton Nodes**
```
从 scope.tasks → taxonomy_nodes(dimension=task, status=candidate)
从 scope.seed_methods → method_nodes(maturity=seed, status=candidate)
从 scope.seed_datasets → taxonomy_nodes(dimension=dataset, status=candidate)
从 scope.paradigms → 匹配/创建 paradigm_templates
从 scope.seed_models → taxonomy_nodes(dimension=model_family, status=candidate)
```

**Step 2: Query Expansion**
```
组合 tasks × methods × modalities 生成搜索关键词
示例: ["video QA GRPO", "long video RL reward", "VLM reinforcement learning",
       "video understanding PPO", "multimodal DPO training"]
+ seed_methods 直接作为搜索词
+ seed_datasets 直接作为搜索词
```

**Step 3: Wide Metadata Harvest → paper_candidates**
```
a. GitHub awesome repo 搜索 → 解析 README → 候选
b. arXiv API 搜索 (多个关键词组合) → 候选
c. OpenAlex topic/concept 搜索 → 候选
d. S2 推荐 (从 seed paper titles) → 候选
e. 去重 (normalized_title + arxiv_id + DOI)
```

**Step 4: Batch DiscoveryScore** (冷启动权重: DomainMatch 0.35, GraphProximity 0.05)

**Step 5: Anchor Selection**
```
Top K (K = budget_deep_ingest, 默认 50) 自动 promote
优先级: seed papers > direct baseline > dataset 原论文 > 高 DiscoveryScore
```

**Step 6: Deep Ingest Anchors** — 对 anchor 跑完整 agent pipeline

**Step 7: Build Initial Graph** — 从提取结果创建 T/M/C/D 节点 + 边 + profiles

**Step 8: Selective Recursive Expansion** — ReferenceRoleMap 驱动递归

### 3.8 absorption_level ↔ PaperState 映射

| absorption_level | PaperState | 含义 | 在 papers 表? |
|-----------------|------------|------|-------------|
| L0 (metadata) | — | 只在 paper_candidates | 否 |
| L1 (shallow) | WAIT → L2_PARSED → L3_SKIMMED | shallow agent 完成后 | 是 |
| L2 (visible graph) | L3_SKIMMED | 有 shallow card + 图谱可见 | 是 |
| L3 (full paper) | L4_DEEP | deep agent 完成 + 完整报告 | 是 |
| L4 (anchor) | CHECKED | anchor 审核通过 | 是 |

L0 候选不创建 Paper 记录（只在 paper_candidates 表）。promote 到 L1 时才创建 Paper。

---

## 第四部分：Agent 架构

### 4.1 Agent 总览 (15 个)

```
PaperKnowledgeWorkflow
├── Phase: Import
│   ├── IdentityResolver        — 去重、身份合并 (deterministic)
│   └── SourceEnricher          — 8 API 元数据补全 (deterministic)
│
├── Phase: Shallow Ingest
│   ├── PDFParser               — GROBID + PyMuPDF (deterministic)
│   ├── EvidenceIndexer         — 细粒度证据建索引 (deterministic)
│   ├── ShallowPaperAgent       — PaperEssence + TaskFacet + MechanismFacet (LLM, 6-12K)
│   ├── ReferenceRoleAgent      — 引用分类 + 递归策略 (LLM, 10-30K)
│   ├── MethodDeltaAgent-lite   — 初步 baseline/slot 分析 (LLM, 8-15K)
│   └── ScoreAgent              — 提取 score signals (LLM, 4-8K)
│
├── Phase: Deep Ingest
│   ├── MethodDeltaAgent-full   — 完整 method delta (LLM, 15-30K)
│   ├── ExperimentAgent         — 实验矩阵 + ablation (LLM, 10-25K)
│   ├── FormulaFigureAgent      — 公式推导 + 图表角色 (LLM/VLM, 15-30K)
│   └── GraphCandidateAgent     — 节点/边候选 (LLM, 10-20K)
│
├── Phase: Profile
│   ├── NodeProfileAgent        — 节点介绍生成 (LLM, 8-15K)
│   └── EdgeProfileAgent        — 边上下文描述 (LLM, 6-12K)
│
├── Phase: Report
│   ├── PaperReportAgent        — 完整 10-section 报告 (LLM, 30-80K)
│   └── LineageStoryAgent       — 方法演化叙事页面 (LLM, 15-25K, 按需)
│
└── Phase: Audit
    └── QualityAuditAgent       — 审核 + 标记异常 (LLM, 8-15K)
```

### 4.2 Context Pack Builder

每个 Agent 接收不同的上下文预算：

```python
class ContextPackBuilder:
    """为不同 Agent 组装上下文包。
    4 层上下文: Global → Domain → Paper → Run"""

    PACKS = {
        "shallow_paper": {
            "global": ["node_types", "relation_types"],
            "domain": ["scope", "existing_tasks", "existing_methods"],
            "paper": ["abstract", "introduction_first_3_paragraphs",
                      "method_section_summary", "experiment_section_summary",
                      "figure_table_captions"],
            "run": [],
            "token_budget": 12_000,
        },
        "reference_role": {
            "global": ["reference_role_definitions"],
            "domain": ["anchor_paper_ids"],
            "paper": ["reference_list", "citation_contexts_grouped",
                      "section_with_citations", "experiment_table_mentions"],
            "run": [],
            "token_budget": 30_000,
        },
        "method_delta_lite": {
            "global": ["slot_types"],
            "domain": ["existing_methods_with_slots", "baselines"],
            "paper": ["method_section", "algorithm_blocks",
                      "pipeline_figure_caption", "formula_contexts"],
            "run": ["paper_essence"],
            "token_budget": 15_000,
        },
        "method_delta_full": {
            "global": ["slot_types", "relation_types"],
            "domain": ["method_node_profiles", "baseline_profiles"],
            "paper": ["method_section_full", "algorithm_blocks",
                      "pipeline_figures", "all_formula_contexts",
                      "experiment_baseline_mentions"],
            "run": ["paper_essence", "reference_role_map"],
            "token_budget": 30_000,
        },
        "experiment": {
            "global": ["experiment_schema"],
            "domain": ["known_benchmarks", "known_baselines"],
            "paper": ["all_result_tables", "experiment_section",
                      "ablation_section", "benchmark_descriptions"],
            "run": ["paper_essence", "method_delta"],
            "token_budget": 25_000,
        },
        "graph_candidate": {
            "global": ["node_types", "relation_types", "edge_confidence_rules"],
            "domain": ["existing_graph_summary", "task_hierarchy",
                       "method_hierarchy"],
            "paper": [],
            "run": ["paper_essence", "method_delta", "task_facet",
                    "mechanism_facet", "dataset_use", "experiment_matrix",
                    "reference_role_map"],
            "token_budget": 20_000,
        },
        "paper_report": {
            "global": ["report_section_schema"],
            "domain": [],
            "paper": ["selected_evidence_snippets"],
            "run": ["ALL_VERIFIED_EXTRACTIONS",
                    "node_profiles", "edge_profiles"],
            "token_budget": 80_000,
        },
    }
```

### 4.3 Extraction Schemas

每个 Agent 输出固定 JSON schema，写入 `paper_extractions` 和 `agent_blackboard_items`。

**PaperEssence** (ShallowPaperAgent):
```json
{
    "problem_statement": "...",
    "core_claim": "...",
    "method_summary": "...",
    "main_contributions": ["..."],
    "what_is_new": ["..."],
    "what_is_reused": ["..."],
    "target_tasks": ["Long Video QA"],
    "target_modalities": ["video", "multimodal"],
    "training_paradigm": ["reinforcement learning"],
    "inference_setting": ["offline"],
    "limitations": ["..."],
    "evidence_refs": [{"source": "abstract", "quote": "..."}]
}
```

**MethodDelta** (MethodDeltaAgent):
```json
{
    "proposed_method_name": "VideoRL-X",
    "baseline_methods": [
        {"name": "GRPO", "role": "adapted_baseline", "evidence_refs": [...]}
    ],
    "changed_slots": [
        {"slot": "reward_function", "change_type": "modified",
         "description": "...", "evidence_refs": [...]}
    ],
    "new_components": [...],
    "removed_components": [...],
    "combined_methods": [...],
    "should_create_method_node": true,
    "should_create_lineage_edge": true
}
```

**ReferenceRoleMap** (ReferenceRoleAgent):
```json
{
    "references": [
        {
            "ref_id": "R12",
            "title": "...",
            "role": "direct_baseline",
            "role_confidence": 0.92,
            "where_mentioned": ["method", "experiment_table"],
            "recommended_ingest_level": "full",
            "reason": "Used as main RL baseline, compared in Table 2."
        }
    ]
}
```

**ExperimentMatrix** (ExperimentAgent):
```json
{
    "main_results": [
        {"benchmark": "VideoMME", "metric": "accuracy",
         "proposed_score": "78.3", "baseline_scores": [...],
         "improvement": "+3.2", "evidence_refs": [...]}
    ],
    "ablations": [
        {"component": "temporal_reward", "finding": "...",
         "supports_claim": true, "evidence_refs": [...]}
    ],
    "costs": {"training_cost": "...", "inference_cost": "...", "model_size": "..."}
}
```

**GraphCandidates** (GraphCandidateAgent):
```json
{
    "node_candidates": [
        {"node_type": "method", "name": "VideoRL-X",
         "one_liner": "...", "confidence": 0.82, "evidence_refs": [...]}
    ],
    "edge_candidates": [
        {"source_type": "paper", "source_ref": "current",
         "relation_type": "modifies_slot", "target_type": "method",
         "target_ref": "GRPO", "slot": "reward_function",
         "one_liner": "本文保留 GRPO 的 group optimization 框架，但修改 reward...",
         "confidence": 0.86, "evidence_refs": [...]}
    ]
}
```

---

## 第五部分：Workflow 伪代码

### 5.1 论文导入阶段

```python
async def import_paper(source, domain_id=None):
    # 1. Identity resolve
    identity = await identity_resolver.resolve(source)
    if identity.duplicate_of:
        return merge_or_skip(identity)

    # 2. Create candidate
    candidate = await candidate_store.create(identity, discovery_source, discovery_reason)

    # 3. Enrich metadata
    metadata = await source_enricher.enrich(candidate)
    candidate.status = "metadata_resolved"

    # 4. Compute DiscoveryScore (deterministic)
    signals = await deterministic_scorer.extract_discovery_signals(candidate, domain_id)
    discovery_score = scoring_engine.compute_discovery_score(signals)

    score_record = CandidateScore(
        candidate_id=candidate.id,
        discovery_score=discovery_score.total,
        discovery_breakdown=discovery_score.breakdown,
        hard_caps_applied=discovery_score.caps,
    )

    # 5. Route
    if discovery_score.total >= 80:
        score_record.decision = "shallow_ingest"
        enqueue("paper.shallow_ingest", candidate.id)
    elif discovery_score.total >= 65:
        score_record.decision = "shallow_ingest"  # 但不立即排队
        candidate.status = "scored"
    elif discovery_score.total >= 45:
        score_record.decision = "metadata_only"
        candidate.status = "metadata_resolved"
    else:
        score_record.decision = "archive"
        candidate.status = "archived"

    return candidate
```

### 5.2 Shallow Ingest 阶段

```python
async def shallow_ingest(candidate_id):
    candidate = await load_candidate(candidate_id)
    paper = await promote_to_paper(candidate, level="shallow")
    # paper.absorption_level = 1

    # 1. Parse PDF (deterministic)
    parsed = await pdf_parser.parse(paper)

    # 2. Index evidence (deterministic)
    await evidence_indexer.index(paper.id, parsed)

    # 3. ShallowPaperAgent (LLM, 6-12K tokens)
    ctx = context_pack_builder.build("shallow_paper", paper, domain)
    paper_essence = await run_agent("ShallowPaperAgent", ctx)
    await save_extraction(paper.id, "paper_essence", paper_essence)

    # 4. ReferenceRoleAgent (LLM, 10-30K tokens) ← 防止指数爆炸的关键
    ctx = context_pack_builder.build("reference_role", paper, domain)
    ref_roles = await run_agent("ReferenceRoleAgent", ctx)
    await save_extraction(paper.id, "reference_role_map", ref_roles)
    await save_reference_role_maps(paper.id, ref_roles)

    # 5. MethodDeltaAgent-lite (LLM, 8-15K tokens)
    ctx = context_pack_builder.build("method_delta_lite", paper, domain,
                                     run_items=["paper_essence"])
    method_delta = await run_agent("MethodDeltaAgentLite", ctx)
    await save_extraction(paper.id, "method_delta", method_delta)

    # 6. ScoreAgent — 提取 signals (LLM, 4-8K)
    score_signals = await run_agent("ScoreAgent", {
        "paper_essence": paper_essence,
        "ref_roles": ref_roles,
        "method_delta": method_delta,
    })
    await save_score_signals(paper.id, score_signals)

    # 7. 计算 DeepIngestScore (deterministic engine)
    deep_score = scoring_engine.compute_deep_ingest_score(score_signals)

    # 8. Route
    if deep_score.total >= 88:
        enqueue("paper.deep_ingest", paper.id)
        candidate.absorption_level = 3  # will become full paper
    elif deep_score.total >= 80:
        enqueue("paper.deep_ingest", paper.id, review_needed=True)
    elif deep_score.total >= 68:
        candidate.absorption_level = 2  # visible graph node
        await create_visible_shallow_node(paper)
    else:
        candidate.absorption_level = 1  # stays as shallow card

    # 9. 递归发现 (基于 ReferenceRoleMap)
    for ref in ref_roles["references"]:
        if ref["recommended_ingest_level"] == "full":
            await import_paper(ref, discovery_source="s2_reference",
                             discovery_reason=ref["role"])
        elif ref["recommended_ingest_level"] == "shallow":
            await import_paper(ref, discovery_source="s2_reference",
                             discovery_reason=ref["role"])
        # metadata_only 和 ignore: 跳过或只记录
```

### 5.3 Deep Ingest 阶段

```python
async def deep_ingest(paper_id, review_needed=False):
    paper = await load_paper(paper_id)

    # 从 Blackboard 读取 shallow 阶段的结果
    paper_essence = await get_extraction(paper_id, "paper_essence")
    ref_roles = await get_extraction(paper_id, "reference_role_map")

    # 1. MethodDeltaAgent-full (LLM, 15-30K)
    ctx = context_pack_builder.build("method_delta_full", paper, domain,
                                     run_items=["paper_essence", "reference_role_map"])
    method_delta = await run_agent("MethodDeltaAgentFull", ctx)

    # 2. ExperimentAgent (LLM, 10-25K)
    ctx = context_pack_builder.build("experiment", paper, domain,
                                     run_items=["paper_essence", "method_delta"])
    experiments = await run_agent("ExperimentAgent", ctx)

    # 3. FormulaFigureAgent (LLM/VLM, 15-30K)
    formulas_figures = await run_agent("FormulaFigureAgent", paper)

    # 4. GraphCandidateAgent (LLM, 10-20K)
    ctx = context_pack_builder.build("graph_candidate", paper, domain,
                                     run_items=["ALL"])
    graph_candidates = await run_agent("GraphCandidateAgent", ctx)

    # 5. 计算 GraphPromotionScore (deterministic)
    for node_cand in graph_candidates["node_candidates"]:
        node_cand["promotion_score"] = scoring_engine.compute_node_promotion_score(node_cand)
    for edge_cand in graph_candidates["edge_candidates"]:
        edge_cand["confidence_score"] = scoring_engine.compute_edge_confidence_score(edge_cand)

    # 6. NodeProfileAgent (LLM, per-node)
    for node in [n for n in graph_candidates["node_candidates"] if n["promotion_score"] >= 75]:
        profile = await run_agent("NodeProfileAgent", node, domain_context)
        await save_node_profile(node, profile)

    # 7. EdgeProfileAgent (LLM, batch per-paper)
    for edge in [e for e in graph_candidates["edge_candidates"] if e["confidence_score"] >= 70]:
        profile = await run_agent("EdgeProfileAgent", edge, paper_context)
        await save_edge_profile(edge, profile)

    # 8. PaperReportAgent (LLM, 30-80K)
    ctx = context_pack_builder.build("paper_report", paper, domain,
                                     run_items=["ALL_VERIFIED"])
    report = await run_agent("PaperReportAgent", ctx)
    await save_paper_report(paper_id, report)

    # 9. QualityAuditAgent
    audit = await run_agent("QualityAuditAgent", paper_id)
    for issue in audit["issues"]:
        await create_review_item(issue)

    # 10. Publish
    await publisher.publish_graph_updates(paper_id, graph_candidates)
    paper.absorption_level = 3
    if not review_needed:
        paper.state = PaperState.CHECKED
```

### 5.4 事件触发链

```python
@on_event("paper.l4_completed")
async def handle_l4(paper_id):
    # 更新相关节点 staleness
    for node in get_related_nodes(paper_id):
        await increment_staleness(node)

@on_event("method.promoted_to_baseline")
async def handle_baseline(method_id):
    await refresh_node_profile(method_id)
    await refresh_related_lineages(method_id)
    await create_review_item("baseline_promotion", method_id)

@on_event("node.staleness_threshold_reached")
async def handle_stale(entity_type, entity_id):
    await enqueue("profile.refresh", entity_type, entity_id)
```

---

## 第六部分：Worker 任务 + 定时任务

### 新增 Worker 任务

```python
# ── 候选系统 ──
task_score_candidates(limit=50)                    # 每 2 小时
task_auto_promote_candidates(threshold=80, limit=20)  # 每日
task_cleanup_archived_candidates(days=90)           # 每月

# ── Profile 刷新 ──
task_refresh_stale_profiles(threshold=3, limit=20)  # 每日
task_generate_missing_profiles(limit=50)            # 每周

# ── 递归发现 ──
task_process_reference_roles(limit=30)              # 每 4 小时
task_arxiv_new_papers_sync(domain_ids=[...])        # 每日
task_citation_refresh(limit=50)                     # 每周
task_awesome_repo_diff(domain_ids=[...])            # 每周
task_dblp_proceedings_sync(venues=[...])            # 每月

# ── 图谱维护 ──
task_recompute_node_promotion_scores()              # 每周
task_detect_duplicate_nodes()                       # 每月
task_lineage_detection()                            # 每周
```

---

## 第七部分：MCP 工具升级

### 新增 MCP 工具

| 工具 | 功能 |
|------|------|
| `rf_domain_cold_start` | 基于 domain manifest 冷启动 |
| `rf_candidate_list` | 查看候选队列 (按分数/状态/domain 筛选) |
| `rf_candidate_promote` | 手动提升候选到指定 absorption level |
| `rf_candidate_reject` | 拒绝候选 + 记录原因 |
| `rf_paper_build_neighborhood` | 触发邻域检索 → 候选 (不自动 ingest) |
| `rf_node_profile_get` | 获取节点 Profile |
| `rf_node_profile_refresh` | 触发刷新节点 Profile |
| `rf_edge_profile_get` | 获取边 Profile |
| `rf_graph_get_subgraph` | 获取子图 (节点+边+profiles) |
| `rf_review_queue` | 查看/处理审核队列 |
| `rf_score_explain` | 解释某个候选的评分明细 |
| `rf_extraction_get` | 获取论文的结构化提取结果 |

### 新增 MCP 资源

| URI | 返回 |
|-----|------|
| `node://{entity_type}/{entity_id}` | 节点元数据 + Profile |
| `candidate://queue/{domain_id}` | 候选队列 |
| `review://queue` | 审核队列 |
| `extraction://{paper_id}/{type}` | 结构化提取结果 |

---

## 第八部分：OSS 存储规划

### Bucket 配置

```
Bucket: researchflow-assets
Region: oss-cn-shanghai (与 ECS 同区域，内网免费)
Endpoint: oss-cn-shanghai-internal.aliyuncs.com (内网)

路径规划:
  pdfs/{arxiv_id_or_uuid}.pdf          # 论文原文
  figures/{paper_id}/{fig_label}.png   # 图表截图
  formulas/{paper_id}/{idx}.png        # 公式截图
  backups/db/{date}.sql.gz             # 数据库备份
  exports/vault/{date}.tar.gz          # Vault 快照
```

### 70 GB 磁盘规划

```
系统 + Docker 基础          ~8 GB
Docker 镜像 (7 个)          ~5 GB
PostgreSQL 数据             ~5 GB (含新表，1000 篇上限)
Redis 数据                  ~0.1 GB
GROBID 容器 + 模型          ~2 GB
代码 + node_modules         ~0.3 GB
/tmp (PDF/图表临时处理)     ~2 GB (自动清理)
日志                        ~1 GB
预留空间                    ~15 GB
─────────────────────────
本地合计                    ~38 GB
可用空间                    ~32 GB

OSS: PDF + 图表 + 公式 + 备份 = 无限
```

**关键策略:** PDF 下载到 `/tmp/rf_pdfs/` → 处理 → 上传 OSS → 删除本地。24 小时自动清理。

---

## 第九部分：实施路线图

```
Phase 0 (3 天): 基础设施
  ├── Alembic migration v016: 16 张新表
  ├── OSS bucket 配置 + object_storage.py 适配
  └── PDF 生命周期改造 (处理完删除本地)

Phase 1 (2 周): 候选队列 + 评分引擎
  ├── paper_candidates CRUD
  ├── ScoringEngine (deterministic, 4 类分数)
  ├── Hard caps + Boosts 规则
  ├── 改造 discovery_service (不自动 ingest → 创建 candidate)
  ├── candidates API router (8 端点)
  └── MCP 工具 (candidate_list, promote, reject, score_explain)

Phase 2 (2 周): Shallow Agent 管线
  ├── ContextPackBuilder
  ├── AgentRun + Blackboard 基础设施
  ├── ShallowPaperAgent (PaperEssence)
  ├── ReferenceRoleAgent (ReferenceRoleMap) ← 防爆炸核心
  ├── MethodDeltaAgent-lite
  ├── ScoreAgent (signal extraction)
  ├── DeepIngestScore 计算
  └── 递归发现 (基于 ReferenceRoleMap)

Phase 3 (2 周): Deep Agent 管线
  ├── MethodDeltaAgent-full
  ├── ExperimentAgent
  ├── FormulaFigureAgent
  ├── GraphCandidateAgent
  ├── GraphPromotionScore + EdgeConfidenceScore
  └── graph_node/edge_candidates 管理

Phase 4 (1 周): Profile 层
  ├── kb_node_profiles + kb_edge_profiles
  ├── NodeProfileAgent
  ├── EdgeProfileAgent
  ├── staleness 刷新机制
  └── MCP 工具 (profile_get, profile_refresh)

Phase 5 (1 周): Paper Report
  ├── paper_reports + paper_report_sections
  ├── PaperReportAgent (10 section, 基于已验证提取)
  ├── QualityAuditAgent
  └── review_queue_items

Phase 6 (1 周): Obsidian 导出 V6
  ├── vault_export_v6 (读取 profiles + edge one-liners)
  ├── 新增节点页面: D__ / Lab__
  ├── Lineage story pages
  └── Obsidian 同步优化

Phase 7 (持续): 增量更新
  ├── 新 worker 任务 (arXiv sync, citation refresh, etc.)
  ├── 事件触发链
  ├── 冷启动流程 (domain manifest → 全量)
  └── 月度维护 (taxonomy cleanup, duplicate merge)
```

**总计: ~10 周，16 张新表，15 个 Agent，12 个 MCP 工具，~15 个 Worker 任务**

---

## 第十部分：关键设计决策总结

| # | 决策 | 选择 | 原因 |
|---|------|------|------|
| 1 | 论文是否自动进入 KB | **不自动** — 全部走候选池 | 防止质量稀释、成本爆炸 |
| 2 | 分数是 1 个还是多个 | **4 类** (Discovery/DeepIngest/GraphPromotion/Anchor) | 不同阶段需要不同判据 |
| 3 | 分数由谁算 | **LLM 提取 signals → 规则引擎算分** | 可审计、可调参、确定性 |
| 4 | Agent 是大还是小 | **15 个小 Agent** + Blackboard | 各自可重试、可跳过、成本可控 |
| 5 | Agent 间如何通信 | **Blackboard (JSONB 表)** + Context Pack | 避免上下文漂移 |
| 6 | 统一 kb_nodes 还是保留现有 | **保留现有表** + Profile 层 | 不破坏 45 个 service |
| 7 | 递归扩展策略 | **ReferenceRoleAgent 决定** | 防止指数爆炸 |
| 8 | PDF 存哪里 | **只放 OSS** | 70GB 限制 |
| 9 | 后端 vs Claude Code 分工 | **后端执行，CC 触发/审查** | 可重复、可审计 |
| 10 | 边是否有介绍 | **kb_edge_profiles.one_liner** | 节点页面 wikilink 后自动注入 |
