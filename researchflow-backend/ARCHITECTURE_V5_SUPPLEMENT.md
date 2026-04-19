# ResearchFlow Architecture v5 补充 (2026-04-19)

> 基于 v4.0 ARCHITECTURE.md 的增量更新。核心原则不变（PostgreSQL 唯一真相源，DeltaCard 不可变层），
> 新增四层提取架构、多源元数据、Faceted 分类、VLM 图表分析。

---

## 1. 四层架构

```
Layer 1: 确定性后端 (CPU, 免费)
├── PyMuPDF: 文本/section/图片/caption 提取
├── GROBID (Docker): 结构化 authors/affiliations/references/formulas
├── 公式区域检测: 数学符号聚类 + 区域截图
└── Figure 区域检测: caption 锚定 + 向上扫描

Layer 2: 来源适配器 (6 个 API)
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
├── Acceptance 冲突判断: 按需
└── 必须用 OpenAI SDK + streaming (apicursor 不支持非 streaming)

Layer 4: Agent 编排
├── Skills: 领域知识 + 工作流
├── MCP Tools (22 个): 数据操作接口
└── Review Gates: 低置信度 → 人工审核
```

## 2. Pipeline 完整流程

```
run_full_pipeline(paper_id):
  Step 0: Triage (keep_score, analysis_priority, tier)
  Step 1: Download PDF (arXiv)
  Step 2: Enrich metadata (10 步):
    2.1 arXiv API → title/abstract/authors/year/keywords/comments
    2.2 arXiv comments → 会议中稿解析 ("Accepted at ICLR 2025")
    2.3 Crossref → DOI/venue/year (跳过 placeholder title)
    2.4 OpenAlex → venue/citations/open_access (跳过 placeholder)
    2.5 Semantic Scholar → citation_count/venue
    2.6 GitHub → code_url + README 中稿/数据集提取
    2.7 HuggingFace → models + datasets
    2.8 GitHub README → acceptance + dataset links
    2.9 Project page → acceptance check
    2.10 PDF 首页文本 → acceptance detection
    * 所有结果写 metadata_observations (观察账本)
    * Placeholder title (arxiv ID) 保护：不用它搜其他 API
  Step 2.5: Venue resolution
    → OpenReview SDK (ICLR/NeurIPS decisions + review scores)
    → DBLP proceedings lookup
    → LLM 冲突判断 (多源不一致时)
    → canonical resolver (按 authority_rank 选最优)
  Step 3: L2 Parse (parser ensemble)
    → GROBID: authors(含机构) + references(结构化) + formulas(坐标)
    → PyMuPDF: sections + figure images + captions
    → Figure extraction: caption 锚定 + VLM 分类补漏
    → Formula extraction: 区域检测 + 合并 + VLM OCR → LaTeX
    → 结果合并，冲突标记
  Step 4: L3 Skim (1 次 LLM)
    → problem_summary, method_summary, worth_deep_read, is_plugin_patch
  Step 5: L4 Deep (2 次 LLM)
    Step 1: extract_evidence (key_equations, key_figures, evidence_units)
    Step 2: build_delta_card (changed_slots, structurality_score)
    Step 3: build_compare_set (DB 驱动 baseline 选择)
    Step 4: propose_lineage (builds_on/extends/replaces DAG)
    Step 5: synthesize_concept (CanonicalIdea linking)
    Step 6: reconcile_neighbors (更新相关论文)
  Step 5.5: Post-L4
    → 回填 paper 字段: core_operator, primary_logic, claims
    → 推断 ring (baseline/structural/plugin) 从 structurality_score
    → 设置 role_in_kb
    → Taxonomy assignment: tags/category/keywords → paper_facets
  Step 6: Citation discovery
    → S2 references + citations → 自动 ingest
```

## 3. 新增数据模型 (v5, 11 张表)

### 3.1 Metadata Observation Ledger

```
metadata_observations
├── entity_type: paper / author / venue
├── entity_id: UUID
├── field_name: venue / status / authors / citation_count / code_url
├── value_json: JSONB (原始值)
├── source: arxiv / crossref / openalex / semantic_scholar / dblp / openreview / github
├── authority_rank: 1=最高权威, 10=最低
└── 不直接覆盖 Paper 字段，通过 canonical resolver 选最优

canonical_paper_metadata
├── 从 observations 中 resolve 的 canonical 值
├── unresolved_conflicts: [{field, sources, values}]
└── resolver_version

权威优先级:
  会议中稿: official_conf > openreview > dblp > crossref > arxiv > s2
  引用数: s2 > openalex > crossref > google_scholar
  作者机构: pdf_grobid > openalex > crossref > s2
```

### 3.2 Faceted Taxonomy DAG

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

### 3.3 Problem Nodes (瓶颈下沉到 Task)

```
problem_nodes
├── parent_task_id → 挂在 taxonomy_nodes(task) 下
├── symptom + root_cause + why_common
├── solution_families: [{name, papers}]
└── 不再是顶层图谱节点

problem_claims
├── paper_id + problem_id + claim_type
└── claim_type: mentions / solves / partially_solves / reveals
```

### 3.4 Method Evolution

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

method_applications
├── paper_id + method_id + role
├── role: baseline / adapted_baseline / proposed_method / component
└── task_id + scenario_id + dataset_ids
```

## 4. MCP Tools (22 个)

### v4 原有 (18 个)
search_research_kb, search_ideas, get_paper_report, compare_papers,
import_research_sources, get_digest, get_reading_plan, propose_directions,
run_full_pipeline, discover_related_papers, build_domain,
enqueue_analysis, refresh_assets, record_user_feedback,
get_paper_detail, get_graph_stats, review_queue, submit_review_decision

### v5 新增 (4 个)
- **resolve_venue**: OpenReview + DBLP + arXiv → 会议中稿检测
- **get_paper_citations**: GROBID refs + S2 citing papers
- **get_paper_figures**: 图表 + OSS URLs + VLM 描述
- **get_metadata_conflicts**: 多源元数据冲突查看

## 5. API Routes (116 个)

### v5 新增
- `GET /taxonomy/nodes` — 按维度/状态筛选
- `GET /taxonomy/tree` — 层级树
- `GET /taxonomy/dimensions` — 维度统计
- `GET /taxonomy/paper/{id}/facets` — 论文的分类标签
- `GET /taxonomy/problems` — 任务下的共性问题
- `GET /methods/nodes` — 方法列表
- `GET /methods/nodes/{id}` — 方法详情 (slots + 演化边 + 论文)
- `GET /methods/lineage/{id}` — 方法演化 DAG

## 6. Arq Worker 任务

### v5 新增
- `task_venue_resolve_batch`: 批量会议中稿检测 (daily 07:00)
- `task_fetch_hf_daily_papers`: HuggingFace 趋势论文 (08:00 + 20:00)
- `task_parse_batch`: L2 parse 未处理论文 (每 2 小时)

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

## 8. Obsidian Vault v5 结构

```
00_Home/           导航
10_Tasks/          T__ 任务节点 (含 Common Problems)
20_Methods/        M__ 方法节点 (含 Mermaid 演化图)
30_Mechanisms/     C__ 可复用技术概念
40_Papers/         P__ A/B/C/D 分级
50_Datasets/       D__ 数据集和 Benchmark
80_Assets/         图片资源
90_Views/          动态表格
```

Paper wikilinks 预算: 6-10 个 (T + M + C + D + P)
其余 facets 放 frontmatter YAML, 不生成 wikilinks

## 9. Skills (17 个)

### v5 新增
- **paper-report-v2**: 深度报告 — 公式逐步推导 + pipeline 模块分解 + 递归 Related Work

## 10. 服务器配置

详见 DEPLOY_QUICK.md。关键参数:
- API 容器 1536MB | GROBID 3072MB (独立容器, compose 管理)
- LLM: apicursor.com/v1, model=op-4.6 (支持 vision, 必须 streaming)
- OpenReview: SDK with cached client
- 代理: mihomo allow-lan=true, HF/Docker/OpenReview 走 Proxy 规则
- Docker daemon proxy: /etc/systemd/system/docker.service.d/http-proxy.conf
