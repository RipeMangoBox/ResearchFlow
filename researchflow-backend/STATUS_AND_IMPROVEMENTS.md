# ResearchFlow 系统现状分析与改进路线

> 截至 2026-04-18 的完整状态评估

---

## 一、已实现功能清单

### 核心管线 (完整)

| 功能 | 实现 | 质量 |
|------|------|------|
| 论文导入 (链接/PDF/批量) | ingestion_service + import API | 生产可用 |
| 去重 (arxiv_id/doi/title) | ingestion_service._find_duplicate | 生产可用 |
| 元数据补全 (arXiv/Crossref) | enrich_service | 生产可用，需配 API key |
| 4 维评分 (keep/priority/struct/ext) | triage_service | 生产可用 |
| L2 PDF 解析 (章节/公式/表格/图注) | parse_service + pymupdf | 生产可用 |
| L3 轻量分析 (skim card) | analysis_service + LLM | 需 API key |
| L4 深度分析 (full report) | analysis_service + LLM | 需 API key |
| IdeaDelta 自动生成 | graph_service + analysis post-hook | 需 API key |
| GraphEdge 自动创建 | graph_service.create_edges_for_idea | 生产可用 |
| 发布审核 (evidence gate) | graph_service.check_publish | 生产可用 |
| 向量嵌入 (OpenAI/mock) | embedding_service | 需 API key |
| 混合搜索 (keyword+semantic+structured) | search_service | 生产可用 |
| 5 路图查询 (citation/bottleneck/mechanism/transfer/synthesis) | graph_query_service | 生产可用 |
| 汇报报告 (30s/5min/deep) | report_service | 需 API key |
| 分层阅读推荐 | reading_planner | 生产可用 |
| 方向推荐 (propose + expand) | direction_service | 需 API key |
| 日/周/月总结 | digest_service | 需 API key |
| 用户反馈/收藏/事件 | feedback_service | 生产可用 |
| 库外输入状态机 (ephemeral) | ingestion_service | 生产可用 |
| 过期清理 | ingestion_service.cleanup_expired | 生产可用 |

### 基础设施 (完整)

| 功能 | 实现 |
|------|------|
| 23 张 DB 表 | 4 次 Alembic 迁移 |
| Docker Compose (6 服务) | 生产配置 + dev 配置 |
| 对象存储 (本地 + COS) | object_storage.py |
| MCP Server (10 工具) | mcp/server.py |
| arq Worker (5 task + 3 cron) | workers/arq_app.py |
| LLM 追踪 (model_runs) | llm_service.py |
| 一键部署脚本 | deploy.sh |
| 前端 7 页 | Next.js standalone |

### 知识图谱 (核心已实现)

| 功能 | 实现 |
|------|------|
| 4 个 ParadigmFrame | RL/VLM/Agent/MotionGen |
| 25 个类型化 Slot | 按领域分配 |
| 19 个 MechanismFamily | 层级分组 |
| IdeaDelta 核心对象 | 自动从 L4 生成 |
| 统一边表 (12 种边) | graph_edges |
| assertion_source 区分 | 每条边标注来源 |
| Evidence gate | publish_status 硬约束 |

---

## 二、未实现 / 需改进的功能

### P0 — 高优先级（影响核心质量）

#### 1. EvidenceUnit 独立持久化尚未完全打通

**现状**：L4 分析后 `_build_idea_graph` 会调用 `persist_evidence_units`，但 mock 模式下 LLM 返回的 evidence_units 为空数组，导致实际没有 evidence 行被创建。

**需要**：
- 真实 LLM 运行验证（配 Anthropic key 后测试）
- L4 prompt 确保 evidence_units 字段总是返回数组
- 添加 fallback：如果 LLM 没返回 evidence，从 confidence_notes 自动提取

**工作量**：~2h

#### 2. Entity Resolution 服务

**现状**：设计了但未实现。当前不同论文对同一个 mechanism 可能用不同名字（如 "flow matching" vs "flow_matching" vs "continuous normalizing flow"），不会自动归一。

**需要**：
- `entity_resolution_service.py`：别名归一 + 近义 MechanismFamily 归并
- Slot 名称标准化（同一概念不同叫法）
- 论文标题去重增强（fuzzy match）

**工作量**：~4h

#### 3. 真实 LLM 端到端验证

**现状**：所有 LLM 功能都在 mock 模式下验证了结构正确性，但未配 API key 跑过真实分析。

**需要**：
- 配 Anthropic API key
- 对 5 篇论文跑完整 L3→L4→IdeaDelta→Evidence→Edge 管线
- 验证：delta_statement 质量、changed_slots 准确性、evidence 是否真的链接了
- 调优 L4 prompt 确保 JSON 输出稳定

**工作量**：~3h

### P1 — 中优先级（提升使用体验）

#### 4. OpenAlex 对接

**现状**：`papers.openalex_id` 字段已存在，但无自动填充和数据拉取。

**需要**：
- `openalex_service.py`：按 DOI/title 查 OpenAlex API
- 自动填充 cited_by_count、authors、topic hierarchy
- 作为 enrich worker 的一步

**工作量**：~4h

#### 5. Paper-to-Paper Citation 边

**现状**：graph_edges 支持 cites 边类型，但无自动创建逻辑。

**需要**：
- 从 L2 extracted_sections["references"] 解析引用列表
- 匹配库内论文，创建 cites 边
- 或从 OpenAlex citation data 导入

**工作量**：~4h

#### 6. 前端图查询页面

**现状**：后端 graph API 已有 12 个端点，但前端没有对应的图查询页面。

**需要**：
- `/graph` 页面：图统计 + 查询界面
- 按 mechanism/bottleneck/paradigm 浏览 IdeaDeltas
- 可视化边关系（简单表格即可，不需要力导向图）

**工作量**：~4h

#### 7. IdeaDelta-to-IdeaDelta 边

**现状**：schema 支持 same_mechanism_as / patch_of / structural_variant_of / contradicts / transferable_to，但无自动生成逻辑。

**需要**：
- 比较同领域 IdeaDelta 的 changed_slots，自动推断关系
- 用 embedding 相似度 + slot overlap 判断 same_mechanism_as
- 用 structurality_score 差异判断 patch_of vs structural_variant_of

**工作量**：~6h

### P2 — 低优先级（长期增强）

#### 8. 评测集 (Gold Set)

**现状**：设计文档提到 200 篇 gold set，未实现。

**需要**：
- 选 50 篇标杆论文，人工标注 primary_bottleneck、changed_slots、top-3 ideas
- 10 个查询 + 10 个重构场景
- 每次改 prompt/model/taxonomy 后回放

#### 9. 多模态证据增强

**现状**：EvidenceUnit.atom_type 支持 formula/table/figure，但 L2 解析的公式/表格/图注未自动关联到 evidence。

**需要**：
- L2 解析结果自动生成 EvidenceUnit (basis=text_stated)
- 公式块 → formula 类型 evidence
- 表格标题 → table 类型 evidence

#### 10. 图可视化（Neo4j projection）

**现状**：PostgreSQL 是唯一真相源，无图可视化。

**需要**：
- Neo4j read-model：从 PostgreSQL 同步 graph_edges
- 前端可视化组件
- 或用 D3.js 直接渲染 JSON

#### 11. 增量图更新

**现状**：每次 L4 分析都是全量生成 IdeaDelta + edges。

**需要**：
- 检测已有 IdeaDelta，只更新变化部分
- 保留人工验证状态（human_verified 的边不被覆盖）
- 版本追踪

#### 12. 跨领域迁移检测

**现状**：transfer_atoms 表存在，graph_edges 支持 transferable_to，但无自动检测。

**需要**：
- 比较不同领域 IdeaDelta 的 mechanism 相似性
- LLM 辅助判断 "这个 RL insight 能否用在 VLM"
- 自动生成 transferable_to 边

#### 13. 三角色复核管线

**现状**：设计了 extractor/auditor/taxonomy_reviewer 三角色，未实现。

**需要**：
- extractor：当前 L4 analysis 就是
- auditor：检查 evidence 够不够、是否误读
- taxonomy_reviewer：检查 slot 分配是否准确
- 可做成 arq 任务，高 importance 论文自动触发

#### 14. 阿里云 OSS 对象存储后端

**现状**：COS 后端已实现，OSS 未实现。

**需要**：
- `object_storage.py` 增加 OSSStorage 类
- 使用 oss2 SDK

---

## 三、系统架构可能的风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| **LLM JSON 输出不稳定** | IdeaDelta/evidence 解析失败 | _parse_json_response 已有 fallback，但需真实测试 |
| **单 PostgreSQL 承载图+向量+OLTP** | 大规模时性能下降 | 5000 篇内够用；之后拆 Qdrant 或 Neo4j |
| **Mock 模式掩盖 prompt 问题** | 上线后发现 LLM 输出格式不对 | 先配 key 跑 5 篇真实论文 |
| **graph_edges 无外键约束** | 孤儿边（指向已删除节点） | 定期审计 + 软删除 |
| **Slot 匹配依赖硬编码映射** | 新领域需手动添加映射 | 后续加 LLM-based frame matching |
| **4C8G 内存紧张** | 高并发时 OOM | 监控 + 减 API workers 到 1 |

---

## 四、建议的下一步优先序

```
1. 配 Anthropic API key → 跑 5 篇真实论文端到端验证  [~3h]
2. 部署到服务器 (deploy.sh)                         [~2h]
3. Entity resolution 基础版                          [~4h]
4. OpenAlex 对接                                     [~4h]
5. 前端图查询页面                                     [~4h]
6. Citation 边自动创建                                [~4h]
7. IdeaDelta-to-IdeaDelta 边                         [~6h]
8. 评测集 (50 篇 gold set)                           [~8h]
```

---

## 五、代码库结构

```
researchflow-backend/
├── backend/
│   ├── models/          # 12 files, 23 model classes
│   │   ├── paper.py     # Paper, PaperAsset, PaperVersion
│   │   ├── analysis.py  # PaperAnalysis, MethodDelta, ParadigmTemplate
│   │   ├── graph.py     # IdeaDelta, Slot, MechanismFamily, GraphEdge, ImplementationUnit
│   │   ├── evidence.py  # EvidenceUnit, TransferAtom
│   │   ├── research.py  # ProjectBottleneck, SearchSession, ReadingPlan
│   │   ├── direction.py # DirectionCard, UserBookmark, UserEvent
│   │   ├── digest.py    # Digest
│   │   ├── system.py    # Job, ModelRun, ExecutionMemory, UserFeedback
│   │   └── enums.py     # 9 enums, 45 values
│   ├── services/        # 17 service modules, 58 functions
│   ├── api/             # 9 router files, 42 endpoints
│   ├── workers/         # arq: 5 tasks + 3 cron
│   ├── mcp/             # 10 MCP tools
│   └── utils/           # frontmatter, sanitize, pdf_extract
├── frontend/            # Next.js: 7 pages + layout
├── alembic/             # 4 migrations
├── migration/           # 4 scripts (csv→db, md→db, seed, validate)
├── compatibility/       # DB→Markdown export
├── caddy/               # Caddyfile
├── docker-compose.yml        # dev
├── docker-compose.prod.yml   # production (6 services)
├── deploy.sh                 # one-click deploy
├── ARCHITECTURE.md           # this doc (v3)
├── DEPLOY_GUIDE.md           # step-by-step deployment
└── STATUS_AND_IMPROVEMENTS.md # current status + roadmap
```
