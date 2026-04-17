# ResearchFlow Backend

Web-first 科研操作系统，以 **单核多投影架构 (v3.1)** 为核心。

> **DeltaCard 是中间真相层，IdeaDelta 是可复用知识原子，Paper 是容器，GraphAssertion 是检索加速器。**

---

## 系统规模

| 组件 | 数量 |
|------|------|
| 数据库表 | 31 (5 次 Alembic 迁移) |
| API 端点 | 63 (11 个 router) |
| MCP 工具 | 15 |
| 前端页面 | 7 |
| 后端 Service | 21 |
| 后台任务 | 5 task + 3 cron |
| 知识图谱: ParadigmFrame | 4 (RL/VLM/Agent/MotionGen) |
| 知识图谱: Slot | 25 |
| 知识图谱: MechanismFamily | 19 |
| 知识图谱: 边类型 | 12 (GraphAssertion 生命周期管理) |

---

## 快速开始

```bash
cp .env.example .env  # 编辑设置密码和 API key

# 本地开发
make db               # 启动 Postgres + Redis
make migrate          # 运行 5 次 Alembic 迁移
make migrate-all      # 导入数据 + 种子

# 或生产部署
bash deploy.sh
```

---

## 核心架构：知识图谱七层

```
Layer 6: Serving Views       — Report / Digest / ReadingPlan / Direction
Layer 5: Quality Control     — ReviewTask / HumanOverride / Assertion 生命周期
Layer 4: Evidence/Impl       — EvidenceUnit (FK→IdeaDelta+DeltaCard) + GraphAssertionEvidence
Layer 3: Idea Layer          — DeltaCard (真相层) + IdeaDelta (知识原子) + GraphAssertion
Layer 2: Canonical Domain    — ParadigmFrame + Slot + MechanismFamily + Bottleneck + Alias
Layer 1: Scholarly Backbone  — Paper + cites 边 + OpenAlex 对接
Layer 0: Asset Layer         — PDF / Repo / HTML (对象存储)
```

---

## 分析管线 (16 步)

```
ingest → canonicalize → enrich → fetch_assets → parse
→ skim_extract (L3) → deep_extract (L4)
→ delta_card_build → entity_resolution → assertion_propose
→ evidence_audit → review → publish → index → export → digest
```

---

## v3.1 新增 (Phase 1-4)

| 新增 | 说明 |
|------|------|
| delta_cards 表 | 中间真相层 — 从 L4 一次构建，多次渲染 |
| graph_nodes + graph_assertions | 替代 graph_edges，支持断言生命周期 |
| review_tasks + human_overrides | 审核队列 + 人工覆盖追踪 |
| aliases 表 | 实体别名归一 |
| delta_card_service | 完整 pipeline: build → evidence → idea → assert → publish |
| assertion_service | propose → audit → review → publish 生命周期 |
| review_service | 审核队列管理 + 级联审批/拒绝 |
| entity_resolution_service | 3 级解析: exact → alias → fuzzy |
| 15 个 assertion API | /assertions/* (断言/审核/覆盖/别名) |
| 5 个 MCP 新工具 | search_ideas, propose_directions, graph_stats, review_queue, submit_review |
| DeltaCard-aware consumers | report/digest/reading_planner/direction/search 全部重构 |
| Export 增强 | Markdown 导出含 DeltaCard 段落 + 前端元数据 |

---

## 文档

| 文档 | 内容 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 系统架构 v3.1 (七层 + 16步管线 + 8条硬约束) |
| [STATUS_AND_IMPROVEMENTS.md](STATUS_AND_IMPROVEMENTS.md) | 现状分析 + 改进路线 |
| [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md) | 阿里云部署指南 |
| [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) | 分阶段开发方案 |

---

## 部署

目标环境：阿里云轻量 4C8G + OSS 对象存储。

```bash
bash deploy.sh  # 一键部署 (Docker + 迁移 + 健康检查)
```

## License

MIT
