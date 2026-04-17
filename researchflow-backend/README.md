# ResearchFlow Backend

Web-first 科研操作系统，以 **Schema-guided Hierarchical Scholarly Idea Graph** 为核心。

> **IdeaDelta 是主对象，Paper 是容器，Evidence 是锚点，Graph 是检索加速器。**

---

## 系统规模

| 组件 | 数量 |
|------|------|
| 数据库表 | 23 |
| API 端点 | 42 |
| MCP 工具 | 10 |
| 前端页面 | 7 |
| 后端 Service | 17 (58 函数) |
| 后台任务 | 5 task + 3 cron |
| 知识图谱: ParadigmFrame | 4 (RL/VLM/Agent/MotionGen) |
| 知识图谱: Slot | 25 |
| 知识图谱: MechanismFamily | 19 |
| 知识图谱: 边类型 | 12 |

---

## 快速开始

```bash
cp .env.example .env  # 编辑设置密码和 API key

# 本地开发
make db               # 启动 Postgres + Redis
make migrate          # 运行 4 次 Alembic 迁移
make migrate-all      # 导入数据 + 种子

# 或生产部署
bash deploy.sh
```

---

## 核心架构：知识图谱五层

```
Layer 5: Serving Views    — Report / Digest / ReadingPlan / Direction
Layer 4: Evidence/Impl    — EvidenceUnit (独立行, FK→IdeaDelta) + ImplementationUnit
Layer 3: Idea Layer       — IdeaDelta (核心主对象) + GraphEdge (12种边)
Layer 2: Canonical Domain — ParadigmFrame + Slot + MechanismFamily + Bottleneck
Layer 1: Scholarly Backbone — Paper + cites 边 + OpenAlex 对接
Layer 0: Asset Layer      — PDF / Repo / HTML (对象存储)
```

---

## 分析管线

```
paper → L1 (API补全) → L2 (PDF解析) → L3 (LLM skim) → L4 (LLM deep)
  → frame_assign → idea_extract → evidence_persist → edge_create → publish_check
```

---

## 文档

| 文档 | 内容 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 系统架构 v3 (知识图谱五层 + 6层质量保障) |
| [STATUS_AND_IMPROVEMENTS.md](STATUS_AND_IMPROVEMENTS.md) | 现状分析 + 14项改进路线 |
| [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md) | 阿里云部署指南 (购买→配置→一键部署) |
| [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) | 分阶段开发方案 |

---

## 部署

目标环境：阿里云轻量 4C8G + OSS 对象存储。年化约 2300 元 + LLM token。

```bash
bash deploy.sh  # 一键部署 (Docker + 迁移 + 健康检查)
```

## License

MIT
