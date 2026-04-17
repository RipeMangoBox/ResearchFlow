<p align="center">
  <img src="./assets/LOGO.png" alt="ResearchFlow logo" width="280"/>
</p>

<h1 align="center">ResearchFlow</h1>

<p align="center"><strong>给一篇论文 → 自动构建领域知识图谱 → 追踪方法演化 → 智能探索</strong></p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a>
</p>

---

## 它做什么

```
你: "这篇论文是关于 VLM 的 RLHF"
         ↓
Step 1:  找到领域的 awesome 仓库 → 导入 72 篇论文 → 评分排序
Step 2:  下载 PDF → 分析 (L2解析 → L3速读 → L4深度) → 提取 DeltaCard
Step 3:  构建方法演化 DAG: GRPO → GRPO+LP → GDPO → GDPO+image_thinking
Step 4:  自动分类: 3 个结构性改进, 5 个插件型, 2 个 reward 改进
Step 5:  你迭代探索，系统追踪你的 pivot 并建议下一步
```

**核心理念**: 方法之间是 DAG (有向无环图)，不是扁平列表。系统追踪哪些改进变成了新 baseline，哪些只是插件，以及范式如何演化。

---

## 快速开始

```bash
git clone https://github.com/RipeMangoBox/ResearchFlow.git
cd ResearchFlow/researchflow-backend
cp .env.example .env              # 设置 ANTHROPIC_API_KEY
make db && make migrate && make up
```

```bash
# 从 awesome 仓库初始化领域
curl -X POST localhost:8000/api/v1/pipeline/init-domain \
  -d '{"domain": "RLHF VLM"}'

# 或给一篇论文，自动发现相关论文
curl -X POST localhost:8000/api/v1/import/links \
  -d '{"items": [{"url": "https://arxiv.org/abs/2402.03300"}]}'
curl -X POST localhost:8000/api/v1/pipeline/{paper_id}/discover
```

Web 前端: `http://localhost:3000` | Claude Code: 自动发现 `.mcp.json`

---

## 知识图谱结构

```
Paper → DeltaCard (中间真相层) → IdeaDelta (知识原子) → GraphAssertions (图谱边)
          │
          ├── parent_delta_card_ids    (DAG 继承: 基于哪些方法)
          ├── method_category          (structural / plugin / reward / ...)
          ├── improvement_type         (fundamental_rethink / additive_plugin / ...)
          └── bottleneck_addressed     (自动提取的研究瓶颈)
```

**自动升级**: 当一个改进被 ≥3 篇论文用作 baseline → 标记为 `established_baseline`。如果还是结构性改进 → 可升级为新版范式。

### 论文过滤优先级

| 层级 | 条件 | 权重 |
|------|------|------|
| 最高 | 开数据 (Tier A) | 0.40 |
| 次高 | 开代码 (Tier B) | 0.30 |
| 中 | 中稿无代码 (Tier C) | 0.20 |
| 最低 | 预印本 (Tier D) | 0.10 |
| 加分 | 顶会 + 重要度 + 时间衰减 | +0.05~0.25 |

### 方法分类 (L4 自动提取)

```
method/structural_architecture    — 改了核心架构
method/plugin_module             — 加了一个模块
method/reward_design             — 改了奖励函数
method/training_recipe           — 改了训练方法
improvement/fundamental_rethink  — 根本性重新思考
improvement/additive_plugin      — 加插件
```

---

## 完整流程

### 16 步管线

```
ingest → triage → download_pdf → enrich → parse_L2
→ skim_L3 → deep_L4 → delta_card_build → link_parent_baselines
→ entity_resolution → assertion_propose → evidence_audit
→ review → publish → index → export
```

### 研究探索会话

```
POST /explore/start   → "RL 优势消失问题"
POST /explore/search  → 搜索 + 分类: structural=1, plugin=6
POST /explore/step    → pivot: "都是插件型，要看根本性方案"
POST /explore/search  → "think with image agentic GDPO"
GET  /explore/{id}    → 完整探索路径 + 论文分类 + 下一步建议
```

---

## 系统规模

| 组件 | 数量 |
|------|------|
| 数据库表 | 31 (7 次迁移) |
| API 路由 | 81 (13 Router) |
| MCP 工具 | 18 |
| Service 模块 | 25 |
| 测试 | 29 |
| 范式模板 | 4 内置 + LLM 动态发现 |

## 仓库结构

```
ResearchFlow/
├── researchflow-backend/        # 核心后端 (唯一写入目标)
│   ├── backend/                 # FastAPI + ORM + Services + MCP
│   ├── frontend/                # Next.js (7 页)
│   ├── alembic/                 # 迁移 (001-007)
│   ├── tests/                   # pytest (29 tests)
│   └── ARCHITECTURE.md          # 完整架构文档
├── paperAnalysis/               # 导出: 分析笔记
├── paperCollection/             # 导出: 索引 + 导航
├── paperIDEAs/                  # 导出: 研究产出
├── .claude/skills/              # Claude Code 技能
├── scripts/                     # 工具脚本
└── AGENTS.md                    # Agent 接入指南
```

## 文档

| 文档 | 内容 |
|------|------|
| [ARCHITECTURE.md](researchflow-backend/ARCHITECTURE.md) | 完整架构: 知识图谱 + 方法演化 + 过滤 + 流程 |
| [Backend README](researchflow-backend/README.md) | 后端开发: 安装 + 功能 + API |
| [DEPLOY_GUIDE.md](researchflow-backend/DEPLOY_GUIDE.md) | 云端部署 |
| [AGENTS.md](AGENTS.md) | MCP 工具 + 技能路由 |

## License

MIT
