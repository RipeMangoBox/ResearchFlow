<p align="center">
  <img src="./assets/LOGO.png" alt="ResearchFlow logo" width="280"/>
</p>

<h1 align="center">ResearchFlow</h1>

<p align="center"><strong>Web-first 科研操作系统。<br/>导入论文 → 自动分析 → 生成汇报报告 → 推荐阅读顺序 → 日周月总结。<br/>普通用户用网页，高级用户走 Claude Code / Codex 专家模式。</strong></p>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README_CN.md">中文</a>
</p>

<p align="center">
  <img alt="Web-first" src="https://img.shields.io/badge/Web--first-Research%20OS-1f6feb?style=flat-square"/>
  <img alt="FastAPI backend" src="https://img.shields.io/badge/FastAPI-backend-0891b2?style=flat-square"/>
  <img alt="PostgreSQL+pgvector" src="https://img.shields.io/badge/PostgreSQL-pgvector-0f766e?style=flat-square"/>
  <img alt="MCP compatible" src="https://img.shields.io/badge/MCP-compatible-d97706?style=flat-square"/>
  <img alt="Claude Code" src="https://img.shields.io/badge/Claude%20Code-expert%20mode-7c3aed?style=flat-square"/>
  <img alt="Codex CLI" src="https://img.shields.io/badge/Codex%20CLI-expert%20mode-cc2936?style=flat-square"/>
  <img alt="Zotero compatible" src="https://img.shields.io/badge/Zotero-compatible-475569?style=flat-square"/>
  <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-111827?style=flat-square"/>
</p>

---

> **ResearchFlow = Web 产品 + 自有后端编排 + MCP 兼容层 + 专家模式（Claude/Codex）**
>
> 普通用户只用网页：导入论文、看报告、看总结、选方向、给反馈。高级用户再通过 Claude Code / Codex 走专家入口。系统本体是自有后端（PostgreSQL + 对象存储 + 任务队列），不是任何 AI agent 会话。
>
> 架构是 **workflow-first**，不是 agent-first。自建轻量编排层，需要 LLM 能力时通过 Agent SDK 调入，同时通过 MCP 暴露给 Claude Code / Codex。

---

## 8 个核心用户功能

| # | 功能 | 说明 |
|---|------|------|
| 1 | **导入任意研究输入** | awesome 列表 / 论文链接 / PDF / GitHub 仓库 / Zotero → 统一入库 |
| 2 | **一键生成汇报报告** | 30 秒版 / 5 分钟汇报版 / 深度对比版，先对照范式再说改了什么 |
| 3 | **Repo × Paper 联合深剖** | 公式→代码映射、shape trace、可修改性分析，带置信度 |
| 4 | **增量更新与资产补全** | 自动补全 repo/project page/数据/配置，记录来源与置信度 |
| 5 | **方向推荐** | 输入子方向 → 1-3 个方向卡片（成本/风险/复现条件）→ 展开可行性方案 |
| 6 | **分层阅读推荐** | canonical baseline → 结构性改进 → 强团队跟进 → patch → 负例 |
| 7 | **日/周/月总结** | 产品一等公民，自动生成，覆盖新增论文/方向变化/策略调整 |
| 8 | **反馈与导出** | 收藏 / 批注 / 纠错 / 导出组会版 / 分享版 / Obsidian 版 |

---

## 系统架构

```text
┌────────────────────────────────────────────────────────┐
│  交互层    Web 前端 (Next.js) + Claude Code/Codex (MCP) │
├────────────────────────────────────────────────────────┤
│  Presentation   报告渲染 / 总结 / 方向推荐              │
├────────────────────────────────────────────────────────┤
│  Workflow/Job   15 种异步任务 (arq + Redis)             │
├────────────────────────────────────────────────────────┤
│  Retrieval      PostgreSQL + pgvector 混合搜索          │
├────────────────────────────────────────────────────────┤
│  Parse/Extract  PDF 解析 + LLM 四级分析 (L1→L4)        │
├────────────────────────────────────────────────────────┤
│  Ingestion      规范化 / 去重 / 资产补全                │
└────────────────────────────────────────────────────────┘
      ↕ 对象存储 (COS/OSS)          ↕ PostgreSQL (19 张表)
```

### 四级分析管线

| 级别 | 处理方式 | Token 消耗 | 输出 |
|------|----------|-----------|------|
| L1 metadata | Crossref/arXiv API 补全 | 0 | 完整元数据 |
| L2 parse | pymupdf 本地解析 | 0 | 章节/公式/表格/图注 |
| L3 skim | LLM 轻量卡片 | ~2K | skim card + delta card |
| L4 deep | LLM 全文分析 | ~10-20K | 完整报告 + 证据原子 |

晋升由多维评分控制，不自动全量推进。

### 兼容层

数据库是事实源，Markdown 是导出物。`paperAnalysis/` `paperCollection/` `paperIDEAs/` 保留为兼容导出层，现有 `.claude/skills/` 继续可用。

---

## 🏗️ 仓库结构

```text
ResearchFlow/
├── researchflow-backend/           # 后端服务 (新增)
│   ├── backend/                    # FastAPI + ORM + Workers + MCP
│   ├── frontend/                   # Next.js 前端 (7 页)
│   ├── alembic/                    # 数据库迁移
│   ├── migration/                  # 数据迁移脚本
│   ├── compatibility/              # DB→Markdown 导出
│   ├── docker-compose.yml
│   ├── ARCHITECTURE.md             # 完整架构设计
│   └── DEVELOPMENT_PLAN.md         # 分阶段开发方案
├── paperAnalysis/                  # 兼容层：结构化分析笔记
│   └── analysis_log.csv
├── paperPDFs/                      # 兼容层：原始 PDF
├── paperCollection/                # 兼容层：索引与导航
│   └── index.jsonl
├── paperIDEAs/                     # 兼容层：研究产出
├── .claude/skills/                 # Skill 定义 (17 个)
├── scripts/                        # 辅助脚本
├── linkedCodebases/                # 外部代码库符号链接
├── AGENTS.md                       # Agent 指南
└── README_CN.md
```

---

## 用户角色与入口

| 角色 | 入口 | 能做什么 |
|------|------|----------|
| 普通研究生/老师 | **Web 前端** | 导入论文、看报告、看总结、选方向、给反馈 |
| 高级研究用户 | Web + **Claude Code/Codex** | 深度交互、自定义检索、跨论文对比、idea 收敛 |
| 维护者/开发者 | **Claude Code/Codex** + 后台 | 开发 skill、修规则、审结果、管理 taxonomy |

Claude Code / Codex 通过 MCP 连接后端，共享同一套知识库和工作流。

---

## MCP Server（10 个工具）

| 工具 | 功能 |
|------|------|
| `import_research_sources` | 导入任意研究输入 |
| `search_research_kb` | 关键词+语义+结构化搜索 |
| `get_paper_report` | 获取论文报告 (30s/5min/deep) |
| `get_repo_paper_report` | Repo×Paper 联合分析 |
| `compare_papers` | 2-5 篇论文对比 |
| `refresh_assets` | 资产补全与增量更新 |
| `propose_directions` | 方向推荐卡片 |
| `expand_feasibility_plan` | 展开可行性方案 |
| `get_digest` | 日/周/月总结 |
| `record_user_feedback` | 纠错/收藏/标签修改 |

---

## 🚀 快速开始

### 前置条件

- Docker + Docker Compose（后端服务）
- Python 3.12+（本地开发）
- Node.js 20+（前端开发）

### 1. 克隆并启动后端

```bash
git clone https://github.com/<your-username>/ResearchFlow.git
cd ResearchFlow/researchflow-backend

cp .env.example .env
# 编辑 .env，设置密码和 API keys

make db          # 启动 Postgres + Redis
make migrate     # 创建数据库表
make migrate-all # 导入现有数据 (CSV + MD)
make up          # 启动全部服务
```

### 2. 访问 Web 前端

打开 `http://localhost:3000`，即可使用全部功能。

### 3. 专家模式（可选）

Claude Code / Codex 用户通过 MCP 接入同一后端：

```text
# Claude Code 会自动发现 .mcp.json 配置
# Codex CLI 使用 .codex/config.toml 配置
```

### 4. 可选：Obsidian 可视化

如果你希望在 Obsidian 中可视化浏览知识库，见 [Obsidian 配置](#obsidian-config)。

---

## 📖 使用方式

### 普通用户（Web 前端）

1. 打开 Web 前端
2. 在导入中心粘贴 awesome 列表链接或上传 PDF
3. 系统自动：规范化 → 去重 → 补全元数据 → 多维评分 → 分级分析
4. 查看自动生成的汇报报告（30s / 5min / 深度对比版）
5. 按推荐顺序阅读论文
6. 查看日/周/月总结

### 高级用户（Claude Code / Codex 专家模式）

通过 MCP 接入后端，可使用 skill 和工具进行深度研究交互。

### Skill 快速参考（专家模式）

| 场景 | Skill |
|------|-------|
| 不确定该做什么 | `research-workflow` |
| 从 GitHub 收集论文 | `papers-collect-from-github-awesome` |
| 分析 PDF | `papers-analyze-pdf` |
| 检索/对比论文 | `papers-query-knowledge-base` |
| 生成 idea | `research-brainstorm-from-kb` |
| 审稿人压测 | `reviewer-stress-test` |

> 完整 skill 列表见 [`.claude/skills/User_README_CN.md`](.claude/skills/User_README_CN.md)。

---

## 文档

| 文档 | 说明 |
|------|------|
| [`researchflow-backend/ARCHITECTURE.md`](researchflow-backend/ARCHITECTURE.md) | 完整系统架构设计（六层架构、数据流、ER 图、部署） |
| [`researchflow-backend/DEVELOPMENT_PLAN.md`](researchflow-backend/DEVELOPMENT_PLAN.md) | 分阶段开发方案（8 个 Phase、任务分解、验收标准） |
| [`researchflow-backend/RESTRUCTURE_PLAN.md`](researchflow-backend/RESTRUCTURE_PLAN.md) | 重构设计文档 v2（完整 DB schema、API、MCP） |
| [`.claude/skills/User_README_CN.md`](.claude/skills/User_README_CN.md) | Skill 快速路由（专家模式用） |

---

## 部署

目标环境：腾讯云 Lighthouse 2C4G / 100GB SSD + COS 对象存储。年化成本约 1500-1900 元（不含 LLM token）。

```bash
cd researchflow-backend
docker compose up -d
```

## License

MIT
