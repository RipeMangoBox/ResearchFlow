<p align="center">
  <img src="./assets/LOGO.png" alt="ResearchFlow" width="260"/>
</p>
<h1 align="center">ResearchFlow</h1>
<p align="center">
  <strong>给一篇论文，自动生成领域知识图谱。</strong>
</p>
<p align="center">
  <a href="README.md">English</a> · <a href="README_CN.md">中文</a>
</p>

---

ResearchFlow 是一个研究操作系统，将学术论文转化为结构化、持续演化的知识图谱。给它一篇种子论文或一个研究主题 — 它会自动发现相关工作，通过 6 步 LLM 管线分析每篇论文（内置怀疑机制），以 DAG 而非扁平列表追踪方法之间的改进关系，并导出可导航的 Obsidian vault。

**核心差异：** ResearchFlow 不只是总结论文。它先读方法和实验，*再*核对摘要声明；从数据库自动构建比较集（不靠论文自述）；高价值结论必须有证据锚点；新论文到来时全图谱保持一致性。

## 已完成功能

### 领域冷启动
给一个研究主题 → GitHub awesome 仓库发现 → 自动导入 50-100 篇论文 → triage 评分 → 批量分析 → 完整知识图谱。一个 API 调用即可启动。

### 6 步分析管线
每篇论文经过 6 个独立可重试的分析步骤，不再是一次 LLM 调用：

| 步骤 | 功能 | 为什么重要 |
|------|------|-----------|
| **extract_evidence** | 公式、图表、证据锚点 | 先读方法/实验，再核对摘要 |
| **build_delta_card** | 基线对比、改动 slot、机制 | 基于 Step 1 证据，无法 hallucinate |
| **build_compare_set** | 从 DB 自动补齐比较集 | 4 个来源，不只是论文自述 |
| **propose_lineage** | builds_on / extends / replaces | 方法形成 DAG，支持多继承 |
| **synthesize_concept** | 更新跨论文 CanonicalIdea | 概念跨论文累积，不孤立 |
| **reconcile_neighbors** | 反向更新相关论文 | 知识图谱全局一致 |

### 10 步元数据补全 (8 个 API)
arXiv → Crossref → OpenAlex → Semantic Scholar → DBLP → OpenReview → GitHub → HuggingFace。结果存入观察账本 (observation ledger)，按权威等级排序 — 冲突自动解决，不盲目覆盖。

### Parser Ensemble (L2)
GROBID（作者、机构、参考文献、公式坐标）+ PyMuPDF（章节、图表、标题）+ VLM（图表分类、公式 OCR → LaTeX）。先确定性提取，LLM 只处理机器解析不了的。

### 方法演化 DAG
论文不是扁平列表 — 它们形成有向无环图，追踪方法之间的改进关系：

```
GRPO (基线, depth=0, 7 篇下游)
├── GRPO+LP (插件, depth=1)
│   └── GRPO-LP+sampling (depth=2)
├── GDPO (结构性, depth=1, parents=[GRPO, DPO])  ← 多继承
│   └── GDPO+image_thinking (depth=2)
```

3+ 篇论文用某方法作 baseline → 自动标记为已确立基线 → 可升级为新范式版本。所有晋升经过审核门控。

### Faceted Taxonomy (15 维度, 75 种子节点)
论文在 domain/modality/task/learning_paradigm/mechanism/method_baseline/model_family/dataset/benchmark/metric/lab/venue 等维度标注 — 不只是一个 category。DAG 结构支持 `is_a`/`part_of`/`uses`/`optimizes` 关系。

### Obsidian Vault 导出
一键导出结构化 Obsidian vault，每篇论文 6-10 个 wikilinks（不是毛线团）：

```
00_Home/           导航 + 阅读顺序建议
10_Lineages/       L__ 方法演化链 (ASCII 树)
20_Concepts/       C__ 机制 + 标准思想合并
30_Bottlenecks/    B__ 跨论文综合 (症状 → 根因 → 解法分层)
40_Papers/
  A__Baselines/    必读 baseline (struct ≥ 0.7)
  B__Structural/   结构性改进 (struct ≥ 0.5)
  C__Plugins/      插件型改进 (struct ≥ 0.3)
  D__Peripheral/   外围参考
80_Assets/         PDF 提取的图表
90_Views/          静态 Markdown 表格 (不依赖 Dataview)
```

### 候选队列 + 多 Agent 管线 (V6)
5 级吸收：`new → shallow → reference_done → deep → graph_ready`。16 个专用 LLM Agent（12 个 prompt 文件）配合 Context Pack Builder。4 层评分引擎（DiscoveryScore → DeepIngestScore → GraphPromotionScore → AnchorScore）。节点/边 profile 用于知识图谱实体画像。冷启动工作流与增量同步（7 个函数）。

### MCP 集成 (35 个工具)
完整 MCP 服务器：35 个工具 + 6 个资源 + 4 个提示模板。Claude Code 自动发现 — 用自然语言交互：

```
> 搜索 "reward hacking in RLHF" 相关论文
> 对前 3 个结果跑完整分析管线
> 比较这些论文的方法
> 导出 vault 到 Obsidian
```

### 21 个 Claude Code Skills
研究工作流自动化：从 GitHub/Web/Zotero 收集论文、分析 PDF、查询知识库、头脑风暴、聚焦假设、审稿压力测试、生成公式推导深度报告、写日志。

### 交互式研究探索
多跳认知迭代：搜索 → 分类（结构性 vs 插件型）→ gap 分析 → pivot → 拓展。系统记住拒绝模式并建议新方向。

### Web 控制台
Next.js 15 前端：论文管理、搜索、图谱可视化、演化链查看、审核队列、摘要阅读、瓶颈探索、导入工具。

## 系统规模

| 组件 | 数量 |
|------|------|
| 数据库表 | 58 + 4 物化视图 |
| API 端点 | 130 (16 个路由器) |
| MCP | 35 工具 + 6 资源 + 4 提示模板 |
| Service 模块 | 55 |
| Worker 任务 | 22 |
| ORM 模型文件 | 24 (含 15 个 V6 类) |
| Agent Prompt | 12 |
| Claude Code Skills | 21 |
| 元数据 API | 8 (arXiv/Crossref/OpenAlex/S2/DBLP/OpenReview/GitHub/HF) |
| 内置范式 | 4 (RL/VLM/Agent/MotionGen) + LLM 动态发现 |
| 数据库迁移 | 16 个版本 |
| 枚举类型 | 9 个 (PaperState 15 个状态值等) |

## 快速开始

```bash
# 1. 启动系统
cd researchflow-backend && cp .env.example .env  # 设置 ANTHROPIC_API_KEY
docker compose up -d
docker compose exec api alembic upgrade head

# 2. 初始化领域
curl -X POST localhost:8000/api/v1/pipeline/init-domain \
  -H "Content-Type: application/json" \
  -d '{"domain": "RLHF for VLM"}'

# 3. 批量分析优先论文
curl -X POST localhost:8000/api/v1/pipeline/batch?limit=10

# 4. 导出 Obsidian vault
curl -X POST localhost:8000/api/v1/pipeline/export/obsidian-vault
```

API 文档: `http://localhost:8000/api/v1/docs`

## 通过 MCP 连接 Claude Code

### 远程连接

```json
{
  "mcpServers": {
    "researchflow": {
      "url": "https://your-domain/sse"
    }
  }
}
```

### 本地运行

```json
{
  "mcpServers": {
    "researchflow-local": {
      "command": "python",
      "args": ["-m", "backend.mcp.server"],
      "cwd": "researchflow-backend",
      "env": {"PYTHONPATH": "."}
    }
  }
}
```

## 同步 Obsidian Vault

```bash
# 服务器上导出
curl -X POST localhost:8000/api/v1/pipeline/export/obsidian-vault

# rsync 到本地
rsync -avz --delete -e ssh \
  root@your-server:/opt/researchflow/obsidian-vault/ \
  ./obsidian-vault/

# 用 Obsidian 打开 → Graph View (Cmd+G)
# 推荐颜色: Papers=蓝, Concepts=绿, Bottlenecks=红, Lineages=橙
```

## 仓库结构

```
researchflow-backend/            # 核心后端 (唯一写入目标)
  backend/
    api/                         #   16 个路由器 (130 端点)
    models/                      #   24 个 ORM 模型文件 (58 张表)
    services/                    #   55 个服务模块
    mcp/                         #   MCP 服务器 (35 工具 + 6 资源 + 4 提示)
    workers/                     #   ARQ 后台任务队列
    utils/                       #   PDF 提取、GROBID 客户端、frontmatter
  alembic/                       #   16 次数据库迁移
  frontend/                      #   Next.js 15 + Tailwind Web 控制台
  ARCHITECTURE.md                #   完整技术文档 (v6)
  DEPLOY.md                      #   生产部署指南
obsidian-vault/                  # 自动生成的 Obsidian vault (只读)
paperAnalysis/                   # 导出的分析笔记 (只读)
paperCollection/                 # 收集索引 + 导航 (只读)
paperIDEAs/                      # 研究想法笔记 (只读)
scripts/                         # 维护和工具脚本
.claude/skills/                  # 21 个 Claude Code 技能定义
.mcp.json                       # MCP 服务器配置
AGENTS.md                       # Agent/MCP 接入指南
```

## 技术栈

| 层 | 技术 |
|----|------|
| 后端 | FastAPI (async) + SQLAlchemy 2.0 (async) |
| 数据库 | PostgreSQL 16 + pgvector (1536d 向量) |
| 任务队列 | ARQ + Redis 7 |
| 前端 | Next.js 15 + Tailwind CSS |
| PDF 解析 | PyMuPDF + GROBID 0.8.1 (ensemble) |
| VLM | Claude Vision (图表分类 + 公式 OCR) |
| LLM | Anthropic Claude / OpenAI (streaming) |
| 元数据 | arXiv + Crossref + OpenAlex + S2 + DBLP + OpenReview + GitHub + HuggingFace |
| MCP | Python MCP SDK (stdio + SSE) |
| 存储 | Tencent COS / Alibaba OSS / Local |
| 部署 | Docker Compose + Caddy (自动 HTTPS) |

## 文档

| 文档 | 读者 | 内容 |
|------|------|------|
| [ARCHITECTURE.md](researchflow-backend/ARCHITECTURE.md) | 开发者 | 数据模型 · 四层提取 · 6 步管线 · DB Schema · 全部 API · 55 个 Service |
| [AGENTS.md](AGENTS.md) | Agent 开发 | 35 MCP 工具 · 6 资源 · 4 提示 · 21 Skills · 使用规则 |
| [DEPLOY.md](researchflow-backend/DEPLOY.md) | 运维 | Docker 配置 · 容器架构 · 日常部署 · 代理 · 故障排查 |

## License

MIT
