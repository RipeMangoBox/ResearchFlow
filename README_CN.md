<p align="center">
  <img src="./assets/LOGO.png" alt="ResearchFlow" width="260"/>
</p>
<h1 align="center">ResearchFlow</h1>
<p align="center"><a href="README.md">English</a> · <a href="README_CN.md">中文</a></p>

---

**给一篇论文 → 自动构建领域知识图谱，追踪方法之间的改进链。**

ResearchFlow 是一个研究操作系统。导入论文 → LLM 分析 → 构建结构化知识图谱（记录方法之间如何层层改进）→ 支持迭代式研究探索。

## 完整示例：从零到知识图谱

```bash
# 1. 启动系统
cd researchflow-backend && cp .env.example .env  # 设置 ANTHROPIC_API_KEY
make db && make migrate && make up

# 2. 从 awesome 仓库初始化领域
curl -X POST localhost:8000/api/v1/pipeline/init-domain \
  -d '{"domain": "RLHF for VLM"}'
# → 找到 awesome 仓库 → 导入 72 篇论文 → 评分排序

# 3. 分析最高优先级的 10 篇
curl -X POST localhost:8000/api/v1/pipeline/batch?limit=10
# → 下载 PDF → L2 解析 → L3 速读 → L4 深度分析
# → 构建 DeltaCard → IdeaDelta → 方法演化 DAG

# 4. 迭代探索
curl -X POST localhost:8000/api/v1/explore/start \
  -d '{"query": "RL 优势消失问题"}'
# → 分类结果: 3 结构性改进, 5 插件型, 2 reward 改进
# → 建议: "都是插件型，试试相邻领域"
```

Web 前端: `localhost:3000` | Claude Code / Codex: 自动发现 `.mcp.json`

## 核心设计

### 方法之间是 DAG，不是列表

系统追踪 GRPO → GRPO+LP → GDPO 是一条改进链，GDPO 同时继承 GRPO 和 DPO。当 3+ 篇论文用某方法作 baseline → 自动标记为"已确立基线"，可升级为新版范式。

```
GRPO (基线, depth=0, 7篇下游)
├── GRPO+LP (插件, depth=1)
│   └── GRPO-LP+sampling (depth=2)
├── GDPO (结构性, depth=1, parent=[GRPO, DPO])  ← 多继承
│   └── GDPO+image_thinking (depth=2)
```

### 论文按证据质量过滤

| 优先级 | 条件 | 权重 |
|--------|------|------|
| 最高 | 开数据 | 0.40 |
| 高 | 开代码 | 0.30 |
| 中 | 中稿无代码 | 0.20 |
| 低 | 预印本 | 0.10 |
| 加分 | 顶会 + 新鲜度 + 团队 | +0.05–0.25 |

### L4 自动提取方法分类

- `method/structural_architecture` · `method/plugin_module` · `method/reward_design` ...
- `improvement/fundamental_rethink` · `improvement/additive_plugin` ...
- 自动创建 ProjectBottleneck (这篇论文在解决什么瓶颈)

## 系统规模

| | 数量 |
|-|------|
| 数据库表 | 40 (+4 物化视图) |
| API 路由 | 95+ |
| MCP 工具 | 18 (+ 3 资源, 2 提示模板) |
| Service | 30 |
| 测试 | 29+ |
| 内置范式 | 4 (RL/VLM/Agent/MotionGen) + LLM 候选发现 |

## 仓库结构

```
researchflow-backend/          # 核心后端 (唯一写入目标)
  backend/                     #   FastAPI + ORM + 30 Services + MCP
  alembic/                     #   数据库迁移 (001-010)
  frontend/                    #   Next.js 15 Web UI (论文/搜索/审核/报告)
  compatibility/               #   DB → Markdown/CSV 导出工具
  tests/                       #   pytest 测试
  ARCHITECTURE.md              #   ← 完整技术文档 (v3.2)
  DEPLOY_GUIDE.md              #   部署指南
paperAnalysis/                 # 只读导出: 分析笔记 (按领域/会议分目录)
paperCollection/               # 只读导出: 索引 + Obsidian 导航
paperIDEAs/                    # 只读导出: 研究产出
.claude/skills/                # Claude Code 技能 (18 个)
scripts/                       # 本地工具与维护脚本
assets/                        # Logo、Banner 图片
linkedCodebases/               # 外部仓库符号链接
AGENTS.md                      # Agent 接入指南
```

## 文档

| 文档 | 读者 | 内容 |
|------|------|------|
| **[ARCHITECTURE.md](researchflow-backend/ARCHITECTURE.md)** | 开发者 | 知识图谱结构 · 方法 DAG · 管线 · API · DB Schema |
| **[AGENTS.md](AGENTS.md)** | Agent 开发 | MCP 工具列表 · 技能路由 · 使用规则 |
| **[DEPLOY_GUIDE.md](researchflow-backend/DEPLOY_GUIDE.md)** | 运维 | 云端部署 · Docker · 成本 |

## License

MIT
