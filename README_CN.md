<p align="center">
  <img src="./assets/LOGO.png" alt="ResearchFlow" width="260"/>
</p>
<h1 align="center">ResearchFlow</h1>
<p align="center"><a href="README.md">English</a> · <a href="README_CN.md">中文</a></p>

---

**给一篇论文 → 自动构建领域知识图谱，追踪方法之间的改进链。**

ResearchFlow 是一个研究操作系统。导入论文 → 6 步 LLM 管线分析 → 构建结构化知识图谱 → 导出干净的 Obsidian vault 供人类导航。

## 核心功能

### 6 步分析管线 (v4.0)

不再一次 LLM 调用出 23 个字段，而是 6 个独立步骤，各自可重试：

| 步骤 | 内容 | 防线 |
|------|------|------|
| 1. extract_evidence | 公式、图表、证据锚点 | 先读 method/实验，再核对 abstract |
| 2. build_delta_card | 基线、改动 slot、机制、瓶颈 | 用 Step 1 证据作 grounding |
| 3. build_compare_set | 从 DB 自动补齐比较集 | 不靠论文自述 |
| 4. propose_lineage | builds_on / extends / replaces | DAG 结构 |
| 5. synthesize_concept | 更新跨论文 CanonicalIdea | 概念累积 |
| 6. reconcile_neighbors | 反向更新旧论文 | 图谱一致性 |

### Obsidian Vault 导出 (v4.0)

只有 5 类笔记，各带前缀：

```
00_Home/
  00_方向总览.md          # 纯导航，不参与主图
  01_阅读顺序.md          # 分层阅读建议
10_Lineages/
  L__DPO_Family.md        # 方法演化主线：A→B→C
20_Concepts/
  C__Direct_Preference.md # Mechanism + CanonicalIdea 合并
30_Bottlenecks/
  B__Credit_Assignment.md # 跨论文综合 insight
40_Papers/
  A__Baselines/           # 必读 baseline
  B__Structural/          # 结构性改进
  C__Plugins/             # 插件型
  D__Peripheral/          # 外围参考
80_Assets/figures/        # PDF 提取的图表
90_Views/                 # Dataview 查询视图
```

**Paper Note 正文 wikilinks 预算 6-8 个**，不链接 Domain Overview / Paradigm（只放 frontmatter 属性）。Graph View 不再是毛线团。

### 方法演化 DAG

方法之间是 DAG，不是列表：

```
GRPO (基线, depth=0, 7篇下游)
├── GRPO+LP (插件, depth=1)
│   └── GRPO-LP+sampling (depth=2)
├── GDPO (结构性, depth=1, parent=[GRPO, DPO])  ← 多继承
│   └── GDPO+image_thinking (depth=2)
```

当 3+ 篇论文用某方法作 baseline → 自动标记为已确立基线 → 可升级为新版范式。

## 快速开始

```bash
# 1. 启动系统
cd researchflow-backend && cp .env.example .env  # 设置 ANTHROPIC_API_KEY
make db && make migrate && make up

# 2. 初始化领域
curl -X POST localhost:8000/api/v1/pipeline/init-domain \
  -H "Content-Type: application/json" \
  -d '{"domain": "RLHF for VLM"}'

# 3. 分析论文
curl -X POST localhost:8000/api/v1/pipeline/batch?limit=10

# 4. 导出 Obsidian vault
curl -X POST localhost:8000/api/v1/pipeline/export/obsidian-vault
```

## 通过 MCP 连接 Claude Code

ResearchFlow 暴露 MCP 服务器（22 个工具 + 6 个资源 + 4 个提示模板）。Claude Code 自动发现 `.mcp.json`。

### 方式 A：远程连接（连接已部署的服务器）

在项目根目录 `.mcp.json` 中添加：

```json
{
  "mcpServers": {
    "researchflow": {
      "url": "https://researchflow.xyz/sse"
    }
  }
}
```

### 方式 B：本地运行

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

### 方式 C：同时连接本地 + 远程

```json
{
  "mcpServers": {
    "researchflow-local": {
      "command": "python",
      "args": ["-m", "backend.mcp.server"],
      "cwd": "researchflow-backend",
      "env": {"PYTHONPATH": "."}
    },
    "researchflow-remote": {
      "url": "http://47.101.167.55:8001/sse"
    }
  }
}
```

### 使用示例

```
> 用 researchflow 搜索 "reward hacking in RLHF" 相关论文
> 对前 3 个结果跑完整分析管线
> 导出 vault 同步到我的 Obsidian
```

Claude Code 会自动调用 MCP 工具执行这些操作。

## 同步 Obsidian 知识图谱

### 从服务器同步到本地

```bash
# 1. 在服务器上导出 vault
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55 \
  "cd /opt/researchflow && curl -s -X POST localhost:8000/api/v1/pipeline/export/obsidian-vault"

# 2. rsync 到本地
rsync -avz --delete \
  -e "ssh -i ~/.ssh/autoresearch.pem" \
  root@47.101.167.55:/opt/researchflow/obsidian-vault/ \
  ./obsidian-vault/

# 3. 用 Obsidian 打开
# File → Open Vault → 选择 ./obsidian-vault/
```

### Obsidian 配置建议

1. **安装 Dataview 插件**：Settings → Community Plugins → Dataview（90_Views/ 需要）
2. **Graph View**：Cmd+G 查看知识图谱
3. **Graph 颜色分组**：
   - `path:40_Papers` → 蓝色（论文）
   - `path:20_Concepts` → 绿色（概念）
   - `path:30_Bottlenecks` → 红色（瓶颈）
   - `path:10_Lineages` → 橙色（演化链）

### 笔记类型说明

| 类型 | 前缀 | 内容 |
|------|------|------|
| Paper | `P__` | 一眼看懂 + baseline 对照表 + 公式图表 + 阅读建议 + 详细分析 |
| Concept | `C__` | 机制 + 标准思想合并，含所有相关论文的对比表 |
| Bottleneck | `B__` | 跨论文综合：症状 → 根因 → 结构性/插件型解法分层 |
| Lineage | `L__` | ASCII 演化树 + 每步改了什么 + 分叉点 |
| Overview | — | 纯导航（方向总览 + 阅读顺序） |

## 系统规模

| 组件 | 数量 |
|------|------|
| 数据库表 | 42 (+4 物化视图) |
| API 路由 | 99 |
| MCP | 22 工具 + 6 资源 + 4 提示模板 |
| Service | 34 |
| 前端页面 | 13 |
| Claude Code 技能 | 18 |
| 内置范式 | 4 (RL/VLM/Agent/MotionGen) + LLM 动态发现 |

## 仓库结构

```
researchflow-backend/            # 核心后端（唯一写入目标）
  backend/                       #   FastAPI + ORM + 34 Services + MCP
    api/                         #   14 个路由器 (99 端点)
    mcp/                         #   MCP 服务器 (22 工具 + 6 资源)
    services/                    #   34 个服务模块
      analysis_steps.py          #   v4.0: Step 1+2 聚焦 prompt
      baseline_comparator_service.py  # v4.0: Step 3 比较集
      concept_synthesizer_service.py  # v4.0: Step 5 概念合成
      incremental_reconciler_service.py # v4.0: Step 6 邻居调和
      export_service.py          #   v4.0: 5 类笔记导出
  alembic/                       #   数据库迁移 (001-011)
  frontend/                      #   Next.js 15 Web UI
  ARCHITECTURE.md                #   完整技术文档
  DEPLOY_GUIDE.md                #   部署指南
obsidian-vault/                  # 导出的 Obsidian vault（自动生成）
paperAnalysis/                   # 只读导出：分析笔记
paperCollection/                 # 只读导出：索引 + 导航
paperIDEAs/                      # 只读导出：研究产出
.claude/skills/                  # Claude Code 技能 (18 个)
.mcp.json                       # MCP 服务器配置
AGENTS.md                       # Agent 接入指南
```

## 部署

### 本地开发

```bash
cd researchflow-backend
cp .env.example .env             # 设置 API key
make db                          # 启动 PostgreSQL + Redis
make migrate                     # 运行迁移
make up                          # 启动 API + Worker + Frontend + MCP
# API: localhost:8000 | Frontend: localhost:3000 | MCP: localhost:8001
```

### 生产环境 (Docker)

```bash
ssh root@your-server
git clone https://github.com/RipeMangoBox/ResearchFlow.git
cd ResearchFlow/researchflow-backend
cp .env.example .env             # 配置生产环境
bash deploy.sh                   # 一键：构建 + 迁移 + 启动
```

详见 [DEPLOY_GUIDE.md](researchflow-backend/DEPLOY_GUIDE.md)。

## 文档

| 文档 | 读者 | 内容 |
|------|------|------|
| [ARCHITECTURE.md](researchflow-backend/ARCHITECTURE.md) | 开发者 | 知识图谱 · 6 步管线 · API · DB Schema |
| [AGENTS.md](AGENTS.md) | Agent 开发 | MCP 工具 · 技能路由 · 使用规则 |
| [DEPLOY_GUIDE.md](researchflow-backend/DEPLOY_GUIDE.md) | 运维 | 云端部署 · Docker · 服务器配置 |

## License

MIT
