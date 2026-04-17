# ResearchFlow Backend

Web-first 科研操作系统的后端服务。

> **ResearchFlow = Web 产品 + 自有后端编排 + MCP 兼容层 + 专家模式（Claude/Codex）**

---

## 快速开始

### 前置条件

- Docker + Docker Compose
- Python 3.12+（本地开发时）

### 1. 启动数据库

```bash
cp .env.example .env
# 编辑 .env，设置 POSTGRES_PASSWORD 和 API keys
make db
```

### 2. 运行迁移

```bash
# 安装 Python 依赖（本地开发）
pip install -r requirements.txt

# 创建数据库表
make migrate

# 导入现有数据
make migrate-all
```

### 3. 启动 API

```bash
# Docker 方式
make up

# 或本地开发
uvicorn backend.main:app --reload --port 8000
```

### 4. 验证

```bash
curl http://localhost:8000/api/v1/health
# {"status": "ok"}
```

---

## 架构概览

```
┌─────────────────────────────────────────────┐
│  Web 前端 (Next.js)  │  Claude/Codex (MCP)  │
├─────────────────────────────────────────────┤
│  Presentation   报告渲染 / 总结 / 方向推荐   │
├─────────────────────────────────────────────┤
│  Feedback/Eval  纠错 / 行为采集 / 重分析     │
├─────────────────────────────────────────────┤
│  Workflow/Job   15 种异步任务 (arq + Redis)  │
├─────────────────────────────────────────────┤
│  Retrieval      Postgres + pgvector 混合搜索 │
├─────────────────────────────────────────────┤
│  Parse/Extract  PDF 解析 + LLM 分析 (L1-L4) │
├─────────────────────────────────────────────┤
│  Ingestion      规范化 / 去重 / 资产补全     │
└─────────────────────────────────────────────┘
```

详见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 目录结构

```
researchflow-backend/
├── backend/
│   ├── models/        # 19 张表 ORM 模型
│   ├── schemas/       # Pydantic 请求/响应
│   ├── api/           # FastAPI routers
│   ├── services/      # 业务逻辑
│   ├── workers/       # arq 异步任务
│   ├── mcp/           # MCP server (10 个工具)
│   └── utils/         # 工具模块
├── frontend/          # Next.js 前端 (7 页)
├── alembic/           # 数据库迁移
├── migration/         # 一次性数据迁移脚本
├── compatibility/     # DB→Markdown 导出 (兼容层)
├── caddy/             # 反向代理配置
├── tests/             # 测试
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## 核心功能

| 功能 | 说明 |
|------|------|
| 任意输入导入 | 链接/PDF/Repo/Awesome list/Zotero → 统一入库 |
| 四级分析管线 | L1 metadata → L2 parse → L3 skim → L4 deep |
| 汇报报告 | 30 秒版 / 5 分钟汇报版 / 深度对比版 |
| Canonical Delta Card | 对照领域标准范式，明确改了哪些槽位 |
| 语义搜索 | 关键词 + 向量 + 结构化混合检索 |
| 分层阅读推荐 | baseline → structural → follow-up → patch → negative |
| 日/周/月总结 | 自动生成，产品一等公民 |
| MCP Server | 10 个高层工具，接 Claude Code / Codex |
| 反馈闭环 | 用户纠错 → 触发重分析 |

---

## 常用命令

```bash
make up              # 启动所有服务
make down            # 停止所有服务
make db              # 只启动数据库
make migrate         # 运行 Alembic 迁移
make migrate-all     # 完整迁移 (创建表 + 导入 CSV + 导入 MD + 校验)
make shell           # 打开 psql
make test            # 运行测试
make export-md       # 导出 DB → Markdown
make export-csv      # 导出 DB → CSV
```

---

## 文档

- [ARCHITECTURE.md](ARCHITECTURE.md) — 完整系统架构设计
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) — 分阶段开发方案
- [RESTRUCTURE_PLAN.md](RESTRUCTURE_PLAN.md) — 重构设计文档 v2

---

## 部署

目标环境：腾讯云 Lighthouse 2C4G / 100GB SSD + COS 对象存储。

年化成本约 1500-1900 元（不含 LLM token）。

```bash
# 生产部署
docker compose up -d

# 检查服务状态
docker compose ps
docker compose logs -f api
```

---

## License

MIT
