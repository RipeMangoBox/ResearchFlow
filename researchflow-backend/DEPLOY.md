# ResearchFlow 部署指南

> 经过生产验证 (2026-04-19)，适用于阿里云 ECS 4C8G 或同等配置。

---

## 1. 服务器要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 2 核 | 4 核 |
| 内存 | 4 GB | 8 GB |
| 磁盘 | 40 GB SSD | 70 GB SSD |
| 系统 | Ubuntu 22.04+ | Ubuntu 24.04 |
| 网络 | 开放 80/443 | 有域名 + SSL |

---

## 2. 一键部署

```bash
# 1. 上传代码
rsync -avz --exclude '.git' --exclude 'node_modules' --exclude '__pycache__' \
  ./ root@your-server:/opt/researchflow/

# 2. SSH 登录
ssh root@your-server
cd /opt/researchflow/researchflow-backend

# 3. 配置环境变量
cp .env.example .env
vim .env   # 必填: POSTGRES_PASSWORD, ANTHROPIC_API_KEY, DOMAIN

# 4. 一键启动
bash deploy.sh
# → 自动: Docker build → PostgreSQL 等待 → pgvector 扩展 → Alembic 迁移 → 健康检查
```

部署完成后访问:
- Web UI: `https://your-domain/`
- API Docs: `https://your-domain/api/v1/docs`
- MCP SSE: `https://your-domain/sse`

---

## 3. 容器架构 (8 + 1 容器)

| 容器 | 镜像 | 内存限制 | 说明 |
|------|------|---------|------|
| **postgres** | pgvector/pgvector:pg16 | 1280 MB | PostgreSQL 16 + pgvector 向量检索 |
| **redis** | redis:7-alpine | 256 MB | 任务队列 + 缓存，生产开启 appendonly |
| **api** | 自构建 (Python 3.12) | 1536 MB | FastAPI，生产 2 worker |
| **worker** | 同 api | 1024 MB | ARQ 后台任务 (长分析任务) |
| **mcp** | 同 api | 256 MB | MCP SSE 服务器 (端口 8001) |
| **frontend** | Node.js (Next.js 15) | 256 MB | Web UI (端口 3000) |
| **caddy** | caddy:2-alpine | ~50 MB | 反向代理 + 自动 HTTPS |
| **grobid** | grobid/grobid:0.8.1 | 3072 MB | PDF 结构化解析 (独立管理) |

**总内存占用**: ~7.7 GB (含系统)

### GROBID 独立管理

GROBID 需要 3 GB 内存加载 ML 模型 (启动约 3 分钟)，独立于 docker-compose 管理：

```bash
# 启动 GROBID
docker run -d --name grobid \
  --restart unless-stopped \
  -p 8070:8070 \
  --memory=3g \
  -e JAVA_OPTS="-Xms512m -Xmx2560m" \
  grobid/grobid:0.8.1

# 验证
curl -s http://localhost:8070/api/isalive  # 返回 true
```

---

## 4. 日常部署 (代码更新)

### 场景 A: Python 代码变更 (最常见，~8 秒)

代码通过 volume 挂载，无需 rebuild：

```bash
# 本地 → 服务器
rsync -avz --exclude '.git' --exclude '__pycache__' \
  ./ root@your-server:/opt/researchflow/

# 服务器上重启
ssh root@your-server "cd /opt/researchflow/researchflow-backend && docker compose restart api worker mcp"
```

### 场景 B: docker-compose.yml 变更

```bash
docker compose up -d --force-recreate
```

### 场景 C: requirements.txt 或 Dockerfile 变更

```bash
docker compose build api && docker compose up -d
```

### 场景 D: 数据库迁移

```bash
docker compose exec api alembic upgrade head
```

---

## 5. 环境变量

```bash
# === 必填 ===
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql+asyncpg://researchflow:${POSTGRES_PASSWORD}@postgres:5432/researchflow
REDIS_URL=redis://redis:6379/0
ANTHROPIC_API_KEY=sk-ant-...
DOMAIN=researchflow.xyz              # Caddy HTTPS 域名

# === LLM ===
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1   # 或兼容端点
OPENAI_MODEL=gpt-4o                          # 或 op-4.6 等

# === 对象存储 (可选，图表/PDF 云存储) ===
OBJECT_STORAGE_PROVIDER=cos                  # cos | oss | local
OBJECT_STORAGE_BUCKET=your-bucket
OBJECT_STORAGE_SECRET_ID=xxx
OBJECT_STORAGE_SECRET_KEY=xxx
OBJECT_STORAGE_REGION=ap-shanghai

# === 外部 API (可选) ===
OPENREVIEW_USERNAME=your_email
OPENREVIEW_PASSWORD=your_password
MCP_AUTH_TOKEN=your_mcp_token
```

---

## 6. 网络与代理

### 直连可用的 API

arXiv API、DBLP、GitHub API、OpenAlex、Crossref、HuggingFace — 均可直连无需代理。

### 可能需要代理的服务

Semantic Scholar、OpenReview、Anthropic API — 视服务器网络环境而定。

### 代理配置 (可选，mihomo)

```yaml
# /opt/clash/config.yaml
allow-lan: true           # Docker 容器通过宿主机 IP 访问
bind-address: "*"
mixed-port: 7890
```

Docker daemon 代理 (如需拉取镜像)：

```ini
# /etc/systemd/system/docker.service.d/http-proxy.conf
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
```

容器内使用代理：

```bash
# .env 中设置
HTTP_PROXY=http://172.18.0.1:7890    # Docker 网桥 IP
HTTPS_PROXY=http://172.18.0.1:7890
```

---

## 7. Dockerfile 说明

```dockerfile
# Python 3.12-slim
# apt 使用阿里云镜像 (国内服务器加速)
# pip 使用清华镜像 (国内服务器加速)
# 暴露 8000 端口
# 健康检查: curl http://localhost:8000/api/v1/health
```

国内服务器构建无需代理，Dockerfile 内置了镜像源替换。

---

## 8. 数据库管理

```bash
# 查看当前迁移版本
docker compose exec api alembic current

# 应用所有迁移
docker compose exec api alembic upgrade head

# 回滚一步
docker compose exec api alembic downgrade -1

# 备份
docker compose exec postgres pg_dump -U researchflow researchflow > backup_$(date +%Y%m%d).sql

# 恢复
cat backup.sql | docker compose exec -T postgres psql -U researchflow researchflow
```

当前 Alembic 版本: v015 (15 次迁移)

---

## 9. 运维命令

```bash
# 查看所有容器状态
docker compose ps

# 查看日志 (实时)
docker compose logs -f api
docker compose logs -f worker --tail=100

# 重启单个服务
docker compose restart api

# 清理磁盘
docker system prune -f

# 进入容器 Python 环境
docker compose exec api python
```

### 在容器内直接执行 Pipeline

对于长时间运行的任务，避免 HTTP 超时：

```bash
docker compose exec api python -c "
import asyncio
from backend.services.pipeline_service import PipelineService
from backend.database import async_session_factory
async def run():
    async with async_session_factory() as session:
        svc = PipelineService(session)
        await svc.run_full_pipeline('paper-uuid-here')
asyncio.run(run())
"
```

---

## 10. 资源与成本参考

| 项目 | 费用 |
|------|------|
| 阿里云 ECS 4C8G | ~2000-3000 CNY/年 |
| 域名 (.xyz) | ~25 CNY/年 |
| LLM API | ~$5-20/月 (按分析量) |
| 对象存储 | ~10 CNY/月 (图表/PDF) |

---

## 11. 故障排查

| 问题 | 解决方案 |
|------|---------|
| GROBID 返回 502 | 检查 `docker logs grobid`，通常是内存不足，确保 3 GB |
| API 启动失败 | `docker compose logs api`，常见: DB 连接失败、迁移未执行 |
| PDF 下载失败 | 检查 arXiv 访问，尝试设置代理 |
| LLM 调用超时 | 检查 API key 和 base_url 配置 |
| 端口冲突 | `lsof -i :8000` 查看占用进程 |
| 迁移冲突 | `alembic current` 确认版本，`alembic upgrade head` 补齐 |
