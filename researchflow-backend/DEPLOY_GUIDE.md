# ResearchFlow 部署指南

## 当前部署信息

| 项目 | 值 |
|------|-----|
| 服务器 | 阿里云 ECS 4C8G / 70G SSD |
| 公网 IP | 47.101.167.55 |
| 域名 | researchflow.xyz |
| HTTPS | Let's Encrypt 自动证书 (Caddy) |
| 前端 | https://researchflow.xyz/ |
| API | https://researchflow.xyz/api/v1/ |
| SSH | `ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55` |

## 一键部署（从零开始）

### 前置条件

- Ubuntu 22.04 + Docker + Docker Compose 已安装
- 域名 DNS A 记录指向服务器 IP
- 代理（可选）：mihomo 在 `/opt/clash/` 运行，端口 7890

### Step 1: 上传代码

```bash
# 从本地 rsync（不用 git clone，因为 GitHub 在国内被墙）
rsync -avz --delete \
  --exclude='.venv' --exclude='node_modules' --exclude='.next' \
  --exclude='__pycache__' --exclude='.git' --exclude='.pytest_cache' \
  --exclude='storage/' --exclude='.env' \
  -e "ssh -i ~/.ssh/autoresearch.pem" \
  /path/to/researchflow-backend/ \
  root@47.101.167.55:/root/researchflow-backend/
```

### Step 2: 配置环境变量

```bash
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55
cd /root/researchflow-backend
cp .env.deploy .env
nano .env
```

必须设置：
```env
POSTGRES_PASSWORD=<强密码>
ANTHROPIC_API_KEY=sk-ant-xxx    # 或 OPENAI_API_KEY
```

### Step 3: 构建并启动

```bash
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
```

### Step 4: 初始化数据库

```bash
# 启用 pgvector
docker compose -f docker-compose.prod.yml exec -T postgres \
  psql -U rf -d researchflow -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 运行迁移（7 次）
docker compose -f docker-compose.prod.yml exec -T -e PYTHONPATH=/app api \
  alembic -c alembic/alembic.ini upgrade head

# 种入范式数据
docker compose -f docker-compose.prod.yml exec -T -e PYTHONPATH=/app api \
  python -m migration.seed_paradigm_frames
```

### Step 5: 验证

```bash
curl https://researchflow.xyz/api/v1/health       # {"status":"ok"}
curl https://researchflow.xyz/api/v1/graph/stats   # 应有 slots:25, mechanisms:19
```

---

## Docker 构建注意事项（中国大陆服务器）

| 组件 | 镜像源 | 说明 |
|------|--------|------|
| apt | mirrors.aliyun.com | Dockerfile 内 sed 替换 |
| pip | pypi.tuna.tsinghua.edu.cn | Dockerfile 内 `-i` 参数 |
| npm | registry.npmmirror.com | Frontend Dockerfile 内 `npm config set` |
| Docker Hub | mirror.ccs.tencentyun.com | `/etc/docker/daemon.json` registry-mirrors |
| GitHub | 不可用 | 用 rsync 替代 git clone |

**关键**：不要给 Docker daemon 设全局代理（会导致容器内 DNS 解析国内域名失败）。

---

## 代理配置（可选，用于访问 GitHub/Semantic Scholar）

```bash
# mihomo 已安装在 /opt/clash/
cd /opt/clash && nohup ./mihomo -d . &>/tmp/mihomo.log &

# 验证
curl -x http://127.0.0.1:7890 https://github.com  # 应返回 200
```

代理仅用于：
- API 容器内访问 Semantic Scholar（论文发现）
- API 容器内访问 GitHub（awesome 仓库解析）

容器内使用代理需要在 `.env` 添加（暂未配置）：
```env
HTTP_PROXY=http://172.17.0.1:7890
HTTPS_PROXY=http://172.17.0.1:7890
```

---

## 日常运维

```bash
# 查看日志
docker compose -f docker-compose.prod.yml logs -f api

# 重启（代码更新后）
rsync ...  # 重新上传代码
docker compose -f docker-compose.prod.yml up -d --build api worker

# 备份数据库
docker compose -f docker-compose.prod.yml exec -T postgres \
  pg_dump -U rf researchflow > backup_$(date +%Y%m%d).sql

# 磁盘清理
docker builder prune -af
docker image prune -f
```

---

## 服务器资源分配

| 服务 | 内存限制 | CPU | 说明 |
|------|---------|-----|------|
| PostgreSQL | 1280 MB | 共享 | shared_buffers 自动 |
| Redis | 256 MB | 共享 | maxmemory=200mb |
| API (FastAPI) | 400 MB | 共享 | 2 workers |
| Worker (arq) | 1024 MB | 共享 | PDF 解析 + LLM 调用 |
| Frontend | 256 MB | 共享 | Next.js standalone |
| Caddy | ~50 MB | 共享 | HTTPS 反向代理 |
| 系统 + mihomo | ~1.8 GB | 共享 | OS + 代理 |

总计 ~4 GB used / 8 GB total。

---

## 费用参考

| 项目 | 费用 |
|------|------|
| 阿里云 ECS 4C8G | ~2000-3000 元/年 |
| 域名 researchflow.xyz | ~25 元/年 |
| Anthropic API | ~$5-20/月（取决于分析论文量）|
| 总计 | ~3000 元/年 + token 费 |
