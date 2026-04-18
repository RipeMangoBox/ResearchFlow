# ResearchFlow 快速部署指南

## 部署路径

```
本地代码: /Users/hzh/Desktop/简历/ResearchFlow/researchflow-backend/
服务器路径: /opt/researchflow/researchflow-backend/
SSH: ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55
```

## 快速部署（代码改动后）

### 1. 同步代码 (~3秒)
```bash
rsync -avz --delete \
  --exclude='.venv' --exclude='node_modules' --exclude='.next' \
  --exclude='__pycache__' --exclude='.git' --exclude='.pytest_cache' \
  --exclude='storage/' --exclude='.env' --exclude='debug_figures' \
  --exclude='obsidian-vault/' --exclude='paperPDFs/' --exclude='paperAnalysis/' \
  -e "ssh -i ~/.ssh/autoresearch.pem" \
  /Users/hzh/Desktop/简历/ResearchFlow/researchflow-backend/ \
  root@47.101.167.55:/opt/researchflow/researchflow-backend/
```

### 2. 重建容器 (~30秒, 有缓存时)
```bash
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55 \
  "cd /opt/researchflow/researchflow-backend && \
   docker compose build api worker mcp && \
   docker compose up -d --force-recreate api worker mcp"
```

如果代码改动很小（只改了 Python 文件，没改 requirements.txt），可以用热重载代替重建：
```bash
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55 \
  "cd /opt/researchflow/researchflow-backend && \
   docker compose restart api worker mcp"
```

### 3. 验证
```bash
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55 \
  "curl -s http://localhost:8000/api/v1/health"
```

## 一键部署脚本
```bash
# 保存为 deploy.sh
#!/bin/bash
set -e
SERVER="root@47.101.167.55"
SSH="ssh -i ~/.ssh/autoresearch.pem"
REMOTE="/opt/researchflow/researchflow-backend"

echo "=== Syncing code ==="
rsync -avz --delete \
  --exclude='.venv' --exclude='node_modules' --exclude='.next' \
  --exclude='__pycache__' --exclude='.git' --exclude='storage/' \
  --exclude='.env' --exclude='debug_figures' --exclude='obsidian-vault/' \
  -e "$SSH" \
  /Users/hzh/Desktop/简历/ResearchFlow/researchflow-backend/ \
  $SERVER:$REMOTE/

echo "=== Restarting containers ==="
$SSH $SERVER "cd $REMOTE && docker compose restart api worker mcp"

echo "=== Waiting ==="
sleep 5

echo "=== Health check ==="
$SSH $SERVER "curl -s http://localhost:8000/api/v1/health"
echo ""
echo "DONE"
```

## 瓶颈分析

| 步骤 | 耗时 | 原因 |
|------|------|------|
| rsync | ~3秒 | 无瓶颈 |
| docker build (有缓存) | ~10秒 | pip install 已缓存 |
| docker build (无缓存) | ~60秒 | pip install 从头装 |
| docker build --no-cache | ~120秒 | **避免使用**, 只在 requirements.txt 变化时用 |
| docker restart | ~5秒 | **推荐**, 代码改动用这个 |
| 容器启动 | ~5-10秒 | 等 postgres healthy |

**结论**: 日常改动用 `rsync + docker restart` (~10秒), 不用 `build --no-cache` (~120秒)

## 网络限制 (关键！)

### 容器内可直连
- arXiv API (export.arxiv.org) ✅
- OpenAlex API ✅
- Crossref API ✅
- DBLP API ✅
- HuggingFace API ✅

### 容器内需要代理
- **arXiv PDF 下载** (arxiv.org/pdf/) — 经常被限速/墙
- **GitHub API** — 部分地区限流
- **Semantic Scholar API** — 频繁调用被 429
- **OpenReview API** — 403 (可能需要认证或代理)
- **Google Scholar** — 完全被墙

### 配置容器代理
在 docker-compose.yml 的 api/worker 服务中加：
```yaml
environment:
  HTTP_PROXY: http://host.docker.internal:7890
  HTTPS_PROXY: http://host.docker.internal:7890
  NO_PROXY: localhost,postgres,redis,127.0.0.1
```

或者在 .env 中设置（如果 compose 使用 env_file）：
```env
HTTP_PROXY=http://172.17.0.1:7890
HTTPS_PROXY=http://172.17.0.1:7890
NO_PROXY=localhost,postgres,redis,127.0.0.1
```

注意：容器内不能用 `host.docker.internal`（Linux 不支持），用 Docker 网关 IP `172.17.0.1` 或主机 IP。

### 验证代理
```bash
docker compose exec -T api python3 -c "
import httpx, asyncio
async def t():
    async with httpx.AsyncClient(proxy='http://172.17.0.1:7890', timeout=10) as c:
        r = await c.get('https://arxiv.org/pdf/2507.02259')
        print(f'arXiv PDF: {r.status_code} size={len(r.content)}')
asyncio.run(t())
"
```

## 常见错误

### 1. docker compose YAML 格式错误
**原因**: sed/python 修改 YAML 时格式破坏
**解决**: 总是保留 .bak 备份, 修改后用 `python3 -c "import yaml; yaml.safe_load(open('docker-compose.yml')); print('OK')"` 验证

### 2. Alembic migration chain 断裂
**原因**: 本地和服务器的 migration 版本不一致
**解决**: 检查 `SELECT version_num FROM alembic_version;`, 新 migration 的 down_revision 要指向服务器当前版本
**服务器当前版本**: `015`

### 3. Docker 镜像拉取 403
**原因**: 中国 Docker 镜像源限制某些镜像（如 GROBID）
**解决**: 
- 用 `docker pull` 加代理: `HTTP_PROXY=http://127.0.0.1:7890 docker pull lfoppiano/grobid:0.8.1`
- 或改用其他镜像源

### 4. arXiv API 返回错误论文
**原因**: 从 URL ingest 时 paper.title 只是 arxiv ID (如 "2507.02259"), enrich 时 arXiv API 返回正确数据但被 title 验证拒绝
**已修复**: 检测 title 是否是 placeholder (纯数字.数字格式), 如果是则跳过验证

### 5. Pipeline HTTP 超时
**原因**: curl --max-time 不够长, LLM 调用需要 30-60 秒/次, L3+L4 共 2-3 次调用
**解决**: 直接在容器内运行 Python 脚本, 不经过 HTTP:
```bash
docker compose exec -T api python3 -c "
import asyncio, json
from uuid import UUID
from backend.database import async_session
from backend.services.pipeline_service import run_full_pipeline
async def main():
    async with async_session() as session:
        r = await run_full_pipeline(session, UUID('PAPER_ID'))
        print(json.dumps(r, indent=2, default=str))
asyncio.run(main())
"
```

### 6. 外键级联删除
**原因**: papers 表被 20+ 个表引用, 简单 DELETE 会失败
**解决**: 使用完整的级联删除脚本 (见 scripts/clean_paper.sql)

## 数据库新表 (v2 新增)

| 表 | 用途 |
|----|------|
| metadata_observations | 多源元数据观察记录 |
| canonical_paper_metadata | 元数据冲突解决结果 |
| taxonomy_nodes | 分类节点 (75个种子) |
| taxonomy_edges | 分类关系边 (14条) |
| paper_facets | 论文-分类关联 |
| problem_nodes | 任务下的共性问题 |
| problem_claims | 论文-问题关联 |
| method_nodes | 方法节点 |
| method_slots | 方法组件 |
| method_edges | 方法演化边 |
| method_applications | 论文-方法使用关系 |
