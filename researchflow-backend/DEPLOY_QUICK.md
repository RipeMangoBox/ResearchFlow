# ResearchFlow 快速部署指南 (验证于 2026-04-19)

## 服务器信息

```
服务器: 阿里云 ECS 4C8G / 70G SSD
IP: 47.101.167.55
域名: researchflow.xyz
SSH: ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55
代码路径: /opt/researchflow/researchflow-backend/  ← 注意不是 /root/
代理: mihomo /opt/clash/ (7890/7891)
```

---

## 日常部署 (改了 Python 代码, 10 秒)

代码通过 Docker volume 挂载，rsync 后 restart 即可，**不需要 rebuild**。

```bash
# Step 1: 同步代码 (~3秒)
rsync -avz --delete \
  --exclude='.venv' --exclude='node_modules' --exclude='__pycache__' \
  --exclude='.git' --exclude='storage/' --exclude='.env' \
  --exclude='obsidian-vault/' --exclude='debug_figures' \
  -e "ssh -i ~/.ssh/autoresearch.pem" \
  /Users/hzh/Desktop/简历/ResearchFlow/researchflow-backend/ \
  root@47.101.167.55:/opt/researchflow/researchflow-backend/

# Step 2: 重启 (~5秒)
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55 \
  "cd /opt/researchflow/researchflow-backend && docker compose restart api"

# Step 3: 验证
ssh -i ~/.ssh/autoresearch.pem root@47.101.167.55 \
  "curl -s http://localhost:8000/api/v1/health"
```

### 何时需要 rebuild (而非 restart)

| 改了什么 | 操作 |
|---------|------|
| Python 代码 | `docker compose restart api` |
| docker-compose.yml (内存/端口/环境变量) | `docker compose up -d --force-recreate api` |
| requirements.txt (新增依赖) | `docker compose build api && docker compose up -d --force-recreate api` |
| Dockerfile | `docker compose build --no-cache api && docker compose up -d --force-recreate api` |

**注意**: `restart` 不会重新应用 compose.yml 的改动（如内存限制）。必须用 `up -d --force-recreate`。

---

## 容器架构 (8 个)

| 容器 | 内存 | 端口 | 管理方式 |
|------|------|------|---------|
| api | 1536MB | 8000 | docker compose |
| worker | 1024MB | - | docker compose |
| mcp | 256MB | 8001 | docker compose |
| postgres | 1280MB | 5432 | docker compose |
| redis | 256MB | 6379 | docker compose |
| frontend | 256MB | 3000 | docker compose |
| caddy | - | 80/443 | docker compose |
| **grobid** | **3072MB** | 8070 | **独立 docker run** |

### GROBID 管理 (独立容器)

GROBID 不在 docker-compose.yml 中，需要单独管理：

```bash
# 启动 (首次或重启后)
docker run -d --name grobid --restart unless-stopped \
  --network researchflow-backend_default \
  --memory 3g \
  -p 127.0.0.1:8070:8070 \
  lfoppiano/grobid:0.8.1

# 启动需要 ~3 分钟 (加载 ML 模型)
# 验证
curl -s http://localhost:8070/api/isalive  # 返回 true 才算就绪
```

⚠️ GROBID 需要 **3GB 内存**（2GB 会 OOM 循环重启）。服务器 8GB 总内存刚好够。

---

## 网络连通性 (已验证)

### 容器内直连 (不需要代理)
- ✅ arXiv API + PDF 下载
- ✅ DBLP
- ✅ GitHub
- ✅ OpenAlex
- ✅ Crossref
- ✅ HuggingFace (mihomo `allow-lan: true` 后直连 OK)

### 需要 API key
- ⚠️ Semantic Scholar — 429 限流，需申请免费 API key
- ⚠️ OpenReview — 403，需注册账号

### 不需要在 docker-compose.yml 中配置代理环境变量
之前尝试在容器内配 `HTTP_PROXY` 会导致 httpx TLS 握手失败。
正确做法：mihomo 配置 `allow-lan: true` + 添加域名规则，容器直连即可。

---

## 代理配置 (mihomo)

```bash
# 配置文件
/opt/clash/config.yaml

# 关键配置项
allow-lan: true  # 必须，让容器通过宿主机代理

# 已添加的代理规则 (在 rules 列表最前面)
- DOMAIN-SUFFIX,huggingface.co,Proxy
- DOMAIN-KEYWORD,huggingface,Proxy
- DOMAIN-SUFFIX,openreview.net,Proxy
- DOMAIN-SUFFIX,semanticscholar.org,Proxy
- DOMAIN-SUFFIX,docker.io,Proxy
- DOMAIN-KEYWORD,anthropic,Proxy

# 重启代理
cd /opt/clash && pkill mihomo; sleep 2; nohup ./mihomo -d . &>/tmp/mihomo.log &
```

### Docker daemon 代理 (拉镜像用)

```
文件: /etc/systemd/system/docker.service.d/http-proxy.conf
内容:
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
Environment="NO_PROXY=localhost,127.0.0.1,172.18.0.0/16"

# 生效
systemctl daemon-reload && systemctl restart docker
```

---

## LLM 配置

```env
OPENAI_API_KEY=sk-reapi-xxx
OPENAI_BASE_URL=https://apicursor.com/v1
OPENAI_MODEL=op-4.6
```

- **文本 LLM**: ✅ 通过 OpenAI SDK (streaming 模式)
- **Vision/VLM**: ✅ `op-4.6` 支持多模态，必须用 streaming 模式
- **重要**: apicursor 不支持非 streaming 调用（会返回 str 而非 ChatCompletion 对象）
- 所有 VLM 服务已改为 OpenAI SDK + streaming

---

## Pipeline 运行

### 通过 API (可能超时)
```bash
# 导入
curl -X POST http://localhost:8000/api/v1/import/links \
  -H "Content-Type: application/json" \
  -d '{"items": [{"url": "https://arxiv.org/abs/2507.02259"}], "default_category": "LLM"}'

# 运行 pipeline
curl -X POST http://localhost:8000/api/v1/pipeline/PAPER_ID/run
```

### 直接在容器内运行 (推荐，无超时)
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

### 单独运行 L2 Parse (含 GROBID)
```bash
docker compose exec -T api python3 -c "
import asyncio
from uuid import UUID
from backend.database import async_session
from backend.services.parse_service import parse_paper_pdf
async def main():
    async with async_session() as s:
        a = await parse_paper_pdf(s, UUID('PAPER_ID'))
        if a:
            spans = a.evidence_spans or {}
            print('Parsers:', spans.get('parse_metadata',{}).get('parsers_used'))
            print('GROBID refs:', len(spans.get('grobid_references',[])))
            print('Formulas:', len(a.extracted_formulas or []))
            await s.commit()
asyncio.run(main())
"
```

---

## 论文数据清理 (级联删除)

papers 表被 20+ 个表引用，简单 DELETE 会失败。使用完整脚本：

```sql
DO $$ DECLARE pid UUID := 'YOUR_PAPER_ID'; BEGIN
  DELETE FROM evidence_units WHERE delta_card_id IN (SELECT id FROM delta_cards WHERE paper_id=pid);
  DELETE FROM graph_assertion_evidence WHERE assertion_id IN (SELECT id FROM graph_assertions WHERE from_node_id IN (SELECT id FROM graph_nodes WHERE ref_id IN (SELECT id FROM idea_deltas WHERE paper_id=pid)));
  DELETE FROM graph_assertions WHERE from_node_id IN (SELECT id FROM graph_nodes WHERE ref_id IN (SELECT id FROM idea_deltas WHERE paper_id=pid));
  DELETE FROM graph_nodes WHERE ref_id IN (SELECT id FROM idea_deltas WHERE paper_id=pid);
  DELETE FROM contribution_to_canonical_idea WHERE idea_delta_id IN (SELECT id FROM idea_deltas WHERE paper_id=pid);
  DELETE FROM idea_deltas WHERE paper_id=pid;
  DELETE FROM model_runs WHERE paper_id=pid;
  DELETE FROM delta_card_lineage WHERE child_delta_card_id IN (SELECT id FROM delta_cards WHERE paper_id=pid);
  DELETE FROM delta_cards WHERE paper_id=pid;
  DELETE FROM paper_bottleneck_claims WHERE paper_id=pid;
  DELETE FROM method_deltas WHERE paper_id=pid;
  DELETE FROM paper_analyses WHERE paper_id=pid;
  DELETE FROM paper_assets WHERE paper_id=pid;
  DELETE FROM paper_versions WHERE paper_id=pid;
  DELETE FROM metadata_observations WHERE entity_id=pid;
  DELETE FROM canonical_paper_metadata WHERE paper_id=pid;
  DELETE FROM papers WHERE id=pid;
END $$;
```

---

## 常见错误速查

| 错误 | 原因 | 解决 |
|------|------|------|
| `restart` 后内存限制没变 | restart 不应用 compose 改动 | `up -d --force-recreate api` |
| GROBID OOM 循环重启 | 2GB 不够 | 给 3GB: `--memory 3g` |
| Docker pull 403 | 镜像源被墙 | Docker daemon proxy (见上) |
| arXiv title 是 "2507.02259" | ingest 时没有 title | 已修复：enrich 自动用 arXiv 返回的标题 |
| Pipeline HTTP 超时 | LLM 调用慢 | 容器内直接跑 Python (见上) |
| VLM 400 "Request too large" | 没用 streaming | 已修复：所有 VLM 用 OpenAI SDK streaming |
| alembic migration KeyError | 版本链不匹配 | 服务器 DB 在 `015`，新 migration 需 `down_revision="015"` |
| compose YAML 格式坏 | sed/python 修改出错 | 改前备份，改后 `python3 -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"` |

---

## 数据库

- **当前 alembic 版本**: `015` (v2 表通过 SQL 直接创建，未走 alembic)
- **v2 新增 11 张表**: metadata_observations, canonical_paper_metadata, taxonomy_nodes/edges, paper_facets, problem_nodes/claims, method_nodes/slots/edges/applications
- **Taxonomy 种子**: 75 个节点 (10 个维度), 14 条边

### 重置种子数据
```bash
docker compose exec -T -e PYTHONPATH=/app api python -m migration.seed_taxonomy
```

---

## 已验证的提取能力 (MemAgent 论文 2507.02259)

| 数据 | 无 GROBID | 有 GROBID |
|------|----------|----------|
| Title | ✅ | ✅ |
| Abstract | ✅ | ✅ |
| Authors | ✅ (11, 无机构) | ✅ (12, **含机构**) |
| Year/Venue | ✅ | ✅ |
| Citations | ✅ (116) | ✅ |
| Code/Data URL | ✅ | ✅ |
| Sections | 7 | 7 |
| **引用 (结构化)** | **0** | **19 条** |
| **公式** | **4** | **13** |
| 图表 | 5 | 5 |
| L3 Skim | ✅ | ✅ |
| L4 Deep | ✅ | ✅ |
| VLM (Vision) | ✅ (op-4.6) | ✅ |
