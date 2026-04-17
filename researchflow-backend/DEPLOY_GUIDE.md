# ResearchFlow 部署指南

## 一、云端配置费用分析

### 必选项

| 资源 | 规格 | 年费(元) | 说明 |
|------|------|---------|------|
| **云服务器** | 腾讯云 Lighthouse 2核4GB / 100GB SSD | ~1020 | Ubuntu 22.04, 北京/上海 |
| **域名** | .com 或 .cn | ~60-80 | 首年可能有优惠 |
| **合计最低** | | **~1080-1100** | |

### 可选项（推荐但非必须）

| 资源 | 规格 | 年费(元) | 说明 |
|------|------|---------|------|
| COS 对象存储 200GB | 腾讯云标准存储 | ~283 | PDF 太多时再开，初期用服务器本地盘 |
| COS 对象存储 500GB | | ~708 | 论文超 1000 篇后考虑 |

### 不含在基础设施中的费用

| 项目 | 预估 | 说明 |
|------|------|------|
| Anthropic API (Claude) | ~$5-20/月 | 取决于分析论文数量。L3 skim ~$0.01/篇，L4 deep ~$0.05/篇 |
| OpenAI Embedding | ~$1-5/月 | text-embedding-3-small, 1000 篇约 $0.5 |

### 预算总结

```
最低配置:  服务器 1020 + 域名 70 = 1090 元/年 (+ token 费)
推荐配置:  服务器 1020 + COS 283 + 域名 70 = 1373 元/年 (+ token 费)
```

---

## 二、购买步骤（腾讯云）

### Step 1: 注册腾讯云账号

1. 打开 https://cloud.tencent.com/
2. 注册并完成实名认证（个人即可）

### Step 2: 购买 Lighthouse 轻量应用服务器

1. 打开 https://cloud.tencent.com/product/lighthouse
2. 点击 **立即选购**
3. 选择配置：
   - **地域**: 北京 或 上海（国内访问快）
   - **镜像**: Ubuntu 22.04 LTS
   - **套餐**: **2核 4GB / 100GB SSD**（约 1020 元/年）
   - **购买时长**: 1 年（年付更便宜）
4. 确认购买，等待创建完成（约 1-2 分钟）
5. 记下 **公网 IP 地址**

### Step 3: 注册域名

1. 打开 https://dnspod.cloud.tencent.com/
2. 搜索一个域名（如 `researchflow.site`）
3. 购买（.site 约 25 元/年，.com 约 70 元/年）
4. 完成实名认证（国内域名需要）

### Step 4: 配置 DNS

1. 在 DNSPod 控制台找到你的域名
2. 添加 A 记录：
   - 主机记录: `rf`（或 `@` 用根域名）
   - 记录值: 你的服务器公网 IP
   - TTL: 600
3. 等待 DNS 生效（通常 5-10 分钟）

### Step 5: （可选）开通 COS 对象存储

如果你的论文 PDF 很多（>100 篇），推荐开通：

1. 打开 https://console.cloud.tencent.com/cos
2. 创建存储桶：
   - 名称: `researchflow-你的APPID`
   - 地域: 与服务器同区域（如北京）
   - 访问权限: **私有读写**
3. 获取 API 密钥：
   - 打开 https://console.cloud.tencent.com/cam/capi
   - 创建子用户密钥（推荐，比主账号密钥安全）
   - 记下 `SecretId` 和 `SecretKey`

---

## 三、服务器配置步骤

### Step 1: SSH 连接服务器

```bash
ssh ubuntu@你的公网IP
# 首次连接输入 yes 确认
```

如果是 Lighthouse，也可以在控制台用 **OrcaTerm** 网页终端登录。

### Step 2: 安装 Docker

```bash
# 安装 Docker + Docker Compose
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 重新登录让权限生效
exit
ssh ubuntu@你的公网IP

# 验证
docker --version
docker compose version
```

### Step 3: 安装 Git 并克隆项目

```bash
sudo apt update && sudo apt install -y git

git clone https://github.com/RipeMangoBox/ResearchFlow.git
cd ResearchFlow/researchflow-backend
```

### Step 4: 配置环境变量

```bash
cp .env.production .env
nano .env
```

**必须修改的项**：

```env
# 数据库密码 — 设一个强密码
POSTGRES_PASSWORD=这里改成随机强密码

# 域名 — 你的实际域名
DOMAIN=rf.你的域名.com

# LLM API Key — 至少填一个
ANTHROPIC_API_KEY=sk-ant-你的key
```

**可选修改**（如果开了 COS）：

```env
OBJECT_STORAGE_PROVIDER=cos
OBJECT_STORAGE_BUCKET=researchflow-你的APPID
OBJECT_STORAGE_SECRET_ID=你的SecretId
OBJECT_STORAGE_SECRET_KEY=你的SecretKey
OBJECT_STORAGE_REGION=ap-beijing
```

保存退出（Ctrl+O, Enter, Ctrl+X）。

### Step 5: 一键部署

```bash
bash deploy.sh
```

脚本会自动：
1. 构建 Docker 镜像（首次约 5-10 分钟）
2. 启动 5 个服务（postgres, redis, api, worker, frontend, caddy）
3. 运行数据库迁移
4. 启用 pgvector 扩展
5. 检查服务健康状态

### Step 6: 验证部署

```bash
# 检查服务状态
docker compose -f docker-compose.prod.yml ps

# 检查 API
curl http://localhost:8000/api/v1/health

# 检查 HTTPS（如果配了域名）
curl https://rf.你的域名.com/api/v1/health
```

打开浏览器访问 `https://rf.你的域名.com`，应该能看到 Dashboard。

### Step 7: 导入现有论文数据

如果你之前在本地已有 paperAnalysis/ 数据：

```bash
# 先把数据传到服务器
scp -r paperAnalysis/ ubuntu@你的IP:~/ResearchFlow/

# 然后运行迁移
docker compose -f docker-compose.prod.yml exec api python -m migration.migrate_csv_to_db
docker compose -f docker-compose.prod.yml exec api python -m migration.migrate_md_to_db
docker compose -f docker-compose.prod.yml exec api python -m migration.validate_migration
```

---

## 四、日常运维

### 查看日志

```bash
# 全部日志
docker compose -f docker-compose.prod.yml logs -f

# 只看 API
docker compose -f docker-compose.prod.yml logs -f api

# 只看 Worker
docker compose -f docker-compose.prod.yml logs -f worker
```

### 重启服务

```bash
# 重启 API（代码更新后）
docker compose -f docker-compose.prod.yml restart api

# 重启全部
docker compose -f docker-compose.prod.yml restart

# 停止全部
docker compose -f docker-compose.prod.yml down

# 重新构建并启动（代码大改后）
docker compose -f docker-compose.prod.yml up -d --build
```

### 更新代码

```bash
cd ~/ResearchFlow
git pull origin feat/paper-filter-and-report-generation
cd researchflow-backend
docker compose -f docker-compose.prod.yml up -d --build api worker frontend
```

### 数据库备份

```bash
# 备份
docker compose -f docker-compose.prod.yml exec postgres pg_dump -U rf researchflow > backup_$(date +%Y%m%d).sql

# 恢复
cat backup_20260418.sql | docker compose -f docker-compose.prod.yml exec -T postgres psql -U rf researchflow
```

### 磁盘空间检查

```bash
df -h                                          # 总磁盘
docker system df                               # Docker 占用
docker compose -f docker-compose.prod.yml exec postgres du -sh /var/lib/postgresql/data  # DB 大小
```

---

## 五、防火墙配置

腾讯云 Lighthouse 默认只开 22 端口。需要开放：

1. 打开 Lighthouse 控制台 → 防火墙
2. 添加规则：
   - **TCP 80** — HTTP（Caddy 自动重定向到 HTTPS）
   - **TCP 443** — HTTPS
3. **不要**开放 5432（PostgreSQL）和 6379（Redis）

---

## 六、服务器内存分配

2C4G (4096MB) 实际分配：

| 服务 | 内存 | 说明 |
|------|------|------|
| PostgreSQL | 1280 MB | shared_buffers=256MB |
| Redis | 256 MB | maxmemory=200mb |
| API (FastAPI) | 400 MB | 2 workers |
| Worker (arq) | 1024 MB | PDF 解析 + LLM 调用 |
| Frontend (Next.js) | 256 MB | standalone 模式 |
| Caddy | ~50 MB | 反向代理 |
| OS + buffer | ~830 MB | 系统预留 |
| **总计** | **~4096 MB** | |

如果内存紧张，可以：
- 减少 API workers 从 2 到 1 → 省 ~150MB
- 减少 PostgreSQL 到 1024MB → 省 256MB

---

## 七、Anthropic API Key 获取

1. 打开 https://console.anthropic.com/
2. 注册账号
3. 进入 API Keys 页面
4. 创建新 Key，复制保存
5. 充值（最低 $5）

推荐模型：`claude-sonnet-4-20250514`（性价比最高）

Token 费用参考：
- L3 skim: ~2K input + ~1K output ≈ $0.01/篇
- L4 deep: ~10K input + ~3K output ≈ $0.05/篇
- 报告生成: ~5K input + ~2K output ≈ $0.03/次
- 分析 100 篇论文（L3）+ 20 篇（L4）≈ $2

---

## 八、常见问题

**Q: 域名还没备案怎么办？**
A: 可以先用 IP 直接访问（`http://你的IP:80`），在 Caddyfile 中 DOMAIN 设为 IP 地址。
备案完成后再改为域名。

**Q: 没有 Anthropic Key 能用吗？**
A: 能用。系统会进入 mock 模式，所有 LLM 分析返回占位内容。
导入、搜索、过滤、PDF 解析等不依赖 LLM 的功能正常工作。

**Q: 内存不够用了怎么办？**
A: 1) 升级服务器到 4C8G；2) 或把 PostgreSQL 数据目录移到额外云硬盘。

**Q: 怎么迁移到更大的服务器？**
A: 备份 pgdata 和 redisdata 两个 Docker volume，在新服务器恢复即可。

**Q: COS 费用怎么算？**
A: 标准存储 0.118 元/GB/月。100 篇论文 PDF 约 1-2GB，年费约 1.4-2.8 元。
主要费用在存储量而非请求次数。
