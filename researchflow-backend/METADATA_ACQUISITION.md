# 论文元数据获取方案

> 每个字段的获取是一条**有优先级的 fallback 链**：前一级获取不到或不可信时，才尝试下一级。
> 部分字段需要 **二次审核**（API 交叉验证或 VLM 判断）来确认准确性。

---

## 0. 设计原则

1. **逐级 fallback**：每个字段有 3-5 层获取方案，前一层成功即停止
2. **二次审核**：关键字段（venue、acceptance、code_url）需要多源交叉验证或 VLM 辅助判断
3. **观测记账**：所有获取结果写入 `metadata_observations` 表，附带 source + confidence + authority_rank
4. **最终写入**：只有通过审核的值才写入 `papers` 表对应字段
5. **限速保护**：所有外部 API 通过 `TokenBucketLimiter` 统一限速，429 时指数退避

---

## 1. 字段级 Fallback 链

### 1.1 title

```
L1  venue_index 本地查询 (零 API)
 ↓ 未命中
L2  arXiv API  export.arxiv.org/api/query?id_list={arxiv_id}
     ├─ 返回 title → 写入
     └─ 二次审核: 如果 paper 已有 title，做 _titles_similar() 相似度校验
        ├─ 相似 → 保留 arXiv title
        └─ 不相似 → 丢弃 arXiv 数据，日志告警 title mismatch
 ↓ 无 arxiv_id
L3  Crossref title search → 返回结果做 _titles_similar() 校验
 ↓ 无 DOI
L4  S2 paper/search?query={title} → _titles_similar() 校验
```

**二次审核**: `_titles_similar()` 归一化比较（去标点、小写、去多余空格），阈值 ~0.85。防止 arXiv/S2/Crossref 返回同名但不同论文。

---

### 1.2 abstract

```
L1  venue_index 本地查询
 ↓ 空
L2  arXiv API → abstract 字段
 ↓ 空
L3  OpenAlex API → inverted_index 还原 abstract
 ↓ 空
L4  S2 API → abstract 字段
```

无需二次审核。直接取第一个非空结果。

---

### 1.3 authors + 机构

```
L1  arXiv API → author_detail (name, affiliation)
     └─ arXiv 的 affiliation 通常为空
 ↓ 空/不完整
L2  S2 API → /paper/{id}/authors → name, affiliations[], hIndex
     └─ 补全 affiliation
 ↓ 空
L3  Crossref API → 返回 authors[] with given/family name
 ↓ 空
L4  PDF 首页 PyMuPDF 提取 (正则匹配作者行)
```

**二次审核**: 当 arXiv 和 S2 作者数差异 >2 时，取较多的那个（防止 S2 漏人或合并同名）。

---

### 1.4 venue (会议/期刊)

**这是最复杂的字段**，因为论文可能同时出现在 arXiv (预印本) 和会议。

```
L1  venue_index 预爬数据 (DBLP/ACL Anthology/OpenReview 等已结构化)
     └─ confidence=0.95，直接采信
 ↓ 未命中
L2  arXiv comment 字段解析
     └─ _parse_acceptance_from_comment("Accepted at ICLR 2025")
     └─ confidence=0.7
 ↓ comment 中无 venue 信息
L3  Crossref API → container-title 字段
     └─ confidence=0.75
 ↓ 无 DOI
L4  OpenAlex → primary_location.source.display_name
     └─ confidence=0.7
 ↓ 空
L5  S2 API → venue 字段
     └─ confidence=0.6 (S2 的 venue 常为空或不标准)
```

**二次审核 (多源冲突解决)**:
- 当 ≥2 源给出不同 venue 时，按 authority_rank 排序:
  `official_conf(1) > openreview(2) > dblp(3) > crossref(4) > openalex(5) > arxiv(6) > s2(7)`
- 取 authority_rank 最高的值
- 冲突写入 `metadata_observations.conflict_group_id` 以便人工 review

---

### 1.5 acceptance_status (是否被录用 + oral/poster/spotlight)

**需要最多二次审核的字段**。

```
L1  venue_index → decision 字段
     └─ confidence=0.95
 ↓ 无
L2  PDF 首页文本 (PyMuPDF)
     └─ "Published as a conference paper at ICLR 2026"
     └─ confidence=0.9 (PDF 文字极可靠)
 ↓ 首页无 acceptance 声明
L3  arXiv comment → _parse_acceptance_from_comment()
     └─ 匹配 "Accepted at", "NeurIPS Oral" 等 5 种 regex
     └─ confidence=0.7
 ↓ comment 不含 acceptance
L4  GitHub README → 匹配 badge/text "Accepted at ..."
     └─ confidence=0.75
 ↓ 无 README 或未提及
L5  项目主页 → 抓取页面文本 → 匹配 acceptance 关键词
     └─ confidence=0.8
 ↓ 无项目主页
L6  VLM 判断 (仅在上面全部失败 + 有 OpenReview/项目页截图时)
     └─ call_llm → JSON {accepted, venue, acceptance_type, confidence}
     └─ 要求 model confidence ≥ 0.7 才采信
```

**二次审核**:
- arXiv comment 说 "Accepted at ICLR" 但未指定 oral/poster → 标记 `acceptance_type = "accepted"`
- 如果 OpenReview 有具体 score，可补全 `review_scores`
- 来自 GitHub README 的 acceptance 需要验证 README 属于**本论文官方仓库**（通过 title keyword 匹配验证）

---

### 1.6 year

```
L1  arXiv API → published date → year
 ↓ 无 arxiv_id
L2  OpenAlex → publication_year
 ↓ 空
L3  Crossref → published.date-parts[0][0]
 ↓ 空
L4  从 venue 推断 ("CVPR 2025" → 2025)
```

无需二次审核。

---

### 1.7 DOI

```
L1  arXiv API → doi 字段 (arXiv 元数据常含 DOI)
 ↓ 空
L2  Crossref title search → doi 字段
     └─ 二次审核: _titles_similar() 确认 Crossref 返回的是同一篇
 ↓ 不匹配
L3  OpenAlex → doi 字段
```

**二次审核**: Crossref title search 容易匹配到同名不同论文，必须做 `_titles_similar()` 校验。

---

### 1.8 citation_count

```
L1  S2 API → citationCount
     └─ authority_rank=1 (S2 引用数最权威)
L2  OpenAlex → cited_by_count
     └─ authority_rank=2
```

**二次审核**: 当 S2 与 OpenAlex 差异 >50% 时，日志告警。取 **S2 值**（S2 数据更全面）。

---

### 1.9 引用文献列表 (references)

```
L1  S2 API → /paper/{id}/references  (最多 100 条)
     └─ 返回: title, authors, venue, year, DOI, arxiv_id
     └─ S2 ID 构建: ARXIV:{id} → DOI:{doi} → URL:{openreview} → title search
 ↓ S2 无数据 (新论文/非英语)
L2  arXiv TeX 源码 → \cite{} bibkey 列表
     └─ 仅有 key 名，无结构化元数据
 ↓ 无 TeX
L3  PDF 文本 References section → 正则提取 (非常粗糙)
```

**二次审核**: S2 references 中 title 为空的条目丢弃（占 ~5%）。

---

### 1.10 公式 (formulas)

```
L1  arXiv TeX 源码 → extract_all_from_tex()
     └─ 精确 LaTeX，零 OCR 误差
     └─ 提取: equation/align/gather 环境 + display math + inline math
     └─ 保留: label, env_type, is_numbered
 ↓ 无 TeX 源码 (非 arXiv 论文)
L2  VLM page scan (Kimi K2.6 Vision)
     └─ PyMuPDF 扫描每页数学符号密度 → 选 top 9 页
     └─ 渲染 1.5x 图片 → 3 页/batch × 3 batch
     └─ max_tokens=16384 (防截断)
     └─ 返回: [{latex, label, page, context}]
 ↓ VLM 不可用 (无 API key)
L3  PyMuPDF regex fallback
     └─ 匹配 \frac{}, \sum_, $$...$$ 等 10 种模式
     └─ 仅提取公式行 + ±1 行上下文
     └─ 准确率 ~40%，假阳性高
```

**二次审核**: VLM 返回的 JSON 做 `_parse_vlm_json()` 健壮解析（处理截断、markdown code fence 等）。JSON parse 失败时重试 1 次。

---

### 1.11 图表 (figures + tables)

```
L1  PyMuPDF 确定性检测
     ├─ caption regex → "Figure N." / "Table N."
     ├─ 图片区域聚类 → 2.5x 高清 PNG
     └─ xref 嵌入图 fallback
 ↓ 提升
L2  VLM 分类 + 遗漏恢复 (Kimi K2.6)
     ├─ 候选截图 → VLM 1 次调用 → 分类 (figure/table/other)
     ├─ 补充: semantic_role (pipeline/result/ablation/...)
     ├─ 补充: description (中文)
     └─ 检测 PyMuPDF 漏掉的图 → 整页截图补回
 ↓
L3  表格内容 VLM OCR (仅 type="table" 的区域)
     └─ 截图 → Kimi → Markdown table → 解析 {headers, rows}
     └─ max_tokens=16384
```

**二次审核**: VLM 分类 `is_figure_or_table=false` 的候选直接丢弃（装饰图、logo 等）。

---

### 1.12 section 文字 + 层级

```
L1  PyMuPDF flat extraction → {introduction, method, experiments, ...}
     └─ 27 种 regex 模式匹配
     └─ 每 section 截断 5000 chars
L2  PyMuPDF hierarchical extraction
     └─ 匹配: ^(\d+(?:\.\d+)*)\s*\.?\s+(.+)$
     └─ 输出: [{number, title, level, text, start_char}]
     └─ level: "3" → 1, "3.1" → 2, "3.1.2" → 3
```

无需二次审核。两层同时提取，flat 做向后兼容，hierarchy 做细粒度检索。

---

### 1.13 行内引用上下文

```
L1  PDF 全文 regex → [N], [N,M], [N-M], (Author et al., YYYY)
     └─ 每个 match 取 ±120 chars context
     └─ 判定所在 section
     └─ 展开范围: "[1-3]" → ["1","2","3"]
     └─ 同一 ref_marker 最多 3 条 (优先不同 section)
 ↓ 增强
L2  与 S2 references 交叉映射
     └─ ref_marker [N] → S2 reference list 第 N 条 → 实际论文 title
```

**二次审核**: 过滤 context 长度 <30 chars 的条目（可能是误匹配，如脚注标记、列表编号）。

---

### 1.14 code_url (GitHub 开源)

```
L1  venue_index → code_url (预爬)
 ↓ 无
L2  GitHub search API → title 关键词搜索
     ├─ 提取 title keywords (去 stopwords)
     ├─ 搜索 repos: title keywords + "in:readme"
     ├─ 评分: keyword 在 repo name/description 中的匹配数
     └─ 过滤: 去除 awesome-list、generic repos
 ↓ 未命中
L3  GitHub search → arXiv ID 搜索 (如有)
 ↓ 未命中
L4  arXiv TeX 源码 → \url{github.com/...} 提取
 ↓ 无 TeX
L5  PDF 全文 → github.com URL regex
```

**二次审核**:
1. **keyword 匹配验证**: repo 的 name + description 必须包含 ≥2 个 title keywords，或 repo README 必须包含 ≥3 个 title keywords
2. **README title 验证**: 从 README 中提取论文标题，做 `_titles_similar()` 校验
3. **过滤 awesome-list**: repo name 含 "awesome" 或 star > 10k 且 description 无 title keyword → 跳过
4. **过滤社区复现**: 优先选 star 数最高且标题匹配的 repo

---

### 1.15 dataset_url (数据集链接)

```
L1  PDF 文本检测 (3 层)
     ├─ "Data Availability" / "Datasets" section 中的 URL
     ├─ 120+ 已知 ML dataset 名匹配 (experiments section)
     └─ URL 模式: huggingface.co/datasets/, kaggle.com/, zenodo.org/record/
 ↓ 无 URL
L2  GitHub README → _extract_dataset_links()
     └─ HuggingFace, Google Drive, Zenodo, 直接下载链接
 ↓ 无
L3  HuggingFace API → datasets search by title
     └─ /api/datasets?search={title}&limit=3
 ↓ 无
L4  项目主页 → 抓取页面 → 提取 dataset 链接
```

**二次审核**:
- PDF 中的 `is_released_by_paper` 检测: 匹配 "we release/publish/provide ... dataset" 动词
- HuggingFace 搜索结果需要标题匹配验证（HF 搜索很模糊，容易返回同名不同论文的 dataset）
- 区分 **论文自发布的数据集** vs **论文使用的已有数据集**

---

## 2. 限速 & 反爬应对

### 2.1 TokenBucket 限速器

所有外部 API 通过 `backend/utils/rate_limiter.py` 的 `TokenBucketLimiter` 统一管控。

| API | rate | burst | 429 退避 | 环境变量 |
|-----|------|-------|---------|---------|
| arXiv API | 2.0/s | 3 | 指数退避 base=3s | — |
| arXiv PDF | 2.0/s | 3 | 同上 (共享 limiter) | — |
| S2 (有 key) | 0.9/s | 3 | base=3s, 单次 / base=30s, batch | `S2_API_KEY` |
| S2 (无 key) | 0.3/s | 2 | 同上 | — |
| GitHub (有 token) | 1.0/s | 3 | base=3s | `GITHUB_TOKEN` |
| GitHub (无 token) | 0.15/s | 2 | 同上 | — |
| Crossref | 5.0/s | 10 | base=3s | — |
| OpenAlex | 5.0/s | 10 | base=3s | — |
| HuggingFace | 1.5/s | 3 | base=3s | — |
| Kimi VLM | 0.5/s | 2 | llm_service 内部指数退避 | — |
| DBLP | 2.0/s | 5 | base=3s | — |
| OpenReview | 1.0/s | 3 | base=3s | — |
| Unpaywall | 2.0/s | 5 | base=3s | — |

### 2.2 arXiv 特殊处理

arXiv 的限制策略:
- **API**: 3 req/s，单连接，违反会返回 503 + 30s block
- **PDF 下载**: 无明确 rate limit，但大量下载会触发 IP 临时封禁
- **OAI-PMH**: 3 秒/请求，支持 resumptionToken 增量同步

**合规方案 (不要 IP 池)**:
1. 元数据批量同步: 使用 OAI-PMH (`https://oaipmh.arxiv.org/oai`)
   - 每 3 秒 1 请求 (全局)
   - resumptionToken 断点续爬
   - 每次返回 ~1000 条
   - 配合 venue matcher 筛选顶会论文
2. 实时检索: arXiv API (单篇查询)
   - 全局 TokenBucket 2.0/s
   - 503 时暂停 30s 再重试
3. PDF 批量下载:
   - 历史全量 → Amazon S3 bucket 或 Kaggle 数据集 (不走 API)
   - 增量新论文 → `export.arxiv.org/pdf/{id}` 逐篇下载
   - 大陆走代理 (直连 10KB/s，代理 0.8s/PDF)

**服务器出口 (阿里云)**:
- NAT Gateway + 固定 EIP (不是 IP 池)
- Worker 共享一个 EIP 出口
- 全局 Redis TokenBucket 控制跨 worker 限速

### 2.3 S2 限速策略

| 场景 | 限速 | 退避 |
|------|------|------|
| 单篇查询 | limiter.acquire() + 0.9/s | 429 → sleep 3s + retry |
| batch 查询 (500/批) | 批间 limiter.acquire() | 429 → sleep 30s |
| references/citations | 两次调用间 limiter.acquire() | 429 → sleep 3s + retry |

**推荐**: 配置 `S2_API_KEY` 可提升到 1 req/s 稳定速率。

### 2.4 GitHub 限速策略

| 场景 | 无 token | 有 token |
|------|---------|---------|
| search API | 10 req/min | 30 req/min |
| REST API | 60 req/hr | 5000 req/hr |
| raw.githubusercontent.com | 无限制 | 无限制 |

**推荐**: 配置 `GITHUB_TOKEN` (Personal Access Token, 只需 `public_repo` scope)。

---

## 3. 二次审核机制汇总

| 字段 | 审核方式 | 实现位置 |
|------|---------|---------|
| title | `_titles_similar()` 归一化比较 | enrich_service.py L857 |
| venue | 多源 authority_rank 冲突解决 | metadata_observations 表 |
| acceptance | 5 层 cascade + VLM 末端兜底 | enrich_service.py L1122-1155 |
| DOI | Crossref 返回标题 vs paper.title 相似度校验 | enrich_service.py L933 |
| code_url | GitHub repo README title 匹配 + keyword 评分 + awesome-list 过滤 | enrich_service.py L298-380 |
| dataset_url | is_released_by_paper 动词检测 + HF title 匹配 | pdf_extract.py + enrich_service.py |
| citation_count | S2 vs OpenAlex 差异 >50% 告警 | enrich_service.py L999 |
| references | S2 title 为空的条目丢弃 | parse_service.py L128 |
| formulas (VLM) | JSON parse 健壮化 + 截断重试 | formula_extraction_service.py |
| figures (VLM) | is_figure_or_table 判断过滤 | figure_extraction_service.py |
| citation_contexts | context 长度 <30 chars 过滤 | pdf_extract.py |

---

## 4. 非 arXiv 论文退化表

当论文没有 arxiv_id 时，各能力的退化情况:

| 能力 | 有 arxiv_id | 无 arxiv_id，有 DOI | 无 arxiv_id，无 DOI |
|------|------------|-------------------|-------------------|
| 基础元数据 | arXiv API 直查 | Crossref + OpenAlex | S2 title search |
| TeX 公式 | 精确 LaTeX | ❌ | ❌ |
| S2 引用列表 | `ARXIV:{id}` | `DOI:{doi}` | title search → paperId |
| PDF 下载 | arxiv.org/pdf/{id} | Unpaywall OA PDF | 手动上传 |
| VLM 公式/图表 | 正常 (需 PDF) | 正常 (需 PDF) | 正常 (需 PDF) |
| PyMuPDF 提取 | 正常 | 正常 | 正常 |
| GitHub 搜索 | title + arxiv_id | title only | title only |

**关键**: 非 arXiv 论文的公式准确率从 ~90% (TeX) 降到 ~80% (VLM) 或 ~40% (regex)。

---

## 5. 环境变量参考

```bash
# 必须
OPENAI_API_KEY=sk-kimi-xxx              # Kimi K2.6 (VLM 公式/图表/表格)
OPENAI_BASE_URL=https://api.kimi.com/coding/v1
OPENAI_MODEL=kimi-k2.6

# 推荐 (显著提升限速容量)
S2_API_KEY=your_key                     # S2: 0.3/s → 1/s
GITHUB_TOKEN=ghp_xxxxxx                 # GitHub: 10 req/min → 5000 req/hr

# VLM max_tokens (Kimi 截断防护)
VLM_MAX_TOKENS_HEAVY=16384              # 公式页扫描、大表格
VLM_MAX_TOKENS_MEDIUM=8192              # 图表分类
VLM_MAX_TOKENS_LIGHT=4096               # 单图/单公式
VLM_MAX_TOKENS_TINY=2048                # acceptance 判断

# 代理 (大陆服务器)
HTTP_PROXY=http://172.17.0.1:7890
HTTPS_PROXY=http://172.17.0.1:7890
NO_PROXY=localhost,postgres,redis,api.semanticscholar.org,api.openalex.org,dblp.org
```

---

## 6. 批量处理成本估算

| 规模 | 总 API 调用 | VLM 成本 | 耗时 (串行) | 安全日量 |
|------|-----------|---------|------------|---------|
| 10 篇 | ~200 | ~¥3 | ~50 min | ✅ |
| 100 篇 | ~2000 | ~¥25 | ~8 hr | ✅ |
| 500 篇 | ~10000 | ~¥130 | ~40 hr | ⚠️ 分 5 天 |
| 1000 篇 | ~20000 | ~¥260 | ~80 hr | ⚠️ 分 10 天 |

> 瓶颈不在 VLM 成本，在 S2/arXiv API 限速。100 篇/天是安全阈值。

---

## 7. OAI-PMH 批量元数据同步 (顶会知识库)

对于 CVPR/ICLR/KDD/ACL 等全顶会知识库建设:

```
1. OAI-PMH 拉取 cs.CV / cs.CL / cs.LG 全量元数据
   URL: https://oaipmh.arxiv.org/oai?verb=ListRecords&set=cs&metadataPrefix=arXiv
   限速: 3 秒/请求
   每次: ~1000 条
   断点: resumptionToken

2. venue matcher 筛选顶会论文
   ├─ arXiv comment → _parse_acceptance_from_comment()
   ├─ Crossref DOI → venue
   ├─ DBLP title search → venue
   └─ OpenReview API → venue + decision

3. 入库: 只导入命中顶会的论文
   └─ 标记 state=WAIT, 等待 pipeline 逐篇处理

4. pipeline 逐篇处理 (100 篇/天节奏)
   └─ download → enrich → L2 parse → L3 skim → L4 deep
```

**不要**: 逐篇调 arXiv API 查询论文是否属于顶会。
**应该**: OAI-PMH 批量拉 → 本地筛选 → 命中才入库。
