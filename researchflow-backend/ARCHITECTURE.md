# ResearchFlow 系统架构设计文档

> **产品定位**：ResearchFlow = Web 产品 + 自有后端编排 + MCP 兼容层 + 专家模式（Claude/Codex）
>
> 普通用户只用网页；高级用户再通过 Claude Code / Codex 走专家入口。系统本体是自有后端，不是任何 AI agent 会话。

---

## 1. 系统全景架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用 户 层                                     │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │  Web 前端    │   │ Claude Code  │   │  Codex CLI / IDE       │  │
│  │  (Next.js)   │   │  (专家模式)  │   │  (专家模式)            │  │
│  │              │   │              │   │                        │  │
│  │  普通用户    │   │  高级用户    │   │  维护者/工程师          │  │
│  └──────┬───────┘   └──────┬───────┘   └───────────┬────────────┘  │
│         │                  │                       │               │
│         │ HTTP/REST        │ MCP (JSON-RPC)        │ MCP           │
└─────────┼──────────────────┼───────────────────────┼───────────────┘
          │                  │                       │
          ▼                  ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     接 入 层 (Caddy)                                │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  /           → frontend:3000    (Web 前端)                   │   │
│  │  /api/*      → api:8000         (REST API)                   │   │
│  │  /mcp/*      → api:8000         (MCP Server, 同进程)         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     后 端 六 层 架 构                                │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ ① Presentation 层                                            │  │
│  │    渲染论文卡片 / 汇报报告(30s·5min·deep) / 深度对比 /       │  │
│  │    Repo 解析报告 / 日周月总结 / 方向推荐卡片                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ ② Feedback / Eval 层                                         │  │
│  │    用户纠错 → 候选规则 → 离线评测 → 小流量验证 → 管理员上线   │  │
│  │    用户收藏 / 批注 / 标签修改 / 行为事件采集                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ ③ Workflow / Job 层 (arq + Redis)                            │  │
│  │                                                               │  │
│  │    collect → enrich → triage → parse(L2) → skim(L3)          │  │
│  │         → deep_report(L4) → taxonomy_review                   │  │
│  │         → repo_alignment → report_generate                    │  │
│  │         → direction_propose → feasibility_expand              │  │
│  │         → digest_generate → asset_refresh → reanalyze         │  │
│  │                                                               │  │
│  │    定时任务: digest(23:00/周日/月末) / asset_refresh / embed   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ ④ Retrieval 层 (PostgreSQL + pgvector)                       │  │
│  │    结构化过滤(SQL) + 全文检索(tsvector) + 语义搜索(HNSW)     │  │
│  │    混合排序: BM25 + cosine + 结构化权重                       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ ⑤ Parse / Extract 层                                         │  │
│  │    本地 PDF 解析 (pymupdf) → 章节/公式/表格/图注              │  │
│  │    LLM 分析 → canonical delta card + evidence spans           │  │
│  │             + method slots + limitations + transfer atoms      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ ⑥ Ingestion 层                                               │  │
│  │    输入: 链接/PDF/HTML/Repo/Awesome list/Zotero               │  │
│  │    → 规范化 → 去重 → canonical identity 对齐 → 资产补全 → 入库│  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          │                    │
          ▼                    ▼
┌──────────────────┐  ┌─────────────────────┐
│ PostgreSQL 16    │  │ 对象存储 (COS/OSS)  │
│ + pgvector       │  │                     │
│                  │  │ papers/raw-pdf/     │
│ 19 张表          │  │ papers/raw-html/    │
│ (事实源)         │  │ papers/figures/     │
│                  │  │ papers/extracted/   │
│                  │  │ reports/skim/       │
│                  │  │ reports/deep/       │
│                  │  │ exports/markdown/   │
└──────────────────┘  └─────────────────────┘
          │
          ▼ (导出兼容层)
┌─────────────────────────────────────────────┐
│  paperAnalysis/  paperCollection/  paperIDEAs/ │
│  (Markdown 导出物, 非事实源)                    │
│  → 保持 .claude/skills 和 Obsidian 可用        │
└─────────────────────────────────────────────┘
```

---

## 2. 数据流架构

### 2.1 论文入库流

```
用户输入 (链接/PDF/Repo/Awesome list/Zotero)
    │
    ▼
┌────────────┐     ┌──────────────┐     ┌──────────────┐
│  Ingestion │────▶│   Enrich     │────▶│   Triage     │
│            │     │              │     │              │
│ 规范化     │     │ Crossref API │     │ keep_score   │
│ 去重       │     │ arXiv API    │     │ analysis_pri │
│ identity   │     │ Sem.Scholar  │     │ structural   │
│ 对齐       │     │ 资产补全     │     │ extensionab. │
└────────────┘     └──────────────┘     └──────┬───────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
    ┌───────────────────────────────────────────────────┐
    │           四级分析管线                              │
    │                                                   │
    │  L1 metadata ──▶ L2 parse ──▶ L3 skim ──▶ L4 deep│
    │  (API补全,0T)   (本地PDF,0T)  (LLM,~2K)   (LLM,  │
    │                                            ~10-20K)│
    │                                                   │
    │  晋升决策由 triage_service 的多维评分控制            │
    │  不自动全量推进，"是否值得花 token"是明确决策点      │
    └───────────────────────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │  结构化产物 (写入 DB)             │
    │                                   │
    │  • papers 主记录 (元数据+评分)    │
    │  • paper_analyses (skim/deep)     │
    │  • method_deltas (delta card)     │
    │  • evidence_units (证据原子)      │
    │  • transfer_atoms (迁移原子)      │
    └───────────────────────────────────┘
```

### 2.2 报告生成流

```
用户请求: "帮我分析这 10 篇论文"
    │
    ▼
┌────────────────────┐
│  report_generate   │
│  worker            │
│                    │
│  1. 从 DB 读取     │
│     papers +       │
│     analyses +     │
│     method_deltas  │
│                    │
│  2. 对照 paradigm  │
│     template       │
│     生成 delta     │
│     对比表         │
│                    │
│  3. 按报告类型渲染 │
│     ├─ 30s 版      │
│     ├─ 5min 汇报版 │
│     └─ deep 对比版 │
│                    │
│  4. 缓存到         │
│     report_cache   │
└────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  汇报报告固定结构                            │
│                                             │
│  1. 问题定义: 这组论文在讲什么               │
│  2. Canonical baseline                      │
│  3. 各论文改了哪些槽位 (delta card 表)       │
│  4. 插件 patch vs 结构性改进 分类            │
│  5. 开源情况 / 复现价值 / 证据强度           │
│  6. 建议阅读顺序                            │
│  7. 最值得追的 1-3 条线                      │
└─────────────────────────────────────────────┘
```

### 2.3 搜索会话流（检索分支树）

```
用户: "优势会消失的问题"
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  search_session                                     │
│                                                     │
│  symptom_query: "优势会消失"                         │
│       │                                             │
│       ▼                                             │
│  第一轮检索 → 大量 RL advantage 论文                 │
│  系统发现: 多数只是 plugin 式 patch (改 weighting)   │
│       │                                             │
│       ▼                                             │
│  latent_need 重写:                                  │
│  "需要更可扩展的 credit/reward 管理，                │
│   或更 agentic 的扩展式方案"                         │
│       │                                             │
│       ▼                                             │
│  开新分支:                                          │
│  ├── branch: grouped reward                         │
│  ├── branch: planner-style                          │
│  └── branch: Think-with-Image (agentic)             │
│                                                     │
│  rejected_patterns: ["advantage weighting patch"]    │
└─────────────────────────────────────────────────────┘
```

---

## 3. 数据库 ER 关系图

```
                    ┌──────────────────┐
                    │  paradigm_       │
                    │  templates       │
                    │                  │
                    │  name (unique)   │
                    │  domain          │
                    │  slots (JSONB)   │
                    └────────┬─────────┘
                             │ referenced by
                             ▼
┌──────────┐  1:N  ┌──────────────┐  N:1  ┌──────────────────┐
│ papers   │◄──────│ method_      │──────▶│ paper_analyses   │
│          │       │ deltas       │       │                  │
│ id (PK)  │       │              │       │ level (L1-L4)    │
│ title    │       │ paradigm_name│       │ model_provider   │
│ venue    │       │ slots (JSONB)│       │ prompt_version   │
│ year     │       │ is_structural│       │ confidence       │
│ category │       └──────────────┘       │ problem_summary  │
│ state    │                              │ method_summary   │
│ tags[]   │  1:N  ┌──────────────┐       │ evidence_summary │
│ scores   │◄──────│ evidence_    │       │ full_report_md   │
│ embedding│       │ units        │       │ is_current       │
│          │       │              │       └──────────────────┘
│          │       │ atom_type    │
│          │       │ claim        │  1:N  ┌──────────────────┐
│          │       │ causal_str.  │◄──────│ transfer_atoms   │
│          │       │ embedding    │       │ source_domain    │
│          │       └──────────────┘       │ target_domain    │
│          │                              │ mechanism        │
│          │  1:N  ┌──────────────┐       └──────────────────┘
│          │◄──────│ paper_assets │
│          │       │ asset_type   │       ┌──────────────────┐
│          │       │ object_key   │       │ project_         │
│          │       │ checksum     │       │ bottlenecks      │
│          │       └──────────────┘       │                  │
│          │                              │ symptom_query    │
│          │  1:N  ┌──────────────┐       │ latent_need      │
│          │◄──────│ paper_       │       │ constraints      │
│          │       │ versions     │       │ embedding        │
│          │       └──────────────┘       └────────┬─────────┘
└──────────┘                                       │ 1:N
                                                   ▼
┌──────────────┐       ┌──────────────┐   ┌──────────────────┐
│ search_      │       │ reading_     │   │ direction_cards  │
│ sessions     │       │ plans        │   │                  │
│              │       │              │   │ title            │
│ symptom_query│       │ bottleneck_id│   │ rationale        │
│ latent_need  │       │ canonical[]  │   │ is_structural    │
│ branches     │       │ structural[] │   │ required_assets  │
│ rewrite_hist │       │ followup[]   │   │ feasibility_plan │
└──────────────┘       │ patches[]    │   └──────────────────┘
                       └──────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ digests      │  │ jobs         │  │ model_runs       │
│              │  │              │  │                  │
│ period_type  │  │ job_type     │  │ model_provider   │
│ period_start │  │ status       │  │ input_tokens     │
│ rendered_text│  │ payload      │  │ output_tokens    │
│ metadata     │  │ priority     │  │ cost_usd         │
└──────────────┘  └──────────────┘  └──────────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ user_feedback│  │ user_        │  │ user_events      │
│              │  │ bookmarks    │  │                  │
│ target_type  │  │ target_type  │  │ event_type       │
│ target_id    │  │ target_id    │  │ target_type      │
│ feedback_type│  │ note         │  │ payload (JSONB)  │
└──────────────┘  └──────────────┘  └──────────────────┘

┌──────────────┐  ┌──────────────┐
│ execution_   │  │ repo_        │
│ memories     │  │ analyses     │
│              │  │              │
│ env_fingerpr.│  │ repo_url     │
│ failed_cmd   │  │ flow_diagram │
│ fix_action   │  │ formula_map  │
│ verified     │  │ shape_trace  │
└──────────────┘  └──────────────┘

                  ┌──────────────┐
                  │ report_cache │
                  │              │
                  │ report_type  │
                  │ rendered_md  │
                  │ expires_at   │
                  └──────────────┘
```

**共 19 张表**：
- 核心数据：papers, paper_assets, paper_versions, paper_analyses, method_deltas, paradigm_templates, evidence_units, transfer_atoms
- 研究工作流：project_bottlenecks, search_sessions, reading_plans, direction_cards, digests
- Repo 分析：repo_analyses, report_cache
- 系统：jobs, model_runs, execution_memories
- 用户：user_feedback, user_bookmarks, user_events

---

## 4. 四级分析管线

| 级别 | 输入 | 处理方式 | Token | 输出 |
|------|------|----------|-------|------|
| **L1** metadata | 标题+摘要+venue | Crossref/arXiv API 补全 | 0 | 完整元数据 |
| **L2** parse | PDF 文件 | pymupdf 本地解析 | 0 | extracted_sections (章节/公式/表格/图注) |
| **L3** skim | title+abstract+intro+method+conclusion | LLM 轻量卡片 | ~2K | skim card + changed_slots + is_plugin_patch + worth_deep_read |
| **L4** deep | 全文 | LLM 深度分析 | ~10-20K | 完整 report + evidence_units + method_delta + transfer_atoms |

**晋升规则**：L2→L3 和 L3→L4 由 triage_service 的多维评分决定。"是否值得花 token"是明确决策点，不自动全量推进。

---

## 5. Canonical Delta Card 设计

每篇论文对照领域标准范式，明确写清改了什么：

```json
{
  "paradigm": "motion_generation_diffusion_v1",
  "paper": "DART (ICLR 2025)",
  "slots": {
    "motion_tokenizer": {"changed": false},
    "denoiser": {
      "changed": true,
      "from": "standard U-Net",
      "to": "DiT with AR motion tokens",
      "change_type": "structural"
    },
    "conditioning": {
      "changed": true,
      "from": "text CLIP embedding",
      "to": "text + spatial joint control",
      "change_type": "structural"
    },
    "objective": {"changed": false},
    "sampling": {
      "changed": true,
      "from": "DDPM 1000-step",
      "to": "flow matching 10-step",
      "change_type": "plugin"
    }
  },
  "is_structural": true,
  "primary_gain_source": "denoiser + conditioning"
}
```

这是防止"被论文叙事欺骗"的核心机制。

---

## 6. MCP 适配架构

```
┌─────────────────────────────────────────────────────────────┐
│                   ResearchFlow MCP Server                    │
│                   (10 个高层工具)                             │
│                                                             │
│  import_research_sources    search_research_kb               │
│  get_paper_report           get_repo_paper_report            │
│  compare_papers             refresh_assets                   │
│  propose_directions         expand_feasibility_plan          │
│  get_digest                 record_user_feedback             │
│                                                             │
│  不暴露: 原始 SQL / 对象存储读写 / 底层 CRUD                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ Claude Code  │ │  Codex   │ │  未来 Agent  │
│              │ │  CLI     │ │  框架        │
│ .mcp.json    │ │ config   │ │              │
│ skills/      │ │ .toml    │ │ MCP 标准     │
│ subagents/   │ │ AGENTS.md│ │ 协议接入     │
│ hooks        │ │ skills/  │ │              │
└──────────────┘ └──────────┘ └──────────────┘
```

**原则**：MCP 是对外标准边界。不被某一家锁死，维护中立 skill 源 → 生成 Claude/Codex 包装层。

---

## 7. 部署架构（2C4G / 2000 元年预算）

```
┌─────────────────────────────────────────────┐
│  腾讯云 Lighthouse 2C4G / 100GB SSD         │
│  Ubuntu 24.04 + Docker Compose              │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │ Caddy (反向代理 + HTTPS)            │    │
│  └────────┬──────────┬─────────────────┘    │
│           │          │                      │
│  ┌────────▼──┐  ┌────▼──────┐               │
│  │ frontend  │  │ api       │               │
│  │ (Next.js) │  │ (FastAPI) │               │
│  │ 300 MB    │  │ 400 MB    │               │
│  └───────────┘  │ + MCP     │               │
│                 └────┬──────┘               │
│                      │                      │
│  ┌───────────────────▼──────────────────┐   │
│  │ worker (arq, 2 并发)     1024 MB     │   │
│  │ parse(1) / deep_report(2) / digest(1)│   │
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌──────────────┐  ┌──────────────┐         │
│  │ PostgreSQL   │  │ Redis        │         │
│  │ + pgvector   │  │ 缓存+队列   │         │
│  │ 1280 MB      │  │ 256 MB       │         │
│  └──────────────┘  └──────────────┘         │
│                                             │
│  总内存: ~3.3 GB，留 ~700 MB 给 OS+buffer   │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  腾讯云 COS (对象存储)                       │
│  200-500 GB，~283-708 元/年                  │
│                                             │
│  papers/raw-pdf/                            │
│  papers/raw-html/                           │
│  papers/extracted-text/                     │
│  papers/figures/                            │
│  papers/code-snapshots/                     │
│  reports/skim/                              │
│  reports/deep/                              │
│  exports/markdown/                          │
└─────────────────────────────────────────────┘

预算:
  服务器: ~1020 元/年
  对象存储 200GB: ~283 元/年
  域名+备份: ~180 元/年
  合计: ~1483 元/年 (500GB COS ~1908 元/年)
```

---

## 8. 用户行为监控架构

```
用户操作
    │
    ▼
┌─────────────────────────────────────────────┐
│  user_events (结构化事件, 最小化采集)        │
│                                             │
│  import / view_paper / export_report /      │
│  click_direction / override_label /         │
│  add_tag / rewrite_bottleneck / select_dir  │
└────────────────────┬────────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
┌────────┐    ┌───────────┐    ┌──────────────┐
│ 排序   │    │ 标签/术语 │    │ 重分析触发   │
│ 更新   │    │ 同义词    │    │              │
│        │    │ 映射学习  │    │ 同类判断频繁 │
│        │    │           │    │ 被纠正 → 重跑│
└────────┘    └───────────┘    └──────┬───────┘
                                      │
                               ┌──────▼───────┐
                               │ 候选规则     │
                               │    ↓         │
                               │ 离线评测集   │
                               │    ↓         │
                               │ 小流量验证   │
                               │    ↓         │
                               │ 管理员确认   │
                               │    ↓         │
                               │ 上线生产     │
                               └──────────────┘

红线: 用户行为不直接修改生产 prompt
```

---

## 9. 提取质量保障六层机制

高质量提取靠的是"程序化证据管线 + 多角色复核 + 可回放评测"，不是单纯靠 Claude Code，也不是单纯靠提示词工程。

```
┌─────────────────────────────────────────────────────────────┐
│  第 6 层：评测集回放 + 人工反馈闭环                          │
│  50 篇标杆 + 10 个查询 + 10 个重构场景                      │
│  每次升级 prompt/taxonomy/模型都先回放                        │
├─────────────────────────────────────────────────────────────┤
│  第 5 层：证据锚点 + 置信度分级                              │
│  每条结论标注: confidence (0-1) + basis                      │
│  basis = code_verified | experiment_backed | text_stated     │
│         | inferred | speculative                             │
│  用户能分辨"事实"和"系统推断"                                │
├─────────────────────────────────────────────────────────────┤
│  第 4 层：交叉验证 — 三角色复核                              │
│  extractor (抽信息) → auditor (检查证据) →                   │
│  taxonomy_reviewer (检查分类)                                │
│  同一高价值论文不由单一 agent 决定                            │
├─────────────────────────────────────────────────────────────┤
│  第 3 层：标准范式对齐 (Canonical Delta Card)                │
│  不问"这篇讲了什么"，而问"它改了哪个槽位"                   │
│  强制与领域 paradigm_template 比对                           │
│  区分 structural 改动 vs plugin patch                        │
├─────────────────────────────────────────────────────────────┤
│  第 2 层：先规则解析，再 LLM 提炼                            │
│  程序抽: 标题/摘要/章节/公式/表格/图注/repo 关键文件          │
│  LLM 只做: 槽位归因 + 机制抽象 + 证据解释 + 迁移判断         │
├─────────────────────────────────────────────────────────────┤
│  第 1 层：四级分析管线 — 分步晋升                            │
│  L1 (metadata, 0T) → L2 (parse, 0T) → L3 (skim, ~2K)       │
│  → L4 (deep, ~10-20K)                                       │
│  "是否值得花 token" 是明确决策点                              │
└─────────────────────────────────────────────────────────────┘
```

### 置信度分级 (EvidenceBasis)

每条 evidence_unit 和 confidence_note 都标注 basis：

| basis | 含义 | 示例 |
|-------|------|------|
| `code_verified` | 已在源代码中确认 | "reward 使用了 group averaging，见 `train.py:L142`" |
| `experiment_backed` | 有 ablation/表/图支持 | "去掉 spatial control 后 FID 下降 12%，见 Table 3" |
| `text_stated` | 作者在正文中明确写 | "We adopt flow matching instead of DDPM" |
| `inferred` | 系统根据上下文逻辑推断 | "虽然未提，但 DiT 架构暗示支持变长序列" |
| `speculative` | 系统的猜测，无直接证据 | "设计可能借鉴了 video diffusion 的时序建模" |

---

## 10. 库外输入状态机

普通网页用户可以随时丢入"不在库中的内容"。系统不要求"先在库里"，而是支持"先输入，再入库"。

### 状态流转

```
用户输入 (PDF / 链接 / Repo / 粘贴文本)
    │
    ▼
ephemeral_received ─────────────────────────────────────────┐
    │                                                       │
    │ 识别身份 + 去重                                        │ 30 天未操作
    ▼                                                       │
canonicalized ──────────────────────────────────────────────│
    │                                                       │
    │ 自动补全资产 (Crossref/arXiv/repo/project page)       │
    ▼                                                       │
enriched ───────────────────────────────────────────────────│
    │                                                       │
    │ 自动生成轻量卡片 (L3 skim)                            │
    ▼                                                       ▼
l3_skimmed                                        archived_or_expired
    │                                              (临时对象过期清理)
    │ 用户点"加入知识库" 或 系统判定高价值
    ▼
accepted_to_kb (= wait / l1_metadata)
    │
    ▼ (进入正式管线: L1→L2→L3→L4→checked)
```

### 三类库外输入的处理方式

| 类型 | 场景 | 处理 | 保留期 |
|------|------|------|--------|
| **一次性分析** | 用户随手丢一个 PDF | 轻量提取 + 即时报告 | 默认 30 天，可延长 |
| **候选入库** | 用户给 awesome list / repo | 自动补全 + 初筛 + 进主库或观察池 | 入库后永久 |
| **强制入库** | 管理员指定必须跟踪 | 直接创建正式对象，跳过观察期 | 永久 |

### 过期清理

- `is_ephemeral = true` 且 `expires_at < now()` 的对象由定时任务自动归档
- 归档后 state → `archived_or_expired`，关联分析保留 90 天后彻底删除
- 对象存储中的 PDF/资产同步清理

---

## 11. 技术选型汇总

| 组件 | 选型 | 理由 |
|------|------|------|
| Web 框架 | FastAPI | async 原生，自动 OpenAPI |
| 前端 | Next.js + Tailwind + shadcn/ui | SSR + 现代组件库 |
| ORM | SQLAlchemy 2.0 (async) | 成熟，支持 pgvector |
| 数据库 | PostgreSQL 16 + pgvector | 结构化+全文+向量 all-in-one |
| 任务队列 | arq (Redis) | asyncio 原生，比 Celery 轻量 |
| 缓存 | Redis | 队列复用 |
| PDF 解析 | pymupdf | 快速、本地、无 OCR 依赖 |
| 对象存储 | 腾讯云 COS / 阿里云 OSS | 便宜，SDK 成熟 |
| LLM 调用 | Anthropic Agent SDK + OpenAI API | 灵活切换 |
| MCP | FastMCP (Python) | 官方推荐 |
| 反向代理 | Caddy | 自动 HTTPS，配置简洁 |
| 部署 | Docker Compose | 2C4G 单机，简单可靠 |
| DB 迁移 | Alembic | SQLAlchemy 标配 |

---

## 12. 安全与权限

- **认证**：Phase 9 加入 API key / JWT（初期单团队）
- **MCP 安全**：只暴露高层工具，不暴露底层数据操作
- **人类在环**：MCP 协议本身强调高风险工具调用需人类确认
- **行为数据**：最小化采集，不录用户原始 prompt
- **重分析**：生产 prompt 变更必须经管理员审批
