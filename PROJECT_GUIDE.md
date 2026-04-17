# ResearchFlow 项目框架阅读指南

> 快速理解项目结构和代码组织，方便新成员上手。

---

## 一句话概括

ResearchFlow 是一个**本地科研知识库 + AI 辅助研究工作流**。核心理念是把论文、分析笔记、研究 idea 沉淀为结构化的 Markdown 文件，让 Claude Code / Codex CLI / 其他 AI agent 共享同一份证据层。

**不是**一个需要数据库/Docker/后端服务的平台——整个系统就是**本地文件夹 + Markdown + CSV**。

---

## 核心数据流

```
收集论文候选 → 下载 PDF → 结构化分析 → 建立索引 → 知识检索/Idea 生成
 (collect)    (download)   (analyze)     (build)     (query/ideate)
```

对应状态流转（`analysis_log.csv` 中的 `state` 字段）：

```
Wait → Downloaded → checked
 │        │
 │        ├→ too_large          (PDF 过大)
 │        └→ analysis_mismatch  (分析模板不完整)
 ├→ Skip     (手动排除)
 └→ Missing  (下载失败)
```

---

## 三层数据架构

```
┌─────────────────────────────────────────────┐
│  产出层  paperIDEAs/                        │
│          idea、研究方案、review 记录          │
├─────────────────────────────────────────────┤
│  索引层  paperCollection/                   │
│          index.jsonl (agent 检索用)          │
│          by_task/ by_technique/ by_venue/    │
│          (Obsidian 导航页)                   │
├─────────────────────────────────────────────┤
│  证据层  paperAnalysis/ + paperPDFs/        │
│          结构化分析笔记 + 原始 PDF            │
│          所有 skill 的主数据源               │
└─────────────────────────────────────────────┘
```

---

## 目录结构速查

```
ResearchFlow/
├── .claude/
│   ├── skills/                  # ⭐ skill 定义（唯一维护源，核心逻辑所在）
│   │   ├── README.md            # skill 总览
│   │   ├── User_README.md       # 用户快速路由（英文）
│   │   ├── User_README_CN.md    # 用户快速路由（中文）
│   │   ├── STATE_CONVENTION.md  # 论文状态流转定义
│   │   ├── research-workflow/   # 统一入口：识别当前阶段，推荐下一步
│   │   ├── papers-collect-from-web/       # 从网页收集候选论文
│   │   ├── papers-collect-from-github-awesome/  # 从 GitHub awesome 仓库收集
│   │   ├── papers-sync-from-zotero/       # 从 Zotero 同步
│   │   ├── papers-download-from-list/     # 批量下载 PDF
│   │   ├── papers-analyze-pdf/            # PDF → 结构化 Markdown 笔记
│   │   ├── papers-build-collection-index/ # 构建 agent 索引 + Obsidian 导航页
│   │   ├── papers-audit-metadata-consistency/  # 元数据一致性审计
│   │   ├── papers-query-knowledge-base/   # 知识库检索 + 论文对比
│   │   ├── code-context-paper-retrieval/  # 改代码前检索论文依据
│   │   ├── research-brainstorm-from-kb/   # 基于知识库生成 idea
│   │   ├── idea-focus-coach/              # 把宽泛 idea 收敛为可执行方案
│   │   ├── reviewer-stress-test/          # 审稿人视角压测
│   │   ├── notes-export-share-version/    # 笔记导出为可分享版
│   │   ├── write-daily-log/               # 生成/更新研究日志
│   │   ├── rf-obsidian-markdown/          # Obsidian Markdown 规范
│   │   ├── skill-fit-guard/               # skill 误匹配诊断
│   │   └── domain-fork/                   # 领域迁移工具
│   └── skills-config.json       # skill 注册描述与路由元数据
│
├── paperAnalysis/               # ⭐ 核心数据：结构化论文分析笔记
│   ├── analysis_log.csv         #    论文候选列表与状态追踪
│   └── <Category>/<Venue_Year>/ #    按分类/会议年份组织的 .md 笔记
│
├── paperPDFs/                   # 原始 PDF 文件（按相同目录结构）
│
├── paperCollection/             # 生成的索引与导航层
│   ├── index.jsonl              #    agent 索引（build 后生成）
│   ├── by_task/                 #    按任务分类的 Obsidian 导航页
│   ├── by_technique/            #    按技术分类
│   └── by_venue/                #    按会议分类
│
├── paperIDEAs/                  # 研究产出：idea、方案、review 记录
│
├── linkedCodebases/             # 外部代码仓库的符号链接
│   └── README.md                #    使用 scripts/link_codebase.py 创建
│
├── scripts/                     # 辅助脚本
│   ├── auto_download_papers.py  #    自动下载
│   ├── playwright_download.py   #    Playwright 下载
│   ├── setup_shared_skills.py   #    生成 Codex 兼容入口
│   ├── link_codebase.py         #    链接外部代码库
│   ├── paper_analysis_maintenance/  # 分析笔记维护脚本
│   └── maintenance/             #    其他维护脚本
│
├── AGENTS.md                    # Agent 指南（Codex 入口）
├── README.md / README_CN.md     # 项目说明
├── assets/                      # Logo 等静态资源
└── .gitignore
```

---

## 关键文件说明

### 每篇论文的分析笔记结构（`paperAnalysis/*.md`）

每个 `.md` 文件包含 YAML frontmatter + 正文三部分：

```yaml
---
title: "论文标题"
venue: "CVPR 2025"
year: 2025
tags: [tag1, tag2]
category: "Human_Object_Interaction"
core_operator: "核心算子描述"
primary_logic: "主要逻辑描述"
pdf_ref: "paperPDFs/Category/Venue_Year/filename.pdf"
---
```

正文结构（中文模式）：
- **Part I：问题与挑战** — 这篇论文要解决什么问题
- **Part II：方法与洞察** — 怎么解决的
- **Part III：证据与局限** — 实验结果与不足
- **核心直觉** — 一句话抓住本质

### `analysis_log.csv`

论文管线的核心追踪文件，记录每篇论文从收集到分析完成的状态。关键列：
- `title` — 论文标题
- `state` — 当前状态（Wait/Downloaded/checked/Skip/Missing/too_large/analysis_mismatch）
- `venue` / `year` — 来源会议与年份
- `link` — 论文链接

### `skills-config.json`

注册所有 skill 的描述与文件路径，agent 通过它判断该调用哪个 skill。

---

## Skill 体系速查

### 按使用场景分类

| 场景 | 该用哪个 skill |
|------|---------------|
| 不确定当前该做什么 | `research-workflow` |
| 从网页收集论文 | `papers-collect-from-web` |
| 从 GitHub awesome 仓库收集 | `papers-collect-from-github-awesome` |
| 从 Zotero 导入 | `papers-sync-from-zotero` |
| 下载 PDF | `papers-download-from-list` |
| 分析 PDF 生成笔记 | `papers-analyze-pdf` |
| 重建索引和导航页 | `papers-build-collection-index` |
| 检查元数据质量 | `papers-audit-metadata-consistency` |
| 检索/对比论文 | `papers-query-knowledge-base` |
| 改代码前找论文依据 | `code-context-paper-retrieval` |
| 生成研究 idea | `research-brainstorm-from-kb` |
| 收敛 idea 为可执行方案 | `idea-focus-coach` |
| 审稿人视角压测 | `reviewer-stress-test` |
| 导出可分享笔记 | `notes-export-share-version` |
| 写研究日志 | `write-daily-log` |
| 迁移到其他领域 | `domain-fork` |

### Skill 文件结构

每个 skill 是一个文件夹，核心是 `SKILL.md`：
```
.claude/skills/<skill-name>/
├── SKILL.md        # skill 定义与工作流指令
├── README.md       # 可选：额外说明
└── references/     # 可选：参考文档
```

---

## 多 Agent 协作模式

```
ResearchFlow/（共享知识层）
├── paperCollection/index.jsonl  ← agent 快速筛选入口
├── paperAnalysis/               ← 所有 agent 的证据源
├── paperPDFs/                   ← 原始文献
└── paperIDEAs/                  ← 各 agent 写入的产出

Agent A (Claude Code)  ── 读 index.jsonl → paperAnalysis/ → 生成 idea
Agent B (Codex CLI)    ── 读 index.jsonl → paperAnalysis/ → 辅助代码修改
Agent C (自定义)        ── 读 index.jsonl → paperAnalysis/ → 自动化实验
```

- Claude Code / Cursor：直接使用 `.claude/skills`
- Codex CLI：运行 `python3 scripts/setup_shared_skills.py` 生成 `.codex/skills`
- 其他 agent：直接读文件，按 `paperAnalysis/ → paperPDFs/ → paperIDEAs/` 顺序协作

---

## 快速上手路径

1. **想了解全貌** → 读本文档 + `AGENTS.md`
2. **想看 skill 怎么用** → 读 `.claude/skills/User_README_CN.md`
3. **想看某个 skill 的具体逻辑** → 读 `.claude/skills/<skill-name>/SKILL.md`
4. **想看已有的论文分析笔记** → 浏览 `paperAnalysis/` 下任意 `.md`
5. **想看辅助脚本** → 读 `scripts/README.md`
6. **想看状态流转规则** → 读 `.claude/skills/STATE_CONVENTION.md`
