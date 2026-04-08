---
name: domain-fork
description: Migrates ResearchFlow's architecture to a new professional domain (e.g. frontend development, accounting, journalism). Interactive session with the user to map research concepts to domain equivalents, then generates a complete set of adapted skills and folder structure. Explicit trigger only.
---

# Domain Fork

## Purpose

将 ResearchFlow 的核心架构（收集 → 下载 → 分析 → 建库 → 查询 → 对比 → 问题发现 → 方案发散 → 聚焦 → 评审）迁移到用户指定的专业领域，一次性生成对应领域的完整 skill 集合和文件夹结构。

## Trigger

**仅限显式调用**。不会被 description 匹配自动触发。

用户必须明确说出类似：
- "帮我 fork 一个前端开发版本的 ResearchFlow"
- "把 ResearchFlow 迁移到会计领域"
- "用 domain-fork 创建一个新闻采编的知识库"

## Interactive Flow

调用后进入交互式确认流程，**不跳步**：

### Step 1: 领域确认

向用户确认：
- **目标领域**：如"前端开发"、"会计审计"、"新闻采编"
- **仓库名称**：建议 `<Domain>Flow`（如 `FrontendFlow`、`AccountingFlow`、`JournalismFlow`），用户可自定义
- **保存位置**：如果用户未指定，默认保存到 `ResearchFlow/<RepoName>/`，并提示：

> "默认保存到 ResearchFlow/<RepoName>/，是否需要修改路径？"

### Step 2: 概念映射表

生成 ResearchFlow → 目标领域的概念映射表，请用户确认或修改：

| ResearchFlow 概念 | 映射到目标领域 | 说明 |
|-------------------|--------------|------|
| paper / 论文 | （领域等价物） | 如：技术文章、法规文件、新闻稿件 |
| PDF | （领域等价物） | 如：文章网页、PDF 法规、稿件文档 |
| venue / 会议 | （领域等价物） | 如：技术博客/框架版本、法规发布机构、媒体来源 |
| paperAnalysis / 分析笔记 | （领域等价物） | 如：技术笔记、法规解读、采编分析 |
| paperCollection / 索引 | （领域等价物） | 如：技术索引、法规库、选题库 |
| paperPDFs / 原始文档 | （领域等价物） | 如：原始文章、法规原文、稿件原文 |
| paperIDEAs / 研究想法 | （领域等价物） | 如：项目方案、审计策略、选题方案 |
| QuestionBank / 问题库 | （领域等价物） | 如：技术难题库、合规问题库、采编问题库 |
| core_operator | （领域等价物） | 如：核心技术方案、核心法规条款、核心新闻角度 |
| primary_logic | （领域等价物） | 如：技术实现流程、合规检查流程、采编流程 |
| analysis_log.csv | 保持 | 主跟踪日志，列名按领域调整 |
| state: Wait→Downloaded→checked | 保持或调整 | 状态流转可能需要领域适配 |

### Step 3: Skill 映射确认

展示 ResearchFlow 的 17 个 skill 如何映射到目标领域：

| ResearchFlow Skill | 目标领域 Skill | 是否保留 | 调整说明 |
|-------------------|---------------|---------|---------|
| papers-collect-from-web | `<domain>-collect-from-web` | ✅ | 收集来源改为领域网站 |
| papers-collect-from-github-awesome | `<domain>-collect-from-curated-list` | ✅/❌ | 视领域是否有 curated list |
| papers-download-from-list | `<domain>-download-from-list` | ✅ | 下载对象改为领域文档 |
| pdfs-compress-large-files | `compress-large-files` | ✅/❌ | 视文档类型决定 |
| papers-analyze-pdf | `<domain>-analyze-document` | ✅ | 分析模板按领域重写 |
| papers-audit-metadata-consistency | `<domain>-audit-metadata` | ✅ | 元数据字段按领域调整 |
| papers-build-collection-index | `<domain>-build-index` | ✅ | 索引维度按领域调整 |
| papers-query-knowledge-base | `<domain>-query-kb` | ✅ | 查询维度按领域调整 |
| papers-compare-table | `<domain>-compare-table` | ✅ | 对比字段按领域调整 |
| code-context-paper-retrieval | `code-context-<domain>-retrieval` | ✅/❌ | 仅代码相关领域保留 |
| research-question-bank | `<domain>-question-bank` | ✅ | 问题维度按领域调整 |
| research-brainstorm-from-kb | `<domain>-brainstorm-from-kb` | ✅ | 发散方向按领域调整 |
| idea-focus-coach | `<domain>-focus-coach` | ✅ | 聚焦维度按领域调整 |
| reviewer-stress-test | `<domain>-stress-test` | ✅ | 评审标准按领域调整 |
| research-workflow | `<domain>-workflow` | ✅ | stage 名称按领域调整 |
| notes-export-share-version | `notes-export-share-version` | ✅ | 通用，无需改 |
| skill-fit-guard | `skill-fit-guard` | ✅ | 通用，无需改 |

用户可以：
- 删除不需要的 skill
- 修改 skill 名称
- 调整映射关系

### Step 4: 确认并生成

用户确认后，一次性生成：

1. **目录结构**

```
<RepoName>/
├── .claude/
│   ├── skills-config.json
│   └── skills/
│       ├── User_README.md
│       ├── README.md
│       ├── STATE_CONVENTION.md
│       └── <all mapped skills>/
├── <AnalysisDir>/          # 对应 paperAnalysis
│   └── tracking_log.csv    # 对应 analysis_log.csv
├── <CollectionDir>/        # 对应 paperCollection
│   ├── by_<dim1>/
│   ├── by_<dim2>/
│   └── by_<dim3>/
├── <SourceDir>/            # 对应 paperPDFs
├── <IdeaDir>/              # 对应 paperIDEAs
├── QuestionBank/
└── README.md
```

2. **每个 skill 的 SKILL.md**：基于 ResearchFlow 对应 skill 重写，替换所有领域术语
3. **skills-config.json**：注册所有生成的 skill
4. **STATE_CONVENTION.md**：状态流转按领域调整
5. **User_README.md / README.md**：按领域重写导航和说明
6. **tracking_log.csv**：空模板，列名按领域调整
7. **仓库 README.md**：按领域重写

## Generation Principles

1. **结构对称**：保持 ResearchFlow 的 collect → build → use 三阶段架构
2. **术语一致**：同一领域概念在所有 skill 中使用相同术语
3. **最小可用**：只生成 skill 定义（SKILL.md），不生成脚本——agent 按需现场编写
4. **状态机保持**：`Wait → Downloaded → checked` 的三态主流程保持，状态名可按领域调整
5. **不修改 ResearchFlow**：domain-fork 只读取 ResearchFlow 作为模板，不修改任何 ResearchFlow 文件

## Boundaries

- 只生成 skill 定义和目录结构，不生成实际内容数据
- 不自动填充 tracking_log.csv（那是用户使用新仓库后的事）
- 不复制 ResearchFlow 的 paperAnalysis / paperCollection / paperPDFs 内容
- 不复制 `.obsidian.zip`（用户可自行从 ResearchFlow 复制并调整）

## Example

用户："帮我 fork 一个前端开发版本"

Step 1 确认：
- 领域：前端开发
- 仓库名：FrontendFlow
- 路径：ResearchFlow/FrontendFlow/

Step 2 映射：
- paper → 技术文章/规范文档
- PDF → 文章网页/PDF 规范
- venue → 技术博客/框架版本/W3C 标准
- paperAnalysis → articleAnalysis
- paperCollection → articleCollection
- paperPDFs → articleSources
- core_operator → 核心技术方案
- primary_logic → 实现流程

Step 3 skill 映射：
- `papers-collect-from-web` → `articles-collect-from-web`
- `papers-collect-from-github-awesome` → `articles-collect-from-curated-list`
- `code-context-paper-retrieval` → `code-context-article-retrieval`（保留，前端是代码领域）
- `reviewer-stress-test` → `code-review-stress-test`（映射为 code review 视角）
- ...

Step 4：生成 FrontendFlow/ 完整结构。
