# User README（技能使用说明）

本文件用于**快速选 skill**。
仅作导航，不参与执行。

## 1. 简略导航（先看这段）

- 不确定怎么开始：`research-workflow`
- 从网页/GitHub 收集论文：`papers-collect-from-web` / `papers-collect-from-github-awesome`
- 批量下载与修复 PDF：`papers-download-from-list`
- 分析 PDF 入库：`papers-analyze-pdf`
- 重建索引并检索：`papers-build-collection-index` / `papers-query-knowledge-base`
- 论文对比表：`papers-compare-table`
- 研究想法发散：`research-brainstorm-from-kb`
- 研究方向问题清单：`research-question-bank`
- 研究想法收敛：`idea-focus-coach`
- 严格审稿式质询：`reviewer-stress-test`
- 发现 skill 调用不贴题且反复发生：`skill-fit-guard`
- 迁移 ResearchFlow 到其他领域：`domain-fork`

## 2. 分类与技能说明（精简版）

### 2.1 KB 构建（收集/下载/分析/索引）

- `papers-collect-from-web`
  - 何时用：已有网页列表，想批量筛论文。
  - 输入：URL + 关键词/会议约束。
  - 产出：可继续处理的 triage 列表。

- `papers-collect-from-github-awesome`
  - 何时用：来源是 awesome/curated 仓库。
  - 输入：GitHub 仓库链接。
  - 产出：与本地分析流程对齐的候选清单。

- `papers-download-from-list`
  - 何时用：已有人工筛选列表，需落地 PDF。
  - 输入：`paperAnalysis/*.txt` 等候选列表。
  - 产出：下载/修复后的本地 PDF 与状态结果。

- `pdfs-compress-large-files`
  - 何时用：PDF 太大影响存储或传输。
  - 输入：`paperPDFs/`（自动扫描）。
  - 产出：压缩报告与更新后的 PDF。

- `papers-analyze-pdf`
  - 何时用：对单篇或批量 PDF 做结构化解读。
  - 输入：本地 PDF 路径。
  - 产出：`paperAnalysis/` 下结构化 Markdown。

- `papers-audit-metadata-consistency`
  - 何时用：怀疑日志与分析笔记不一致。
  - 输入：当前 `paperAnalysis/`。
  - 产出：一致性审计报告（质量问题清单）。

- `papers-build-collection-index`
  - 何时用：新增/修改分析后需要刷新索引。
  - 输入：`paperAnalysis/` frontmatter。
  - 产出：更新后的 `paperCollection/`。

### 2.2 KB 查询与代码上下文检索

- `papers-query-knowledge-base`
  - 何时用：按任务/技术/会议找论文，做比较总结。
  - 输入：查询问题（可附方向与约束）。
  - 产出：基于本地 KB 的引用式回答。

- `papers-compare-table`
  - 何时用：需要多篇论文的结构化对比表。
  - 输入：论文标题列表、查询条件或 analysis 路径。
  - 产出：Markdown 或 CSV 格式的对比表。

- `code-context-paper-retrieval`
  - 何时用：你在改代码，想看相关论文支撑。
  - 输入：当前代码上下文（或目标模块）。
  - 产出：brief/deep 两种论文检索结果。
  - 注意：会优先从 codebase 的 environment.yml 等文件检测环境，找不到会主动询问。

### 2.3 Paper Idea 与研究方案

- `research-brainstorm-from-kb`
  - 何时用：需要发散候选 idea。
  - 输入：问题/方向草案。
  - 产出：结构化 idea 候选与关联工作。

- `research-question-bank`
  - 何时用：想摸清一个方向的问题全景，再决定做什么。
  - 输入：研究方向 + 粗略任务描述（可选目标 venue）。
  - 产出：`QuestionBank/` 下的结构化问题/挑战清单 Markdown。

- `idea-focus-coach`
  - 何时用：idea 太宽，想逐步收敛到可执行。
  - 输入：初始想法、目标偏好。
  - 产出：聚焦目标、非目标、优先假设、MVP 实验。
  - 独立使用，不依赖 brainstorm 或 reviewer 的输出。

- `reviewer-stress-test`
  - 何时用：想提前承受 ICLR/CVPR/SIGGRAPH 风格质询。
  - 输入：idea/roadmap/full paper。
  - 产出：major/minor 风险 + 对应修复动作。
  - 独立使用，不依赖 focus 的输出。

### 2.4 Pipeline 编排与过程保障

- `research-workflow`
  - 何时用：不确定当前处于哪一阶段。
  - 输入：你现在的任务描述。
  - 产出：阶段判定 + 下一步 skill 建议。

- `notes-export-share-version`
  - 何时用：内部笔记要对外分享。
  - 输入：待导出的笔记。
  - 产出：去内部痕迹的分享版 Markdown。

- `skill-fit-guard`
  - 何时用：某次 skill 输出明显不匹配，且可能重复发生。
  - 输入：该次失配现象。
  - 产出：失配原因 + 修订选项 + 是否立即修订询问。

### 2.5 领域迁移

- `domain-fork`
  - 何时用：想把 ResearchFlow 的架构迁移到其他专业领域（如前端开发、会计、新闻等）。
  - 输入：目标领域名称。
  - 产出：交互式确认后，生成完整的领域适配仓库（skill 集合 + 目录结构 + README）。
  - 仅限显式调用。

## 3. 触发策略

所有 skill 均通过 description 匹配或显式调用触发，不存在文件变更后自动运行的机制。下表给出每个 skill 的推荐触发模式：

| Skill | 触发模式 | 触发时机 |
|-------|---------|---------|
| `papers-collect-from-web` | 显式 | 用户给出 URL + 关键词时 |
| `papers-collect-from-github-awesome` | 显式 | 用户给出 GitHub repo URL 时 |
| `papers-download-from-list` | 显式 / 建议式 | collect 完成后建议"是否下载？" |
| `pdfs-compress-large-files` | 建议式 | download 完成后检测到 >20MB PDF 时建议 |
| `papers-analyze-pdf` | 显式 / 建议式 | download 完成后建议"是否分析？" |
| `papers-audit-metadata-consistency` | 建议式 | 批量 analyze 完成后建议"是否审计一致性？" |
| `papers-build-collection-index` | 建议式 | analyze 完成后建议"是否重建索引？" |
| `papers-query-knowledge-base` | 显式 / 静默 | 用户查询时显式；brainstorm/focus/review 内部静默调用 |
| `papers-compare-table` | 显式 | 用户要求对比时 |
| `code-context-paper-retrieval` | 显式 / 建议式 | 代码修改前，检测到涉及模型/方法的任务时建议"是否先检索论文？" |
| `research-brainstorm-from-kb` | 显式 | 用户提出研究问题时 |
| `research-question-bank` | 显式 | 用户给出研究方向想生成问题清单时 |
| `idea-focus-coach` | 显式 | 用户 idea 模糊想收敛时 |
| `reviewer-stress-test` | 显式 | 用户有成型 idea 想压测时 |
| `research-workflow` | 显式 / 建议式 | 用户不确定下一步时；每个 stage 完成后建议下一步 |
| `notes-export-share-version` | 显式 | 用户要分享笔记时 |
| `skill-fit-guard` | 建议式 | agent 检测到 skill 输出与用户意图明显不匹配时主动建议 |
| `domain-fork` | 显式 | 用户明确要求迁移 ResearchFlow 到其他领域时 |

触发模式说明：
- 显式：用户主动调用或 agent 根据 description 匹配
- 建议式：agent 在特定上下文中主动建议"是否要运行 X？"，用户确认后执行
- 静默：作为其他 skill 的内部依赖被调用，不单独提示

## 4. 调用方式

- 直接说目标（推荐）。
- 直接点名 skill（如"用 papers-download-from-list"）。
- slash 方式（如 `/papers-analyze-pdf`）。

## 5. 调用安全说明

- 实际调用只依赖 `.claude/skills-config.json` 与各 skill 的 `SKILL.md`。
- 本 `User_README.md` 不在注册表中，不会影响调用。
