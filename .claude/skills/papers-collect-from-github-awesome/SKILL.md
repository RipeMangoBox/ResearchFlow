---
name: papers-collect-from-github-awesome
description: Collects paper candidates from a GitHub awesome / curated repo README. Agent fetches the README, analyzes its format, writes a one-off parser, and outputs rows aligned with analysis_log.csv. No pre-built scripts — agent adapts to each repo's format on the fly.
---

# Papers Collect From GitHub Awesome

## Purpose

从 GitHub awesome / curated 仓库的 README 中提取论文候选列表，输出格式对齐 `paperAnalysis/analysis_log.csv`。

由于不同 awesome 列表的格式差异很大（Markdown 表格、bullet list、混合 HTML 等），本 skill **不附带固定脚本**。Agent 每次根据目标 README 的实际格式，自行编写一次性解析逻辑。

## Input

用户提供：
- **GitHub repo URL**：如 `https://github.com/Foruck/Awesome-Human-Motion`
- **include/exclude 关键词**（可选）：如 include `motion generation`，exclude `survey`
- **目标 category**（可选）：如 `Motion_Generation_Text_Speech_Music_Driven`

## Output Format

输出必须对齐 `paperAnalysis/analysis_log.csv` 的列格式：

```
state,importance,paper_title,venue,project_link_or_github_link,paper_link,sort,pdf_path
```

各字段说明：

| 列 | 首次收集默认值 | 说明 |
|----|-------------|------|
| state | `Wait` | 后续人工筛选后改为 `Skip`，或保持 `Wait` 等待下载 |
| importance | 空 | 后续人工标注 S/A/B/C |
| paper_title | 从 README 解析 | 论文标题 |
| venue | 从 README 解析，无法识别则 `Unknown` | 如 `CVPR 2025`、`ICLR 2026`；仅在 arXiv 等公开平台发表则写 `arXiv YYYY` |
| project_link_or_github_link | 项目页或 GitHub 链接 | 优先项目页；确认无开源则填 `N/A` |
| paper_link | arXiv / OpenReview 链接 | 优先 arXiv abs 链接 |
| sort | README 中的 section/category heading | 用 `_` 连接，如 `Motion_Generation` |
| pdf_path | 空 | 下载后由 `papers-download-from-list` 填充 |

参考示例见仓库中的 `paperAnalysis/analysis_log.csv`。

## Workflow

### 1. Fetch README

- 从 GitHub repo URL 获取 README 原文
- 优先用 `https://raw.githubusercontent.com/<owner>/<repo>/<branch>/README.md`
- 保存原始 README 到 `paperAnalysis/processing/github_awesome/<repo_slug>/README.raw.md`（可选，便于调试）

### 2. Analyze Format

Agent 阅读 README 内容，判断其结构：
- Markdown 表格？bullet list？混合格式？
- 论文标题在哪个位置？链接如何组织？
- venue 信息是内联的还是按 section 分组的？

### 3. Write One-Off Parser

根据分析结果，agent 编写一次性 Python 脚本（或直接在对话中用代码块处理），提取论文信息。

脚本要求：
- 输出 CSV 格式，列顺序与 `analysis_log.csv` 一致
- 处理常见链接模式：arXiv abs/pdf、OpenReview、GitHub project page
- venue 解析：尝试从文本中提取 `(CVPR 2025)` / `[ICLR 2026]` 等模式
- 去重：同一论文多次出现时只保留一条

### 4. Filter & Dedup

- 应用用户的 include/exclude 关键词过滤
- 与现有 `analysis_log.csv` 去重（按 paper_title 模糊匹配）
- 输出新增候选行数

### 5. Append to analysis_log.csv

- 将新候选追加到 `paperAnalysis/analysis_log.csv`
- 不修改已有行
- 向用户汇报：新增 N 条候选，来自哪些 section

### 6. Completeness Check & Suggestion

追加完成后，agent 扫描本次新增行，统计缺失字段：

| 缺失字段 | 检测条件 | 补全方式 | Fallback 值 |
|---------|---------|---------|------------|
| venue | 值为空或 `Unknown` | 用 paper_link 查 arXiv metadata / Semantic Scholar API | 未被会议收录但在公开平台发表：`arXiv YYYY`（如 `arXiv 2025`） |
| paper_link | 值为空 | 用 paper_title 搜索 arXiv / Google Scholar | 确实找不到：留空 |
| project_link_or_github_link | 值为空 | 用 paper_title 搜索 GitHub / Papers With Code | 确认无开源：`N/A` |
| sort | 值为空 | 从 README section heading 推断，或请用户指定 | 留空待用户填写 |

扫描完成后，向用户汇报缺失情况并建议：

> "本次新增 N 条候选，其中 X 条缺少 venue，Y 条缺少 paper_link，Z 条缺少 project_link。是否需要多源搜索补全？（全部补全 / 仅补 venue / 仅补 link / 跳过）"

用户确认后：
- **全部补全**：依次用 arXiv API、Semantic Scholar、Google Scholar、Papers With Code 搜索补全
- **仅补特定字段**：只搜索对应来源
- **跳过**：保持现状，后续手动处理

补全时的约束：
- 每次搜索间隔 ≥ 3 秒（遵守 API rate limit）
- 搜索结果需与 paper_title 做相似度匹配，阈值 > 0.8 才自动填入，否则标记 `[需确认]` 让用户复核
- 补全结果直接更新 `analysis_log.csv` 中对应行

## Environment Detection

如需运行 Python 脚本，遵循 `code-context-paper-retrieval` 中的 Environment Detection 规则：
- 优先从 codebase 的 `environment.yml` / `requirements.txt` 等检测环境
- 找不到则主动询问用户

## Constraints

- 不附带预置脚本——每个 awesome 列表格式不同，agent 现场适配
- 一次性脚本可保存到 `paperAnalysis/processing/github_awesome/<repo_slug>/parse_<timestamp>.py` 供复用参考
- 不自动下载 PDF（那是 `papers-download-from-list` 的职责）
- 不自动生成 analysis note（那是 `papers-analyze-pdf` 的职责）

## Typical Usage

> "帮我从 https://github.com/Foruck/Awesome-Human-Motion 收集论文，只要 motion generation 相关的"

> "把 https://github.com/xxx/awesome-video-diffusion 的论文列表整理到 analysis_log.csv"
