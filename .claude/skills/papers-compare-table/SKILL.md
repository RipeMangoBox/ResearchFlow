---
name: papers-compare-table
description: Generates structured comparison tables for N papers from paperAnalysis frontmatter (core_operator, primary_logic, dataset, metrics, venue). Use when the user asks to compare methods, generate a related-work table, or needs a side-by-side summary of multiple papers.
---

# Papers Compare Table

## Purpose

从 `paperAnalysis/` 的结构化 frontmatter 中提取关键字段，生成论文对比表。适用于：

- 写 Related Work 时需要方法对比
- 选 baseline 时需要横向比较
- 向导师/合作者汇报时需要快速概览

## Input

用户提供以下任一形式：

1. **论文标题列表**：直接给出 N 篇论文标题
2. **查询条件**：如"所有 Motion Generation 的 CVPR 2025 论文"（通过 `papers-query-knowledge-base` 先检索再生成表格）
3. **paperAnalysis 路径列表**：直接指定 `.md` 文件路径

可选参数：
- `--fields`：指定对比字段（默认见下方）
- `--format`：输出格式，`markdown`（默认）或 `csv`
- `--output`：输出文件路径（默认输出到对话中，不写文件）

## Default comparison fields

从每篇论文的 `paperAnalysis/*.md` frontmatter 和正文中提取：

| 字段 | 来源 | 说明 |
|------|------|------|
| Title | frontmatter `title` | 论文标题 |
| Venue | frontmatter `venue` + `year` | 如 CVPR 2025 |
| Category | frontmatter `category` | 任务类别 |
| Core Operator | frontmatter `core_operator` | 核心方法一句话 |
| Primary Logic | frontmatter `primary_logic` | 输入→处理→输出流 |
| Key Contribution | Part II 正文 | 1-2 句核心贡献 |
| Dataset | Part III 或 frontmatter | 使用的数据集 |
| Metrics | Part III 或 frontmatter | 评测指标 |

用户可通过 `--fields` 选择子集或添加自定义字段。

## Workflow

1. **定位论文**：根据用户输入，在 `paperAnalysis/` 中找到对应的 `.md` 文件。
   - 如果用户给的是标题，先在 `paperCollection/_AllPapers.md` 中匹配，再定位到 analysis note。
   - 如果用户给的是查询条件，先调用 `papers-query-knowledge-base` 获取候选列表。
2. **提取字段**：读取每篇 `.md` 的 YAML frontmatter（`title`, `venue`, `year`, `core_operator`, `primary_logic`, `category`, `pdf_ref`）。
3. **补充正文字段**：如需 Key Contribution / Dataset / Metrics，从 Part II / Part III 正文中提取。
4. **生成表格**：按指定格式输出。

## Output contract

### Markdown 表格（默认）

```markdown
| Paper | Venue | Core Operator | Primary Logic | Dataset | Metrics |
|-------|-------|---------------|---------------|---------|---------|
| Paper A | CVPR 2025 | ... | ... | ... | ... |
| Paper B | ICLR 2026 | ... | ... | ... | ... |
```

### CSV（可选）

逗号分隔，首行为表头，适合导入 Excel / Google Sheets。

## Constraints

- 最多一次对比 **20 篇**论文（超过时提示用户缩小范围）
- 如果某篇论文的 analysis note 不存在，在表格中标注 `[未分析]` 并跳过
- 不自动生成新的 analysis note（那是 `papers-analyze-pdf` 的职责）

## Typical usage

### 1) 对比特定论文

> "帮我对比 InterMoE、TIMotion、Interact2Ar 这三篇 interaction 论文"

### 2) 按条件批量对比

> "把所有 Motion Generation 类别的 ICLR 2026 论文做个对比表"

### 3) 自定义字段

> "对比这五篇论文，只要 title、core_operator 和 dataset"

### 4) 输出到文件

> "生成对比表并保存到 paperAnalysis/compare_motion_gen_2026.md"
