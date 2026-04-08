---
name: research-workflow
description: Unified entry for the research pipeline. Maps current work to one stage (collect/download/analyze/build/query/compare/ideate/focus/review/audit/export), recommends the right existing skill/command, supports step-by-step and end-to-end guidance, and keeps stage boundaries clear without duplicating underlying capabilities.
---

# Research Workflow Entry

Use this as the single entry for the local research workflow orchestration.

It does **not** replace underlying skills. It routes to them with a clear stage model.

## Stage model

KB 构建链：
- collect → download → analyze → build

KB 使用链：
- query / compare → question-bank → ideate → focus → review

辅助链：
- audit / export / code-context

## Purpose

- Provide one understandable 入口 for the full workflow.
- Support both:
  - **step-by-step** execution (指定 stage)
  - **end-to-end** guidance (auto detect next stage)
- Clarify each stage's input/output contract.
- Avoid overlap: orchestration here, execution in existing stage skills.

## Stage mapping (reuse existing skills)

- collect
  - `papers-collect-from-web`
  - `papers-collect-from-github-awesome`
- download
  - `papers-download-from-list`
- analyze
  - `papers-analyze-pdf`
  - note: analyze 结束后仅提示可调用 build，不自动 build
- build
  - `papers-build-collection-index`
- query
  - `papers-query-knowledge-base`
  - `code-context-paper-retrieval`
- compare
  - `papers-compare-table`
- question-bank
  - `research-question-bank`
- ideate
  - `research-brainstorm-from-kb`
- focus
  - `idea-focus-coach`（独立使用，不依赖 ideate 阶段的输出）
- review
  - `reviewer-stress-test`（独立使用，不依赖 focus 阶段的输出）
- audit
  - `papers-audit-metadata-consistency`
- export
  - `notes-export-share-version`

## Input contract by stage

- collect: URLs 或 GitHub repo URL + venue/year + include/exclude
- download: triage/log 文件路径
- analyze: PDF 路径或 `Downloaded` 队列
- build: 无额外输入（默认当前 当前仓库）
- query: 任务描述/关键词（可选 changed files, mode=brief/deep）
- compare: 论文列表（标题、路径或查询条件）
- question-bank: 研究方向 + 粗略任务描述（可选目标 venue）
- ideate: 研究问题陈述
- focus: 初始想法 + 目标偏好（可从 ideate 输出接入，也可独立输入）
- review: idea / roadmap / full paper（可从 focus 输出接入，也可独立输入）
- audit: 无额外输入（扫描当前 paperAnalysis）
- export: 待导出的笔记路径

## Output contract by stage

For each stage, return:

1. 当前阶段
2. 输入要求
3. 推荐执行（skill 或命令模板）
4. 产出路径
5. 下一阶段建议

## Typical usage

### 1) I don't know what to run next

- 描述你当前在做什么
- workflow 会判断阶段并推荐 skill

### 2) I only want one stage

- 指定 stage 名称
- 执行对应 skill

### 3) End-to-end pass

- Start from auto
- Execute stage by stage
- At each stage, verify outputs exist before advancing

## Non-goals

- 不替代任何底层 skill 的执行逻辑
- 不自动串联多个 stage（每个 stage 结束后提示下一步，由用户决定是否继续）

## State Convention

`analysis_log.csv` 中论文状态遵循统一约定，详见 `STATE_CONVENTION.md`：

```
主流程：Wait → Downloaded → checked
非主流程：Skip（人工跳过）、Missing（下载失败）
```
