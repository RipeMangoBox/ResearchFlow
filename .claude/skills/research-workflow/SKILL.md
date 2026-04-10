---
name: research-workflow
description: Unified entry for the research pipeline. Maps current work to one stage (collect/download/analyze/build/query/compare/ideate/focus/review/audit/export), recommends the right existing skill/command, supports step-by-step and end-to-end guidance, and keeps stage boundaries clear without duplicating underlying capabilities. Note: question-bank and pdfs-compress have been retired; their functions are covered by brainstorm and the download/analyze pipeline respectively.
---

# Research Workflow Entry

Use this as the single entry for local research workflow orchestration.

It does **not** replace underlying skills. It routes to them with a clear stage model.

## Stage model

KB build chain:
- collect → download → analyze → (optional build)

KB usage chain:
- query / compare → ideate → focus → review

Support chain:
- audit / export / code-context

## Purpose

- Provide one understandable entry point for the full workflow.
- Support both:
  - **step-by-step** execution (specify a stage)
  - **end-to-end** guidance (auto-detect next stage)
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
  - note: after analyze, you can go directly to query; run build only if you need statistics/navigation pages
- build
  - `papers-build-collection-index` (optional, refreshes `paperCollection/` statistics/navigation pages)
- query
  - `papers-query-knowledge-base`
  - `code-context-paper-retrieval`
- compare
  - `papers-compare-table`
- ideate
  - `research-brainstorm-from-kb`
- focus
  - `idea-focus-coach` (can be used independently; does not depend on ideate output)
- review
  - `reviewer-stress-test` (can be used independently; does not depend on focus output)
- audit
  - `papers-audit-metadata-consistency`
- export
  - `notes-export-share-version`

## Input contract by stage

- collect: URLs or GitHub repo URL + venue/year + include/exclude
- download: triage/log file path
- analyze: PDF path or `Downloaded` queue
- build: no extra input (defaults to current repository)
- query: task description/keywords (optional changed files, mode=brief/deep)
- compare: paper list (title, path, or query filter)
- ideate: research problem statement
- focus: initial idea + goal preferences (can come from ideate or be independent input)
- review: idea / roadmap / full paper (can come from focus or be independent input)
- audit: no extra input (scan current paperAnalysis)
- export: note path to export

## Output contract by stage

For each stage, return:

1. Current stage
2. Input requirements
3. Recommended execution (skill or command template)
4. Output paths
5. Suggested next stage

## Typical usage

### 1) I don't know what to run next

- Describe what you are currently doing
- The workflow will identify the stage and recommend the right skill

### 2) I only want one stage

- Specify the stage name
- Run the corresponding skill

### 3) End-to-end pass

- Start from auto
- Execute stage by stage
- At each stage, verify outputs exist before advancing

## Non-goals

- Do not replace execution logic of any underlying skill
- Do not auto-chain multiple stages (after each stage, suggest next step and let the user decide whether to continue)

## State Convention

`analysis_log.csv` paper states follow a unified convention. See `STATE_CONVENTION.md`:

```
Main pipeline: Wait → Downloaded → checked
Out-of-band states: Skip (manually skipped), Missing (download failed)
```
