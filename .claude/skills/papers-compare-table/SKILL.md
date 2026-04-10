---
name: papers-compare-table
description: "Generates structured comparison tables for N papers from paperAnalysis frontmatter (core_operator, primary_logic, dataset, metrics, venue). Use when the user needs a side-by-side table to decide between design alternatives, write a Related Work section, select baselines, or present a method overview to collaborators. Not for free-text summaries (use papers-query-knowledge-base instead)."
---

# Papers Compare Table

## Purpose

Extract key fields from structured frontmatter and content in `paperAnalysis/` to generate paper comparison tables.

### When to use (concrete scenarios)

1. **Design decision**: you are choosing between two or more operators/representations/losses and need a structured side-by-side view of their core_operator, primary_logic, and applicable scenarios.
2. **Related Work table**: you are writing a paper and need a formatted comparison table ready to paste into LaTeX or Markdown.
3. **Baseline selection**: you need to compare datasets, metrics, and reported numbers across candidate baselines before running experiments.
4. **Advisor / collaborator update**: you want a quick, scannable overview of N papers without reading full analysis notes.

### When NOT to use

- You only need a text summary or evidence synthesis → use `papers-query-knowledge-base`.
- You want to generate new ideas from the comparison → use `research-brainstorm-from-kb` after getting the table.

## Input

The user may provide any of the following:

1. **Paper title list**: directly provide N paper titles
2. **Query condition**: e.g., "all Motion Generation papers from CVPR 2025" (retrieve with `papers-query-knowledge-base` first, then generate the table)
3. **paperAnalysis path list**: directly provide `.md` file paths

Optional parameters:
- `--fields`: comparison fields to include (defaults below)
- `--format`: output format, `markdown` (default) or `csv`
- `--output`: output file path (default is in-chat output; no file write)

## Default comparison fields

Extract from each `paperAnalysis/*.md` frontmatter and body:

| Field | Source | Description |
|------|------|------|
| Title | frontmatter `title` | paper title |
| Venue | frontmatter `venue` + `year` | e.g., CVPR 2025 |
| Category | frontmatter `category` | task category |
| Core Operator | frontmatter `core_operator` | one-line core method |
| Primary Logic | frontmatter `primary_logic` | input→process→output flow |
| Key Contribution | Part II body | 1-2 sentence core contribution |
| Dataset | Part III or frontmatter | datasets used |
| Metrics | Part III or frontmatter | evaluation metrics |

Users can select a subset with `--fields` or add custom fields.

## Workflow

1. **Locate papers**: find matching `.md` files under `paperAnalysis/`.
   - If input is titles, match titles in `paperAnalysis/` first.
   - If overview/navigation help is needed, reference `paperCollection/`.
   - If input is query conditions, call `papers-query-knowledge-base` to get candidates first.
2. **Extract fields**: read YAML frontmatter (`title`, `venue`, `year`, `core_operator`, `primary_logic`, `category`, `pdf_ref`).
3. **Extract body fields**: if needed, read Key Contribution / Dataset / Metrics from Part II / Part III.
4. **Generate table**: output in the requested format.

## Output contract

### Markdown table (default)

```markdown
| Paper | Venue | Core Operator | Primary Logic | Dataset | Metrics |
|-------|-------|---------------|---------------|---------|---------|
| Paper A | CVPR 2025 | ... | ... | ... | ... |
| Paper B | ICLR 2026 | ... | ... | ... | ... |
```

### CSV (optional)

Comma-separated with header row; suitable for Excel / Google Sheets import.

## Constraints

- Compare at most **20 papers** per run (if exceeded, ask the user to narrow scope)
- If an analysis note is missing for a paper, mark `[Not analyzed]` in the table and skip it
- Do not auto-generate new analysis notes (that is `papers-analyze-pdf` responsibility)

## Typical usage

### 1) Compare specific papers

> "Compare InterMoE, TIMotion, and Interact2Ar for interaction"

### 2) Batch comparison by condition

> "Build a comparison table for all ICLR 2026 papers in Motion Generation"

### 3) Custom fields

> "Compare these five papers with only title, core_operator, and dataset"

### 4) Save to file

> "Generate the comparison table and save to paperAnalysis/compare_motion_gen_2026.md"
