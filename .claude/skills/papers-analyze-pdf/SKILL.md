---
name: papers-analyze-pdf
follows: rf-obsidian-markdown
description: Analyze local academic PDFs into structured Markdown notes under `paperAnalysis/`. The primary contract is canonical PDF path plus high-quality analysis note. Optional source, citation, or artifact sidecars are secondary and should be added only when explicitly needed.
---

# Paper Analysis

Primary workflow:

```text
canonical PDF -> analysis note
```

The note is the main deliverable. Optional bundles or extraction artifacts are support material, not the default success criterion.

## Scope

- Input: one local PDF, one directory of PDFs, or rows from `analysis_log.csv`
- Output: one analysis note per PDF under `paperAnalysis/`
- Read only local PDFs. Do not fetch paper content from the web. Do not fabricate figures, tables, formulas, or citations.

Assume the repository root contains `paperPDFs/` and `paperAnalysis/`.

## Path Rules

- Canonical PDF path: `paperPDFs/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.pdf`
- Canonical note path: `paperAnalysis/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.md`
- Derive the note path by mirroring the final canonical PDF path

If the PDF is already under `paperPDFs/`:

- keep the canonical PDF path
- write or update the mirrored note path only

If the PDF is outside `paperPDFs/`:

- infer category, venue, and year when reliable
- otherwise ask the user before moving or renaming files
- use the ingest helper when metadata is known:

```bash
python .claude/skills/papers-analyze-pdf/scripts/ingest_pdfs.py <path> --category <cat> --venue <Venue> --year <Year> [--title <Title>]
```

## Analysis Goal

Every note must answer three questions:

1. What is the real bottleneck, and why does it matter?
2. What causal knob did the paper change?
3. What capability gain is actually supported by the evidence?

Write for retrieval and reuse, not for exhaustiveness.

- Prioritize: intuition > mechanism > evidence
- Keep details only when they support the core idea, contribution, or reusable operator
- Avoid formula derivations, long hyperparameter dumps, and table-by-table restatement
- If formulas matter, explain only the mechanism-level role of the key objective or constraint

## Language Control

Use `analysis_language` from repo-level `AGENTS.md` unless the user explicitly overrides it in the current run.

| `analysis_language` | Body language | Required headings | Required Part II subsection |
| --- | --- | --- | --- |
| `zh` | Chinese | `## Part I：问题与挑战` / `## Part II：方法与洞察` / `## Part III：证据与局限` | `### 核心直觉` |
| `en` | English | `## Part I: Problem & Challenge` / `## Part II: Method & Insight` / `## Part III: Evidence & Limits` | `### The "Aha!" Moment` |

Rules:

- YAML frontmatter keys stay in English
- `tags`, `category`, file names, wiki links, and `pdf_ref` remain English-compatible
- `core_operator` and `primary_logic` follow the selected analysis language

## Required Structure

Each note should use the same structure.
If Part I, Part II, Part III, or the required Part II subsection is missing or clearly too thin, regenerate once.
If the paper still does not fit cleanly, keep the best version and mark it with `analysis_mismatch`.

### 1. YAML frontmatter

Required fields:

- `title`
- `venue`
- `year`
- `tags`
- `core_operator`
- `primary_logic`
- `claims`
- `pdf_ref`
- `category`

Minimal example:

```yaml
---
title: "Short Title of the Paper"
venue: CVPR
year: 2025
tags:
  - Motion_Generation_Text_Speech_Music_Driven
  - task/text-to-motion
  - diffusion
  - dataset/HumanML3D
core_operator: 用一句话说明核心机制
primary_logic: |
  输入条件 → 关键变换 → 输出能力
claims:
  - "Claim 1: verifiable statement"
  - "Claim 2: verifiable statement"
pdf_ref: paperPDFs/Category/Venue_Year/Year_Title.pdf
category: Motion_Generation_Text_Speech_Music_Driven
---
```

Tag guidance:

- keep exactly one category tag
- keep 1-2 `task/` tags for the main task
- keep 1-3 technique tags for the defining method
- add `dataset/` tags for main experiments
- add `repr/` only when the representation is part of the contribution
- add one `opensource/` tag when the status is clear

`claims` should contain 2-3 verifiable statements covering the main contribution or strongest experimental conclusions.

### 2. Title and TL;DR

- use the paper title as the level-1 heading
- add `> [!abstract] **Quick Links & TL;DR**`
- include:
  - links such as arXiv, project page, or repo if known
  - one-sentence summary of the core contribution
  - 1-2 key result bullets

### 3. Part I

Use the exact heading required by the selected language.

Focus on:

- what the paper is trying to make possible
- the true source of difficulty
- the input/output interface
- where the method works and where it breaks

### 4. Part II

Use the exact heading required by the selected language.

Focus on:

- the main design idea
- the key change relative to prior work
- why that change works
- the main trade-off

Must include:

- `zh`: `### 核心直觉`
- `en`: `### The "Aha!" Moment`

Make the causal chain explicit:

`what changed -> which bottleneck or constraint changed -> which capability changed`

### 5. Part III

Use the exact heading required by the selected language.

Focus on:

- the strongest evidence for the capability gain
- 1-2 key numbers only when they truly support the main claim
- failure modes or limitations
- reusable operators, modules, or ideas

### 6. Local PDF embed

End with:

```text
![[paperPDFs/<Category>/<Venue_Year>/<filename>.pdf]]
```

This must match `pdf_ref`.

## Batch Helpers

When the task is driven from `analysis_log.csv`, the active default is:

- process `Downloaded` rows first
- retry `analysis_mismatch` only when explicitly requested
- retry `api_failed` only when explicitly requested

Automation wrappers are optional. The public contract for this skill is still:

```text
analysis_log.csv rows or local PDFs -> analysis notes under paperAnalysis/
```

Public batch helper entrypoints in this repository:

```bash
python scripts/batch_analyze_api.py --log paperAnalysis/analysis_log.csv --filter regular --batch-size 4
python scripts/batch_analyze_api_claude.py --log paperAnalysis/analysis_log.csv --filter regular --batch-size 4
```

## Optional Sidecars And Artifacts

These are optional, not required for a normal paper-analysis run:

- `paperAnalysis/processing/incremental/...`
- Semantic Scholar refresh
- preserved parsed markdown, figures, tables, or formulas

Use them only when the task explicitly asks for source refresh, citation refresh, or artifact preservation.

If you need them, available helpers include:

- `python .claude/skills/papers-analyze-pdf/scripts/prepare_incremental_bundle.py <pdf>`

Do not treat these optional helpers as the default completion condition for `papers-analyze-pdf`.

## Execution Checklist

For each PDF:

1. Resolve the canonical PDF path and mirrored note path
2. Read the main PDF and extract the bottleneck, causal knob, and evidence
3. Read supplementary material only if the main PDF is insufficient for the core questions
4. Write the note using the required structure
5. Check that the note stays focused on core idea, contribution, and evidence
6. If running from `analysis_log.csv`, update status after writing:
   - success -> `checked`
   - unreadable or missing PDF -> keep current state or mark the concrete file problem
   - structure still mismatched after one retry -> `analysis_mismatch`
