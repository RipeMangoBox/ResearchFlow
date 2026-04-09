---
name: research-question-bank
description: Given a research direction and rough task description, generates a structured question/challenge list grounded in local KB and web search, covering domain open problems, top-venue reviewer concerns, and actionable research cuts. Saves output to QuestionBank/ as a reusable Markdown file. Use when the user wants to map out the question landscape for a research area before committing to a specific idea.
---

# Research Question Bank

## Purpose

Given a research direction and rough task description, the agent retrieves evidence from the local paper knowledge base (primarily `paperAnalysis`, with `paperCollection` only for statistics/navigation support) plus web search, and generates a structured question/challenge list covering:

1. Core open problems in the domain
2. Top-venue reviewer concern perspectives
3. Executable research entry points

Output is saved under `QuestionBank/` as a long-term reusable question asset.

## Positioning

- This is a **question discovery** tool, not an idea generation tool (that is `research-brainstorm-from-kb`)
- It is not reviewer simulation (that is `reviewer-stress-test`)
- Position in pipeline: upstream of ideation — first clarify "what questions are worth asking in this direction," then choose what to build

## Input

User provides:
- **Research direction**: for example "human motion generation", "multi-modal interaction", "video diffusion evaluation"
- **Rough task description** (optional): for example "improve text-to-motion evaluation", "explore interaction-aware generation"
- **Target venue** (optional): for example CVPR, ICLR, SIGGRAPH (influences reviewer perspective emphasis)

## Knowledge Protocol

1. **KB-first**: retrieve from local KB via `papers-query-knowledge-base`
   - Prioritize title, task path, tags, venue, year, `core_operator`, `primary_logic` in `paperAnalysis/`
   - Read `core_operator` and `primary_logic` from relevant `paperAnalysis/` to extract method commonalities and differences
   - Reference `paperCollection/` only when overview pages/statistics/Obsidian navigation aid is needed
2. **Web supplement**: search latest progress in this direction
   - Trends in accepted papers at top venues over the last 1-2 years
   - Latest direction shifts from leading teams
   - Public reviewer comments / OpenReview discussions (if available)
3. **Synthesis**: generate the question list based on KB evidence + web information

## Output Structure

Generated Markdown should follow this structure:

```markdown
---
created: {{ISO_DATE}}
updated: {{ISO_DATE}}
direction: {{ResearchDirection}}
tags:
  - question-bank
  - {{DirectionTag}}
---

# Question Bank: {{ResearchDirection}}

> Generated from local knowledge-base retrieval primarily on `paperAnalysis` plus web search; `paperCollection` is referenced only when statistics/navigation support is needed.

## 1. Core open problems in the domain

Group by sub-domain/sub-task. For each question include:
- question statement (1-2 sentences)
- why it matters (1 sentence)
- current progress snapshot (cite KB papers or web sources)

## 2. Top-venue reviewer concern perspectives

Organize by review dimensions (aligned with target venue standards):

### 2.1 Novelty & Non-triviality
- typical reviewer questions
- common "looks new but is not" traps in this direction

### 2.2 Technical Soundness
- common technical weaknesses in this direction
- assumptions that are easily challenged

### 2.3 Experimental Rigor
- baseline consensus and debates in this direction
- known flaws of evaluation metrics
- ablation types commonly requested by reviewers

### 2.4 Significance & Impact
- reviewer fatigue points (what kind of work is already oversaturated)
- what contributions are more likely to be recognized

## 3. Executable research cuts

Based on the questions above, provide 3-5 concrete research-cut suggestions:
- cut description (1-2 sentences)
- corresponding core question(s) (reference IDs above)
- estimated difficulty and required resources
- minimum validation plan

## 4. Reusable question templates

Provide 5-7 general question templates users can reuse when exploring new ideas later.
```

## File Naming and Storage

- Directory: `QuestionBank/` (under repository root)
- Filename: `YYYY-MM-DD_<topic>.md`
  - `YYYY-MM-DD`: creation date, e.g. `2026-04-08`
  - `<topic>`: lowercase English with underscores, e.g. `motion_generation`, `video_diffusion_evaluation`
  - Example: `2026-04-08_motion_generation.md`
- If same-date same-topic file already exists: append new sections instead of overwriting

## Workflow

1. Receive user direction and task description
2. Retrieve local KB via `papers-query-knowledge-base`
3. Use web search to supplement latest progress and reviewer perspectives
4. Generate the question list following the Output Structure
5. Write under `QuestionBank/`
6. Report back: number of questions generated, covered dimensions, and recommended priorities

## Boundaries

- Do not generate specific ideas or full plans (that belongs to `research-brainstorm-from-kb` and `idea-focus-coach`)
- Do not run reviewer scoring simulation (that belongs to `reviewer-stress-test`)
- Do not depend on project-specific code or experiment results
- Keep the question list at direction level (generalizable), not bound to a single user project
