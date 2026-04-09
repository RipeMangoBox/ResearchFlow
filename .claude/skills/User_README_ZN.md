# User README (Skill Usage Guide)

[English](User_README.md) | [Chinese](User_README_ZN.md)

This file is a **quick skill selection guide**.
It is for navigation only and does not participate in execution.

## 1. Quick routing

- Not sure where to start: `research-workflow`
- Collect papers from the web or GitHub: `papers-collect-from-web` / `papers-collect-from-github-awesome`
- Batch download and repair PDFs: `papers-download-from-list`
- Analyze PDFs into the knowledge base: `papers-analyze-pdf`
- Rebuild statistics/navigation pages or query the knowledge base: `papers-build-collection-index` / `papers-query-knowledge-base`
- Build a paper comparison table: `papers-compare-table`
- Expand research ideas: `research-brainstorm-from-kb`
- Generate a question map for a direction: `research-question-bank`
- Narrow an idea into an executable plan: `idea-focus-coach`
- Run strict reviewer-style stress testing: `reviewer-stress-test`
- Diagnose recurring skill mismatch: `skill-fit-guard`
- Migrate ResearchFlow into another domain: `domain-fork`

## 2. Categories and skill summaries

### 2.1 Knowledge base construction

- `papers-collect-from-web`
  - When to use: you already have a list of pages and want to triage papers in batch.
  - Input: URLs plus keyword and venue constraints.
  - Output: a triage list that can be processed downstream.

- `papers-collect-from-github-awesome`
  - When to use: the source is an awesome or curated repository.
  - Input: a GitHub repository URL.
  - Output: a candidate list aligned with the local analysis workflow.

- `papers-download-from-list`
  - When to use: you already have a manually filtered candidate list and need local PDFs.
  - Input: candidate lists such as `paperAnalysis/*.txt`.
  - Output: downloaded or repaired local PDFs plus status results.

- `pdfs-compress-large-files`
  - When to use: PDFs are too large for storage or transfer.
  - Input: `paperPDFs/` (auto-scanned).
  - Output: a compression report and updated PDFs.

- `papers-analyze-pdf`
  - When to use: you need structured analysis for one PDF or a batch.
  - Input: local PDF paths.
  - Output: structured Markdown under `paperAnalysis/`.

- `papers-audit-metadata-consistency`
  - When to use: you suspect inconsistencies between logs and analysis notes.
  - Input: current `paperAnalysis/`.
  - Output: a consistency audit report and quality issue list.

- `papers-build-collection-index`
  - When to use: analysis notes changed and `paperCollection/` statistics/navigation pages need refresh.
  - Input: frontmatter in `paperAnalysis/`.
  - Output: refreshed `paperCollection/`.

### 2.2 Knowledge base query and code-context retrieval

- `papers-query-knowledge-base`
  - When to use: you need to find papers, gather evidence, or produce comparison summaries from the local knowledge base.
  - Input: a research question, optionally with direction and constraints.
  - Output: evidence-grounded answers from the local knowledge base.

- `papers-compare-table`
  - When to use: you need a structured comparison table across multiple papers.
  - Input: paper titles, query conditions, or analysis paths.
  - Output: a Markdown or CSV comparison table.

- `code-context-paper-retrieval`
  - When to use: you are preparing a code change and want paper evidence first.
  - Input: the current code context or target module.
  - Output: brief or deep retrieval results.
  - Note: it prioritizes environment detection from codebase files such as `environment.yml`, and proactively asks the user if detection fails.

### 2.3 Paper ideas and research planning

- `research-brainstorm-from-kb`
  - When to use: you want divergent candidate ideas.
  - Input: a problem statement or direction draft.
  - Output: structured idea candidates with related-work support.

- `research-question-bank`
  - When to use: before deciding what to build, you want a map of important questions in a direction.
  - Input: a research direction and rough task description, optionally a target venue.
  - Output: a structured challenge list written under `QuestionBank/`.

- `idea-focus-coach`
  - When to use: an idea is too broad and needs step-by-step narrowing.
  - Input: an initial idea and goal preferences.
  - Output: focused goals, non-goals, prioritized hypotheses, and MVP experiments.
  - Independent use: does not depend on brainstorm or reviewer output.

- `reviewer-stress-test`
  - When to use: you want to stress-test an idea in ICLR/CVPR/SIGGRAPH style.
  - Input: an idea, roadmap, or full paper.
  - Output: major/minor risks and corresponding repair actions.
  - Independent use: does not depend on focus output.

### 2.4 Pipeline orchestration and process safeguards

- `research-workflow`
  - When to use: you are not sure which stage you are currently in.
  - Input: a description of your current task.
  - Output: stage identification and the recommended next skill.

- `notes-export-share-version`
  - When to use: internal notes need to be shared externally.
  - Input: notes to export.
  - Output: a shareable Markdown note with internal traces removed.

- `skill-fit-guard`
  - When to use: a skill output is clearly misaligned and that mismatch is likely to recur.
  - Input: symptoms of the mismatch.
  - Output: likely causes, revision options, and a prompt asking whether to revise now.

### 2.5 Domain migration

- `domain-fork`
  - When to use: you want to migrate the ResearchFlow architecture into another professional domain, such as frontend engineering, accounting, or journalism.
  - Input: target domain name.
  - Output: after interactive confirmation, a complete domain-adapted repository including renamed skills, folder structure, and README.
  - Trigger mode: explicit only.

## 3. Trigger strategy

All skills are triggered by description matching or explicit invocation. There is no automatic trigger based on file changes. The table below gives recommended trigger modes.

| Skill | Trigger mode | Typical timing |
|-------|----------|----------|
| `papers-collect-from-web` | explicit | When the user provides URLs and topic constraints |
| `papers-collect-from-github-awesome` | explicit | When the user provides a GitHub repository URL |
| `papers-download-from-list` | explicit / suggestive | Suggest after collection is completed |
| `pdfs-compress-large-files` | suggestive | Suggest when downloaded PDFs exceed 20 MB |
| `papers-analyze-pdf` | explicit / suggestive | Suggest after download is completed |
| `papers-audit-metadata-consistency` | suggestive | Suggest after batch analysis is completed |
| `papers-build-collection-index` | suggestive | Suggest after analysis if statistics/navigation pages need refresh |
| `papers-query-knowledge-base` | explicit / silent | Explicit for user queries; silent when used as an internal dependency |
| `papers-compare-table` | explicit | When the user asks for comparisons |
| `code-context-paper-retrieval` | explicit / suggestive | Suggest before model/method-related code changes |
| `research-brainstorm-from-kb` | explicit | When the user asks for research idea generation |
| `research-question-bank` | explicit | When the user wants to map the question landscape first |
| `idea-focus-coach` | explicit | When the user has a fuzzy idea and wants gradual narrowing |
| `reviewer-stress-test` | explicit | When the user has a fairly formed idea and wants pressure testing |
| `research-workflow` | explicit / suggestive | When the user is unsure of next steps |
| `notes-export-share-version` | explicit | When the user wants to share notes externally |
| `skill-fit-guard` | suggestive | When the agent detects clear repeated skill mismatch |
| `domain-fork` | explicit | When the user explicitly asks to migrate ResearchFlow into another domain |

Trigger mode notes:

- explicit: directly invoked by the user or triggered by description matching
- suggestive: the agent recommends running the skill, then executes after user confirmation
- silent: invoked as an internal dependency of another skill

## 4. Invocation methods

- Describe your goal directly (recommended).
- Name the skill directly, for example: "use papers-download-from-list".
- Use slash style, for example: `/papers-analyze-pdf`.

## 5. Invocation safety notes

- Actual routing depends only on `.claude/skills-config.json` and each skill's `SKILL.md`.
- `.claude/skills` is the only maintained source. If you also need `.agents/skills` / `.codex/skills` compatibility paths, run `python3 scripts/setup_shared_skills.py` or `py -3 scripts\setup_shared_skills.py`.
- This `User_README_ZN.md` is navigation-only and does not affect execution.
- `User_README.md` is also navigation-only and is not in the registry.
