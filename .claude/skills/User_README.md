# User README (Skill Usage Guide)

[English](User_README.md) | [Chinese](User_README_CN.md)

This file is a **quick skill selection guide**.
It is navigation-only and does not participate in execution.

## 1. Quick routing

- Not sure how to start: `research-workflow`
- Collect papers from the web or GitHub: `papers-collect-from-web` / `papers-collect-from-github-awesome`
- Batch download and repair PDFs: `papers-download-from-list`
- Analyze PDFs into the KB: `papers-analyze-pdf`
- Rebuild statistics / navigation pages or search the KB: `papers-build-collection-index` / `papers-query-knowledge-base`
- Build a paper comparison table: `papers-compare-table`
- Explore research ideas when the direction is still open-ended: `research-brainstorm-from-kb`
- Narrow an existing direction into an executable plan: `idea-focus-coach`
- Run strict reviewer-style pressure testing when you want challenge rather than co-creation: `reviewer-stress-test`
- Diagnose repeated skill mismatch: `skill-fit-guard`
- Fork ResearchFlow into another domain: `domain-fork`

## 2. Categories and skill summaries

### 2.1 KB construction

- `papers-collect-from-web`
  - When to use: you already have a page list and want to filter papers in batch.
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

- `papers-analyze-pdf`
  - When to use: you want structured analysis for one PDF or a batch of PDFs.
  - Input: local PDF paths.
  - Output: structured Markdown under `paperAnalysis/`.
  - Language note: output language follows `analysis_language` in `AGENTS.md` (`zh` by default, switch to `en` for English notes). An explicit user request can override the default for one run.

- `papers-audit-metadata-consistency`
  - When to use: you suspect the log and analysis notes are inconsistent.
  - Input: the current `paperAnalysis/`.
  - Output: a consistency audit report and quality issue list.

- `papers-build-collection-index`
  - When to use: analysis notes changed and `paperCollection/` statistics / navigation pages need refresh.
  - Input: frontmatter in `paperAnalysis/`.
  - Output: refreshed `paperCollection/`.

### 2.2 KB query and code-context retrieval

- `papers-query-knowledge-base`
  - When to use: you need papers, evidence, or a text synthesis across papers from the local KB.
  - Input: a research question, optionally with direction and constraints.
  - Output: KB-grounded answers with local evidence.
  - Routing note: if you need a structured table, use `papers-compare-table`; if this is for an upcoming code change, prefer `code-context-paper-retrieval`.

- `papers-compare-table`
  - When to use: you need to choose between design alternatives (operators, representations, losses), write a Related Work table, select baselines, or present a method overview to collaborators.
  - Input: paper titles, query conditions, or analysis paths.
  - Output: a Markdown or CSV comparison table.
  - Routing note: if you only need a text summary, use `papers-query-knowledge-base`.

- `code-context-paper-retrieval`
  - When to use: you are changing code and want paper support first.
  - Input: the current code context or target module.
  - Output: brief or deep retrieval results.
  - Note: it tries to detect the environment from codebase files such as `environment.yml`, and asks if detection fails.

### 2.3 Paper ideas and research planning

- `research-brainstorm-from-kb`
  - When to use: you need divergent candidate ideas.
  - Input: a question or direction draft.
  - Output: structured idea candidates plus related work support.
  - Routing note: use this when the idea is still open-ended and you want candidate directions rather than scope cuts.

- `idea-focus-coach`
  - When to use: the idea is too broad and needs gradual narrowing.
  - Input: an initial idea and goal preferences.
  - Output: focused goals, non-goals, prioritized hypotheses, and MVP experiments.
  - Independent use: it does not require prior brainstorm or reviewer output.
  - Routing note: use this when you already have a real direction and need scope cuts, hypothesis ranking, or MVP planning.

- `reviewer-stress-test`
  - When to use: you want to pressure test an idea in ICLR/CVPR/SIGGRAPH style.
  - Input: an idea, roadmap, or full paper.
  - Output: major and minor risks plus concrete repair actions.
  - Independent use: it does not require prior focus output.
  - Routing note: use this when you want challenge and rejection-risk exposure rather than co-creation.

### 2.4 Pipeline orchestration and safeguards

- `research-workflow`
  - When to use: you are unsure which stage you are in.
  - Input: a description of the current task.
  - Output: stage identification plus the recommended next skill.

- `notes-export-share-version`
  - When to use: internal notes need to be shared externally.
  - Input: the note to export.
  - Output: a shareable Markdown version with internal traces removed.

- `skill-fit-guard`
  - When to use: a skill output is obviously misaligned and the mismatch may recur.
  - Input: the mismatch symptom.
  - Output: likely causes, revision options, and a prompt asking whether to revise immediately.

### 2.5 Domain migration

- `domain-fork`
  - When to use: you want to migrate the ResearchFlow architecture to another professional domain, such as frontend development, accounting, or journalism.
  - Input: the target domain name.
  - Output: after interactive confirmation, a complete domain-adapted repository with renamed skills, folder structure, and README.
  - Trigger mode: explicit only.

## 3. Trigger strategy

All skills are triggered either by description matching or explicit invocation. There is no automatic file-change trigger. The table below gives the recommended trigger mode for each skill.

| Skill | Trigger mode | Typical timing |
|-------|--------------|----------------|
| `papers-collect-from-web` | explicit | When the user provides URLs plus topic constraints |
| `papers-collect-from-github-awesome` | explicit | When the user provides a GitHub repository URL |
| `papers-download-from-list` | explicit / suggestive | Suggest after collection is complete |
| `papers-analyze-pdf` | explicit / suggestive | Suggest after download is complete |
| `papers-audit-metadata-consistency` | suggestive | Suggest after a batch analysis pass |
| `papers-build-collection-index` | suggestive | Suggest after analysis is complete if refreshed statistics / navigation pages are needed |
| `papers-query-knowledge-base` | explicit / silent | Explicit for user queries; silent as an internal dependency |
| `papers-compare-table` | explicit | When the user asks for a comparison |
| `code-context-paper-retrieval` | explicit / suggestive | Suggest before model- or method-related code edits |
| `research-brainstorm-from-kb` | explicit | When the user asks for open-ended candidate directions |
| `idea-focus-coach` | explicit | When the user has a real direction and wants to narrow it into scope cuts or MVPs |
| `reviewer-stress-test` | explicit | When the user has a formed idea and wants challenge rather than co-creation |
| `research-workflow` | explicit / suggestive | When the user is unsure of the next step |
| `notes-export-share-version` | explicit | When the user wants to share notes externally |
| `skill-fit-guard` | suggestive | When the agent detects strong repeated mismatch |
| `domain-fork` | explicit | When the user clearly asks to migrate ResearchFlow into another domain |

Trigger mode notes:

- explicit: directly invoked by the user or matched by description
- suggestive: the agent proposes running the skill in context, then waits for confirmation
- silent: called as an internal dependency of another skill

## 4. Invocation styles

- Describe the goal directly. Recommended.
- Name the skill directly, such as "use papers-download-from-list".
- Use slash-style invocation, such as `/papers-analyze-pdf`.

## 5. Safety note

- Actual routing depends only on `.claude/skills-config.json` and each skill's `SKILL.md`.
- `.claude/skills` is the maintained source of truth. Run `python3 scripts/setup_shared_skills.py` or `py -3 scripts\setup_shared_skills.py` if you also need `.codex/skills` compatibility aliases.
- This `User_README.md` is not part of the registry and does not affect execution.
- `User_README_CN.md` is also navigation-only and does not affect execution.
