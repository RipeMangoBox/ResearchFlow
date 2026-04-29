## ResearchFlow Skills Overview

Single source of truth: keep the maintained skill library in `.claude/skills`.
If you also want Codex-compatible paths, run `python3 scripts/setup_shared_skills.py` on macOS/Linux or `py -3 scripts\setup_shared_skills.py` on Windows to generate `.codex/skills` aliases without copying.

This skills directory supports the local paper workflow covering **sync -> collect -> download -> analyze -> build -> query -> emerge -> ideate -> focus -> review**.

### 1. Workflow entry

- **research-workflow**
  - Routes work to one stage among sync / collect / download / analyze / build / query / emerge / ideate / focus / review / audit / export.
  - Returns the current stage, required inputs, suggested command or skill, expected outputs, and next step.

### 2. Paper pipeline skills

- **papers-sync-from-zotero**
  - Sync papers from a Zotero library (via pyzotero / Zotero API) or import from a flat PDF folder (fallback).
  - Copies PDFs into `paperPDFs/`, writes rich metadata to `paperAnalysis/processing/zotero/manifest.jsonl`, appends lightweight rows to `analysis_log.csv`.
  - Supports incremental sync.
- **papers-collect-from-web**
  - Collect candidate papers from non-GitHub web pages (conference sites, lab homepages, proceedings) and generate a triage list under `paperAnalysis/`.
- **papers-collect-from-github-awesome**
  - Parse any GitHub repository (awesome lists, survey companion repos, lab paper lists, conference accepted-paper repos, etc.) into an `analysis_log.csv`-aligned candidate list.
- **papers-download-from-list**
  - Download, verify, repair, and deduplicate PDFs into `paperPDFs/`.
- **papers-analyze-pdf**
  - Convert local PDFs into structured `paperAnalysis/*.md` notes.
  - Default output language is Chinese. Change `analysis_language` in `AGENTS.md` to `en`, or explicitly request English output in the current prompt if needed.
- **papers-audit-metadata-consistency**
  - Audit consistency between logs and analysis notes.
- **papers-build-collection-index**
  - Rebuild `paperCollection/index.jsonl` (agent index) and `paperCollection/` Obsidian navigation pages from `paperAnalysis/` frontmatter.

### 3. KB query and code context

- **papers-query-knowledge-base**
  - Query the knowledge base primarily from `paperAnalysis/`; when present, reads `paperCollection/index.jsonl` first for fast filtering. Includes code-context mode for pre-coding paper retrieval. Also handles comparison requests with honest text-based analysis.
- **code-context-paper-retrieval** *(alias — routes to papers-query-knowledge-base code-context mode)*
  - Retrieve paper evidence relevant to a coding task. Triggers before code modification.

### 4. Research ideation and review

- **idea-emerge**
  - Generate research idea candidates from local KB evidence, domain bottleneck diagnosis, task-core web papers, cross-domain operators, explicit decision rules, implementation traces, and task constraints.
  - Outputs domain bottleneck diagnosis, evidence ledger, cards, candidate score breakdown, `S3` hard gates, rejected or parked ideas, and next-skill handoff.
  - Use this before open-ended brainstorming when the input is evidence and constraints rather than a cleanly stated research question.
- **research-brainstorm-from-kb**
  - Turn a research question into structured idea notes using the local knowledge base.
- **idea-focus-coach** (independent)
  - Co-create and narrow broad ideas into focused goals, scope cuts, and next experiments.
- **reviewer-stress-test** (independent)
  - Run strict-but-fair ICLR/CVPR/SIGGRAPH reviewer-style questioning on an idea, roadmap, or full paper.

### 5. Utility skills

- **notes-export-share-version**
  - Convert internal notes into external-share Markdown versions.
- **skill-fit-guard**
  - Diagnose recurring skill mismatch after a skill call and ask whether that skill should be revised.
- **write-daily-log**
  - Generate or update a structured daily research log from git diffs, artifacts, and conversation context.
  - Core logic: collect evidence that changes future judgment (not activity logs), then compress into a fixed template.
  - Evidence channels: (A) current session context, (B) cross-session git diffs and filesystem artifact scan across all workspace repos.
  - Output sections: 今日进展 / 核心结论 / 问题与思考 / 明日任务.
  - Built-in consistency check: numbers must cite artifacts, old/new env results must not be mixed, 明日任务 must follow from conclusions.

### 6. Domain migration

- **domain-fork**
  - Migrate ResearchFlow's architecture to any professional domain, such as frontend development, accounting, or journalism.
  - Interactive flow: confirm the target domain -> define concept mapping -> adapt skills -> generate folders and README.

### 7. Choosing a skill

- Need to import papers from Zotero or a local PDF folder -> `papers-sync-from-zotero`
- Need candidate papers from sites or review pages -> `papers-collect-from-web`
- Need candidate papers from a GitHub repository (awesome lists, survey repos, etc.) -> `papers-collect-from-github-awesome`
- Already have candidate rows and need PDFs -> `papers-download-from-list`
- Already have PDFs and need analysis notes -> `papers-analyze-pdf`
- Changed or added notes and want refreshed indexes → `papers-build-collection-index`
- Need a metadata quality pass -> `papers-audit-metadata-consistency`
- Need to search, summarize, compare, or cite papers in prose -> `papers-query-knowledge-base`
- Need paper suggestions before coding -> `papers-query-knowledge-base` (code-context mode)
- Need to turn KB evidence, domain bottleneck diagnosis, task-core papers, cross-domain operators, and decision rules into idea candidates -> `idea-emerge`
- Need idea generation backed by the knowledge base while the direction is still open-ended -> `research-brainstorm-from-kb`
- Need collaborative narrowing from a broad-but-real idea to an executable plan -> `idea-focus-coach`
- Need strict reviewer-mode challenge with repair paths -> `reviewer-stress-test`
- A recently used skill seems repeatedly unfit -> `skill-fit-guard`
- Need to write or update a daily research log -> `write-daily-log`
- Want to create a domain-adapted version of ResearchFlow -> `domain-fork`
