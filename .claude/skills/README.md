## ResearchFlow Skills Overview

Single source of truth: keep the maintained skill library in `.claude/skills`.
If you also want Codex-compatible paths, run `python3 scripts/setup_shared_skills.py` on macOS/Linux or `py -3 scripts\setup_shared_skills.py` on Windows to generate `.codex/skills` aliases without copying.

This skills directory supports the local paper workflow covering **collect -> download -> analyze -> build -> query -> compare -> ideate -> focus -> review**.

### 1. Workflow entry

- **research-workflow**
  - Routes work to one stage among collect / download / analyze / build / query / compare / ideate / focus / review / audit / export.
  - Returns the current stage, required inputs, suggested command or skill, expected outputs, and next step.

### 2. Paper pipeline skills

- **papers-collect-from-web**
  - Collect candidate papers from web pages and generate a triage list under `paperAnalysis/`.
- **papers-collect-from-github-awesome**
  - Parse GitHub curated lists into an `analysis_log.csv`-aligned candidate list.
- **papers-download-from-list**
  - Download, verify, repair, and deduplicate PDFs into `paperPDFs/`.
- **papers-analyze-pdf**
  - Convert local PDFs into structured `paperAnalysis/*.md` notes.
- **papers-audit-metadata-consistency**
  - Audit consistency between logs and analysis notes.
- **papers-build-collection-index**
  - Rebuild `paperCollection/` from `paperAnalysis/` frontmatter for statistics, Obsidian navigation, and backlink-friendly browsing.

### 3. KB query and code context

- **papers-query-knowledge-base**
  - Query the knowledge base primarily from `paperAnalysis/`, with `paperCollection/` as optional navigation support.
- **papers-compare-table**
  - Generate structured comparison tables for design decisions, Related Work writing, baseline selection, or method overviews.
- **code-context-paper-retrieval**
  - Retrieve paper evidence relevant to a coding task. Triggers before code modification.

### 4. Research ideation and review

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

### 6. Domain migration

- **domain-fork**
  - Migrate ResearchFlow's architecture to any professional domain, such as frontend development, accounting, or journalism.
  - Interactive flow: confirm the target domain -> define concept mapping -> adapt skills -> generate folders and README.

### 7. Choosing a skill

- Need candidate papers from sites or review pages -> `papers-collect-from-web`
- Need candidate papers from a GitHub curated repository -> `papers-collect-from-github-awesome`
- Already have candidate rows and need PDFs -> `papers-download-from-list`
- Already have PDFs and need analysis notes -> `papers-analyze-pdf`
- Changed or added notes and want refreshed statistics / navigation pages -> `papers-build-collection-index`
- Need a metadata quality pass -> `papers-audit-metadata-consistency`
- Need to search, summarize, or cite papers -> `papers-query-knowledge-base`
- Need a side-by-side comparison table -> `papers-compare-table`
- Need paper suggestions before coding -> `code-context-paper-retrieval`
- Need idea generation backed by the knowledge base while the direction is still open-ended -> `research-brainstorm-from-kb`
- Need collaborative narrowing from a broad-but-real idea to an executable plan -> `idea-focus-coach`
- Need strict reviewer-mode challenge with repair paths -> `reviewer-stress-test`
- A recently used skill seems repeatedly unfit -> `skill-fit-guard`
- Want to create a domain-adapted version of ResearchFlow -> `domain-fork`
