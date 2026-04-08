## ResearchFlow Skills Overview (.claude/skills)

This skills directory supports a local paper workflow covering **collect → download → analyze → build → query → compare → ideate → focus → review**.

### 1. Workflow entry

- **research-workflow**
  - Routes work to one stage among collect / download / analyze / build / query / compare / ideate / focus / review / audit / export.
  - Returns current stage, required inputs, suggested command/skill, outputs, and next step.

### 2. Paper pipeline skills (KB 构建)

- **papers-collect-from-web**
  - Collect candidate papers from web pages and generate a triage list under `paperAnalysis/`.
- **papers-collect-from-github-awesome**
  - Parse GitHub curated lists into an `analysis_log.csv`-aligned candidate list.
- **papers-download-from-list**
  - Download, verify, repair, and deduplicate PDFs into `paperPDFs/`.
- **pdfs-compress-large-files**
  - Compress oversized PDFs in `paperPDFs/`.
- **papers-analyze-pdf**
  - Convert local PDFs into structured `paperAnalysis/*.md` notes.
- **papers-audit-metadata-consistency**
  - Audit consistency between logs and analysis notes.
- **papers-build-collection-index**
  - Rebuild `paperCollection/` from `paperAnalysis/` frontmatter.

### 3. KB query & code context

- **papers-query-knowledge-base**
  - Query the knowledge base by task, technique, and venue.
- **papers-compare-table**
  - Generate structured comparison tables for multiple papers.
- **code-context-paper-retrieval**
  - Retrieve paper evidence relevant to a coding task. Triggers BEFORE code modification.

### 4. Research ideation & review

- **research-brainstorm-from-kb**
  - Turn a research question into structured idea notes using the local knowledge base.
- **research-question-bank**
  - Generate a structured question/challenge list for a research direction, grounded in KB and web search. Saves to `QuestionBank/`.
- **idea-focus-coach** (independent)
  - Co-create and narrow broad ideas into focused goals, scope cuts, and next experiments.
- **reviewer-stress-test** (independent)
  - Run strict-but-fair ICLR/CVPR/SIGGRAPH reviewer-style questioning on idea/roadmap/full paper.

### 5. Utility skills

- **notes-export-share-version**
  - Convert internal notes into external-share Markdown versions.
- **skill-fit-guard**
  - Diagnose recurring skill mismatch after a skill call and ask whether to revise that skill.

### 6. Domain migration

- **domain-fork**
  - Migrate ResearchFlow's architecture to any professional domain (frontend dev, accounting, journalism, etc.)
  - Interactive: confirms domain → concept mapping → skill adaptation → folder generation

### 7. Choosing a skill

- Need candidate papers from sites or review pages → `papers-collect-from-web`
- Need candidate papers from a GitHub curated repo → `papers-collect-from-github-awesome`
- Already have candidate rows and need PDFs → `papers-download-from-list`
- Already have PDFs and need analysis notes → `papers-analyze-pdf`
- Changed or added notes and need indexes → `papers-build-collection-index`
- Need a metadata quality pass → `papers-audit-metadata-consistency`
- Need to search, compare, or cite papers → `papers-query-knowledge-base`
- Need a side-by-side comparison table → `papers-compare-table`
- Need paper suggestions before coding → `code-context-paper-retrieval`
- Need idea generation backed by the knowledge base → `research-brainstorm-from-kb`
- Need to map out questions/challenges for a research direction → `research-question-bank`
- Need collaborative narrowing from broad idea to executable plan → `idea-focus-coach`
- Need strict reviewer-mode challenge with repair paths → `reviewer-stress-test`
- A recently used skill seems repeatedly unfit → `skill-fit-guard`
- Want to create a domain-adapted version of ResearchFlow → `domain-fork`
