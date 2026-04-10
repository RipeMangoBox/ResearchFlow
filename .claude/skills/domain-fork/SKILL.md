---
name: domain-fork
description: Migrates ResearchFlow's architecture to a new professional domain (e.g. frontend development, accounting, journalism). Interactive session with the user to map research concepts to domain equivalents, then generates a complete set of adapted skills and folder structure. Explicit trigger only.
---

# Domain Fork

## Purpose

Migrate the core ResearchFlow architecture (collect → download → analyze → build KB → query → compare → question discovery → ideation divergence → focus → review) into a user-specified professional domain, and generate a complete domain-adapted skill set plus folder structure in one pass.

## Trigger

**Explicit invocation only**. It should not be auto-triggered by description matching.

User must clearly request something like:
- "Fork a frontend development version of ResearchFlow"
- "Migrate ResearchFlow to the accounting domain"
- "Use domain-fork to create a journalism knowledge-base workflow"

## Interactive Flow

After invocation, enter an interactive confirmation flow with **no skipped steps**:

### Step 1: Domain confirmation

Confirm with the user:
- **Target domain**: e.g., "frontend development", "accounting/audit", "news production"
- **Repository name**: suggest `<Domain>Flow` (e.g., `FrontendFlow`, `AccountingFlow`, `JournalismFlow`), user can override
- **Save location**: if not specified, default to `ResearchFlow/<RepoName>/` and prompt:

> "Default output path is ResearchFlow/<RepoName>/. Do you want to change it?"

### Step 2: Concept mapping table

Generate a ResearchFlow → target-domain concept mapping table for user confirmation/edit:

| ResearchFlow concept | Mapped target-domain concept | Notes |
|-------------------|--------------|------|
| paper | (domain equivalent) | e.g., technical article, regulation document, news story |
| PDF | (domain equivalent) | e.g., webpage article, regulation PDF, manuscript document |
| venue | (domain equivalent) | e.g., technical blog/framework version, regulator source, media source |
| paperAnalysis | (domain equivalent) | e.g., technical notes, regulation interpretation, editorial analysis |
| paperCollection | (domain equivalent) | e.g., technical index, regulation index, topic index |
| paperPDFs | (domain equivalent) | e.g., original articles, source regulations, source manuscripts |
| paperIDEAs | (domain equivalent) | e.g., project proposals, audit strategies, topic plans |
| QuestionBank | (domain equivalent) | e.g., technical problem bank, compliance issue bank, editorial issue bank |
| core_operator | (domain equivalent) | e.g., core technical strategy, core regulation clause, core news angle |
| primary_logic | (domain equivalent) | e.g., implementation flow, compliance-check flow, editorial flow |
| analysis_log.csv | keep | main tracking log, with domain-adjusted columns |
| state: Wait→Downloaded→checked | keep or adapt | state machine may need domain adaptation |

### Step 3: Skill mapping confirmation

Show how ResearchFlow's 17 skills map to target-domain skills:

| ResearchFlow Skill | Target-domain Skill | Keep? | Adjustment |
|-------------------|---------------|---------|---------|
| papers-collect-from-web | `<domain>-collect-from-web` | ✅ | source websites become domain-specific |
| papers-collect-from-github-awesome | `<domain>-collect-from-curated-list` | ✅/❌ | depends on whether curated lists exist |
| papers-download-from-list | `<domain>-download-from-list` | ✅ | download objects become domain documents |
| pdfs-compress-large-files | `compress-large-files` | ✅/❌ | depends on document types |
| papers-analyze-pdf | `<domain>-analyze-document` | ✅ | analysis template rewritten for domain |
| papers-audit-metadata-consistency | `<domain>-audit-metadata` | ✅ | metadata fields adjusted for domain |
| papers-build-collection-index | `<domain>-build-index` | ✅ | index dimensions adjusted for domain |
| papers-query-knowledge-base | `<domain>-query-kb` | ✅ | query dimensions adjusted for domain |
| papers-compare-table | `<domain>-compare-table` | ✅ | comparison fields adjusted for domain |
| code-context-paper-retrieval | `code-context-<domain>-retrieval` | ✅/❌ | keep only for code-related domains |
| research-question-bank | `<domain>-question-bank` | ✅ | question dimensions adjusted for domain |
| research-brainstorm-from-kb | `<domain>-brainstorm-from-kb` | ✅ | ideation dimensions adjusted for domain |
| idea-focus-coach | `<domain>-focus-coach` | ✅ | focus dimensions adjusted for domain |
| reviewer-stress-test | `<domain>-stress-test` | ✅ | review standards adjusted for domain |
| research-workflow | `<domain>-workflow` | ✅ | stage naming adjusted for domain |
| notes-export-share-version | `notes-export-share-version` | ✅ | generic, no change needed |
| skill-fit-guard | `skill-fit-guard` | ✅ | generic, no change needed |

User may:
- remove unnecessary skills
- rename skills
- adjust mapping relationships

### Step 4: Confirm and generate

After user confirmation, generate in one pass:

1. **Directory structure**

```
<RepoName>/
├── .claude/
│   ├── skills-config.json
│   └── skills/
│       ├── User_README.md
│       ├── README.md
│       ├── STATE_CONVENTION.md
│       └── <all mapped skills>/
├── <AnalysisDir>/          # maps from paperAnalysis
│   └── tracking_log.csv    # maps from analysis_log.csv
├── <CollectionDir>/        # maps from paperCollection
│   ├── by_<dim1>/
│   ├── by_<dim2>/
│   └── by_<dim3>/
├── <SourceDir>/            # maps from paperPDFs
├── <IdeaDir>/              # maps from paperIDEAs
├── QuestionBank/
└── README.md
```

2. **SKILL.md for each skill**: rewritten from corresponding ResearchFlow skill with all domain terminology replaced
3. **skills-config.json**: register all generated skills
4. **STATE_CONVENTION.md**: state transitions adapted for the target domain
5. **User_README.md / README.md**: domain-adapted navigation and instructions
6. **tracking_log.csv**: empty template with domain-adjusted column names
7. **Repository README.md**: domain-adapted overview

## Generation Principles

1. **Structural symmetry**: keep the collect → build → use three-stage architecture
2. **Terminology consistency**: use one consistent term per concept across all skills
3. **Minimum viable generation**: generate only skill definitions (`SKILL.md`), not scripts (agent writes scripts on demand)
4. **State machine continuity**: keep the three-state main flow `Wait → Downloaded → checked`, with optional domain-specific labels
5. **Do not modify ResearchFlow**: domain-fork reads ResearchFlow as template only and must not edit any ResearchFlow file

## Boundaries

- Generate skill definitions and folder structure only; do not generate real content data
- Do not auto-populate `tracking_log.csv` (that happens when users use the new repository)
- Do not copy ResearchFlow `paperAnalysis` / `paperCollection` / `paperPDFs` data
- Do not copy `.obsidian.zip` (user can copy/adjust it manually if needed)

## Example

User: "Fork a frontend development version"

Step 1 confirmation:
- domain: frontend development
- repo name: FrontendFlow
- path: ResearchFlow/FrontendFlow/

Step 2 mapping:
- paper → technical article/spec document
- PDF → webpage article/PDF spec
- venue → technical blog/framework version/W3C standard
- paperAnalysis → articleAnalysis
- paperCollection → articleCollection
- paperPDFs → articleSources
- core_operator → core technical strategy
- primary_logic → implementation flow

Step 3 skill mapping:
- `papers-collect-from-web` → `articles-collect-from-web`
- `papers-collect-from-github-awesome` → `articles-collect-from-curated-list`
- `code-context-paper-retrieval` → `code-context-article-retrieval` (kept; frontend is code-centric)
- `reviewer-stress-test` → `code-review-stress-test` (mapped to code review perspective)
- ...

Step 4: generate the complete `FrontendFlow/` structure.
