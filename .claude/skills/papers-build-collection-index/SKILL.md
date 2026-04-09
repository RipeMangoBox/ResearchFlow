---
name: papers-build-collection-index
description: Builds/refreshes `paperCollection/` index notes from `paperAnalysis/` frontmatter. Use when the user asks to update/rebuild `paperCollection`, regenerate indexes (by task/technique/venue), or after adding/editing analysis notes and PDF refs.
---

# Build paperCollection

## What this skill does

Regenerates the Obsidian index layer under `paperCollection/` by scanning `paperAnalysis/**/*.md`, extracting frontmatter and body hashtags (`#tag`), and emitting:

- `paperCollection/README.md` (home)
- `paperCollection/_AllPapers.md` (grouped view)
- `paperCollection/by_task/*.md`
- `paperCollection/by_technique/_Index.md` + `paperCollection/by_technique/*.md`
- `paperCollection/by_venue/_Index.md` + `paperCollection/by_venue/*.md`

## Preconditions (what gets indexed)

Only analysis notes that look like “real papers” are indexed:

- Frontmatter includes `pdf_ref` that **starts with** `paperPDFs/`
- and `pdf_ref` **ends with** `.pdf`

Notes missing/invalid `pdf_ref` are skipped.

## Workflow

1. Run the builder from the vault root (the folder that contains `paperAnalysis/` and `paperCollection/`):

```bash
python .claude/skills/papers-build-collection-index/scripts/build_paper_collection.py
```

2. Tag explosion guard (built-in):
   - If technique tag count is over `500`, the builder first runs knowledge-base-wide tag normalization / consolidation from the colocated scripts directory, then rebuilds.
   - If tag count is still high, it also emits a tag audit report from the same colocated scripts directory for further cleanup.

3. Confirm it succeeded:
   - Expect console output like `[OK] papers: ...` and `[OK] output: .../paperCollection`.
   - Spot-check that `paperCollection/README.md` and `paperCollection/_AllPapers.md` are updated.

## How categories / tags are interpreted

- **Task (category)**: prefers the folder name under `paperAnalysis/` (e.g. `paperAnalysis/Motion_Generation_Text_Speech_Music_Driven/...` → task = `Motion_Generation_Text_Speech_Music_Driven`).
- **Technique tags**: merges frontmatter `tags` with body hashtags (`#tag`), then excludes:
  - the task itself
  - any tag starting with `status/`
  - heading / code-fence / inline-code noise

## Examples (triggers)

- "Refresh `paperCollection` / rebuild the index"
- "I just added several `paperAnalysis` notes; regenerate `paperCollection`"
- "The by_technique / by_venue pages are incomplete; fix and rebuild"
