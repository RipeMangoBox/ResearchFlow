---
name: papers-build-collection-index
follows: rf-obsidian-markdown
description: Builds/refreshes `paperCollection/index.jsonl` (agent index) and `paperCollection/` Obsidian navigation pages from `paperAnalysis/` frontmatter. Use when the user asks to update/rebuild indexes, regenerate navigation pages (by task/technique/venue), or after adding/editing analysis notes and PDF refs.
---

# Build Index

## What this skill does

Regenerates two index layers by scanning `paperAnalysis/**/*.md` and extracting frontmatter and body hashtags (`#tag`):

1. **Agent index** — `paperCollection/index.jsonl`: one JSONL line per paper with retrieval-dimension fields only. Agents read this first to narrow thousands of notes down to a small candidate set.
2. **Obsidian navigation** — `paperCollection/`: Markdown navigation pages for human browsing in Obsidian.

This skill is the single writer for both outputs. `papers-analyze-pdf` does not append to `index.jsonl`.

This skill assumes the note/PDF path layer is already canonical. It does **not** repair stale paths by itself.

### paperCollection/index.jsonl format

```jsonl
{"path":"paperAnalysis/Cat/Venue_Year/Year_Title.md","title":"...","category":"Cat","venue":"CVPR","year":2026,"tags":["task/x","technique"],"core_operator":"...","primary_logic":"...","claims_count":2,"pdf_ref":"paperPDFs/Cat/Venue_Year/Year_Title.pdf"}
```

### paperCollection/ output

- `paperCollection/README.md` (home)
- `paperCollection/_AllPapers.md` (grouped view)
- `paperCollection/by_task/*.md`
- `paperCollection/by_technique/_Index.md` + `paperCollection/by_technique/*.md`
- `paperCollection/by_venue/_Index.md` + `paperCollection/by_venue/*.md`

## Preconditions (what gets indexed)

Only analysis notes that look like “real papers” are indexed:

- Frontmatter includes `pdf_ref` that **starts with** `paperPDFs/`
- and `pdf_ref` **ends with** `.pdf`
- and `pdf_ref` points to the **post-repair canonical location** if files were renamed or normalized

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
   - Spot-check that `paperCollection/index.jsonl`, `paperCollection/README.md` and `paperCollection/_AllPapers.md` are updated.

## Required ordering after path repair

When papers were renamed, moved out of `Unknown*` directories, or had `pdf_ref` changed:

1. finish the full path synchronization first:
   - `paperPDFs/`
   - `paperAnalysis/`
   - note frontmatter / embeds
   - `analysis_log.csv`
   - any derived CSV caches
2. then run this builder

Do **not** hand-edit `paperCollection/index.jsonl` or navigation pages as a substitute for fixing the underlying note/PDF paths. This skill is the canonical refresh step after repair.

## How categories / tags are interpreted

- **Task (category)**: prefers the folder name under `paperAnalysis/` (e.g. `paperAnalysis/Motion_Generation_Text_Speech_Music_Driven/...` → task = `Motion_Generation_Text_Speech_Music_Driven`).
- **Technique tags**: merges frontmatter `tags` with body hashtags (`#tag`), then excludes:
  - the task itself
  - any tag starting with `status/`
  - heading / code-fence / inline-code noise

## Examples (triggers)

- "Refresh indexes / rebuild the index"
- "I just added several `paperAnalysis` notes; regenerate indexes"
- "The by_technique / by_venue pages are incomplete; fix and rebuild"
- "Update `paperCollection/index.jsonl` for agent retrieval"
