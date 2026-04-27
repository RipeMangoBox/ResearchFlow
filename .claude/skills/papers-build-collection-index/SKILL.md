---
name: papers-build-collection-index
follows: rf-obsidian-markdown
description: Builds/refreshes `paperCollection/index.jsonl` (agent index) and `paperCollection/by_venue/` Obsidian navigation pages from `paperAnalysis/` frontmatter. Use when the user asks to update/rebuild the agent index, regenerate venue pages, or after adding/editing analysis notes and PDF refs. Task & technique browsing now lives in Obsidian tag/backlink navigation.
---

# Build Index

## What this skill does

Regenerates two index layers by scanning `paperAnalysis/**/*.md` and extracting frontmatter and body hashtags (`#tag`):

1. **Agent index** — `paperCollection/index.jsonl`: one JSONL line per paper with retrieval-dimension fields only. Agents read this first to narrow thousands of notes down to a small candidate set.
2. **Obsidian navigation** — `paperCollection/`: Markdown navigation pages for human browsing in Obsidian.

This skill is the single writer for both outputs. `papers-analyze-pdf` does not append to `index.jsonl`.

### paperCollection/index.jsonl format

```jsonl
{"path":"paperAnalysis/Cat/Venue_Year/Year_Title.md","title":"...","category":"Cat","venue":"CVPR","year":2026,"tags":["task/x","technique"],"core_operator":"...","primary_logic":"...","claims_count":2,"pdf_ref":"paperPDFs/Cat/Venue_Year/Year_Title.pdf"}
```

### paperCollection/ output

- `paperCollection/README.md` (home)
- `paperCollection/_AllPapers.md` (grouped view)
- `paperCollection/by_venue/_Index.md` + `paperCollection/by_venue/*.md`

> `by_task/` and `by_technique/` were retired. Task pages and technique pages
> were duplicating what Obsidian's built-in tag panel and backlink view
> already provide off the `paperAnalysis/` frontmatter. The builder now
> deletes any leftover `by_task/` / `by_technique/` directories on each run.

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
   - Spot-check that `paperCollection/index.jsonl`, `paperCollection/README.md` and `paperCollection/_AllPapers.md` are updated.

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
