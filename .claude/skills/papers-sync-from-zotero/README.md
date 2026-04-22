# papers-sync-from-zotero

## Overview

Syncs papers from a Zotero library into ResearchFlow's structured layout through `.claude/skills/papers-sync-from-zotero/scripts/zotero_to_rf.py`.

- **Mode A (recommended)**: Zotero Web API or local Zotero API
- **Mode B (lightweight follow-up)**: append Zotero highlights / comments into existing RF analysis notes from the manifest produced by Mode A

## Quick start

Local Zotero API:

```bash
python .claude/skills/papers-sync-from-zotero/scripts/zotero_to_rf.py sync \
  --repo-root /path/to/isolated/Zotero \
  --local-api \
  --library-type user \
  --library-id 0 \
  --append-annotations-to-md
```

Web API:

```bash
python .claude/skills/papers-sync-from-zotero/scripts/zotero_to_rf.py sync \
  --repo-root /path/to/isolated/Zotero \
  --library-type user \
  --library-id <YOUR_LIBRARY_ID> \
  --api-key <READ_ONLY_KEY> \
  --collection "Video Generation"
```

Backfill Zotero annotations into already-written analysis notes:

```bash
python .claude/skills/papers-sync-from-zotero/scripts/zotero_to_rf.py append-annotations
```

## What it does

1. Connects to Zotero Web API or local Zotero API
2. Keeps only paper-like item types by default: `conferencePaper`, `journalArticle`, `preprint`
3. Resolves PDF attachments, copies them into `paperPDFs/<Category>/<Venue_Year>/`
4. Deduplicates against existing Zotero manifest records and `paperAnalysis/analysis_log.csv`
5. Writes rich metadata, annotations, and Zotero notes to `paperAnalysis/processing/zotero/manifest.jsonl`
6. Appends or repairs lightweight rows in `paperAnalysis/analysis_log.csv`
7. Optionally appends Zotero highlights/comments into existing analysis notes after Part III
8. Records per-item sync failures to `paperAnalysis/processing/zotero/sync_errors.jsonl`

## What it does NOT do

- Does not analyze PDFs (use `papers-analyze-pdf` after sync)
- Does not modify or move Zotero originals
- Does not access Zotero SQLite directly (API only)
- Does not auto-run `papers-build-collection-index`

## Output files

| File | Content |
|------|---------|
| `paperPDFs/<Category>/<Venue_Year>/<Year>_<Title>.pdf` | Copied PDFs in RF naming convention |
| `paperAnalysis/processing/zotero/manifest.jsonl` | Rich metadata, dedup keys, annotations, Zotero notes |
| `paperAnalysis/processing/zotero/sync_state.json` | Incremental sync state |
| `paperAnalysis/processing/zotero/sync_errors.jsonl` | Per-item failures encountered during the last sync run |
| `paperAnalysis/processing/zotero/category_map.json` | Zotero collection/tag -> RF category mapping |
| `paperAnalysis/analysis_log.csv` | Lightweight rows (state=Downloaded) |

## After sync

1. Run `papers-analyze-pdf` on the newly synced `Downloaded` entries.
2. If you did not use `--append-annotations-to-md` during sync, run `append-annotations` later to backfill Zotero reading traces into the generated Markdown notes.
3. Run `papers-build-collection-index` if you want refreshed indexes.

## Real-world notes

- If you want to avoid polluting an existing vault root, point `--repo-root` at an isolated sub-vault such as `MyVault/Zotero`.
- A Zotero attachment with `linkMode=linked_file` often returns `404` on the Zotero Web API `/file` endpoint even though metadata and annotations are readable.
- The sync script now falls back to the attachment `url` when it looks like a direct PDF URL. This recovers many arXiv / OpenReview / CVF PDFs while still preserving Zotero annotations in the manifest.
- Some publisher-hosted PDFs still fail on fallback due to access control (`403` from ACM / other paywalled sites). These items are kept in `sync_errors.jsonl` for manual handling or local Zotero sync.
