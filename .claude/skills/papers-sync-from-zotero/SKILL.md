---
name: papers-sync-from-zotero
description: "Syncs papers from a Zotero library into RF's structured layout via Zotero Web API / local API, writes rich metadata + annotations to `paperAnalysis/processing/zotero/manifest.jsonl`, appends lightweight rows to `analysis_log.csv`, and can backfill Zotero highlights/comments into existing analysis notes."
---

# Papers Sync From Zotero

## Purpose

Bridge Zotero (bibliographic source of truth) into ResearchFlow (analytical source of truth) by syncing papers, PDFs, and metadata into RF's structured layout.

ResearchFlow should not replicate Zotero. Zotero handles collection, metadata cleaning, PDF/annotation, and citation styles. RF handles structured analysis, comparison, brainstorming, and agent retrieval. This skill connects the two.

## Pipeline position

```
Zotero library
  -> papers-sync-from-zotero (this skill)
    -> paperPDFs/<Category>/<Venue_Year>/<Year>_<Title>.pdf
    -> paperAnalysis/processing/zotero/manifest.jsonl (rich metadata)
    -> paperAnalysis/analysis_log.csv (lightweight rows)
  -> papers-analyze-pdf
  -> papers-build-collection-index
  -> papers-query-knowledge-base
```

## Two operating modes

### Mode A: Zotero API sync (primary, implemented)

Uses `.claude/skills/papers-sync-from-zotero/scripts/zotero_to_rf.py sync` against Zotero Web API or local API.

Advantages:
- Complete metadata (title, authors, venue, DOI, tags, collections, dates, abstract)
- Incremental sync via `since` / `version` parameters and `sync_state.json`
- Identity tracking via `zotero_key` + `zotero_library_id`
- Attachment-level annotation extraction
- Dedup against both Zotero manifest and `analysis_log.csv`
- Supports `--repo-root` so sync can target an isolated sub-vault such as `MyVault/Zotero`

Requirements:
- Zotero API key (for Web API) or Zotero running locally (for local API)
- User provides: library type + library ID + API key, or confirms local Zotero is running

### Mode B: Annotation backfill into RF notes (post-sync, implemented)

Uses `.claude/skills/papers-sync-from-zotero/scripts/zotero_to_rf.py append-annotations`.

Advantages:
- Reuses the manifest produced in Mode A
- Adds Zotero highlights / comments / notes after Part III in existing analysis Markdown
- Idempotent: rerun replaces the Zotero appendix instead of duplicating it

## Input

User provides:

**Mode A (API sync):**
- **Library type**: `user` or `group`
- **Library ID**: Zotero library ID (for local user library, use `0`)
- **API key**: Zotero API key (for Web API) — or confirm local Zotero is running
- **Collection filter** (optional): sync only a specific Zotero collection
- **Category hint** (optional): override RF category for all items in this sync
- **Tag filter** (optional): sync only items with specific Zotero tags
- **Append annotations to notes** (optional): whether to immediately append Zotero highlights/comments into existing analysis `.md`
- **Repo root** (optional): isolated target root for all RF outputs; recommended when the host vault already contains another RF knowledge base

> **Security recommendation:** Prefer the local Zotero API when possible. If Web API is used, create a **read-only** API key in Zotero settings (Settings → Feeds/API → Create new private key → uncheck "Allow write access"). Never store the API key in committed files. The agent should not echo the key in output or logs.

**Mode B (append annotations):**
- Existing `paperAnalysis/processing/zotero/manifest.jsonl`
- Existing RF analysis notes under `paperAnalysis/`
- Optional filter: one `analysis_md` path or one `pdf_ref`

## Output

### 1. PDFs in RF structure

```
paperPDFs/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.pdf
```

Naming convention:
- SanitizedTitle: words joined by `_`, no spaces/commas/hyphens/colons
- Example: `2025_MotionDiffuse_Text_Driven_Human_Motion_Generation.pdf`
- Venue_Year: `CVPR_2025`, `ICLR_2026`, `arXiv_2025`, `unknown_2026`

### 2. Rich metadata manifest

```
paperAnalysis/processing/zotero/manifest.jsonl
```

One JSON object per line, containing all Zotero metadata:

```json
{
  "zotero_key": "ABC12345",
  "zotero_library_id": "12345",
  "zotero_version": 42,
  "title": "MotionDiffuse: Text-Driven Human Motion Generation",
  "authors": ["Zhang, M.", "Cai, Z.", "Pan, L."],
  "year": 2025,
  "venue": "CVPR",
  "doi": "10.1109/CVPR.2025.xxxxx",
  "abstract": "We present MotionDiffuse...",
  "citekey": "zhang2025motiondiffuse",
  "zotero_tags": ["motion-generation", "diffusion"],
  "zotero_collections": ["Motion Generation"],
  "pdf_source_path": "/home/user/Zotero/storage/ABC12345/Zhang_2025.pdf",
  "rf_pdf_path": "paperPDFs/Motion_Generation/CVPR_2025/2025_MotionDiffuse_Text_Driven_Human_Motion_Generation.pdf",
  "rf_analysis_path": "paperAnalysis/Motion_Generation/CVPR_2025/2025_MotionDiffuse_Text_Driven_Human_Motion_Generation.md",
  "rf_category": "Motion_Generation_Text_Speech_Music_Driven",
  "rf_sort": "Motion_Generation",
  "pdf_import_source": "attachment_url_fallback",
  "annotation_count": 3,
  "annotations": [
    {
      "annotation_key": "WA3J78PQ",
      "annotation_type": "highlight",
      "page_label": "4",
      "annotation_text": "Diffusion in latent motion space enables fast generation.",
      "annotation_comment": "Potential reusable operator."
    }
  ],
  "synced_at": "2026-04-13T18:00:00",
  "sync_version": 1
}
```

### 3. Lightweight rows in analysis_log.csv

```
state,importance,paper_title,venue,project_link_or_github_link,paper_link,sort,pdf_path
```

| Column | Value |
|--------|-------|
| state | `Downloaded` |
| importance | (empty) |
| paper_title | from Zotero metadata |
| venue | `<Venue> <Year>` (e.g. `CVPR 2025`) |
| project_link_or_github_link | (empty — fill later) |
| paper_link | DOI link or arXiv link if available |
| sort | derived from RF category |
| pdf_path | RF-convention path |

## Workflow

### Mode A: Zotero API sync

**Step 1: Connect and authenticate**
- Use `zotero_to_rf.py sync`
- Connect to Zotero library using Web API or local API
- Verify connection by fetching collections / first page of items

**Step 2: Fetch items (incremental)**
- Read last sync version from `paperAnalysis/processing/zotero/sync_state.json`:
  ```json
  {"library_id": "12345", "last_version": 42, "last_synced": "2026-04-13T18:00:00"}
  ```
- If first sync: fetch all items
- If incremental: fetch only items with `version > last_version` using `since` parameter
- Apply collection/tag filters if specified
- Keep only paper-like item types by default: `conferencePaper`, `journalArticle`, `preprint`
- Require a PDF attachment; skip metadata-only entries
- Fetch attachment children to collect Zotero annotations / comments
- If the chosen attachment is `linked_file`, metadata and annotations may still be readable even though `/items/<attachment>/file` returns `404`

**Step 3: Map metadata to RF structure**
- For each Zotero item:
  - Map `itemType` + `publicationTitle` / `conferenceName` / `proceedingsTitle` → RF `venue`
  - Map Zotero `tags` + `collections` → RF `category` (configurable mapping table, plus fuzzy match to existing RF categories)
  - Build canonical RF paths
  - Dedup against existing manifest + `analysis_log.csv` by `zotero_key`, DOI, arXiv ID, `pdf_path`, and normalized `title + year`

**Step 4: Download/copy PDFs**
- Locate the preferred PDF attachment
- Download attachment to RF path if the canonical target does not already exist
- If Zotero Web API file download fails and the attachment `url` looks like a direct PDF URL, fall back to downloading from that URL
- Never move or mutate the Zotero original
- Skip ambiguous collisions instead of overwriting unknown local files

**Step 5: Write manifest + log**
- Upsert rich metadata to `paperAnalysis/processing/zotero/manifest.jsonl`
- Write per-item failures to `paperAnalysis/processing/zotero/sync_errors.jsonl`
- Upsert lightweight row to `paperAnalysis/analysis_log.csv`
- Preserve existing non-empty log fields; fill blanks instead of clobbering
- Set log state to `checked` if the analysis note already exists, otherwise `Downloaded`
- Update `sync_state.json` with new version number

### Mode B: append annotations into existing analysis notes

**Step 1: Read manifest**
- Load `paperAnalysis/processing/zotero/manifest.jsonl`
- Select all records with `annotations` / `zotero_notes`

**Step 2: Find corresponding analysis note**
- Use `rf_analysis_path` from the manifest
- Skip missing `.md` files; report them for later re-run after `papers-analyze-pdf`

**Step 3: Insert Zotero appendix**
- Write a new section after Part III and before the local PDF section
- Heading:
  - `zh`: `## 附：Zotero 高亮与笔记`
  - `en`: `## Appendix: Zotero Highlights & Notes`
- Re-running replaces the old appendix instead of duplicating it

## Category mapping

For Mode A, Zotero collections/tags are mapped to RF categories. Default mapping can be configured in `paperAnalysis/processing/zotero/category_map.json`:

```json
{
  "zotero_collection_to_rf_category": {
    "Motion Generation": "Motion_Generation_Text_Speech_Music_Driven",
    "Video Generation": "Video_Generation",
    "3D Gaussian Splatting": "3D_Gaussian_Splatting"
  },
  "zotero_tag_to_rf_category": {
    "motion-gen": "Motion_Generation_Text_Speech_Music_Driven",
    "video-diffusion": "Video_Generation"
  },
  "default_category": "Uncategorized"
}
```

If no mapping matches, agent infers from title/abstract keywords or asks user.

## Optional note enrichment

After sync, the following optional fields become available for `papers-analyze-pdf` or post-processing note enrichment:

| Field | Source | Required? |
|-------|--------|-----------|
| authors | Zotero API | optional |
| doi | Zotero API | optional |
| citekey | Better BibTeX / Zotero | optional |
| zotero_key | Zotero API | optional |
| abstract | Zotero API | optional |
| annotations | Zotero attachment children | optional |
| zotero_notes | Zotero child notes | optional |

These fields are stored in `manifest.jsonl`. `annotations` and `zotero_notes` are intended for the optional body appendix, not for required frontmatter.

## Constraints

- **Copy, never move** source PDFs (Zotero library stays intact)
- **Incremental by default** (Mode A) — only sync new/updated items
- **No Zotero database direct access** — use API only (Zotero docs explicitly warn against direct SQLite access)
- **No PDF analysis** — that is `papers-analyze-pdf`
- **Idempotent** — re-running updates the manifest and replaces the Zotero appendix instead of duplicating it
- **analysis_log.csv stays lightweight** — rich metadata lives in manifest.jsonl
- **Web API caveat** — `linked_file` attachments often expose annotations/metadata but not the local file bytes; expect `404` on `/file` and use local API or URL fallback when possible

## Relationship to other skills

| Skill | Role | Boundary |
|-------|------|----------|
| `papers-sync-from-zotero` (this) | Source adapter: Zotero -> RF structure | Stops after PDF copy + manifest + log append; optional appendix write if note already exists |
| `papers-collect-from-github-awesome` | Source adapter: GitHub repos -> analysis_log.csv | Collects metadata only, no PDFs |
| `papers-collect-from-web` | Source adapter: web pages -> analysis_log.csv | Collects metadata only, no PDFs |
| `papers-download-from-list` | Downloads PDFs from URLs in triage lists | Handles online sources, not local files |
| `papers-analyze-pdf` | Analyzes PDFs into structured Markdown notes | Assumes PDFs already at RF paths |

## Typical Usage

> `python .claude/skills/papers-sync-from-zotero/scripts/zotero_to_rf.py sync --local-api --library-type user --library-id 0 --append-annotations-to-md`

> `python .claude/skills/papers-sync-from-zotero/scripts/zotero_to_rf.py sync --library-type user --library-id 12345 --api-key xxx --collection "Video Generation"`

> `python .claude/skills/papers-sync-from-zotero/scripts/zotero_to_rf.py append-annotations`
