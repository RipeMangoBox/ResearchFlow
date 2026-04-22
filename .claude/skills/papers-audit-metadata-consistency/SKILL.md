---
name: papers-audit-metadata-consistency
follows: rf-obsidian-markdown
description: Runs a unified metadata consistency audit across paperAnalysis logs and analysis notes, checking title/venue/year/category consistency, link/pdf_ref completeness, duplicates, and structure anomalies; writes quality_report_*.md under paperAnalysis.
---

# Metadata Consistency Audit

## What this skill does

Runs one unified audit entry for step5 quality governance:

- checks `paperAnalysis/*.txt` list/log entries;
- checks `paperAnalysis/**/*.md` analysis notes;
- checks cross-layer consistency between log/list and analysis notes;
- outputs `paperAnalysis/quality_report_*.md` (optionally `.json`).

## Coverage

This skill covers **structural and metadata hygiene only**.

- title / venue / year / category consistency
- link completeness (`paper_link`, `project_link`/`github_link`, `pdf_ref`)
- path-layer consistency for local assets and notes:
  - canonical `paperPDFs/<Category>/<Venue_Year>/<Year_Title>.pdf`
  - canonical `paperAnalysis/<Category>/<Venue_Year>/<Year_Title>.md`
  - stale `UnknownYear_*`, `Unknown_<year>`, and `paperPDFs/Unknown/...` path segments
- duplicate paper detection in txt logs
- analysis note structure anomalies (missing frontmatter keys, missing Part I/II/III)
- unmatched records (`only-in-txt`, `only-in-md`)

## Current limits

- does not evaluate whether conclusions are sufficiently supported by evidence
- does not score citation quality in idea notes or share notes
- does not audit completeness of comparison tables or downstream summaries
- does not detect stale analyses that should be refreshed
- broader semantic-layer governance belongs to a future knowledge-hygiene / evidence-governance extension

## Path Repair Protocol

When the user asks to **fix** metadata or path issues discovered by the audit, use the following generic repair protocol.

This skill remains the **diagnosis entry**. The repair itself must follow these rules exactly:

1. Determine canonical target paths first:
   - PDF: `paperPDFs/<CurrentCategory>/<Venue_Year>/<Year_Title>.pdf`
   - note: `paperAnalysis/<CurrentCategory>/<Venue_Year>/<Year_Title>.md`
2. Preserve the **current top-level category directory** by default.
   - Do **not** reclassify papers across categories based only on frontmatter `category`, tags, or a secondary source.
   - Only move across categories if the user explicitly requests taxonomy cleanup.
3. Distinguish metadata dirt from literal title text.
   - `UnknownYear_*`, `Unknown_<year>`, and `paperPDFs/Unknown/...` are path anomalies and should be normalized.
   - A paper title that literally contains the word `Unknown` is **not** a path anomaly and must not be renamed away.
4. Synchronize all affected layers together. A path repair is incomplete unless it updates all of:
   - the physical PDF path under `paperPDFs/`
   - the physical note path under `paperAnalysis/`
   - note frontmatter `venue`, `year`, and `pdf_ref`
   - note body embeds / local wikilinks that point to the moved PDF or note
   - every affected `analysis_log.csv` row's `venue` and `pdf_path`
   - derived caches such as `paperAnalysis/processing/unified_paper_index.csv` and `paperAnalysis/processing/unified_paper_duplicates.csv`
5. Handle collisions conservatively.
   - If the canonical PDF target already exists and is the same paper, keep the canonical target and delete the stale duplicate.
   - Treat PDFs as duplicates when hash matches; if hash differs but size and first-page text match, treat them as the same paper unless the user asks for archival retention.
   - If the canonical note target already exists, prefer the canonical note as the survivor, merge the path/frontmatter fixes into it, and avoid silently overwriting a richer note with a thinner one.
   - If two candidates at the target path appear materially different, stop and surface the conflict instead of guessing.
6. Never invent missing assets.
   - If a log row points to a PDF that does not exist locally, do not fabricate a file just to satisfy path normalization.
   - In that case, normalize metadata only when safe, keep the missing state explicit, and report it.
7. Final verification after repair:
   - no stale old path strings remain in vault text files
   - no `UnknownYear_*`, `Unknown_<year>`, or `paperPDFs/Unknown/...` remain unless the remaining `Unknown` token is part of the literal paper title
   - `analysis_log.csv` no longer points to pre-repair paths
   - rebuild `paperCollection` with `papers-build-collection-index`

## Run

From the repository root:

```bash
python3 .claude/skills/papers-audit-metadata-consistency/scripts/audit_metadata_consistency.py
```

Optional JSON output:

```bash
python3 .claude/skills/papers-audit-metadata-consistency/scripts/audit_metadata_consistency.py --json
```

Canonical path normalization after audit:

```bash
python3 .claude/skills/papers-audit-metadata-consistency/scripts/normalize_unknown_paths.py
```

Apply changes and rebuild collection indexes:

```bash
python3 .claude/skills/papers-audit-metadata-consistency/scripts/normalize_unknown_paths.py --apply --rebuild-index
```

## Output

- `paperAnalysis/quality_report_<YYYY-MM-DD_HHMM>.md`
- optional `paperAnalysis/quality_report_<YYYY-MM-DD_HHMM>.json`

## When to use

- after collect/download/analyze batches
- before rebuilding collection indexes
- periodic KB quality checks for open-source readiness
- before any large-scale `Unknown*` path repair or canonical rename pass
