---
name: papers-collect-from-web
description: Collects paper candidates from web URLs by fetching and storing pages locally, then generating a pipe-delimited triage list (status | title | venue&time | paper link | project/github link | category) under `paperAnalysis/`. Use when the user provides URLs + keyword constraints + venue/year and wants a processing list like ICLR_2026.txt.
---

# Paper Collector (Online)

Fetch web pages to local storage, extract paper candidates, and generate a triage list like `paperAnalysis/ICLR_2026.txt`.

## Scope

- **Input**: one or more URLs + include/exclude keyword constraints + target venue/year label.
- **Output**:
  - saved source pages under `paperSources/<run_id>/...`
  - a pipe-delimited list at `paperAnalysis/<VENUE>_<YEAR>.txt`

## Output format (exact columns)

Each line must be:

`state | title | venue&time | paper link | project link/github link | category`

Defaults:
- **state**: `Wait` (newly collected) or `Skip` (manually marked skip)
- **venue&time**: user-provided label such as `ICLR 2026`
- **category**: leave blank if unknown; user can fill per item later

> State convention is defined in `STATE_CONVENTION.md`: main pipeline `Wait → Downloaded → checked`; out-of-band states `Skip` / `Missing`.

## Workflow

1. Ask user for:
   - URLs (one or many)
   - include keywords (optional) and exclude keywords (optional)
   - venue/time label and target output filename (for example `ICLR_2026.txt`)
   - whether to overwrite or append if output file already exists
2. Run collector script (preferred default):
   - `python3 ".claude/skills/papers-collect-from-web/scripts/paper_collector_online/collect_from_urls.py" --venue-time "ICLR 2026" --out "paperAnalysis/ICLR_2026.txt" --urls "<URL1>" "<URL2>" --include "motion;diffusion" --exclude "workshop;dataset" --append`
3. Verify generated list:
   - columns present and in correct order
   - dedup applied by normalized paper link
   - links are well-formed (no surrounding punctuation)
4. If extraction quality is poor for a URL:
   - rerun with tighter keywords, or
   - run per-source URLs separately, or
   - manually provide a short snippet list and extend script patterns only if needed.

## Rules and heuristics (defaults)

- **Fetch**: store HTML as-is (no JS rendering).
- **Dedup**: by normalized `paper link` (for example normalize arXiv pdf/abs).
- **Project link**: best-effort heuristic; may be empty.
- **Keywords**: match against title + nearby anchor text (case-insensitive).
- **Safety**: respect reasonable timeouts; do not brute-force crawl.

## Notes

- This skill intentionally does **not** download PDFs. It only produces triage lists for later processing.
- Repository root is the folder containing `paperAnalysis/` and `paperPDFs/`.
