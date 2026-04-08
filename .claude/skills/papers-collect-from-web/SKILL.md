---
name: papers-collect-from-web
description: Collects paper candidates from web URLs by fetching and storing pages locally, then generating a pipe-delimited triage list (status | title | venue&time | paper link | project/github link | category) under `paperAnalysis/`. Use when the user provides URLs + keyword constraints + venue/year and wants a processing list like ICLR_2026.txt.
---

# Paper Collector (Online)

Fetch web pages to local storage, extract paper candidates, and generate a triage list similar to `paperAnalysis/ICLR_2026.txt`.

## Scope

- **Input**: one or more URLs + include/exclude keyword constraints + target venue/year label.
- **Output**:
  - Saved source pages under `paperSources/<run_id>/...`
- A pipe-delimited list at `paperAnalysis/<VENUE>_<YEAR>.txt`

## Output format (exact columns)

Each line must be:

`状态 | title | 会议&时间 | paper link | project link/github link | 文章分类`

Defaults:
- **状态**: `Wait`（新收集）或 `Skip`（人工标记跳过）
- **会议&时间**: user-provided label like `ICLR 2026`
- **文章分类**: leave blank if not known; user can fill per item later

> 状态约定见 `STATE_CONVENTION.md`：主流程 `Wait → Downloaded → checked`，非主流程 `Skip` / `Missing`。

## Workflow

1. Ask the user for:
   - URLs (one or many)
   - Include keywords (optional) and exclude keywords (optional)
   - Venue/time label, and target output filename (e.g. `ICLR_2026.txt`)
   - Whether to overwrite or append if the output file exists
2. Run the collector script (preferred default):
   - `python3 ".claude/skills/papers-collect-from-web/scripts/paper_collector_online/collect_from_urls.py" --venue-time "ICLR 2026" --out "paperAnalysis/ICLR_2026.txt" --urls "<URL1>" "<URL2>" --include "motion;diffusion" --exclude "workshop;dataset" --append`
3. Verify the generated list:
   - Columns present and in order
   - Dedup applied by normalized paper link
   - Links are well-formed (no surrounding punctuation)
4. If extraction quality is poor for a particular URL:
   - Re-run with tighter keywords; or
   - Run per-source URLs separately; or
   - Manually paste a short snippet list into a text file and extend the script patterns (only if needed).

## Rules and heuristics (defaults)

- **Fetch**: store HTML as-is (no JS rendering).
- **Dedup**: by normalized `paper link` (e.g. arXiv pdf/abs normalized).
- **Project link**: best-effort heuristic; may be empty.
- **Keywords**: matched against title + nearby anchor text (case-insensitive).
- **Safety**: respect reasonable timeouts; do not brute-force crawl.

## Notes

- This skill intentionally does **not** download PDFs. It only produces a triage list for later analysis/ingest.
- The repository root is the folder that contains `paperAnalysis/` and `paperPDFs/`.

