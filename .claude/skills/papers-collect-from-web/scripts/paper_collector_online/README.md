---
created: 1970-01-01T08:00
updated: 2026-03-17T19:54
---
# paper_collector_online

Generate triage lists like `paperAnalysis/ICLR_2026.txt` by fetching web pages, saving HTML locally, extracting paper candidates, and writing a pipe-delimited list.

## Quick start

```bash
python3 ".claude/skills/papers-collect-from-web/scripts/paper_collector_online/collect_from_urls.py" \
  --venue-time "ICLR 2026" \
  --out "paperAnalysis/ICLR_2026.txt" \
  --urls "https://example.com/papers.html" "https://another.example.org/list" \
  --include "motion;diffusion" \
  --exclude "workshop;dataset" \
  --append
```

## Output columns

`state | title | venue&time | paper link | project link/github link | category`

Notes:
- `state` defaults to `Wait` (use `--status checked` if you want).
- `category` is left blank for per-item manual fill.

## Where HTML is stored

Default:

`paperSources/<venue_time>_<timestamp>/...`
