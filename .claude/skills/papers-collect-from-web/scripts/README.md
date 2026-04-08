# papers-collect-from-web scripts

## Quick start

Generate a triage list like `paperAnalysis/ICLR_2026.txt`:

```bash
python3 ".claude/skills/papers-collect-from-web/scripts/collect_from_urls.py" \
  --venue-time "ICLR 2026" \
  --out "paperAnalysis/ICLR_2026.txt" \
  --urls "https://example.com/papers.html" "https://another.example.org/list" \
  --include "motion;diffusion" \
  --exclude "workshop;dataset" \
  --append
```

## Output columns (pipe-delimited)

`状态 | title | 会议&时间 | paper link | project link/github link | 文章分类`

Notes:
- `状态` defaults to `Wait`. Use `--status checked` if you want.
- `文章分类` is intentionally left blank for per-item manual fill.

## Where HTML is stored

Fetched pages are stored under:

`paperSources/<venue_time>_<timestamp>/...`

## Tips for better extraction

- Prefer pages that already contain direct arXiv/OpenReview links.
- Run per-source URL separately if one page is too noisy.
- Tighten `--include/--exclude` keywords to reduce false positives.

