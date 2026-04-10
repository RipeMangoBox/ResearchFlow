# papers-collect-from-web scripts

## Quick start

Append paper candidates to `paperAnalysis/analysis_log.csv`:

```bash
python3 ".claude/skills/papers-collect-from-web/scripts/paper_collector_online/collect_from_urls.py" \
  --venue-time "ICLR 2026" \
  --urls "https://example.com/papers.html" "https://another.example.org/list" \
  --include "motion;diffusion" \
  --exclude "workshop;dataset" \
  --append
```

## Output columns (CSV, same as analysis_log.csv)

`state,importance,paper_title,venue,project_link_or_github_link,paper_link,sort,pdf_path`

Notes:
- `state` defaults to `Wait`. Use `--status checked` if needed.
- `importance`, `sort`, `pdf_path` are left blank for later manual fill.
- Output is now unified with `papers-collect-from-github-awesome`.

## Where HTML is stored

Fetched pages are stored under:

`paperSources/<venue_time>_<timestamp>/...`

## Tips for better extraction

- Prefer pages that already contain direct arXiv/OpenReview links.
- Run each source URL separately if one page is too noisy.
- Tighten `--include/--exclude` keywords to reduce false positives.
