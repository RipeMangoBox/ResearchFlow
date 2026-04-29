---
created: 1970-01-01T08:00
updated: 2026-04-29T17:00
---
# paper_collector_online

Collect candidates into an `analysis_log.csv`-compatible CSV while saving source payloads.

## Supported inputs

- Live source URLs:
  - OpenReview conference group pages / forum URLs
  - arXiv search pages / paper URLs
  - Semantic Scholar search pages
  - arbitrary static HTML pages as fallback
- Preset paper lists:
  - `CSV`
  - `TSV`
  - `pipe-delimited`
  - `JSONL`

## Quick start

Live source only:

```bash
python3 ".claude/skills/papers-collect-from-web/scripts/paper_collector_online/collect_from_urls.py" \
  --venue-time "<VENUE YEAR>" \
  --out "paperAnalysis/<run_name>.csv" \
  --urls "https://openreview.net/group?id=<GROUP_ID>" \
         "https://arxiv.org/search/?query=<QUERY>&searchtype=all&source=header" \
  --include "<keyword1>;<keyword2>"
```

Preset universe + live refresh:

```bash
python3 ".claude/skills/papers-collect-from-web/scripts/paper_collector_online/collect_from_urls.py" \
  --preset-list "paperAnalysis/<preset_list>.csv" \
  --out "paperAnalysis/<run_name>.csv" \
  --urls "https://openreview.net/group?id=<GROUP_ID>" \
         "https://arxiv.org/search/?query=<QUERY>&searchtype=all&source=header" \
  --include "<keyword1>;<keyword2>"
```

When the preset list already contains `openreview_id` or `arxiv_id` information, the collector also performs direct ID lookups. This keeps preset/live merge less sensitive to broader group/search URL pagination or truncation.

Persist source preference / API choice:

```bash
python3 ".claude/skills/papers-collect-from-web/scripts/paper_collector_online/collect_from_urls.py" \
  --configure-only \
  --preferred-sources "openreview,arxiv,semantic_scholar,html" \
  --semantic-scholar-api-key "<KEY>" \
  --remember-config
```

## Output columns

`state,importance,paper_title,venue,project_link_or_github_link,paper_link,sort,pdf_path`

## Extra artifacts

Under `paperSources/<run_id>/...` the collector writes:

- raw source payloads (`.html`, `.json`, `.xml`)
- `preset_records.jsonl` when `--preset-list` is used
- `discovered_live_records.jsonl`
- `merged_records.jsonl`

## Source behavior

- Supported sources use structured APIs first:
  - `OpenReview` group/forum URLs use `api2.openreview.net` and preserve venue labels such as `ICLR 2026 Poster` / `ICLR 2026 Oral`.
  - `arXiv` search URLs use the arXiv API first, then the rendered search page only as a supplemental signal when capacity remains.
  - `Semantic Scholar` search URLs use the Graph API; an API key is optional but helps with rate limits.
- `--venue-time` is a fallback for generic venue pages. It is not stamped onto arXiv, Semantic Scholar, or OpenReview records that lack their own venue.
- Filtering is limited to caller-provided `--include` / `--exclude` keywords. The collector does not ship fixed topic tags or task-specific topic gates.

## API notes

- OpenReview: no key required for supported flows.
- arXiv: no key required for supported flows.
- Semantic Scholar: can run without an API key, but unauthenticated requests may hit `429` rate limits.
- Persistent config path:
  - `~/.config/researchflow/papers_collect_from_web.json`
