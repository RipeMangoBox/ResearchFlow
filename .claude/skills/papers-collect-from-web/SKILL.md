---
name: papers-collect-from-web
description: "Collects paper candidates from non-GitHub web URLs — conference sites, lab homepages, proceedings pages, Google Scholar results, blog posts with paper lists, etc. Fetches and stores pages locally, then appends rows to `paperAnalysis/analysis_log.csv`. Use when the user provides web URLs (not GitHub repos) + keyword constraints + venue/year and wants candidates added to the unified log. For GitHub repos, use `papers-collect-from-github-repo` instead."
---

# Paper Collector (Online / Web)

Collect paper candidates from web sources into an `analysis_log.csv`-compatible file.

The collector now supports two modes:

- **Live source discovery** from web URLs
- **Preset universe + live refresh** where you provide a paper list first, then let the collector merge it with live sources

Use this skill for **non-GitHub web sources**: conference sites, lab homepages, proceedings pages (ACL Anthology, IEEE Xplore, etc.), Google Scholar results, blog posts with paper lists, etc.

For **GitHub repositories** (awesome lists, survey repos, lab paper repos, conference paper repos), use `papers-collect-from-github-repo` instead — it handles raw Markdown and multi-file repo structures better.

## Scope

- **Input**:
  - one or more web URLs, and/or
  - a preset paper list (`CSV` / `TSV` / `pipe-delimited` / `JSONL`)
  - optional include/exclude keyword constraints
  - optional venue/year label
- **Output**:
  - saved source payloads under `paperSources/<run_id>/...`
  - a new or appended CSV aligned with `paperAnalysis/analysis_log.csv`

## Output format (CSV columns, same as analysis_log.csv)

```
state,importance,paper_title,venue,project_link_or_github_link,paper_link,sort,pdf_path
```

Defaults:
- **state**: `Wait` (newly collected) or `Skip` (manually marked skip)
- **importance**: empty (user fills later)
- **venue**: source-provided venue when available; otherwise a user-provided fallback such as `ICLR 2026` only for generic venue pages
- **sort**: leave blank if unknown; user can fill per item later
- **pdf_path**: empty (filled later by `papers-download-from-list`)

> State convention is defined in `STATE_CONVENTION.md`: main pipeline `Wait → Downloaded → checked`; out-of-band states `Skip` / `Missing`.

## Workflow

1. Ask user for:
   - URLs (one or many)
   - include keywords (optional) and exclude keywords (optional)
   - venue/time label (for example `ICLR 2026`)
   - whether to overwrite or append if rows already exist for this venue
2. Run collector script (preferred default):
   - `python3 ".claude/skills/papers-collect-from-web/scripts/paper_collector_online/collect_from_urls.py" --venue-time "<VENUE YEAR>" --urls "<URL1>" "<URL2>" --include "<required keyword or phrase>;..." --exclude "<excluded keyword or phrase>;..." --append`
3. When a preset paper list exists:
   - `python3 ".claude/skills/papers-collect-from-web/scripts/paper_collector_online/collect_from_urls.py" --preset-list "paperAnalysis/<preset_list>.csv" --urls "<URL1>" "<URL2>" --include "<required keyword or phrase>;..." --out "paperAnalysis/<run_name>.csv"`
4. Verify generated rows:
   - columns present and in correct order
   - dedup applied by normalized paper link plus exact-title live-source merge
   - links are well-formed (no surrounding punctuation)
   - keyword filtering did not leave obvious unrelated papers in the output
   - OpenReview venue labels keep presentation type when available, for example `ICLR 2026 Poster` / `ICLR 2026 Oral`
5. If extraction quality is poor for a URL:
   - rerun with tighter keywords, or
   - run per-source URLs separately, or
   - prefer source adapters over raw HTML when available, or
   - extend the source adapter if the site is important and recurring.

## Rules and heuristics (defaults)

- **Supported source adapters**:
  - `OpenReview`: conference group pages and forum URLs via `api2.openreview.net`
  - `arXiv`: search pages use arXiv API first, then HTML search-page results only as a supplemental signal when capacity remains
  - `Semantic Scholar`: search pages via the Graph API
  - fallback arbitrary static HTML link extraction for everything else
- **Fetch**: store raw source payloads under `paperSources/<run_id>/...`.
- **Dedup**: by normalized `paper link`, then exact-title merge using preferred source order.
- **Preset confirmation**: when `--preset-list` is present, known `openreview_id` / `arxiv_id` entries are directly re-queried so that preset/live merge is less sensitive to pagination windows in broader source URLs.
- **Venue labels**: `--venue-time` is only a fallback for generic venue pages. It is not applied to arXiv, Semantic Scholar, or OpenReview records that lack their own venue. OpenReview API venue strings with presentation type are preserved during preset/live merge.
- **Project link**: best-effort heuristic; may be empty.
- **Keywords**: match against title + abstract + venue + authors + links (case-insensitive).
- **No task-specific topic logic**: do not add fixed research tags, topic word lists, or one-off deterministic topic gates to this skill or its scripts. Keep query constraints in runtime inputs such as `--include`, `--exclude`, URLs, and preset lists.
- **Safety**: respect reasonable timeouts; do not brute-force crawl.

## Persistent config and APIs

- Default persistent config path:
  - `~/.config/researchflow/papers_collect_from_web.json`
- Save source preference / API keys:
  - `python3 ".../collect_from_urls.py" --configure-only --preferred-sources "openreview,arxiv,semantic_scholar,html" --semantic-scholar-api-key "<KEY>" --remember-config`
- **OpenReview** and **arXiv** do not require API keys for the supported flows.
- **Semantic Scholar** can run without an API key, but unauthenticated requests are rate-limited. Store a key with `--remember-config` only when higher rate limits are needed.

## Notes

- This skill intentionally does **not** download PDFs. It only appends candidate rows to `analysis_log.csv` for later processing.
- Output format is unified with `papers-collect-from-github-repo` — both write to `analysis_log.csv`.
- Repository root is the folder containing `paperAnalysis/` and `paperPDFs/`.

## Relationship to papers-collect-from-github-repo

- **This skill** (`papers-collect-from-web`): optimized for recurring web sources plus preset paper universes; it prefers source-specific adapters when possible and falls back to static HTML extraction
- **`papers-collect-from-github-repo`**: optimized for GitHub repos — fetches raw Markdown, understands repo structure, handles multi-file layouts, writes one-off parsers
