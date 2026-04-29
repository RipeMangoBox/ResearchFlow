# papers-collect-from-web scripts

## Main entrypoint

```bash
python3 ".claude/skills/papers-collect-from-web/scripts/paper_collector_online/collect_from_urls.py" ...
```

## What it does

- collects from supported live sources (`OpenReview`, `arXiv`, `Semantic Scholar`, fallback static HTML)
- prefers structured/API adapters over raw HTML whenever a supported source exposes one
- merges against an optional preset paper universe
- preserves OpenReview presentation labels such as `Poster` / `Oral` in the venue field
- applies only caller-provided include/exclude keyword filters, with no fixed topic tags or task-specific gates
- writes an `analysis_log.csv`-compatible CSV
- supports persistent local config for source preference and optional API keys

## Core flags

- `--urls ...`
- `--preset-list <path>`
- `--preferred-sources "openreview,arxiv,semantic_scholar,html"`
- `--semantic-scholar-api-key <key>`
- `--remember-config`
- `--configure-only`

## Output columns

`state,importance,paper_title,venue,project_link_or_github_link,paper_link,sort,pdf_path`

## Persistent config

Default:

`~/.config/researchflow/papers_collect_from_web.json`
