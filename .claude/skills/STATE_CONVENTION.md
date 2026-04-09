# Paper State Convention

Unified definitions for the `state` column in `analysis_log.csv`. All skills must follow this convention.

## Main pipeline states

```
Wait → Downloaded → checked
```

| state | Meaning | Written by stage |
|-------|------|--------------|
| `Wait` | Newly collected candidate, waiting for download | collect (from-web / from-github-awesome) |
| `Downloaded` | PDF downloaded to `paperPDFs/`, waiting for analysis | download (`papers-download-from-list`) |
| `checked` | Structured analysis completed; corresponding `.md` exists in `paperAnalysis/` | analyze (`papers-analyze-pdf`) |

## Out-of-band states

| state | Meaning | Notes |
|-------|------|------|
| `Skip` | Manually filtered out and not processed | Does not enter the main pipeline; kept in the log for later review |
| `Missing` | PDF still unavailable after repeated download attempts | Kept in the log for manual addition or future retry |

## Rules

1. Each record state can only move forward along the main pipeline: `Wait → Downloaded → checked`
2. `Skip` and `Missing` may be set from `Wait` and cannot be reverted to `Wait` (unless manually edited by the user)
3. Downloads automatically compress oversized PDFs (>20MB)
4. Each skill only processes states for its own stage:
   - download processes only `Wait`
   - analyze processes only `Downloaded`
   - build-collection-index processes only `checked`

## Field fallback values

| Field | Fallback | Notes |
|------|---------|------|
| venue | `arXiv YYYY` | For works not accepted by a venue but published on open platforms such as arXiv, e.g., `arXiv 2025` |
| project_link_or_github_link | `N/A` | Confirmed no open-source code or project page |
