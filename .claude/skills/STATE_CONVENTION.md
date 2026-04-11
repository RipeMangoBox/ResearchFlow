# Paper State Convention

Unified definitions for the `state` column in `analysis_log.csv`. All skills must follow this convention.

## State flow

```
Wait → Downloaded → checked
  │        │
  │        ├→ too_large          (PDF exceeds size limit after compression)
  │        └→ analysis_mismatch  (analysis template incomplete after retry)
  │
  ├→ Skip     (user manually excluded)
  └→ Missing  (download failed, PDF unavailable)
```

## Main pipeline states

| state | Meaning | Written by | Next action |
|-------|---------|------------|-------------|
| `Wait` | Newly collected candidate, waiting for download | collect (from-web / from-github-awesome) | Run download |
| `Downloaded` | PDF downloaded to `paperPDFs/`, waiting for analysis | download (`papers-download-from-list`) | Run analyze |
| `checked` | Structured analysis completed; `.md` exists in `paperAnalysis/` | analyze (`papers-analyze-pdf`) | Ready for query / build index |

## Abnormal states (from analyze stage)

| state | Meaning | Written by | Recovery |
|-------|---------|------------|----------|
| `analysis_mismatch` | Analysis generated but required sections (Part I/II/III or Aha! Moment) are missing or too thin after one retry | analyze (`papers-analyze-pdf`) | Re-run analyze on this entry, or manually edit the `.md` then set state to `checked` |
| `too_large` | PDF exceeds 20 MB after compression (`/ebook` then `/screen`); skipped | analyze (`papers-analyze-pdf`) | Manually compress or split the PDF, then set state back to `Downloaded` |

## Out-of-band states

| state | Meaning | Written by | Recovery |
|-------|---------|------------|----------|
| `Skip` | Manually filtered out, not processed | User (manual edit) | Set back to `Wait` if reconsidered |
| `Missing` | PDF unavailable after repeated download attempts | download (`papers-download-from-list`) | Retry later, or manually place PDF then set to `Downloaded` |

## Rules

1. Main pipeline moves forward only: `Wait → Downloaded → checked`.
2. `Skip` and `Missing` are set from `Wait`; `too_large` and `analysis_mismatch` are set from `Downloaded`.
3. Only the user may revert a state (e.g. `Missing → Wait`, `too_large → Downloaded`).
4. Downloads automatically compress PDFs > 20 MB before analysis.
5. Each skill only processes entries at its own input state:
   - download processes `Wait`
   - analyze processes `Downloaded`
   - build-collection-index processes `checked`

## Field fallback values

| Field | Fallback | Notes |
|-------|----------|-------|
| venue | `arXiv YYYY` | For works not accepted by a venue but published on open platforms such as arXiv, e.g., `arXiv 2025` |
| project_link_or_github_link | `N/A` | Confirmed no open-source code or project page |
