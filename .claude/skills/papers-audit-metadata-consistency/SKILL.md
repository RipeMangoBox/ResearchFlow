---
name: papers-audit-metadata-consistency
description: Runs a unified metadata consistency audit across paperAnalysis logs and analysis notes, checking title/venue/year/category consistency, link/pdf_ref completeness, duplicates, and structure anomalies; writes quality_report_*.md under paperAnalysis.
---

# Metadata Consistency Audit

## What this skill does

Runs one unified audit entry for step5 quality governance:

- checks `paperAnalysis/*.txt` list/log entries;
- checks `paperAnalysis/**/*.md` analysis notes;
- checks cross-layer consistency between log/list and analysis notes;
- outputs `paperAnalysis/quality_report_*.md` (optionally `.json`).

## Coverage

This skill covers **structural and metadata hygiene only**.

- title / venue / year / category consistency
- link completeness (`paper_link`, `project_link`/`github_link`, `pdf_ref`)
- duplicate paper detection in txt logs
- analysis note structure anomalies (missing frontmatter keys, missing Part I/II/III)
- unmatched records (`only-in-txt`, `only-in-md`)

## Current limits

- does not evaluate whether conclusions are sufficiently supported by evidence
- does not score citation quality in idea notes or share notes
- does not audit completeness of comparison tables or downstream summaries
- does not detect stale analyses that should be refreshed
- broader semantic-layer governance belongs to a future knowledge-hygiene / evidence-governance extension

## Run

From the repository root:

```bash
python3 .claude/skills/papers-audit-metadata-consistency/scripts/audit_metadata_consistency.py
```

Optional JSON output:

```bash
python3 .claude/skills/papers-audit-metadata-consistency/scripts/audit_metadata_consistency.py --json
```

## Output

- `paperAnalysis/quality_report_<YYYY-MM-DD_HHMM>.md`
- optional `paperAnalysis/quality_report_<YYYY-MM-DD_HHMM>.json`

## When to use

- after collect/download/analyze batches
- before rebuilding collection indexes
- periodic KB quality checks for open-source readiness
