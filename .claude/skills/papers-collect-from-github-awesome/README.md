# papers-collect-from-github-awesome

## Overview

Collects paper candidates from GitHub awesome / curated repository READMEs and outputs rows aligned with `paperAnalysis/analysis_log.csv`.

Because each awesome list uses a different structure (Markdown tables, bullet lists, mixed HTML, etc.), this skill does **not** ship fixed parsing scripts. Instead, the agent analyzes each README format on the fly and writes a one-off parser.

## Output format (CSV columns)

```
state,importance,paper_title,venue,project_link_or_github_link,paper_link,sort,pdf_path
```

- `state` defaults to `Wait`
- `importance` and `pdf_path` are left blank during collection
- `sort` inherits section headings from the GitHub repo, using `_` separators (for example `Motion_Generation`)
- See `paperAnalysis/analysis_log.csv` for reference examples

## Typical usage

```
1. "Collect papers from https://github.com/Foruck/Awesome-Human-Motion, only motion generation related."
2. "Focus on Motion Customization, Long Video / Film Generation, and Video Generation with 3D/Physical Prior under https://github.com/showlab/Awesome-Video-Diffusion?tab=readme-ov-file#motion-customization; generate a paper list and save to 'paperAnalysis/analysis_log.csv'."
```

The agent will:
1. Fetch the raw README
2. Analyze its format
3. Write a one-off parser
4. Append new candidates to `analysis_log.csv`

## Processing artifacts (optional)

One-off parsing scripts and raw README snapshots may be saved under:

```
paperAnalysis/processing/github_awesome/<repo_slug>/
```

These are for debugging reference only and are not required for the workflow.
