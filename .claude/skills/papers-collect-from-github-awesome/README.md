# papers-collect-from-github-awesome

## Overview

Collects paper candidates from GitHub awesome / curated repo READMEs and outputs rows aligned with `paperAnalysis/analysis_log.csv`.

Since every awesome list has a different format (Markdown tables, bullet lists, mixed HTML, etc.), this skill does **not** ship pre-built parsing scripts. Instead, the agent analyzes each README's structure on the fly and writes a one-off parser.

## Output format (CSV columns)

```
state,importance,paper_title,venue,project_link_or_github_link,paper_link,sort,pdf_path
```

- `state` defaults to `Wait`
- `importance` and `pdf_path` are left blank during collection
- `sort` inherits GitHub repo section headings, using `_` separators (e.g. `Motion_Generation`)
- See `paperAnalysis/analysis_log.csv` for reference examples

## Typical usage

```
1. "帮我从 https://github.com/Foruck/Awesome-Human-Motion 收集论文，只要 motion generation 相关的".
2. 聚焦https://github.com/showlab/Awesome-Video-Diffusion?tab=readme-ov-file#motion-customization的 Motion Customization, Long Video / Film Generation, Video Generation with 3D/Physical Prior 子专题,生成论文清单,保存到'paperAnalysis/analysis_log.csv'.
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

These are for debugging reference only and not required for the workflow.
