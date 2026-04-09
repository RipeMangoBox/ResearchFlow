---
name: papers-collect-from-github-awesome
description: Collects paper candidates from a GitHub awesome / curated repo README. Agent fetches the README, analyzes its format, writes a one-off parser, and outputs rows aligned with analysis_log.csv. No pre-built scripts — agent adapts to each repo's format on the fly.
---

# Papers Collect From GitHub Awesome

## Purpose

Extract paper candidates from GitHub awesome / curated repository READMEs and output rows aligned with `paperAnalysis/analysis_log.csv`.

Because awesome-list formats vary widely (Markdown tables, bullet lists, mixed HTML, etc.), this skill **does not include fixed scripts**. The agent writes one-off parsing logic for each target README.

## Input

User provides:
- **GitHub repo URL**: for example `https://github.com/Foruck/Awesome-Human-Motion`
- **include/exclude keywords** (optional): for example include `motion generation`, exclude `survey`
- **target category** (optional): for example `Motion_Generation_Text_Speech_Music_Driven`

## Output Format

Output must align with the column format of `paperAnalysis/analysis_log.csv`:

```
state,importance,paper_title,venue,project_link_or_github_link,paper_link,sort,pdf_path
```

Field notes:

| Column | Initial default | Description |
|----|-------------|------|
| state | `Wait` | Later can be changed manually to `Skip`, or remain `Wait` for download |
| importance | empty | Later manually labeled as S/A/B/C |
| paper_title | parsed from README | paper title |
| venue | parsed from README, fallback `Unknown` | e.g., `CVPR 2025`, `ICLR 2026`; if only on open platforms, use `arXiv YYYY` |
| project_link_or_github_link | project page or GitHub link | prefer project page; if confirmed no open source, use `N/A` |
| paper_link | arXiv / OpenReview link | prefer arXiv abs link |
| sort | section/category heading in README | join words with `_`, e.g. `Motion_Generation` |
| pdf_path | empty | filled later by `papers-download-from-list` |

See `paperAnalysis/analysis_log.csv` in the repository for examples.

## Workflow

### 1. Fetch README

- Fetch raw README from the GitHub repo URL
- Prefer `https://raw.githubusercontent.com/<owner>/<repo>/<branch>/README.md`
- Optionally save raw README to `paperAnalysis/processing/github_awesome/<repo_slug>/README.raw.md` for debugging

### 2. Analyze Format

Agent reads README and determines structure:
- Markdown table, bullet list, or mixed format?
- Where paper titles appear and how links are organized?
- Whether venue metadata is inline or grouped by sections?

### 3. Write One-Off Parser

Based on the structure analysis, the agent writes a one-off Python script (or processes directly in conversation code blocks) to extract paper information.

Script requirements:
- Output CSV with the same column order as `analysis_log.csv`
- Handle common link patterns: arXiv abs/pdf, OpenReview, GitHub project page
- Parse venues from text patterns like `(CVPR 2025)` / `[ICLR 2026]`
- Deduplicate repeated appearances of the same paper

### 4. Filter and Deduplicate

- Apply user include/exclude keyword filtering
- Deduplicate against existing `analysis_log.csv` entries (fuzzy match by `paper_title`)
- Report number of newly added candidate rows

### 5. Append to analysis_log.csv

- Append new rows to `paperAnalysis/analysis_log.csv`
- Do not modify existing rows
- Report to user: number of new candidates and source sections

### 6. Completeness Check and Suggestion

After append, scan newly added rows for missing fields:

| Missing field | Detection condition | Completion method | Fallback |
|---------|---------|---------|------------|
| venue | empty or `Unknown` | lookup by paper_link via arXiv metadata / Semantic Scholar API | if unpublished at venue but public: `arXiv YYYY` (e.g., `arXiv 2025`) |
| paper_link | empty | search arXiv / Google Scholar by paper_title | if not found: keep empty |
| project_link_or_github_link | empty | search GitHub / Papers With Code by paper_title | if confirmed no open source: `N/A` |
| sort | empty | infer from README section heading or ask user | keep empty for manual fill |

Then report and suggest:

> "Added N new candidates: X missing venue, Y missing paper_link, Z missing project_link. Do you want multi-source completion? (complete all / venue only / links only / skip)"

After user confirmation:
- **complete all**: use arXiv API, Semantic Scholar, Google Scholar, Papers With Code in sequence
- **specific fields only**: search only matching sources
- **skip**: keep current state for later manual handling

Constraints for completion:
- Keep at least 3 seconds between searches (respect API rate limits)
- Match search results to `paper_title`; only auto-fill when similarity > 0.8, otherwise mark `[needs confirmation]`
- Write completion results back to matching rows in `analysis_log.csv`

## Environment Detection

If Python scripts are needed, follow Environment Detection rules from `code-context-paper-retrieval`:
- prioritize environment detection from codebase `environment.yml` / `requirements.txt` etc.
- if unresolved, proactively ask user

## Constraints

- No prebuilt parser scripts; each awesome list format is adapted live
- One-off scripts may be saved to `paperAnalysis/processing/github_awesome/<repo_slug>/parse_<timestamp>.py` for future reference
- Do not auto-download PDFs (that is `papers-download-from-list`)
- Do not auto-generate analysis notes (that is `papers-analyze-pdf`)

## Typical Usage

> "Collect papers from https://github.com/Foruck/Awesome-Human-Motion, only motion generation related"

> "Organize papers from https://github.com/xxx/awesome-video-diffusion into analysis_log.csv"
