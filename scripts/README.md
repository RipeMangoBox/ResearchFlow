# ResearchWY scripts

This folder centralizes all executable maintenance/collection scripts for the ResearchWY vault.

## Structure

- `paper_collector_online/`
  - `collect_from_urls.py`: fetch URLs → save HTML to `paperSources/` → generate triage list in `paperAnalysis/*.txt`
- `paper_analysis_maintenance/`
  - `salad_format_audit.py`
  - `check_part_sections.py`
  - `mark_wait_for_incomplete_parts.py`
  - `fill_project_github_in_abstract.py`
  - `update_status_by_parts.py`
- `audit_metadata_consistency.py`
  - unified metadata consistency audit across `paperAnalysis/*.txt` and `paperAnalysis/**/*.md`
  - writes `paperAnalysis/quality_report_*.md` (optional `.json`)
- `paper_download_tools/` (scripts with same-directory imports)
  - `check_paper_downloads.py` (core module used by others)
  - `download_wait_papers.py`
  - `dedupe_paperpdfs.py`
  - `mark_missing_wait.py`
  - `fix_wrong_downloads_from_log.py`
  - `redownload_correct_pdfs.py`
  - `check_pdfs_against_log.py`
- `maintenance/`
  - `rename_dart.py`
  - `get_missing_md.py`

## Notes

- Prefer running scripts from the repository root, e.g.:
  - `python3 scripts/paper_collector_online/collect_from_urls.py --help`
- Shared skill aliases are managed by `scripts/setup_shared_skills.py`.
  - Maintained source of truth: `.claude/skills` and `.claude/skills-config.json`
  - Codex compatibility aliases: `.codex/skills`, `.codex/skills-config.json`
  - Example: `python3 scripts/setup_shared_skills.py --check`
- Some scripts depend on optional packages (e.g. `requests`, `pypdf`).
