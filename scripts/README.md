# scripts/

Utility scripts for ResearchFlow. Backend API handles most operations now; these scripts are for local-only tasks and diagnostics.

## Active scripts

| Script | Purpose |
|--------|---------|
| `setup_shared_skills.py` | Create .agents/ and .codex/ skill aliases |
| `link_codebase.py` | Create symlinks under linkedCodebases/ |
| `auto_download_papers.py` | Download PDFs from triage logs |
| `playwright_download.py` | Headless browser PDF download fallback |
| `update_download_log.py` | Normalize download log format |
| `find_pdfs.py` | Locate PDFs on disk |
| `audit_knowledge_batch.py` | Check analysis file structure |
| `fix_analysis_md_issues.py` | Repair broken frontmatter |
| `fix_missing_venue_year.py` | Fill missing venue/year |
| `review_analysis_mismatch.py` | Compare CSV log vs .md files |

## Maintenance subdirectories

- `maintenance/get_missing_md.py` — Find PDFs without analysis notes
- `paper_analysis_maintenance/check_part_sections.py` — Audit Part I/II/III headers
- `paper_analysis_maintenance/fill_project_github_in_abstract.py` — Auto-fill GitHub links
- `paper_analysis_maintenance/mark_wait_for_incomplete_parts.py` — Mark incomplete analyses
- `paper_analysis_maintenance/salad_format_audit.py` — Strict format compliance check

## Backend equivalents

Most script functionality is now available via backend API:

| Script task | Backend API |
|-------------|-------------|
| Import papers | `POST /api/v1/import/links` |
| Download PDFs | `POST /api/v1/pipeline/{id}/download-pdf` |
| Analyze papers | `POST /api/v1/pipeline/{id}/run` |
| Search/query | `POST /api/v1/search/hybrid` |
| Quality audit | `GET /api/v1/graph/quality` |
