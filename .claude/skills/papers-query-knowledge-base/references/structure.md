# paperCollection & paperAnalysis — Paths and schema

For tooling, scripts, or Agent skill that need exact paths and frontmatter fields. Paths are relative to the repository root that contains `paperCollection/`, `paperAnalysis/`, and `paperPDFs/`.

## Root paths

- `paperCollection/` — index layer
- `paperAnalysis/` — analysis notes
- `paperPDFs/` — PDFs (linked from analysis only)

Use the matching absolute repository path for your machine when invoking from another workspace.

## paperCollection layout

| Path | Description |
| ------ | ------ |
| `paperCollection/README.md` | Home: links to _AllPapers, by_task, by_technique, by_venue |
| `paperCollection/_AllPapers.md` | All papers grouped by task → venue+year |
| `paperCollection/by_task/<task>.md` | Papers for one task (task = category) |
| `paperCollection/by_technique/_Index.md` | List of all technique tags |
| `paperCollection/by_technique/<tag>.md` | Papers that have this technique tag |
| `paperCollection/by_venue/_Index.md` | List of all venues |
| `paperCollection/by_venue/<venue>.md` | Papers for one venue, grouped by year |

Task and tag filenames are sanitized (e.g. spaces → single space, unsafe chars → underscore).

## paperCollection page frontmatter

- `type: paper-index`
- `dimension: all | task | technique | venue`
- Optional: `task:`, `technique:`, `venue:`
- `generated: <ISO date>`

## paperAnalysis note frontmatter

- **Required for indexing**: `pdf_ref` (path like `paperPDFs/.../file.pdf`) — only notes with valid pdf_ref are included in paperCollection.
- **Display/navigation**: title, venue, year, category, tags
- **Research/citation**: core_operator, primary_logic (optional; string or multi-line scalar)
- **Optional**: created, updated, status, note

Tags: list of strings. Filter out `status/*` and category-name tag when building technique index.

## paperAnalysis note body structure

- Quick Links & TL;DR (Summary, Key Performance)
- Part I: problem / skill signature
- Part II: method / "Aha!" moment
- Part III: technical deep dive
- Local Reading (link/embed to PDF)

## Relationship

- paperCollection pages **link to** paperAnalysis notes via `[[paperAnalysis/.../file.md|...]]` and to PDFs via `[[paperPDFs/.../file.pdf|PDF]]`.
- paperAnalysis notes are **not** modified by the collection build; they are only read (frontmatter + path) to build the index.
