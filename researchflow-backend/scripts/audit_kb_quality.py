"""Audit knowledge-base quality and produce a markdown report.

Surfaces rows that are likely garbage and need human review before vault
export. Does NOT modify any data; the report ends with a SQL "soft-delete
suggestions" block the user can copy-paste after review.

Run:
    python -m scripts.audit_kb_quality
    python -m scripts.audit_kb_quality --out paperAnalysis/quality_report.md

Categories surfaced:
  1. Venue anomalies — full-name venues, suspicious tokens, NULL.
  2. Title anomalies — arXiv-id placeholders, oversized titles, P__ stubs.
  3. Missing artifacts — no L2 figures, no L4 full_report_md.
  4. Quality flags — source_quality != 'normal'/'published',
                     source contains 'test'.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path

from sqlalchemy import text

from backend.database import async_session

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Heuristics ─────────────────────────────────────────────────────────

SUSPICIOUS_VENUE_TOKENS = (
    "Cornell University", "Cambridge University Press", "eBooks",
    "Proceedings of the", "Trans. ", "Society", "Wiley", "Elsevier",
)

ARXIV_ID_TITLE_RE = re.compile(r"^\s*\d{4}\.\d{4,5}(v\d+)?\s*$")
P_STUB_TITLE_RE = re.compile(r"^P__\d+$", re.IGNORECASE)
TITLE_MAX_LEN = 140


def _venue_issue(venue: str | None) -> str | None:
    if not venue:
        return "venue_null"
    if any(tok in venue for tok in SUSPICIOUS_VENUE_TOKENS):
        return f"venue_suspicious_token: {venue!r}"
    if len(venue) > 80:
        return f"venue_oversize ({len(venue)} chars)"
    return None


def _title_issue(title: str | None) -> str | None:
    if not title or not title.strip():
        return "title_empty"
    if ARXIV_ID_TITLE_RE.match(title):
        return "title_is_arxiv_id"
    if P_STUB_TITLE_RE.match(title):
        return "title_is_P_stub"
    if len(title) > TITLE_MAX_LEN:
        return f"title_oversize ({len(title)} chars)"
    return None


# ── Queries ────────────────────────────────────────────────────────────

ALL_PAPERS_SQL = text("""
    SELECT
      p.id, p.title, p.venue, p.year, p.category, p.source,
      COALESCE(p.source_quality, 'normal') AS source_quality,
      EXISTS (
        SELECT 1 FROM paper_analyses
        WHERE paper_id = p.id AND level = 'l2_parse' AND is_current = true
          AND extracted_figure_images IS NOT NULL
          AND jsonb_typeof(extracted_figure_images) = 'array'
          AND jsonb_array_length(extracted_figure_images) > 0
      ) AS has_l2_figures,
      EXISTS (
        SELECT 1 FROM paper_analyses
        WHERE paper_id = p.id AND level = 'l4_deep' AND is_current = true
          AND full_report_md IS NOT NULL AND length(full_report_md) > 200
      ) AS has_l4_report,
      EXISTS (
        SELECT 1 FROM paper_figures WHERE paper_id = p.id
      ) AS has_paper_figures
    FROM papers p
    WHERE p.state NOT IN ('skip', 'archived_or_expired')
""")


# ── Report ─────────────────────────────────────────────────────────────

def _format_row(p) -> str:
    return f"| `{str(p.id)[:8]}` | {(p.title or '')[:80]!r} | {p.venue or ''} | {p.year or ''} | {p.category or ''} |"


async def main(out_path: Path) -> None:
    async with async_session() as session:
        rows = (await session.execute(ALL_PAPERS_SQL)).fetchall()

    venue_bad: list[tuple] = []   # (issue, row)
    title_bad: list[tuple] = []
    missing_figures: list = []
    missing_report: list = []
    quality_flagged: list = []
    test_marker: list = []

    for r in rows:
        v_issue = _venue_issue(r.venue)
        if v_issue:
            venue_bad.append((v_issue, r))
        t_issue = _title_issue(r.title)
        if t_issue:
            title_bad.append((t_issue, r))
        if not r.has_l2_figures and not r.has_paper_figures:
            missing_figures.append(r)
        if not r.has_l4_report:
            missing_report.append(r)
        if r.source_quality not in ("normal", "published"):
            quality_flagged.append(r)
        if r.source and "test" in r.source.lower():
            test_marker.append(r)

    total = len(rows)
    lines: list[str] = []
    lines.append(f"# KB Quality Audit — {datetime.utcnow().date().isoformat()}")
    lines.append("")
    lines.append(f"Inspected **{total}** papers (state ≠ skip/archived).")
    lines.append("")

    def _section(title: str, items: list, formatter):
        lines.append(f"## {title} ({len(items)})")
        lines.append("")
        if not items:
            lines.append("_None._")
            lines.append("")
            return
        lines.append("| id | title | venue | year | category |")
        lines.append("|----|-------|-------|------|----------|")
        for it in items[:200]:
            lines.append(formatter(it))
        if len(items) > 200:
            lines.append(f"| … | _(+{len(items) - 200} more)_ | | | |")
        lines.append("")

    _section("1. Venue anomalies",
             venue_bad,
             lambda x: f"| `{str(x[1].id)[:8]}` | _{x[0]}_ | {x[1].venue or ''} | {x[1].year or ''} | {x[1].category or ''} |")

    _section("2. Title anomalies",
             title_bad,
             lambda x: f"| `{str(x[1].id)[:8]}` | _{x[0]}_: {(x[1].title or '')[:60]!r} | {x[1].venue or ''} | {x[1].year or ''} | {x[1].category or ''} |")

    _section("3. Missing L2 figures (no L2 JSONB and no paper_figures rows)",
             missing_figures, _format_row)

    _section("4. Missing L4 deep report",
             missing_report, _format_row)

    _section("5. source_quality flagged (not normal/published)",
             quality_flagged,
             lambda r: f"| `{str(r.id)[:8]}` | _{r.source_quality}_ → {(r.title or '')[:60]!r} | {r.venue or ''} | {r.year or ''} | {r.category or ''} |")

    _section("6. source field contains 'test'",
             test_marker,
             lambda r: f"| `{str(r.id)[:8]}` | source=_{r.source}_ | {(r.title or '')[:60]!r} | {r.venue or ''} | {r.year or ''} |")

    # SQL suggestion block (manual gate — never auto-execute)
    lines.append("---")
    lines.append("## Soft-delete suggestions (review before running)")
    lines.append("")
    lines.append("```sql")
    lines.append("-- 1. Mark all venue/title anomalies as low-quality (does NOT delete rows):")
    bad_ids = sorted({str(r[1].id) for r in venue_bad + title_bad})
    if bad_ids:
        sample = bad_ids[:8]
        more = "" if len(bad_ids) <= 8 else f"  -- and {len(bad_ids) - 8} more"
        lines.append("UPDATE papers SET source_quality = 'low'")
        lines.append("WHERE id IN (")
        lines.append(",\n".join(f"  '{i}'" for i in sample) + more)
        lines.append(");")
    else:
        lines.append("-- (no anomalies found)")
    lines.append("")
    lines.append("-- 2. Hide source_quality != normal/published from vault export by")
    lines.append("--    extending vault_export_v6 _load_all_data WHERE clause:")
    lines.append("--    AND COALESCE(p.source_quality,'normal') IN ('normal','published')")
    lines.append("```")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s (%d papers, %d venue, %d title, %d no-fig, %d no-report)",
                out_path, total, len(venue_bad), len(title_bad),
                len(missing_figures), len(missing_report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_out = Path(__file__).resolve().parents[2] / "paperAnalysis" / (
        f"quality_report_{datetime.utcnow().strftime('%Y%m%d')}.md"
    )
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()
    asyncio.run(main(args.out))
