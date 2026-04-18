"""Export service — async DB → paperAnalysis/ + paperCollection/ sync.

Called after pipeline completion to keep Markdown exports in sync with DB.
Also generates paperCollection index from DB on demand.
"""

import logging
from pathlib import Path
from uuid import UUID

import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings

logger = logging.getLogger(__name__)


def _render_frontmatter(data: dict) -> str:
    yaml_str = yaml.dump(
        data, default_flow_style=False, allow_unicode=True,
        sort_keys=False, width=200,
    )
    return f"---\n{yaml_str}---\n"


async def export_paper_analysis(
    session: AsyncSession,
    paper_id: UUID,
    out_dir: str | None = None,
) -> str | None:
    """Export a single paper's analysis to paperAnalysis/ Markdown.

    Called automatically after L4 analysis completes.
    Returns the output file path or None if no analysis exists.
    """
    root = Path(out_dir or settings.paper_analysis_dir)

    row = (await session.execute(text("""
        SELECT
            p.id, p.title, p.venue, p.year, p.category, p.tags,
            p.core_operator, p.primary_logic, p.claims,
            p.title_sanitized, p.paper_link, p.code_url,
            pa.full_report_md, pa.problem_summary, pa.method_summary,
            pa.evidence_summary, pa.core_intuition,
            dc.delta_statement, dc.baseline_paradigm,
            dc.structurality_score AS dc_struct,
            dc.transferability_score AS dc_transfer,
            dc.key_ideas_ranked, dc.assumptions, dc.failure_modes,
            dc.status AS dc_status
        FROM papers p
        LEFT JOIN paper_analyses pa ON pa.paper_id = p.id
            AND pa.is_current = true AND pa.level = 'l4_deep'
        LEFT JOIN delta_cards dc ON dc.id = p.current_delta_card_id
        WHERE p.id = :pid
    """), {"pid": paper_id})).fetchone()

    if not row:
        return None

    category = row.category or "Uncategorized"
    venue_year = f"{row.venue}_{row.year}" if row.venue and row.year else "Unknown"
    filename = f"{row.title_sanitized or str(row.id)}.md"

    out_path = root / category / venue_year / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build frontmatter
    fm = {
        "title": row.title,
        "venue": row.venue,
        "year": row.year,
        "category": category,
        "tags": list(row.tags) if row.tags else [],
        "core_operator": row.core_operator,
        "primary_logic": row.primary_logic,
    }
    if row.dc_struct is not None:
        fm["structurality_score"] = round(float(row.dc_struct), 3)
    if row.dc_transfer is not None:
        fm["transferability_score"] = round(float(row.dc_transfer), 3)
    if row.baseline_paradigm:
        fm["paradigm"] = row.baseline_paradigm
    if row.dc_status:
        fm["delta_card_status"] = row.dc_status
    if row.paper_link:
        fm["paper_link"] = row.paper_link
    if row.code_url:
        fm["code_url"] = row.code_url

    # Build body
    body_parts = []
    if row.full_report_md:
        body_parts.append(row.full_report_md)
    else:
        if row.problem_summary:
            body_parts.append(f"## Part I: 问题与挑战\n\n{row.problem_summary}\n")
        if row.method_summary:
            body_parts.append(f"## Part II: 方法与洞察\n\n{row.method_summary}\n")
        if row.core_intuition:
            body_parts.append(f"### 核心直觉\n\n{row.core_intuition}\n")
        if row.evidence_summary:
            body_parts.append(f"## Part III: 证据与局限\n\n{row.evidence_summary}\n")

    if row.delta_statement:
        body_parts.append(f"\n## Delta Statement\n\n{row.delta_statement}\n")

    content = _render_frontmatter(fm) + "\n" + "\n".join(body_parts)
    out_path.write_text(content, encoding="utf-8")
    logger.info(f"Exported analysis: {out_path}")
    return str(out_path)


async def export_analysis_log_csv(
    session: AsyncSession,
    out_path: str | None = None,
) -> int:
    """Export analysis_log.csv from DB. Returns row count."""
    import csv
    import io

    target = Path(out_path or settings.paper_analysis_dir) / "analysis_log.csv"
    target.parent.mkdir(parents=True, exist_ok=True)

    rows = (await session.execute(text("""
        SELECT p.title, p.venue, p.year, p.category, p.state,
               p.paper_link, p.code_url, p.title_sanitized
        FROM papers p
        WHERE p.state NOT IN ('archived_or_expired', 'skip')
        ORDER BY p.category, p.venue, p.year
    """))).fetchall()

    with open(target, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "venue", "year", "category", "state", "paper_link", "code_url", "filename"])
        for r in rows:
            writer.writerow([r.title, r.venue, r.year, r.category, r.state, r.paper_link, r.code_url, r.title_sanitized])

    return len(rows)


async def build_collection_index(
    session: AsyncSession,
    out_dir: str | None = None,
) -> dict:
    """Build paperCollection/index.jsonl + navigation pages from DB.

    Returns stats about what was generated.
    """
    import json

    root = Path(out_dir or settings.paper_collection_dir)
    root.mkdir(parents=True, exist_ok=True)

    rows = (await session.execute(text("""
        SELECT p.id, p.title, p.venue, p.year, p.category, p.tags,
               p.mechanism_family, p.structurality_score,
               p.keep_score, p.importance, p.state,
               dc.delta_statement, dc.baseline_paradigm
        FROM papers p
        LEFT JOIN delta_cards dc ON dc.id = p.current_delta_card_id
        WHERE p.state NOT IN ('archived_or_expired', 'skip')
        ORDER BY p.category, p.venue, p.year
    """))).fetchall()

    # Write index.jsonl
    index_path = root / "index.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for r in rows:
            entry = {
                "id": str(r.id),
                "title": r.title,
                "venue": r.venue,
                "year": r.year,
                "category": r.category,
                "tags": list(r.tags) if r.tags else [],
                "mechanism_family": r.mechanism_family,
                "structurality_score": float(r.structurality_score) if r.structurality_score else None,
                "keep_score": float(r.keep_score) if r.keep_score else None,
                "importance": r.importance,
                "state": r.state,
                "delta_statement": r.delta_statement[:200] if r.delta_statement else None,
                "paradigm": r.baseline_paradigm,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Generate by_venue navigation page
    venues: dict[str, list] = {}
    for r in rows:
        key = f"{r.venue}_{r.year}" if r.venue else "Unknown"
        venues.setdefault(key, []).append(r.title)

    venue_page = root / "by_venue.md"
    with open(venue_page, "w", encoding="utf-8") as f:
        f.write("# Papers by Venue\n\n")
        for venue, titles in sorted(venues.items()):
            f.write(f"## {venue} ({len(titles)})\n\n")
            for t in titles:
                f.write(f"- {t}\n")
            f.write("\n")

    # Generate by_category navigation page
    categories: dict[str, list] = {}
    for r in rows:
        categories.setdefault(r.category, []).append(r.title)

    cat_page = root / "by_category.md"
    with open(cat_page, "w", encoding="utf-8") as f:
        f.write("# Papers by Category\n\n")
        for cat, titles in sorted(categories.items()):
            f.write(f"## {cat} ({len(titles)})\n\n")
            for t in titles:
                f.write(f"- {t}\n")
            f.write("\n")

    return {
        "index_entries": len(rows),
        "venues": len(venues),
        "categories": len(categories),
        "files": ["index.jsonl", "by_venue.md", "by_category.md"],
    }
