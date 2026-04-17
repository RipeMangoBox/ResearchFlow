"""Export DB → paperAnalysis/*.md (backward-compatible Markdown export).

Generates Markdown files with YAML frontmatter matching the original format,
so existing .claude/skills/ continue working.

Usage: python -m compatibility.export_analysis_md [--db-url URL] [--out-dir PATH]
"""

import argparse
import sys
from pathlib import Path

import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings


def render_frontmatter(data: dict) -> str:
    """Render dict as YAML frontmatter block."""
    yaml_str = yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=200,
    )
    return f"---\n{yaml_str}---\n"


def export(db_url: str, out_dir: str) -> None:
    engine = create_engine(db_url)
    root = Path(out_dir)

    with Session(engine) as session:
        # Get all papers with current L4 analysis
        rows = session.execute(text("""
            SELECT
                p.title, p.venue, p.year, p.category, p.tags,
                p.core_operator, p.primary_logic, p.claims,
                p.pdf_path_local, p.title_sanitized,
                pa.full_report_md, pa.problem_summary, pa.method_summary,
                pa.evidence_summary, pa.core_intuition
            FROM papers p
            LEFT JOIN paper_analyses pa ON pa.paper_id = p.id
                AND pa.is_current = true AND pa.level = 'l4_deep'
            WHERE p.state = 'checked'
            ORDER BY p.category, p.venue, p.year
        """)).fetchall()

        exported = 0
        for row in rows:
            (
                title, venue, year, category, tags,
                core_operator, primary_logic, claims,
                pdf_path_local, title_san,
                full_report_md, problem_summary, method_summary,
                evidence_summary, core_intuition,
            ) = row

            # Build frontmatter
            fm = {"title": f'"{title}"'}
            if venue:
                fm["venue"] = venue
            if year:
                fm["year"] = year
            if tags:
                fm["tags"] = list(tags)
            if core_operator:
                fm["core_operator"] = core_operator
            if primary_logic:
                fm["primary_logic"] = primary_logic
            if claims:
                fm["claims"] = list(claims)
            if pdf_path_local:
                fm["pdf_ref"] = pdf_path_local
            if category:
                fm["category"] = category

            # Determine output path
            venue_year = f"{venue}_{year}" if venue and year else "Unknown"
            filename = f"{year}_{title_san}.md" if year else f"{title_san}.md"
            out_path = root / category / venue_year / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # If we have full_report_md from migration, use it directly
            if full_report_md:
                content = render_frontmatter(fm) + "\n" + full_report_md
            else:
                # Reconstruct from sections
                parts = [render_frontmatter(fm), "", f"# {title}", ""]
                if problem_summary:
                    parts.extend(["## Part I：问题与挑战", "", problem_summary, ""])
                if method_summary:
                    parts.extend(["## Part II：方法与洞察", "", method_summary, ""])
                if evidence_summary:
                    parts.extend(["## Part III：证据与局限", "", evidence_summary, ""])
                if core_intuition:
                    parts.extend(["### 核心直觉", "", core_intuition, ""])
                content = "\n".join(parts)

            out_path.write_text(content, encoding="utf-8")
            exported += 1

    print(f"Exported {exported} analysis files to {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DB to analysis Markdown")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent.parent.parent / "paperAnalysis"),
        help="Output directory (default: ../paperAnalysis)",
    )
    parser.add_argument("--db-url", default=settings.database_url_sync, help="Database URL")
    args = parser.parse_args()
    export(args.db_url, args.out_dir)
