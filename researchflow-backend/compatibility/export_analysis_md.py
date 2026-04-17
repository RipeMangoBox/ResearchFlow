"""Export DB → paperAnalysis/*.md (backward-compatible Markdown export).

v3: Exports from DeltaCard (primary) with fallback to PaperAnalysis.
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
        # Get all papers with current L4 analysis + optional DeltaCard
        rows = session.execute(text("""
            SELECT
                p.id, p.title, p.venue, p.year, p.category, p.tags,
                p.core_operator, p.primary_logic, p.claims,
                p.pdf_path_local, p.title_sanitized,
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
            LEFT JOIN delta_cards dc ON dc.paper_id = p.id
                AND dc.status != 'deprecated'
            WHERE p.state IN ('checked', 'l4_deep')
            ORDER BY p.category, p.venue, p.year
        """)).fetchall()

        exported = 0
        for row in rows:
            (
                paper_id, title, venue, year, category, tags,
                core_operator, primary_logic, claims,
                pdf_path_local, title_san,
                full_report_md, problem_summary, method_summary,
                evidence_summary, core_intuition,
                dc_statement, dc_baseline, dc_struct, dc_transfer,
                dc_key_ideas, dc_assumptions, dc_failure_modes, dc_status,
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

            # v3: DeltaCard metadata in frontmatter
            if dc_statement:
                fm["delta_statement"] = dc_statement[:300]
            if dc_baseline:
                fm["baseline_paradigm"] = dc_baseline
            if dc_struct is not None:
                fm["structurality_score"] = round(dc_struct, 3)
            if dc_transfer is not None:
                fm["transferability_score"] = round(dc_transfer, 3)
            if dc_status:
                fm["delta_card_status"] = dc_status

            # Determine output path
            venue_year = f"{venue}_{year}" if venue and year else "Unknown"
            filename = f"{year}_{title_san}.md" if year else f"{title_san}.md"
            out_path = root / category / venue_year / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Build content
            if full_report_md:
                content = render_frontmatter(fm) + "\n" + full_report_md
            else:
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

            # v3: Append DeltaCard section if available
            if dc_statement:
                dc_parts = ["\n## DeltaCard", ""]
                dc_parts.append(f"**核心改动**: {dc_statement}")
                if dc_baseline:
                    dc_parts.append(f"\n**对照基线**: {dc_baseline}")
                if dc_struct is not None or dc_transfer is not None:
                    scores = []
                    if dc_struct is not None:
                        scores.append(f"结构性={dc_struct:.2f}")
                    if dc_transfer is not None:
                        scores.append(f"可迁移性={dc_transfer:.2f}")
                    dc_parts.append(f"\n**评分**: {', '.join(scores)}")
                if dc_key_ideas:
                    dc_parts.append("\n**关键洞察**:")
                    for ki in (dc_key_ideas if isinstance(dc_key_ideas, list) else []):
                        if isinstance(ki, dict):
                            dc_parts.append(f"  {ki.get('rank', '?')}. {ki.get('statement', '')} (conf={ki.get('confidence', '?')})")
                if dc_assumptions:
                    dc_parts.append(f"\n**假设**: {'; '.join(dc_assumptions[:5])}")
                if dc_failure_modes:
                    dc_parts.append(f"\n**失败模式**: {'; '.join(dc_failure_modes[:5])}")
                content += "\n".join(dc_parts) + "\n"

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
