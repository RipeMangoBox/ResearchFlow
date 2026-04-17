"""Migrate paperAnalysis/*.md → paper_analyses table + update papers metadata.

Walks paperAnalysis/<Category>/<Venue_Year>/<Year_Title>.md, parses YAML
frontmatter and body sections, matches to existing papers rows (by
title_sanitized + venue + year), and inserts paper_analyses at L4 level.

Usage: python -m migration.migrate_md_to_db [--dir PATH] [--db-url URL]
"""

import argparse
import re
import sys
from pathlib import Path

import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings


# ── Frontmatter parsing ────────────────────────────────────────

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_md(filepath: Path) -> tuple[dict, str]:
    """Return (frontmatter_dict, body)."""
    raw = filepath.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, raw[m.end():]


def extract_sections(body: str) -> dict[str, str]:
    """Extract Part I/II/III and core intuition."""
    out = {}
    for pattern, key in [
        (r"##\s*Part I[：:][^\n]*\n(.*?)(?=\n##\s*Part II|$)", "problem_summary"),
        (r"##\s*Part II[：:][^\n]*\n(.*?)(?=\n##\s*Part III|$)", "method_summary"),
        (r"##\s*Part III[：:][^\n]*\n(.*?)(?=\n##\s|$)", "evidence_summary"),
    ]:
        match = re.search(pattern, body, re.DOTALL)
        if match:
            out[key] = match.group(1).strip()
    # Core intuition
    aha = re.search(
        r"###\s*(?:核心直觉|The \"Aha!\" Moment)[^\n]*\n(.*?)(?=\n##|\n---|\Z)",
        body, re.DOTALL,
    )
    if aha:
        out["core_intuition"] = aha.group(1).strip()
    return out


def sanitize_title(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", title)
    return re.sub(r"\s+", "_", s.strip())


# ── Main ────────────────────────────────────────────────────────

def migrate(analysis_dir: str, db_url: str) -> None:
    engine = create_engine(db_url)
    root = Path(analysis_dir)
    md_files = sorted(root.rglob("*.md"))
    # Filter out non-analysis files (READMEs, logs, etc.)
    md_files = [
        f for f in md_files
        if f.name not in ("README.md", "analysis_log.csv")
        and "processing" not in str(f)
        and "quality_report" not in f.name
    ]

    print(f"Found {len(md_files)} analysis MD files")

    inserted = 0
    updated = 0
    skipped = 0

    with Session(engine) as session:
        for md_path in md_files:
            fm, body = parse_md(md_path)
            if not fm.get("title"):
                skipped += 1
                continue

            title = fm["title"].strip('"').strip("'")
            title_san = sanitize_title(title)

            # Venue from frontmatter (just the abbreviation, e.g. "CVPR")
            venue = fm.get("venue", "")
            if isinstance(venue, str):
                venue = venue.strip()
            year = fm.get("year")

            # Find matching paper
            row = session.execute(
                text("SELECT id FROM papers WHERE title_sanitized = :ts AND venue = :v AND year = :y"),
                {"ts": title_san, "v": venue, "y": year},
            ).fetchone()

            if not row:
                # Try fuzzy match by title_sanitized only
                row = session.execute(
                    text("SELECT id FROM papers WHERE title_sanitized = :ts LIMIT 1"),
                    {"ts": title_san},
                ).fetchone()

            if not row:
                # Create paper from frontmatter
                category = fm.get("category", md_path.parent.parent.name)
                tags = fm.get("tags", [])
                if isinstance(tags, str):
                    tags = [tags]

                session.execute(
                    text("""
                        INSERT INTO papers (
                            title, title_sanitized, venue, year, category,
                            state, tags, core_operator, primary_logic, claims,
                            pdf_path_local, source
                        ) VALUES (
                            :title, :ts, :venue, :year, :cat,
                            'checked', :tags, :core_op, :primary, :claims,
                            :pdf_ref, 'md_migration'
                        )
                    """),
                    {
                        "title": title,
                        "ts": title_san,
                        "venue": venue,
                        "year": year,
                        "cat": category,
                        "tags": tags,
                        "core_op": fm.get("core_operator"),
                        "primary": fm.get("primary_logic"),
                        "claims": fm.get("claims", []),
                        "pdf_ref": fm.get("pdf_ref"),
                    },
                )
                session.flush()
                row = session.execute(
                    text("SELECT id FROM papers WHERE title_sanitized = :ts AND venue = :v AND year = :y"),
                    {"ts": title_san, "v": venue, "y": year},
                ).fetchone()
                inserted += 1
            else:
                # Update existing paper with frontmatter fields
                tags = fm.get("tags", [])
                if isinstance(tags, str):
                    tags = [tags]
                session.execute(
                    text("""
                        UPDATE papers SET
                            core_operator = COALESCE(:core_op, core_operator),
                            primary_logic = COALESCE(:primary, primary_logic),
                            claims = COALESCE(:claims, claims),
                            tags = COALESCE(:tags, tags),
                            state = CASE WHEN state IN ('wait','downloaded') THEN 'checked' ELSE state END,
                            analyzed_at = NOW(),
                            updated_at = NOW()
                        WHERE id = :id
                    """),
                    {
                        "core_op": fm.get("core_operator"),
                        "primary": fm.get("primary_logic"),
                        "claims": fm.get("claims", []),
                        "tags": tags,
                        "id": row[0],
                    },
                )
                updated += 1

            paper_id = row[0]

            # Extract sections from body
            sections = extract_sections(body)

            # Check if analysis already exists
            existing_analysis = session.execute(
                text("SELECT id FROM paper_analyses WHERE paper_id = :pid AND level = 'l4_deep' AND is_current = true"),
                {"pid": paper_id},
            ).fetchone()

            if existing_analysis:
                continue  # Don't duplicate

            # Insert paper_analyses (L4 deep)
            session.execute(
                text("""
                    INSERT INTO paper_analyses (
                        paper_id, level, model_provider, model_name,
                        prompt_version, schema_version,
                        problem_summary, method_summary, evidence_summary,
                        core_intuition, full_report_md,
                        is_current, generated_at
                    ) VALUES (
                        :pid, 'l4_deep', 'manual_migration', 'legacy',
                        'legacy', 'v0',
                        :problem, :method, :evidence,
                        :intuition, :full_report,
                        true, NOW()
                    )
                """),
                {
                    "pid": paper_id,
                    "problem": sections.get("problem_summary"),
                    "method": sections.get("method_summary"),
                    "evidence": sections.get("evidence_summary"),
                    "intuition": sections.get("core_intuition"),
                    "full_report": body,
                },
            )

        session.commit()

    print(f"Papers: {inserted} created, {updated} updated, {skipped} skipped")
    print(f"Analyses: {inserted + updated - skipped} L4 records inserted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate analysis MDs to DB")
    parser.add_argument(
        "--dir",
        default=str(Path(__file__).resolve().parent.parent.parent / "paperAnalysis"),
        help="Path to paperAnalysis/ directory",
    )
    parser.add_argument("--db-url", default=settings.database_url_sync, help="Database URL")
    args = parser.parse_args()
    migrate(args.dir, args.db_url)
