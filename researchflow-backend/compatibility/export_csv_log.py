"""Export DB → analysis_log.csv (backward-compatible CSV export).

Generates a CSV matching the original 8-column format.

Usage: python -m compatibility.export_csv_log [--db-url URL] [--out PATH]
"""

import argparse
import csv
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings

# Map DB state back to original CSV state values
STATE_EXPORT_MAP = {
    "wait": "",
    "downloaded": "Downloaded",
    "checked": "checked",
    "l1_metadata": "Downloaded",
    "l2_parsed": "Downloaded",
    "l3_skimmed": "checked",
    "l4_deep": "checked",
    "skip": "Skip",
    "missing": "Missing",
    "too_large": "too_large",
    "analysis_mismatch": "analysis_mismatch",
}


def export(db_url: str, out_path: str) -> None:
    engine = create_engine(db_url)

    with Session(engine) as session:
        rows = session.execute(text("""
            SELECT
                state, importance, title, venue, year,
                project_link, paper_link, category, pdf_path_local
            FROM papers
            ORDER BY
                CASE state
                    WHEN 'checked' THEN 0
                    WHEN 'l4_deep' THEN 0
                    WHEN 'l3_skimmed' THEN 0
                    WHEN 'downloaded' THEN 1
                    WHEN 'l2_parsed' THEN 1
                    WHEN 'l1_metadata' THEN 1
                    ELSE 2
                END,
                category, venue, year
        """)).fetchall()

    fieldnames = [
        "state", "importance", "paper_title", "venue",
        "project_link_or_github_link", "paper_link", "sort", "pdf_path",
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            state, imp, title, venue, year, project_link, paper_link, category, pdf_path = row

            # Reconstruct combined venue string
            venue_str = f"{venue} {year}" if venue and year else (venue or "")

            writer.writerow({
                "state": STATE_EXPORT_MAP.get(state, state),
                "importance": imp or "",
                "paper_title": title,
                "venue": venue_str,
                "project_link_or_github_link": project_link or "",
                "paper_link": paper_link or "",
                "sort": category,
                "pdf_path": pdf_path or "",
            })

    print(f"Exported {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DB to CSV log")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "paperAnalysis" / "analysis_log.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--db-url", default=settings.database_url_sync, help="Database URL")
    args = parser.parse_args()
    export(args.db_url, args.out)
