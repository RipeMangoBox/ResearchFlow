"""Validate migration: cross-check DB against original CSV and MD files.

Usage: python -m migration.validate_migration [--db-url URL]
"""

import argparse
import csv
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings


def validate(db_url: str) -> bool:
    engine = create_engine(db_url)
    project_root = Path(__file__).resolve().parent.parent.parent
    ok = True

    with Session(engine) as session:
        # 1. Count papers in DB
        db_count = session.execute(text("SELECT COUNT(*) FROM papers")).scalar()
        print(f"[DB] Total papers: {db_count}")

        # 2. Count rows in CSV
        csv_path = project_root / "paperAnalysis" / "analysis_log.csv"
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                csv_rows = sum(1 for _ in csv.DictReader(f))
            print(f"[CSV] Total rows: {csv_rows}")
        else:
            print("[CSV] analysis_log.csv not found")
            csv_rows = 0

        # 3. Count MD files
        analysis_dir = project_root / "paperAnalysis"
        md_files = [
            f for f in analysis_dir.rglob("*.md")
            if f.name not in ("README.md",)
            and "processing" not in str(f)
            and "quality_report" not in f.name
        ]
        print(f"[MD] Total analysis files: {len(md_files)}")

        # 4. Count analyses in DB
        analysis_count = session.execute(text("SELECT COUNT(*) FROM paper_analyses")).scalar()
        print(f"[DB] Total analyses: {analysis_count}")

        # 5. Check every 'checked' paper has an analysis
        checked_no_analysis = session.execute(text("""
            SELECT p.title FROM papers p
            WHERE p.state = 'checked'
            AND NOT EXISTS (
                SELECT 1 FROM paper_analyses pa
                WHERE pa.paper_id = p.id AND pa.is_current = true
            )
        """)).fetchall()
        if checked_no_analysis:
            print(f"\n[WARN] {len(checked_no_analysis)} checked papers without analysis:")
            for r in checked_no_analysis:
                print(f"  - {r[0]}")
            ok = False

        # 6. Count PDFs on disk
        pdfs_dir = project_root / "paperPDFs"
        pdf_count = len(list(pdfs_dir.rglob("*.pdf"))) if pdfs_dir.exists() else 0
        papers_with_pdf = session.execute(
            text("SELECT COUNT(*) FROM papers WHERE pdf_path_local IS NOT NULL AND pdf_path_local != ''")
        ).scalar()
        print(f"\n[PDF] On disk: {pdf_count}, in DB (pdf_path_local): {papers_with_pdf}")

        # 7. State distribution
        states = session.execute(text(
            "SELECT state, COUNT(*) FROM papers GROUP BY state ORDER BY COUNT(*) DESC"
        )).fetchall()
        print("\n[DB] State distribution:")
        for state, count in states:
            print(f"  {state}: {count}")

        # 8. Category distribution
        cats = session.execute(text(
            "SELECT category, COUNT(*) FROM papers GROUP BY category ORDER BY COUNT(*) DESC"
        )).fetchall()
        print("\n[DB] Category distribution:")
        for cat, count in cats:
            print(f"  {cat}: {count}")

    print(f"\n{'VALIDATION PASSED' if ok else 'VALIDATION HAS WARNINGS'}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate migration")
    parser.add_argument("--db-url", default=settings.database_url_sync, help="Database URL")
    args = parser.parse_args()
    success = validate(args.db_url)
    sys.exit(0 if success else 1)
