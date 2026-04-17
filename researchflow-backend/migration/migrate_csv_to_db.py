"""Migrate analysis_log.csv → papers table.

Usage: python -m migration.migrate_csv_to_db [--csv PATH] [--db-url URL]
"""

import argparse
import csv
import re
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings


# ── Helpers ─────────────────────────────────────────────────────

def parse_venue_year(venue_str: str) -> tuple[str, int | None]:
    """Split 'CVPR 2025' → ('CVPR', 2025)."""
    if not venue_str:
        return "", None
    m = re.match(r"^(.+?)\s+(\d{4})$", venue_str.strip())
    if m:
        return m.group(1).strip(), int(m.group(2))
    return venue_str.strip(), None


def extract_arxiv_id(url: str) -> str | None:
    if not url:
        return None
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", url)
    return m.group(1) if m else None


def sanitize_title(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", title)
    return re.sub(r"\s+", "_", s.strip())


def infer_title_from_pdf_path(pdf_path: str) -> str:
    """Extract a best-guess title from pdf_path like .../2026_Some_Title.pdf."""
    stem = Path(pdf_path).stem
    # Remove leading year prefix
    stem = re.sub(r"^\d{4}_", "", stem)
    return stem.replace("_", " ")


def infer_year_from_pdf_path(pdf_path: str) -> int | None:
    stem = Path(pdf_path).stem
    m = re.match(r"^(\d{4})_", stem)
    return int(m.group(1)) if m else None


def infer_venue_from_pdf_path(pdf_path: str) -> str | None:
    """Extract venue from path like paperPDFs/Cat/ICLR_2026/..."""
    parts = Path(pdf_path).parts
    for p in parts:
        m = re.match(r"^([A-Za-z]+)_(\d{4})$", p)
        if m:
            return m.group(1)
    return None


STATE_MAP = {
    "checked": "checked",
    "Downloaded": "downloaded",
    "Wait": "wait",
    "Skip": "skip",
    "Missing": "missing",
    "too_large": "too_large",
    "analysis_mismatch": "analysis_mismatch",
    "": "wait",  # empty state → wait
}


def map_state(raw: str) -> str:
    return STATE_MAP.get(raw.strip(), "wait")


# ── Main ────────────────────────────────────────────────────────

def migrate(csv_path: str, db_url: str) -> None:
    engine = create_engine(db_url)
    rows_inserted = 0
    rows_skipped = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Read {len(rows)} rows from {csv_path}")

    with Session(engine) as session:
        for row in rows:
            title = row.get("paper_title", "").strip()
            pdf_path = row.get("pdf_path", "").strip()
            sort_cat = row.get("sort", "").strip()

            # Rows with no title — infer from pdf_path
            if not title:
                if not pdf_path:
                    rows_skipped += 1
                    continue
                title = infer_title_from_pdf_path(pdf_path)

            title_san = sanitize_title(title)

            # Venue / year
            raw_venue = row.get("venue", "").strip()
            venue, year = parse_venue_year(raw_venue)
            if not venue and pdf_path:
                venue = infer_venue_from_pdf_path(pdf_path) or ""
            if not year and pdf_path:
                year = infer_year_from_pdf_path(pdf_path)

            # Category
            category = sort_cat or "Uncategorized"

            # State
            state = map_state(row.get("state", ""))

            # Importance
            imp = row.get("importance", "").strip() or None

            # Links
            paper_link = row.get("paper_link", "").strip() or None
            project_link = row.get("project_link_or_github_link", "").strip() or None
            arxiv_id = extract_arxiv_id(paper_link) if paper_link else None

            # Deduplicate by title_sanitized + venue + year
            existing = session.execute(
                text(
                    "SELECT id FROM papers WHERE title_sanitized = :ts AND venue = :v AND year = :y"
                ),
                {"ts": title_san, "v": venue, "y": year},
            ).fetchone()
            if existing:
                rows_skipped += 1
                continue

            session.execute(
                text("""
                    INSERT INTO papers (
                        title, title_sanitized, venue, year, category,
                        state, importance,
                        paper_link, project_link, arxiv_id,
                        pdf_path_local, tags, source
                    ) VALUES (
                        :title, :title_san, :venue, :year, :category,
                        :state, :importance,
                        :paper_link, :project_link, :arxiv_id,
                        :pdf_path, :tags, :source
                    )
                """),
                {
                    "title": title,
                    "title_san": title_san,
                    "venue": venue,
                    "year": year,
                    "category": category,
                    "state": state,
                    "importance": imp,
                    "paper_link": paper_link,
                    "project_link": project_link,
                    "arxiv_id": arxiv_id,
                    "pdf_path": pdf_path or None,
                    "tags": [category] if category else [],
                    "source": "csv_migration",
                },
            )
            rows_inserted += 1

        session.commit()

    print(f"Inserted {rows_inserted} papers, skipped {rows_skipped} duplicates/empty.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate analysis_log.csv to DB")
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parent.parent.parent / "paperAnalysis" / "analysis_log.csv"),
        help="Path to analysis_log.csv",
    )
    parser.add_argument("--db-url", default=settings.database_url_sync, help="Database URL")
    args = parser.parse_args()
    migrate(args.csv, args.db_url)
