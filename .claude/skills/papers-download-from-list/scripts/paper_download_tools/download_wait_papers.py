from __future__ import annotations

import re
from pathlib import Path
from typing import Set

import requests

from check_paper_downloads import (  # type: ignore[import]
    LOG_PATH,
    PAPER_ROOT,
    LogEntry,
    parse_log,
)


def slugify(text: str) -> str:
    """Convert text into a filesystem-friendly slug."""
    text = re.sub(r"[^0-9a-zA-Z]+", " ", text)
    parts = [p for p in text.strip().split() if p]
    return "_".join(parts)


def extract_year(venue: str) -> str:
    """Extract a 4-digit year from the venue string, if present."""
    years = re.findall(r"\d{4}", venue)
    return years[-1] if years else ""


def build_target_path(entry: LogEntry) -> Path:
    """
    Build a target local PDF path for a WAIT entry, following
    paperPDFs/<category>/<venue_year>/<year>_<title>.pdf
    """
    category = entry.category or "Uncategorized"
    venue = entry.venue or "Unknown"

    category_dir = Path(PAPER_ROOT) / category.replace(" ", "_")
    venue_slug = slugify(venue)
    venue_dir = category_dir / venue_slug

    year = extract_year(venue)
    title_slug = slugify(entry.title or "paper")

    if year:
        filename = f"{year}_{title_slug}.pdf"
    else:
        filename = f"{title_slug}.pdf"

    return venue_dir / filename


def main() -> None:
    log_path = Path(LOG_PATH)
    print(f"Using log file: {log_path}")

    entries = parse_log(log_path)

    wait_entries = [e for e in entries if e.status.strip().upper() == "WAIT"]
    print(f"Total WAIT entries: {len(wait_entries)}")

    session = requests.Session()
    succeeded_lines: Set[int] = set()

    for e in wait_entries:
        target = build_target_path(e)
        url = e.pdf_url.strip()
        if not url:
            print(f"[SKIP] Empty URL for line {e.index}: {e.title}")
            continue

        target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.is_file():
            print(f"[SKIP] Already exists for line {e.index}: {target}")
            succeeded_lines.add(e.index)
            continue

        print(f"[DOWNLOAD] line {e.index}: {e.title}")
        print(f"  URL: {url}")
        print(f"  ->  {target}")

        try:
            resp = session.get(url, timeout=180)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  [ERROR] Failed to download: {exc}")
            continue

        content_type = resp.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type:
            print(f"  [WARN] Content-Type not PDF: {content_type}")

        try:
            target.write_bytes(resp.content)
        except Exception as exc:
            print(f"  [ERROR] Failed to write file: {exc}")
            continue

        print("  [OK]")
        succeeded_lines.add(e.index)

    if succeeded_lines:
        print(f"Updating status to Downloaded for {len(succeeded_lines)} lines.")
        lines = log_path.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = []
        for i, line in enumerate(lines, start=1):
            if i in succeeded_lines and line.startswith("WAIT "):
                new_lines.append(line.replace("WAIT", "Downloaded", 1))
            else:
                new_lines.append(line)
        log_path.write_text("".join(new_lines), encoding="utf-8")

    print("Done processing WAIT entries.")


if __name__ == "__main__":
    main()

