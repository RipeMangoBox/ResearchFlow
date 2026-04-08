from __future__ import annotations

from pathlib import Path
from typing import List, Set

from check_paper_downloads import LOG_PATH, parse_log  # type: ignore[import]


def normalize_title(text: str) -> str:
    """Normalize titles for comparison."""
    import re

    s = text.lower()
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def load_actual_titles(paper_list_path: Path) -> Set[str]:
    """
    Load normalized titles from paper_list.txt.

    Each line format:
      category|venue_dir|filename.pdf
    """
    titles: Set[str] = set()
    if not paper_list_path.is_file():
        return titles

    with paper_list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            filename = parts[2]
            if not filename.lower().endswith(".pdf"):
                continue
            stem = filename[:-4]
            if len(stem) > 5 and stem[:4].isdigit() and stem[4] in {"_", " "}:
                stem = stem[5:]
            norm = normalize_title(stem)
            if norm:
                titles.add(norm)
    return titles


def main() -> None:
    log_path = Path(LOG_PATH)
    paper_list_path = log_path.parent / "paper_list.txt"

    print(f"Using log file: {log_path}")
    print(f"Using paper list: {paper_list_path}")

    entries = parse_log(log_path)
    have_titles = load_actual_titles(paper_list_path)

    print(f"Total log entries: {len(entries)}")
    print(f"Distinct titles with local PDFs: {len(have_titles)}")

    missing_indices: Set[int] = set()
    for e in entries:
        norm = normalize_title(e.title)
        if not norm:
            continue
        if norm not in have_titles:
            missing_indices.add(e.index)

    print(f"Entries without matching local PDF (by title): {len(missing_indices)}")
    if not missing_indices:
        print("Nothing to mark as WAIT.")
        return

    lines = log_path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines: List[str] = []
    for i, line in enumerate(lines, start=1):
        if i in missing_indices and line.startswith("Downloaded"):
            new_lines.append(line.replace("Downloaded", "WAIT", 1))
        else:
            new_lines.append(line)

    log_path.write_text("".join(new_lines), encoding="utf-8")
    print("Updated statuses to WAIT for missing entries.")


if __name__ == "__main__":
    main()

