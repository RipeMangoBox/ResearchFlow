from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from check_paper_downloads import LogEntry  # type: ignore[import]
from check_paper_downloads import PAPER_ROOT, parse_log  # type: ignore[import]


def normalize_title(text: str) -> str:
    """Simple normalization to compare titles."""
    import re

    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def compute_hash(path: Path, block_size: int = 1 << 20) -> str:
    """Compute SHA1 hash of a file (streaming)."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_entries_by_title() -> Dict[str, List[LogEntry]]:
    """Index log entries by normalized title."""
    entries = parse_log(Path(PAPER_ROOT) / "download_log_updated.txt")
    by_title: Dict[str, List[LogEntry]] = {}
    for e in entries:
        key = normalize_title(e.title)
        by_title.setdefault(key, []).append(e)
    return by_title


def is_preferred_path(path: Path) -> bool:
    """
    Heuristic for 'good' naming:
    - filename starts with 4 digits and underscore, ends with .pdf
    - parent directory looks like CONFERENCE_YEAR (e.g. CVPR_2024)
    """
    name = path.name
    if not name.lower().endswith(".pdf"):
        return False
    if len(name) < 9 or not name[:4].isdigit() or name[4] != "_":
        return False

    parts = path.parts
    if len(parts) < 3:
        return False
    parent = parts[-2]
    if "_" in parent:
        prefix, _, year = parent.partition("_")
        if prefix.isalpha() and len(year) == 4 and year.isdigit():
            return True
    return False


def choose_canonical(paths: List[Path]) -> Path:
    """Choose which path to keep among duplicates."""
    preferred = [p for p in paths if is_preferred_path(p)]
    if preferred:
        return sorted(preferred, key=lambda p: str(p))[0]
    return sorted(paths, key=lambda p: str(p))[0]


def main() -> None:
    root = Path(PAPER_ROOT)
    print(f"Scanning PDFs under: {root}")

    pdfs = sorted(root.rglob("*.pdf"))
    print(f"Total PDFs found: {len(pdfs)}")

    groups: Dict[Tuple[int, str], List[Path]] = {}
    for p in pdfs:
        try:
            size = p.stat().st_size
        except OSError:
            continue
        h = compute_hash(p)
        groups.setdefault((size, h), []).append(p)

    duplicates: List[List[Path]] = [g for g in groups.values() if len(g) > 1]
    print(f"Duplicate groups detected (by content): {len(duplicates)}")

    total_deleted = 0
    for group in duplicates:
        canonical = choose_canonical(group)
        print("\n[GROUP]")
        for p in group:
            mark = "*" if p == canonical else " "
            print(f" {mark} {p}")

        to_delete = [p for p in group if p != canonical]
        for p in to_delete:
            try:
                p.unlink()
                print(f"  [DEL] {p}")
                total_deleted += 1
            except Exception as exc:
                print(f"  [ERROR] Failed to delete {p}: {exc}")

    print(f"\nDone. Total duplicate files deleted: {total_deleted}")


if __name__ == "__main__":
    main()

