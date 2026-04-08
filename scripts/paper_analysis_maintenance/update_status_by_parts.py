from __future__ import annotations

import csv
import io
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PA_DIR = REPO_ROOT / "paperAnalysis"
LOG_PATH = PA_DIR / "analysis_log.csv"


def load_log() -> list[list[str]]:
    with io.open(LOG_PATH, "r", encoding="utf-8", newline="") as f:
        return [row for row in csv.reader(f)]


def save_log(rows: list[list[str]]) -> None:
    with io.open(LOG_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def get_first_header(text: str) -> str | None:
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            s = s.lstrip("#").strip()
        return s
    return None


def main() -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Missing log: {LOG_PATH}")

    rows = load_log()
    if not rows:
        print("Empty log.")
        return

    data_start = 1 if rows[0] and rows[0][0] == "state" else 0

    updated = 0
    unmatched: list[tuple[str, str]] = []

    for root, _, files in os.walk(PA_DIR):
        for fn in files:
            if not fn.lower().endswith(".md"):
                continue
            path = os.path.join(root, fn)
            with io.open(path, "r", encoding="utf-8") as f:
                txt = f.read()

            has_all = all(p in txt for p in ("Part I", "Part II", "Part III"))
            if has_all:
                continue

            header = get_first_header(txt)
            if not header:
                unmatched.append((path, "NO_HEADER"))
                continue

            hits = []
            for i in range(data_start, len(rows)):
                row = (rows[i] + [""] * 8)[:8]
                if header and header in row[2]:
                    hits.append(i)

            if len(hits) != 1:
                unmatched.append((path, f"MATCHES={len(hits)}"))
                continue

            idx = hits[0]
            row = (rows[idx] + [""] * 8)[:8]
            if row[0].strip() == "checked":
                row[0] = "Wait"
                rows[idx] = row
                updated += 1

    if updated:
        save_log(rows)

    print(f"Updated {updated} log row(s) from checked to Wait.")
    if unmatched:
        print("Unmatched md files (需要手动确认):")
        for p, reason in unmatched:
            print(" -", p, "->", reason)


if __name__ == "__main__":
    main()
