"""
Download Wait entries from ICLR_2026.txt using the explicit pdf_path field.

Column layout (0-indexed):
  0: state | 1: title | 2: venue | 3: paper_link | 4: project_link | 5: sort | 6: pdf_path
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[5]
LOG_PATH = REPO_ROOT / "paperAnalysis" / "ICLR_2026.txt"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}


def parse_entries(log: Path):
    entries = []
    with log.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(" | ")]
            if len(parts) < 7:
                continue
            state, title, venue, paper_link, project_link, sort, pdf_path = (
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
            )
            if state.upper() == "WAIT":
                entries.append({
                    "lineno": lineno,
                    "title": title,
                    "url": paper_link,
                    "pdf_path": REPO_ROOT / pdf_path,
                    "raw": raw,
                })
    return entries


def download(entry: dict, session: requests.Session) -> bool:
    url = entry["url"]
    dest: Path = entry["pdf_path"]
    title = entry["title"]

    if not url:
        print(f"  [SKIP] no URL — {title}")
        return False

    if dest.is_file() and dest.stat().st_size > 10_000:
        print(f"  [EXISTS] {dest.name}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [DL] {title[:70]}")
    print(f"       {url}")
    print(f"       -> {dest}")

    try:
        resp = session.get(url, headers=HEADERS, timeout=120, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

    ct = resp.headers.get("Content-Type", "")
    if "pdf" not in ct.lower() and not url.lower().endswith(".pdf"):
        print(f"  [WARN] Content-Type: {ct} — saving anyway")

    if len(resp.content) < 5_000:
        print(f"  [WARN] Very small response ({len(resp.content)} bytes) — skipping")
        return False

    dest.write_bytes(resp.content)
    print(f"  [OK] {len(resp.content)//1024} KB")
    return True


def update_log(log: Path, succeeded_linenos: set[int]):
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []
    for i, line in enumerate(lines, 1):
        if i in succeeded_linenos and line.startswith("Wait "):
            new_lines.append(line.replace("Wait", "Downloaded", 1))
        else:
            new_lines.append(line)
    log.write_text("".join(new_lines), encoding="utf-8")


def main():
    entries = parse_entries(LOG_PATH)
    print(f"Found {len(entries)} Wait entries in {LOG_PATH.name}\n")

    session = requests.Session()
    succeeded = set()

    for e in entries:
        ok = download(e, session)
        if ok:
            succeeded.add(e["lineno"])
        time.sleep(1.5)  # polite delay

    print(f"\n--- {len(succeeded)}/{len(entries)} downloaded successfully ---")

    if succeeded:
        update_log(LOG_PATH, succeeded)
        print("Log updated: Wait -> Downloaded")


if __name__ == "__main__":
    main()
