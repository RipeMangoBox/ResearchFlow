"""Download PDFs for Wait entries in CVPR_2026.txt.

Format: status | title | venue | paper_link | project_link | sort | pdf_path
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[5]
LOG_PATH = REPO_ROOT / "paperAnalysis" / "CVPR_2026.txt"
PDF_ROOT = REPO_ROOT / "paperPDFs"


def parse_log(path: Path):
    entries = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) < 4:
            continue
        entries.append({
            "lineno": i,
            "status": parts[0],
            "title": parts[1] if len(parts) > 1 else "",
            "venue": parts[2] if len(parts) > 2 else "",
            "paper_link": parts[3] if len(parts) > 3 else "",
            "project_link": parts[4] if len(parts) > 4 else "",
            "sort": parts[5] if len(parts) > 5 else "",
            "pdf_path": parts[6] if len(parts) > 6 else "",
        })
    return entries, lines


def resolve_pdf_path(entry: dict) -> Path:
    """Use the pdf_path column if present, else derive from sort/venue/title."""
    if entry["pdf_path"]:
        p = Path(entry["pdf_path"])
        # pdf_path is relative to repo root (for example: paperPDFs/...).
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p
    # fallback: derive
    sort = entry["sort"].replace(" ", "_") or "Uncategorized"
    venue = entry["venue"].replace(" ", "_")
    title_slug = "_".join(entry["title"].split())[:80]
    return PDF_ROOT / sort / venue / f"{title_slug}.pdf"


def main():
    entries, raw_lines = parse_log(LOG_PATH)
    wait = [e for e in entries if e["status"].strip().lower() == "wait"]
    print(f"WAIT entries: {len(wait)}")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    updated_lines = list(raw_lines)
    changed = 0

    for e in wait:
        url = e["paper_link"]
        if not url:
            print(f"[SKIP] No URL: {e['title']}")
            continue

        target = resolve_pdf_path(e)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.is_file() and target.stat().st_size > 10_000:
            print(f"[EXISTS] {target.name}")
            updated_lines[e["lineno"]] = raw_lines[e["lineno"]].replace("Wait", "Downloaded", 1)
            changed += 1
            continue

        print(f"[DL] {e['title']}")
        print(f"     {url}")
        print(f"  -> {target}")
        try:
            resp = session.get(url, timeout=180, allow_redirects=True)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            continue

        ct = resp.headers.get("Content-Type", "")
        if "pdf" not in ct.lower() and not resp.content[:4] == b"%PDF":
            print(f"  [WARN] Not a PDF (Content-Type: {ct})")

        target.write_bytes(resp.content)
        size_kb = target.stat().st_size // 1024
        print(f"  [OK] {size_kb} KB")
        updated_lines[e["lineno"]] = raw_lines[e["lineno"]].replace("Wait", "Downloaded", 1)
        changed += 1

    if changed:
        LOG_PATH.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
        print(f"\nUpdated {changed} entries to Downloaded in {LOG_PATH.name}")

    print("Done.")


if __name__ == "__main__":
    main()
