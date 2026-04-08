from __future__ import annotations

from pathlib import Path

import requests

from check_paper_downloads import WRONG_LOG_PATH  # type: ignore[import]


def parse_wrong_log(path: Path) -> list[tuple[str, Path, str]]:
    """
    Parse wrong_download.txt lines into (title, local_pdf_path, pdf_url).

    Each non-comment, non-empty line is expected to have the format:
      论文名 | 本地PDF路径 | 原始pdf链接 | 问题说明
    """
    results: list[tuple[str, Path, str]] = []

    if not path.is_file():
        return results

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(" | ")]
            if len(parts) < 3:
                continue

            title = parts[0]
            local = parts[1]
            url = parts[2]

            if not local or local == "-" or not url:
                continue

            results.append((title, Path(local), url))

    return results


def main() -> None:
    wrong_path = Path(WRONG_LOG_PATH)
    print(f"Reading wrong download log: {wrong_path}")

    items = parse_wrong_log(wrong_path)
    try:
        print(f"Entries to (re)download: {len(items)}")
    except UnicodeEncodeError:
        pass

    session = requests.Session()

    for title, local_path, url in items:
        local_path = local_path.resolve()
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"[REDOWNLOAD] {title}")
            print(f"  URL:  {url}")
            print(f"  ->   {local_path}")
        except UnicodeEncodeError:
            pass

        try:
            resp = session.get(url, timeout=120)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  [ERROR] Failed to download: {exc}")
            continue

        content_type = resp.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type:
            print(f"  [WARN] Content-Type not PDF: {content_type}")

        try:
            local_path.write_bytes(resp.content)
        except Exception as exc:
            print(f"  [ERROR] Failed to write file: {exc}")
            continue

        print("  [OK]")

    print("Done re-downloading all entries from wrong_download.txt.")


if __name__ == "__main__":
    main()

