from pathlib import Path

# Infer repository root from this file location:
# scripts/paper_analysis_maintenance/mark_wait_for_incomplete_parts.py -> parents[2] = repo root
ROOT = Path(__file__).resolve().parents[2]
paper_root = ROOT / "paperAnalysis"
log_path = paper_root / "analysis_log_updated.txt"

REQUIRED_MARKERS = [
    "Part I:",
    "Part II:",
    "Part III:",
]


def has_all_parts(md_path: Path) -> bool:
    if not md_path.exists():
        return False
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    return all(m in text for m in REQUIRED_MARKERS)


def main() -> None:
    lines = log_path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            new_lines.append(line)
            continue

        parts = [p.strip() for p in stripped.split("|")]
        if len(parts) < 6:
            new_lines.append(line)
            continue

        status, title, venue, github, pdf_url, category = parts[:6]

        def sanitize_title_for_filename(t: str) -> str:
            out = []
            prev_us = False
            for ch in t:
                if ch.isalnum():
                    out.append(ch)
                    prev_us = False
                else:
                    if not prev_us:
                        out.append("_")
                        prev_us = True
            return "".join(out).strip("_")

        category_dir = category.replace(" ", "_").replace("-", "_")
        venue_dir = venue.replace(" ", "_")
        year = "".join(ch for ch in venue if ch.isdigit()) or "Unknown"
        safe_title = sanitize_title_for_filename(title)

        md_path = paper_root / category_dir / venue_dir / f"{year}_{safe_title}.md"

        if not has_all_parts(md_path):
            parts[0] = "Wait"
        else:
            parts[0] = status

        new_lines.append(" | ".join(parts) + "\n")

    log_path.write_text("".join(new_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

