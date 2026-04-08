from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VAULT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ANALYSIS_DIR = VAULT_ROOT / "paperAnalysis"


FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$")
KEY_LINE = re.compile(r"^([A-Za-z0-9_\-]+):(?:\s*(.*))?$")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def find_frontmatter_bounds(lines: List[str]) -> Tuple[Optional[int], Optional[int]]:
    if not lines or not FRONTMATTER_BOUNDARY.match(lines[0]):
        return (None, None)
    end_idx: Optional[int] = None
    for i in range(1, len(lines)):
        if FRONTMATTER_BOUNDARY.match(lines[i]):
            end_idx = i
            break
    if end_idx is None:
        return (None, None)
    return (0, end_idx)


def parse_frontmatter(lines: List[str]) -> Dict[str, object]:
    """
    Lightweight YAML frontmatter parser (scalar + list), mirroring build_paper_collection.
    """
    fm: Dict[str, object] = {}
    i = 0
    while i < len(lines):
        raw = lines[i]
        if not raw.strip():
            i += 1
            continue

        m = KEY_LINE.match(raw)
        if not m:
            i += 1
            continue

        key = m.group(1)
        rest = (m.group(2) or "").rstrip()

        # list form: key: then indented "- item"
        if rest == "" and i + 1 < len(lines) and lines[i + 1].lstrip().startswith("- "):
            items: List[str] = []
            i += 1
            while i < len(lines):
                li = lines[i]
                if li.lstrip().startswith("- "):
                    items.append(li.lstrip()[2:].strip())
                    i += 1
                else:
                    break
            fm[key] = items
            continue

        # multi-line scalar: key: | or >
        if rest in ("|", ">"):
            block: List[str] = []
            i += 1
            while i < len(lines):
                li = lines[i]
                if li.startswith("  "):
                    block.append(li[2:])
                    i += 1
                elif li.strip() == "":
                    block.append("")
                    i += 1
                else:
                    break
            fm[key] = "\n".join(block).rstrip("\n")
            continue

        # inline scalar
        val = rest.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        fm[key] = val
        i += 1

    return fm


@dataclass
class Fix:
    path: str
    old_venue: str
    old_year: str
    new_venue: str
    new_year: str


def infer_venue_year_from_path(md_path: Path) -> Optional[Tuple[str, str]]:
    """
    Infer (venue, year) from a note's folder structure:
    paperAnalysis/<task>/<VenueToken_YYYY>/<file>.md

    Examples:
      - .../CVPR_2025/...      -> ("CVPR", "2025")
      - .../NeurIPS_2024/...   -> ("NeurIPS", "2024")
      - .../SIGGRAPH_Asia_2024 -> ("SIGGRAPH Asia", "2024")
      - .../TMLR_2025/...      -> ("TMLR 2025", "2025")  (keep year in venue)
    """
    try:
        rel = md_path.relative_to(VAULT_ROOT).as_posix()
    except ValueError:
        return None

    parts = rel.split("/")
    if len(parts) < 4 or parts[0] != "paperAnalysis":
        return None

    venue_seg = parts[2]  # e.g. "CVPR_2025"
    token = venue_seg.replace("_", " ")  # -> "CVPR 2025" / "SIGGRAPH Asia 2024"
    m = re.match(r"(.+?)\s+(\d{4})$", token)
    if not m:
        return None

    base = m.group(1).strip()
    year = m.group(2)

    # Conferences / journals where we keep base name only.
    base_only = {
        "AAAI",
        "CVPR",
        "ICCV",
        "ECCV",
        "ICLR",
        "ICML",
        "NeurIPS",
        "SIGGRAPH",
        "SIGGRAPH Asia",
        "TPAMI",
        "TOG",
    }

    if base in base_only:
        venue = base
    else:
        # For others (e.g., "TMLR 2025") keep year in venue string.
        venue = f"{base} {year}"

    return (venue, year)


def apply_fixes() -> List[Fix]:
    fixes: List[Fix] = []

    if not PAPER_ANALYSIS_DIR.exists():
        return fixes

    for md in sorted(PAPER_ANALYSIS_DIR.rglob("*.md"), key=lambda p: p.as_posix().lower()):
        text = read_text(md)
        lines = text.splitlines()
        fm_start, fm_end = find_frontmatter_bounds(lines)
        if fm_start is None or fm_end is None:
            continue

        fm_lines = lines[fm_start + 1 : fm_end]
        fm = parse_frontmatter(fm_lines)

        current_venue = str(fm.get("venue") or "").strip()
        current_year = str(fm.get("year") or "").strip()

        # Only fix notes that currently have missing/unknown venue or year.
        if current_venue and current_venue != "UnknownVenue" and current_year and current_year != "UnknownYear":
            continue

        inferred = infer_venue_year_from_path(md)
        if not inferred:
            continue

        new_venue, new_year = inferred

        # If nothing actually changes, skip.
        if (current_venue or "") == new_venue and (current_year or "") == new_year:
            continue

        new_fm_lines: List[str] = []
        saw_venue = False
        saw_year = False

        for line in fm_lines:
            m = KEY_LINE.match(line)
            if not m:
                new_fm_lines.append(line)
                continue

            key = m.group(1)
            if key == "venue":
                new_fm_lines.append(f"venue: {new_venue}")
                saw_venue = True
            elif key == "year":
                new_fm_lines.append(f"year: {new_year}")
                saw_year = True
            else:
                new_fm_lines.append(line)

        # If venue/year keys were missing entirely, insert them after title/updated if possible.
        if not saw_venue or not saw_year:
            insert_idx = 0
            for i, l in enumerate(new_fm_lines):
                if l.startswith("title:"):
                    insert_idx = i + 1
                if l.startswith("updated:"):
                    insert_idx = i + 1

            if not saw_venue:
                new_fm_lines.insert(insert_idx, f"venue: {new_venue}")
                insert_idx += 1
            if not saw_year:
                new_fm_lines.insert(insert_idx, f"year: {new_year}")

        new_lines = lines[: fm_start + 1] + new_fm_lines + lines[fm_end:]
        write_text(md, "\n".join(new_lines) + "\n")

        rel_path = md.relative_to(VAULT_ROOT).as_posix()
        fixes.append(
            Fix(
                path=rel_path,
                old_venue=current_venue,
                old_year=current_year,
                new_venue=new_venue,
                new_year=new_year,
            )
        )

    return fixes


def main() -> int:
    fixes = apply_fixes()
    print(f"[OK] fixed notes: {len(fixes)}")
    for fx in fixes:
        print(
            f"- {fx.path}: venue '{fx.old_venue or '-'}' -> '{fx.new_venue}', "
            f"year '{fx.old_year or '-'}' -> '{fx.new_year}'"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

