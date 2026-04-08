#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[4]
PAPER_PDFS_DIR = REPO_ROOT / "paperPDFs"
PAPER_ANALYSIS_DIR = REPO_ROOT / "paperAnalysis"


def to_posix(path: Path) -> str:
    return path.as_posix()


def sanitize_venue(venue: str) -> str:
    s = (venue or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    return s or "UnknownVenue"


def sanitize_year(year: str) -> str:
    s = (year or "").strip()
    m = re.search(r"(19|20)\d{2}", s)
    return m.group(0) if m else "UnknownYear"


def sanitize_title_for_filename(title: str, max_len: int = 140) -> str:
    """Sanitize title for filename: single _ between words, no spaces/commas/hyphens."""
    s = (title or "").strip()
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # – —
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ", "_").replace("-", "_")  # spaces and hyphens -> underscore
    s = re.sub(r"[<>:\"/\\|?*\x00-\x1F,]", "_", s)  # commas and other unsafe chars -> underscore
    s = re.sub(r"_+", "_", s).strip("_. ")
    if not s:
        s = "Untitled"
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def infer_year_title_from_filename(pdf_path: Path) -> Tuple[Optional[str], str]:
    stem = pdf_path.stem
    m = re.match(r"^((19|20)\d{2})[_\-\s]+(.+)$", stem)
    if m:
        year = m.group(1)
        title = m.group(3).strip().replace("_", " ")
        return year, title
    return None, stem.replace("_", " ").strip()


def iter_pdfs(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            yield input_path
        return
    if input_path.is_dir():
        yield from input_path.rglob("*.pdf")
        yield from input_path.rglob("*.PDF")


def is_under(dir_path: Path, maybe_child: Path) -> bool:
    try:
        maybe_child.resolve().relative_to(dir_path.resolve())
        return True
    except Exception:
        return False


def derive_from_paperpdfs_path(pdf_path: Path) -> Optional[Tuple[str, str, str, str]]:
    """
    If already in paperPDFs/<category>/<venue_year>/<year>_<title>.pdf
    return (category, venue, year, title_stem).
    """
    if not is_under(PAPER_PDFS_DIR, pdf_path):
        return None
    rel = pdf_path.resolve().relative_to(PAPER_PDFS_DIR.resolve()).as_posix()
    parts = rel.split("/")
    if len(parts) < 3:
        return None
    category = parts[0]
    venue_year = parts[1]
    filename = Path(parts[-1]).stem
    m = re.match(r"^((19|20)\d{2})_(.+)$", filename)
    year = m.group(1) if m else ""
    title = (m.group(3) if m else filename).replace("_", " ").strip()

    # venue_year like CVPR_2025 or SIGGRAPH_Asia_2025 (venue contains underscores)
    vm = re.match(r"^(.+)_((19|20)\d{2})$", venue_year)
    venue = vm.group(1).replace("_", " ").strip() if vm else venue_year.replace("_", " ").strip()
    if vm and not year:
        year = vm.group(2)
    return category, venue, year, title


def unique_dest(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(2, 1000):
        cand = path.with_name(f"{stem}_v{i}{suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Failed to find unique filename for: {path}")


def compute_paths(category: str, venue: str, year: str, title: str) -> Dict[str, str]:
    cat = category.strip() or "Uncategorized"
    v = sanitize_venue(venue)
    y = sanitize_year(year)
    t = sanitize_title_for_filename(title)

    venue_year_dir = f"{v}_{y}"
    pdf_name = f"{y}_{t}.pdf"
    md_name = f"{y}_{t}.md"

    pdf_abs = PAPER_PDFS_DIR / cat / venue_year_dir / pdf_name
    md_abs = PAPER_ANALYSIS_DIR / cat / venue_year_dir / md_name

    pdf_ref = f"paperPDFs/{cat}/{venue_year_dir}/{pdf_name}"
    analysis_rel = f"paperAnalysis/{cat}/{venue_year_dir}/{md_name}"
    return {
        "category": cat,
        "venue": venue.strip() or "UnknownVenue",
        "venue_dir": v,
        "year": y,
        "title": title.strip() or "Untitled",
        "pdf_abs": str(pdf_abs),
        "analysis_md_abs": str(md_abs),
        "pdf_ref": pdf_ref,
        "analysis_rel": analysis_rel,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Ingest one PDF or a folder of PDFs into paperPDFs with standard naming; prints target pdf_ref and analysis path as JSON lines."
    )
    p.add_argument("input", help="Path to a .pdf file or a directory containing PDFs")
    p.add_argument("--category", default="", help="Target category (task folder under paperPDFs/paperAnalysis)")
    p.add_argument("--venue", default="", help="Venue name (e.g. CVPR, ICCV, NeurIPS)")
    p.add_argument("--year", default="", help="Year (e.g. 2025)")
    p.add_argument("--title", default="", help="Paper title (used for filename); if empty, inferred from filename")
    p.add_argument("--dry-run", action="store_true", help="Do not copy files; only print computed paths")
    args = p.parse_args(argv)

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"[ERR] input not found: {input_path}", file=sys.stderr)
        return 2

    if not PAPER_PDFS_DIR.exists() or not PAPER_ANALYSIS_DIR.exists():
        print(f"[ERR] expected repository folders missing under: {REPO_ROOT}", file=sys.stderr)
        print(f"      missing: {PAPER_PDFS_DIR} or {PAPER_ANALYSIS_DIR}", file=sys.stderr)
        return 3

    pdfs = list(iter_pdfs(input_path))
    if not pdfs:
        print("[ERR] no PDFs found", file=sys.stderr)
        return 4

    for pdf in sorted(set(pdfs), key=lambda x: x.as_posix().lower()):
        existing = derive_from_paperpdfs_path(pdf)
        if existing:
            category, venue, year, title = existing
            info = compute_paths(category=category, venue=venue, year=year, title=title)
            info.update(
                {
                    "input": str(pdf),
                    "action": "already_in_paperPDFs",
                    "copied_to": str(pdf),
                }
            )
            print(json.dumps(info, ensure_ascii=False))
            continue

        year_from_name, title_from_name = infer_year_title_from_filename(pdf)
        year = args.year.strip() or (year_from_name or "")
        title = args.title.strip() or title_from_name
        category = args.category.strip()
        venue = args.venue.strip()

        missing = [k for k, v in (("category", category), ("venue", venue), ("year", year)) if not v.strip()]
        if missing:
            print(
                json.dumps(
                    {
                        "input": str(pdf),
                        "action": "needs_metadata",
                        "missing": missing,
                        "hint": "Provide --category --venue --year (and optionally --title) for PDFs outside paperPDFs/",
                        "title_inferred": title,
                        "year_inferred": year_from_name or "",
                    },
                    ensure_ascii=False,
                )
            )
            continue

        info = compute_paths(category=category, venue=venue, year=year, title=title)
        dest = Path(info["pdf_abs"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        final_dest = unique_dest(dest)

        if not args.dry_run:
            shutil.copy2(pdf, final_dest)

        info.update(
            {
                "input": str(pdf),
                "action": "copied" if not args.dry_run else "dry_run",
                "copied_to": str(final_dest),
            }
        )
        # If unique_dest changed name, recompute refs to match final dest.
        if final_dest != dest:
            year2, title2 = infer_year_title_from_filename(final_dest)
            # final_dest should be under paperPDFs already; derive canonical from path.
            derived = derive_from_paperpdfs_path(final_dest)
            if derived:
                category2, venue2, year3, title3 = derived
                info.update(compute_paths(category=category2, venue=venue2, year=year3 or (year2 or ""), title=title3 or title2))
                info["copied_to"] = str(final_dest)
        print(json.dumps(info, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

