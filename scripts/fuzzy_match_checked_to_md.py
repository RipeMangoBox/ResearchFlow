# -*- coding: utf-8 -*-
"""
Fuzzy match 'checked' log entries that had no exact .md match to existing .md files.
Writes results to fuzzy_match_results.txt for later normalizing both .md and PDF filenames.
Run from repository root: python paperAnalysis/fuzzy_match_checked_to_md.py
"""
import re
import yaml
from pathlib import Path
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parent.parent
PAPER_ANALYSIS = ROOT / "paperAnalysis"
LOG_PATH = PAPER_ANALYSIS / "analysis_log_updated.txt"
OUT_PATH = PAPER_ANALYSIS / "fuzzy_match_results.txt"
MIN_SCORE = 0.45  # accept matches above this (0-1)


def normalize_for_fuzzy(s):
    if not s:
        return ""
    s = re.sub(r"[\s:,\-–—]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def slug_for_filename(title, max_len=100):
    """Build filesystem-safe basename stem from title (no extension)."""
    s = title.strip()
    s = re.sub(r"[\s:]+", " ", s)
    s = re.sub(r'[\\/:*?"<>|]', "", s)
    s = re.sub(r"\s+", "_", s.strip())
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip("_")
    return s


def parse_venue_year(venue_str):
    m = re.match(r"^([A-Za-z\s]+)\s+(\d{4})$", (venue_str or "").strip())
    if m:
        return m.group(1).strip(), int(m.group(2))
    if (venue_str or "").strip().isdigit() and len((venue_str or "").strip()) == 4:
        return "", int((venue_str or "").strip())
    return (venue_str or "").strip(), None


def parse_log():
    checked = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("checked"):
                continue
            parts = [p.strip() for p in line.split("|", 5)]
            if len(parts) != 6:
                continue
            status, title, venue, project_url, pdf_url, category = parts
            if status.strip().lower() != "checked":
                continue
            checked.append({
                "title": title.strip(),
                "venue_raw": venue.strip(),
                "category": category.strip(),
            })
    return checked


def build_md_list():
    """List of (title_in_md, path) for every .md under paperAnalysis."""
    out = []
    for md_path in PAPER_ANALYSIS.rglob("*.md"):
        if md_path.name.startswith("."):
            continue
        try:
            text = md_path.read_text(encoding="utf-8")
        except Exception:
            continue
        title = None
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
        if m:
            try:
                fm = yaml.safe_load(m.group(1))
                if isinstance(fm, dict):
                    title = (fm.get("title") or "").strip()
            except Exception:
                pass
        if not title:
            h1 = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
            if h1:
                title = h1.group(1).strip()
        if title:
            out.append((title, md_path))
    return out


def fuzzy_best(log_title, md_list):
    n_log = normalize_for_fuzzy(log_title)
    if not n_log:
        return None, 0.0
    best_path = None
    best_md_title = None
    best_score = 0.0
    for md_title, md_path in md_list:
        n_md = normalize_for_fuzzy(md_title)
        if not n_md:
            continue
        r = SequenceMatcher(None, n_log, n_md).ratio()
        if r > best_score:
            best_score = r
            best_path = md_path
            best_md_title = md_title
    return (best_path, best_md_title, best_score) if best_score >= MIN_SCORE else (None, None, best_score)


def main():
    checked = parse_log()
    md_list = build_md_list()
    exact_index = {t: p for t, p in md_list}

    not_found_meta = []
    for meta in checked:
        if meta["title"] not in exact_index:
            not_found_meta.append(meta)

    venue_year = {m["title"]: parse_venue_year(m["venue_raw"]) for m in not_found_meta}

    lines = []
    lines.append(
        "# Fuzzy match: log title (no exact .md) -> best matching .md and suggested filenames for normalization"
    )
    lines.append("# Columns: log_title | log_venue | log_category | matched_md_path | md_title_in_file | score | need_review | suggested_md_basename | suggested_pdf_basename")
    lines.append("# need_review: Y = score<0.80 or category mismatch, suggest manual check before renaming.")
    lines.append("# Suggested names use log title + year from venue. Use this file to rename .md and PDF consistently.")
    lines.append("")
    sep = " | "

    for meta in not_found_meta:
        title = meta["title"]
        venue_raw = meta["venue_raw"]
        category = meta["category"]
        path, md_title, score = fuzzy_best(title, md_list)
        venue, year = venue_year.get(title, ("", None))
        year_s = str(year) if year else "YYYY"
        slug = slug_for_filename(title)
        suggested_stem = f"{year_s}_{slug}" if slug else f"{year_s}_Untitled"
        suggested_md = suggested_stem + ".md"
        suggested_pdf = suggested_stem + ".pdf"

        md_path_str = str(path.relative_to(ROOT)) if path else ""
        cat_in_path = ""
        if path and "paperAnalysis" in str(path):
            try:
                parts = path.relative_to(PAPER_ANALYSIS).parts
                if parts:
                    cat_in_path = parts[0]
            except Exception:
                pass
        cat_norm = (category or "").replace(" ", "_")
        cat_path_norm = (cat_in_path or "").replace(" ", "_")
        mismatch = cat_path_norm and cat_norm and cat_path_norm != cat_norm
        need_review = "Y" if (score < 0.80 or mismatch) else "N"

        md_title_safe = (md_title or "").replace("|", ";")
        title_safe = title.replace("|", ";")
        lines.append(sep.join([
            title_safe,
            venue_raw.replace("|", ";"),
            category.replace("|", ";"),
            md_path_str,
            md_title_safe,
            f"{score:.3f}",
            need_review,
            suggested_md,
            suggested_pdf,
        ]))

    text = "\n".join(lines)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"Written: {OUT_PATH.relative_to(ROOT)}")
    print(f"Entries with no exact match: {len(not_found_meta)}")
    with_match = sum(1 for m in not_found_meta if fuzzy_best(m["title"], md_list)[0] is not None)
    print(f"Fuzzy match above threshold ({MIN_SCORE}): {with_match}")


if __name__ == "__main__":
    main()
