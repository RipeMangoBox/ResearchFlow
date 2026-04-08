# -*- coding: utf-8 -*-
"""
Batch normalize all 'checked' paper analysis .md files to AAMDM-style frontmatter.
Usage: run from paperAnalysis folder or repository root: python paperAnalysis/normalize_emergentmind_frontmatter.py
"""
import re
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER_ANALYSIS = ROOT / "paperAnalysis"
LOG_PATH = PAPER_ANALYSIS / "analysis_log_updated.txt"


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
                "pdf_url": pdf_url.strip(),
                "category": category.strip(),
            })
    return checked


def parse_venue_year(venue_str):
    m = re.match(r"^([A-Za-z\s]+)\s+(\d{4})$", venue_str.strip())
    if m:
        return m.group(1).strip(), int(m.group(2))
    if venue_str.strip().isdigit() and len(venue_str.strip()) == 4:
        return "", int(venue_str.strip())
    return venue_str.strip() or "", None


def build_title_to_path_index():
    """Build once: title -> first matching md path (from frontmatter or H1)."""
    index = {}
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
        if title and title not in index:
            index[title] = md_path
    return index


def find_md_for_title(title, index):
    return index.get(title)


def normalize_file(md_path, meta):
    text = md_path.read_text(encoding="utf-8")
    blocks = list(re.finditer(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL | re.MULTILINE))
    if not blocks:
        return False, "no frontmatter"
    first = blocks[0]
    try:
        fm = yaml.safe_load(first.group(1)) or {}
    except Exception:
        fm = {}
    for b in blocks[1:]:
        try:
            extra = yaml.safe_load(b.group(1)) or {}
            for k, v in extra.items():
                if k not in fm:
                    fm[k] = v
        except Exception:
            pass
    body = text[first.end():]
    for b in blocks[1:]:
        body = body.replace(b.group(0), "", 1)
    body = body.lstrip()

    venue, year = parse_venue_year(meta["venue_raw"])
    fm["title"] = fm.get("title") or meta["title"]
    fm["venue"] = venue or fm.get("venue", "")
    if year is not None:
        fm["year"] = year
    else:
        fm.setdefault("year", "")
    cat = (meta["category"] or "").strip().replace(" ", "_")
    fm["category"] = cat or fm.get("category", "")
    if "pdf_ref" not in fm or not (str(fm.get("pdf_ref") or "").strip()):
        fm["pdf_ref"] = meta["pdf_url"]
    tags = fm.get("tags")
    if not isinstance(tags, list):
        tags = [t.strip() for t in str(tags or "").split(",") if t.strip()] if tags else []
    if fm["category"] and fm["category"] not in tags:
        tags.insert(0, fm["category"])
    fm["tags"] = tags or [fm["category"]]
    fm.setdefault("created", "")
    fm.setdefault("updated", "")
    for k in ("core_operator", "primary_logic"):
        fm.setdefault(k, "")

    out = "---\n" + yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False) + "---\n\n" + body
    md_path.write_text(out, encoding="utf-8")
    return True, "ok"


def main():
    entries = parse_log()
    print("Building title -> .md index...")
    index = build_title_to_path_index()
    print(f"Index: {len(index)} titles from .md files. Log checked entries: {len(entries)}")
    normalized = []
    not_found = []
    errors = []
    for meta in entries:
        md_path = find_md_for_title(meta["title"], index)
        if not md_path:
            not_found.append(meta["title"])
            continue
        try:
            ok, msg = normalize_file(md_path, meta)
            if ok:
                normalized.append(str(md_path.relative_to(ROOT)))
            else:
                errors.append((meta["title"], msg))
        except Exception as e:
            errors.append((meta["title"], str(e)))
    report = []
    report.append("=== Normalize report ===")
    report.append(f"Checked entries in log: {len(entries)}")
    report.append(f"Normalized (frontmatter updated): {len(normalized)}")
    report.append(f"MD not found for title: {len(not_found)}")
    report.append(f"Errors: {len(errors)}")
    report.append("")
    report.append("--- Normalized files ---")
    for p in sorted(normalized)[:200]:
        report.append(p)
    if len(normalized) > 200:
        report.append(f"... and {len(normalized) - 200} more")
    report.append("")
    report.append("--- Titles with no matching .md ---")
    for t in not_found[:80]:
        report.append(t)
    if len(not_found) > 80:
        report.append(f"... and {len(not_found) - 80} more")
    report.append("")
    report.append("--- Errors ---")
    for title, msg in errors[:30]:
        report.append(f"  {title}: {msg}")
    if len(errors) > 30:
        report.append(f"... and {len(errors) - 30} more")
    result = "\n".join(report)
    out_path = PAPER_ANALYSIS / "normalize_emergentmind_report.txt"
    out_path.write_text(result, encoding="utf-8")
    print(f"Report saved: {out_path.relative_to(ROOT)}")
    print(f"Summary: normalized={len(normalized)}, not_found={len(not_found)}, errors={len(errors)}")


if __name__ == "__main__":
    main()
