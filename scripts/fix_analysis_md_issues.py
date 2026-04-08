#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VAULT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ANALYSIS_DIR = VAULT_ROOT / "paperAnalysis"
PAPER_PDFS_DIR = VAULT_ROOT / "paperPDFs"

REQUIRED_KEYS = [
    "title",
    "venue",
    "year",
    "tags",
    "core_operator",
    "primary_logic",
    "pdf_ref",
    "category",
]

PART_PATTERNS = {
    "Part I": re.compile(r"(?i)\bPart\s*I\b"),
    "Part II": re.compile(r"(?i)\bPart\s*II\b"),
    "Part III": re.compile(r"(?i)\bPart\s*III\b"),
}

FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$")
KEY_LINE = re.compile(r"^([A-Za-z0-9_\-]+):(?:\s*(.*))?$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix analysis markdown metadata issues")
    p.add_argument("--report-dir", default=str(PAPER_ANALYSIS_DIR), help="Directory for fix reports")
    return p.parse_args()


def normalize_title(title: str) -> str:
    t = (title or "").lower()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def split_frontmatter_and_body(md_text: str) -> Tuple[str, str]:
    lines = md_text.splitlines()
    if not lines or not FRONTMATTER_BOUNDARY.match(lines[0]):
        return "", md_text

    end_idx: Optional[int] = None
    for i in range(1, len(lines)):
        if FRONTMATTER_BOUNDARY.match(lines[i]):
            end_idx = i
            break
    if end_idx is None:
        return "", md_text

    frontmatter = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :])
    return frontmatter, body


def parse_frontmatter(md_text: str) -> Dict[str, object]:
    lines = md_text.splitlines()
    if not lines or not FRONTMATTER_BOUNDARY.match(lines[0]):
        return {}

    end_idx: Optional[int] = None
    for i in range(1, len(lines)):
        if FRONTMATTER_BOUNDARY.match(lines[i]):
            end_idx = i
            break
    if end_idx is None:
        return {}

    fm_lines = lines[1:end_idx]
    data: Dict[str, object] = {}
    i = 0
    while i < len(fm_lines):
        raw = fm_lines[i]
        if not raw.strip():
            i += 1
            continue

        m = KEY_LINE.match(raw)
        if not m:
            i += 1
            continue

        key = m.group(1)
        rest = (m.group(2) or "").rstrip()

        if rest == "" and i + 1 < len(fm_lines) and fm_lines[i + 1].lstrip().startswith("- "):
            items: List[str] = []
            i += 1
            while i < len(fm_lines) and fm_lines[i].lstrip().startswith("- "):
                items.append(fm_lines[i].lstrip()[2:].strip())
                i += 1
            data[key] = items
            continue

        if rest in ("|", ">"):
            block: List[str] = []
            i += 1
            while i < len(fm_lines):
                li = fm_lines[i]
                if li.startswith("  "):
                    block.append(li[2:])
                    i += 1
                elif li.strip() == "":
                    block.append("")
                    i += 1
                else:
                    break
            data[key] = "\n".join(block).rstrip("\n")
            continue

        val = rest.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        data[key] = val
        i += 1

    return data


def dump_frontmatter(data: Dict[str, object]) -> str:
    key_order = [
        "title",
        "venue",
        "year",
        "tags",
        "core_operator",
        "primary_logic",
        "pdf_ref",
        "category",
        "created",
        "updated",
    ]
    extras = [k for k in data.keys() if k not in key_order]
    final_keys = [k for k in key_order if k in data] + sorted(extras)

    lines: List[str] = ["---"]
    for k in final_keys:
        v = data.get(k)
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
            continue

        s = "" if v is None else str(v)
        if "\n" in s:
            lines.append(f"{k}: |")
            for ln in s.splitlines():
                lines.append(f"  {ln}")
        else:
            lines.append(f"{k}: {s}")

    lines.append("---")
    return "\n".join(lines)


def extract_h1(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def strip_year_prefix(stem: str) -> str:
    return re.sub(r"^\d{4}_", "", stem)


def infer_venue_year(venue_dir: str) -> Tuple[str, str]:
    m = re.match(r"^(.*?)[_\- ]?(\d{4})$", venue_dir)
    if m:
        venue = m.group(1).strip("_-") or venue_dir
        year = m.group(2)
        return venue.replace("_", " "), year
    return venue_dir.replace("_", " "), ""


def looks_like_analysis_note(path: Path) -> bool:
    rel = path.relative_to(PAPER_ANALYSIS_DIR)
    if len(rel.parts) != 3:
        return False
    top_dir, venue_dir, file_name = rel.parts
    top_dir_l = top_dir.lower()
    if top_dir_l in {"processing", "emergentmind_paper_analysis"}:
        return False
    if not file_name.lower().endswith(".md"):
        return False
    if not re.search(r"\d{4}", venue_dir):
        return False
    return True


def resolve_pdf_ref(md_path: Path, category: str, venue_dir: str) -> Tuple[Optional[str], str]:
    stem = md_path.stem
    direct_pdf = PAPER_PDFS_DIR / category / venue_dir / f"{stem}.pdf"
    if direct_pdf.exists():
        rel = direct_pdf.relative_to(VAULT_ROOT).as_posix()
        return rel, "direct"

    pdf_dir = PAPER_PDFS_DIR / category / venue_dir
    if not pdf_dir.exists():
        return None, "pdf directory missing"

    target = normalize_title(strip_year_prefix(stem))
    candidates: List[Path] = []
    for p in pdf_dir.glob("*.pdf"):
        if normalize_title(strip_year_prefix(p.stem)) == target:
            candidates.append(p)

    if len(candidates) == 1:
        rel = candidates[0].relative_to(VAULT_ROOT).as_posix()
        return rel, "normalized-title unique"
    if len(candidates) > 1:
        return None, "multiple normalized-title pdf matches"
    return None, "no normalized-title pdf match"


def pdf_ref_exists(pdf_ref: str) -> bool:
    if not pdf_ref:
        return False
    rel = pdf_ref.strip()
    for prefix in (f"{VAULT_ROOT.name}/",):
        if rel.startswith(prefix):
            rel = rel[len(prefix) :]
            break
    return (VAULT_ROOT / rel).exists()


def main() -> int:
    args = parse_args()

    changed: List[str] = []
    skipped: List[str] = []
    unresolved: List[str] = []

    for md_path in sorted(PAPER_ANALYSIS_DIR.rglob("*.md")):
        if "__pycache__" in md_path.parts:
            continue
        if not looks_like_analysis_note(md_path):
            continue

        rel = md_path.relative_to(PAPER_ANALYSIS_DIR)
        category = rel.parts[0]
        venue_dir = rel.parts[1]
        stem = md_path.stem

        venue_infer, year_infer = infer_venue_year(venue_dir)
        title_infer = strip_year_prefix(stem).replace("_", " ").strip()

        text = md_path.read_text(encoding="utf-8", errors="ignore")
        frontmatter, body = split_frontmatter_and_body(text)
        fm = parse_frontmatter(text)

        if not frontmatter:
            fm = {}

        h1_title = extract_h1(body)

        file_changed = False
        applied: List[str] = []

        if not str(fm.get("category", "")).strip():
            fm["category"] = category
            file_changed = True
            applied.append("filled category")

        if not str(fm.get("venue", "")).strip():
            fm["venue"] = venue_infer
            file_changed = True
            applied.append("filled venue")

        if not str(fm.get("year", "")).strip() and year_infer:
            fm["year"] = year_infer
            file_changed = True
            applied.append("filled year")

        if not str(fm.get("title", "")).strip():
            fm["title"] = h1_title or title_infer
            file_changed = True
            applied.append("filled title")

        if "tags" not in fm:
            fm["tags"] = [category]
            file_changed = True
            applied.append("added tags")

        if "core_operator" not in fm:
            fm["core_operator"] = "[TODO]"
            file_changed = True
            applied.append("added core_operator")

        if "primary_logic" not in fm:
            fm["primary_logic"] = "[TODO]"
            file_changed = True
            applied.append("added primary_logic")

        need_pdf_ref_fix = False
        current_pdf_ref = str(fm.get("pdf_ref", "")).strip()
        if not current_pdf_ref:
            need_pdf_ref_fix = True
        elif not pdf_ref_exists(current_pdf_ref):
            need_pdf_ref_fix = True

        if need_pdf_ref_fix:
            resolved_pdf_ref, reason = resolve_pdf_ref(md_path, category, venue_dir)
            if resolved_pdf_ref:
                fm["pdf_ref"] = resolved_pdf_ref
                file_changed = True
                applied.append(f"fixed pdf_ref ({reason})")
            else:
                unresolved.append(
                    f"{md_path.relative_to(VAULT_ROOT)} unresolved pdf_ref ({reason})"
                )

        missing_parts = [name for name, rx in PART_PATTERNS.items() if not rx.search(body)]
        if missing_parts:
            unresolved.append(
                f"{md_path.relative_to(VAULT_ROOT)} missing sections: {', '.join(missing_parts)}"
            )

        if file_changed:
            new_frontmatter = dump_frontmatter(fm)
            if frontmatter:
                new_text = f"{new_frontmatter}\n{body}" if body else f"{new_frontmatter}\n"
            else:
                new_text = f"{new_frontmatter}\n\n{body}" if body else f"{new_frontmatter}\n"
            md_path.write_text(new_text, encoding="utf-8")
            changed.append(f"{md_path.relative_to(VAULT_ROOT)}: {', '.join(applied)}")
        else:
            skipped.append(str(md_path.relative_to(VAULT_ROOT)))

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"fix_analysis_md_issues_report_{ts}.md"

    lines: List[str] = []
    lines.append("# Analysis MD Fix Report")
    lines.append("")
    lines.append(f"- generated: {datetime.now().strftime('%Y-%m-%dT%H:%M')}")
    lines.append(f"- changed: {len(changed)}")
    lines.append(f"- skipped: {len(skipped)}")
    lines.append(f"- unresolved: {len(unresolved)}")
    lines.append("")

    def section(title: str, items: List[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not items:
            lines.append("- none")
        else:
            for item in items:
                lines.append(f"- {item}")
        lines.append("")

    section("Changed", changed)
    section("Skipped", skipped)
    section("Unresolved", unresolved)

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] changed: {len(changed)}")
    print(f"[OK] skipped: {len(skipped)}")
    print(f"[OK] unresolved: {len(unresolved)}")
    print(f"[OK] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
