#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VAULT_ROOT = Path(__file__).resolve().parents[4]
PAPER_ANALYSIS_DIR = VAULT_ROOT / "paperAnalysis"

REQUIRED_FRONTMATTER_KEYS = {
    "title",
    "venue",
    "year",
    "tags",
    "core_operator",
    "primary_logic",
    "pdf_ref",
    "category",
}

PART_PATTERNS = {
    "Part I": re.compile(r"(?i)\bPart\s*I\b"),
    "Part II": re.compile(r"(?i)\bPart\s*II\b"),
    "Part III": re.compile(r"(?i)\bPart\s*III\b"),
}

PAPER_HOST_HINTS = (
    "arxiv.org",
    "openreview.net",
    "cvf.com",
    "thecvf.com",
    "doi.org",
    "acm.org",
    "springer.com",
)

FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$")
KEY_LINE = re.compile(r"^([A-Za-z0-9_\-]+):(?:\s*(.*))?$")


@dataclass
class LogRecord:
    file_path: Path
    line_no: int
    state: str
    importance: str
    title: str
    venue: str
    paper_link: str
    project_link: str
    category: str
    pdf_path: str


@dataclass
class MdRecord:
    file_path: Path
    title: str
    venue: str
    year: str
    category: str
    pdf_ref: str
    missing_keys: List[str]
    missing_parts: List[str]
    invalid_year: bool
    missing_frontmatter: bool
    pdf_missing: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit metadata consistency across paperAnalysis logs and notes")
    p.add_argument("--report-dir", default=str(PAPER_ANALYSIS_DIR), help="Directory for quality reports")
    p.add_argument("--json", action="store_true", help="Also write JSON report")
    return p.parse_args()


def normalize_title(title: str) -> str:
    t = (title or "").lower()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_category(category: str) -> str:
    t = (category or "").lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    return t.strip("_")


def parse_venue(venue: str) -> Tuple[str, str]:
    tokens = (venue or "").strip().split()
    if not tokens:
        return "", ""
    year = tokens[-1] if re.fullmatch(r"\d{4}", tokens[-1]) else ""
    conf = " ".join(tokens[:-1]).strip() if year else " ".join(tokens).strip()
    return normalize_title(conf), year


def is_url(s: str) -> bool:
    return bool(re.match(r"(?i)^https?://", (s or "").strip()))


def is_probably_paper_link(url: str) -> bool:
    u = (url or "").lower()
    if not u:
        return False
    if any(h in u for h in PAPER_HOST_HINTS):
        return True
    return "/pdf" in u or u.endswith(".pdf")


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


def split_frontmatter_and_body(md_text: str) -> Tuple[str, str]:
    lines = md_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", md_text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1 :])
    return "", md_text


def extract_h1(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


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


def parse_log_row(parts: List[str], file_path: Path, line_no: int) -> LogRecord:
    if len(parts) >= 8:
        state, importance, title, venue, c5, c6, category, pdf_path = parts[:8]
        paper_link, project_link = "", ""
        c5_is_paper = is_probably_paper_link(c5)
        c6_is_paper = is_probably_paper_link(c6)
        if c5_is_paper and not c6_is_paper:
            paper_link, project_link = c5, c6
        elif c6_is_paper and not c5_is_paper:
            paper_link, project_link = c6, c5
        else:
            paper_link, project_link = c5, c6
        return LogRecord(file_path, line_no, state, importance, title, venue, paper_link, project_link, category, pdf_path)

    state = parts[0] if len(parts) > 0 else ""
    title = parts[1] if len(parts) > 1 else ""
    venue = parts[2] if len(parts) > 2 else ""
    c4 = parts[3] if len(parts) > 3 else ""
    c5 = parts[4] if len(parts) > 4 else ""
    category = parts[5] if len(parts) > 5 else ""
    pdf_path = parts[6] if len(parts) > 6 else ""
    paper_link, project_link = "", ""
    if is_probably_paper_link(c4):
        paper_link, project_link = c4, c5
    elif is_probably_paper_link(c5):
        paper_link, project_link = c5, c4
    else:
        paper_link, project_link = c4, c5
    return LogRecord(file_path, line_no, state, "", title, venue, paper_link, project_link, category, pdf_path)


def load_log_records() -> Tuple[List[LogRecord], List[str], List[str]]:
    records: List[LogRecord] = []
    parse_errors: List[str] = []
    files_scanned: List[str] = []

    for txt in sorted(PAPER_ANALYSIS_DIR.glob("*.txt")):
        if txt.name.startswith("missing") or "check_task" in txt.name:
            continue
        files_scanned.append(str(txt.relative_to(VAULT_ROOT)))

        lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 6:
                parse_errors.append(f"{txt.relative_to(VAULT_ROOT)}:{i} insufficient fields")
                continue
            records.append(parse_log_row(parts, txt, i))

    return records, parse_errors, files_scanned


def load_md_records() -> List[MdRecord]:
    out: List[MdRecord] = []
    for md in sorted(PAPER_ANALYSIS_DIR.rglob("*.md")):
        if "__pycache__" in md.parts:
            continue
        if not looks_like_analysis_note(md):
            continue

        text = md.read_text(encoding="utf-8", errors="ignore")
        fm_text, body = split_frontmatter_and_body(text)
        fm = parse_frontmatter(text)
        missing_frontmatter = not bool(fm_text)

        missing_keys: List[str] = []
        if missing_frontmatter:
            missing_keys = sorted(REQUIRED_FRONTMATTER_KEYS)
        else:
            missing_keys = sorted([k for k in REQUIRED_FRONTMATTER_KEYS if k not in fm])

        title = str(fm.get("title", "")).strip() if fm else ""
        venue = str(fm.get("venue", "")).strip() if fm else ""
        year = str(fm.get("year", "")).strip() if fm else ""
        category = str(fm.get("category", "")).strip() if fm else ""
        pdf_ref = str(fm.get("pdf_ref", "")).strip() if fm else ""

        if not title:
            title = extract_h1(body)

        invalid_year = bool(year) and not bool(re.fullmatch(r"\d{4}", year))

        missing_parts = [name for name, rx in PART_PATTERNS.items() if not rx.search(body)]

        pdf_missing = False
        if pdf_ref:
            pdf_abs = VAULT_ROOT / pdf_ref
            pdf_missing = not pdf_abs.exists()

        out.append(
            MdRecord(
                file_path=md,
                title=title,
                venue=venue,
                year=year,
                category=category,
                pdf_ref=pdf_ref,
                missing_keys=missing_keys,
                missing_parts=missing_parts,
                invalid_year=invalid_year,
                missing_frontmatter=missing_frontmatter,
                pdf_missing=pdf_missing,
            )
        )
    return out


def render_report(
    generated: str,
    analysis_notes: int,
    missing_refs: List[str],
    structure_issues: List[str],
) -> str:
    lines: List[str] = []
    lines.append("# Metadata Quality Report")
    lines.append("")
    lines.append(f"- generated: {generated}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- analysis notes: {analysis_notes}")
    lines.append(f"- missing links/refs: {len(missing_refs)}")
    lines.append(f"- structure issues: {len(structure_issues)}")
    lines.append("")

    def section(title: str, items: List[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not items:
            lines.append("- none")
        else:
            for x in items:
                lines.append(f"- {x}")
        lines.append("")

    section("Missing links/refs", missing_refs)
    section("Structure issues", structure_issues)

    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    md_records = load_md_records()

    missing_refs: List[str] = []
    structure_issues: List[str] = []

    for md in md_records:
        where = str(md.file_path.relative_to(VAULT_ROOT))
        if md.missing_frontmatter:
            structure_issues.append(f"{where} missing frontmatter block")
        if md.missing_keys:
            structure_issues.append(f"{where} missing frontmatter keys: {', '.join(md.missing_keys)}")
        if md.invalid_year:
            structure_issues.append(f"{where} invalid year: `{md.year}`")
        if md.missing_parts:
            structure_issues.append(f"{where} missing sections: {', '.join(md.missing_parts)}")

        if not md.pdf_ref:
            missing_refs.append(f"{where} missing pdf_ref")
        elif md.pdf_missing:
            missing_refs.append(f"{where} pdf_ref target missing: `{md.pdf_ref}`")

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    generated = datetime.now().strftime("%Y-%m-%dT%H:%M")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    md_report = report_dir / f"quality_report_{ts}.md"

    report_text = render_report(
        generated=generated,
        analysis_notes=len(md_records),
        missing_refs=sorted(missing_refs),
        structure_issues=sorted(structure_issues),
    )
    md_report.write_text(report_text, encoding="utf-8")

    print(f"[OK] analysis notes: {len(md_records)}")
    print(f"[OK] report: {md_report}")

    if args.json:
        json_report = report_dir / f"quality_report_{ts}.json"
        payload = {
            "generated": generated,
            "counts": {
                "analysis_notes": len(md_records),
                "missing_refs": len(missing_refs),
                "structure_issues": len(structure_issues),
            },
            "missing_refs": sorted(missing_refs),
            "structure_issues": sorted(structure_issues),
        }
        json_report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] json report: {json_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
