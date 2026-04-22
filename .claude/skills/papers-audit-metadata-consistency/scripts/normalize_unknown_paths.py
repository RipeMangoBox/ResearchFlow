#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[4]
PAPER_ANALYSIS_DIR = ROOT / "paperAnalysis"
PAPER_PDFS_DIR = ROOT / "paperPDFs"
TEXT_EXTENSIONS = {".md", ".csv", ".jsonl", ".txt"}

FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$")
KEY_LINE = re.compile(r"^([A-Za-z0-9_\-]+):(?:\s*(.*))?$")
YEAR_RX = re.compile(r"\b((?:19|20)\d{2})\b")
UNKNOWN_DIR_RX = re.compile(r"^Unknown(?:_\d{4})?$")
UNKNOWN_YEAR_STEM_RX = re.compile(r"^UnknownYear_(.+)$")
YEAR_STEM_RX = re.compile(r"^(?:19|20)\d{2}_(.+)$")


@dataclass
class LogRowRef:
    log_path: Path
    line_no: int
    row: Dict[str, str]


@dataclass
class CanonicalPlan:
    category: str
    title: str
    venue_base: str
    venue_full: str
    year: str
    source_reason: str
    old_pdf_rel: Optional[str] = None
    new_pdf_rel: Optional[str] = None
    old_md_rel: Optional[str] = None
    new_md_rel: Optional[str] = None
    log_refs: List[LogRowRef] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Normalize UnknownYear/Unknown_<year> paths across paperPDFs, paperAnalysis, and analysis logs."
    )
    p.add_argument("--apply", action="store_true", help="Actually apply changes. Default is dry-run.")
    p.add_argument("--rebuild-index", action="store_true", help="Run papers-build-collection-index after apply.")
    p.add_argument(
        "--report-dir",
        default=str(PAPER_ANALYSIS_DIR),
        help="Directory for the normalization report (default: paperAnalysis).",
    )
    return p.parse_args()


def is_unknown_path(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    if any(part == "Unknown" or UNKNOWN_DIR_RX.match(part) for part in parts):
        return True
    stem = Path(rel_path).stem
    return bool(UNKNOWN_YEAR_STEM_RX.match(stem))


def sanitize_title(title: str) -> str:
    out: List[str] = []
    prev_us = False
    for ch in title or "":
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    return "".join(out).strip("_") or "Untitled"


def split_venue_year(raw_venue: str, raw_year: str) -> Optional[Tuple[str, str, str]]:
    year = (raw_year or "").strip()
    venue = (raw_venue or "").strip()
    if YEAR_RX.fullmatch(year):
        if not venue or venue.startswith("Unknown"):
            return None
        return venue, f"{venue} {year}" if not venue.endswith(year) else venue, year

    m = YEAR_RX.search(venue)
    if not m:
        return None

    year = m.group(1)
    if venue.startswith("Unknown"):
        return None

    base = YEAR_RX.sub("", venue).strip()
    if not base:
        return None
    return base, venue, year


def venue_dir_name(venue_base: str, year: str) -> str:
    base = venue_base.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
    base = base.replace("&", "and")
    base = re.sub(r"_+", "_", base).strip("._")
    return f"{base}_{year}"


def desired_stem(current_stem: str, title: str, year: str) -> str:
    m = UNKNOWN_YEAR_STEM_RX.match(current_stem)
    if m:
        return f"{year}_{m.group(1)}"
    m = YEAR_STEM_RX.match(current_stem)
    if m:
        return f"{year}_{m.group(1)}"
    return f"{year}_{sanitize_title(title)}"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def parse_frontmatter_text(text: str) -> Tuple[Dict[str, object], str]:
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    end_idx: Optional[int] = None
    for i in range(1, len(lines)):
        if FRONTMATTER_BOUNDARY.match(lines[i]):
            end_idx = i
            break
    if end_idx is None:
        return {}, text

    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :])
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
                items.append(fm_lines[i].lstrip()[2:].rstrip())
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
            data[key] = ("block", block)
            continue

        val = rest.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        data[key] = val
        i += 1

    return data, body


def dump_frontmatter(data: Dict[str, object], body: str) -> str:
    lines = ["---"]
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
            continue

        if isinstance(value, tuple) and len(value) == 2 and value[0] == "block":
            lines.append(f"{key}: |")
            for item in value[1]:
                if item == "":
                    lines.append("")
                else:
                    lines.append(f"  {item}")
            continue

        sval = str(value)
        if any(ch in sval for ch in [":", "[", "]", "{", "}", "#"]):
            lines.append(f'{key}: "{sval}"')
        else:
            lines.append(f"{key}: {sval}")
    lines.append("---")
    return "\n".join(lines) + "\n" + body


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def first_page_text(path: Path) -> str:
    try:
        out = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "1", str(path), "-"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    return " ".join(out.stdout.split())[:4000]


def pdfs_are_duplicates(old_path: Path, new_path: Path) -> bool:
    if old_path.stat().st_size == new_path.stat().st_size:
        try:
            if sha256(old_path) == sha256(new_path):
                return True
        except OSError:
            pass
        if first_page_text(old_path) and first_page_text(old_path) == first_page_text(new_path):
            return True
    return False


def list_log_paths() -> List[Path]:
    return [p for p in ROOT.rglob("analysis_log.csv") if "_private/git_backups" not in str(p)]


def load_logs() -> Tuple[List[LogRowRef], Dict[str, List[LogRowRef]], Dict[str, List[LogRowRef]]]:
    refs: List[LogRowRef] = []
    by_pdf: Dict[str, List[LogRowRef]] = defaultdict(list)
    by_title: Dict[str, List[LogRowRef]] = defaultdict(list)

    for log_path in list_log_paths():
        with log_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=2):
                clean = {k: (v or "") for k, v in row.items()}
                ref = LogRowRef(log_path=log_path, line_no=i, row=clean)
                refs.append(ref)

                pdf_path = clean.get("pdf_path", "").strip()
                if pdf_path:
                    by_pdf[pdf_path].append(ref)

                title = clean.get("paper_title", "").strip()
                if title:
                    by_title[title].append(ref)

    return refs, by_pdf, by_title


def choose_log_ref(candidates: Iterable[LogRowRef], category: str, old_pdf_rel: Optional[str]) -> Optional[LogRowRef]:
    scored: List[Tuple[int, LogRowRef]] = []
    for ref in candidates:
        score = 0
        if ref.row.get("sort", "").strip() == category:
            score += 10
        if old_pdf_rel and ref.row.get("pdf_path", "").strip() == old_pdf_rel:
            score += 20
        venue = ref.row.get("venue", "").strip()
        if split_venue_year(venue, ""):
            score += 5
        scored.append((score, ref))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else None


def resolve_plan_from_md(
    md_path: Path,
    by_pdf: Dict[str, List[LogRowRef]],
    by_title: Dict[str, List[LogRowRef]],
) -> Tuple[Optional[CanonicalPlan], List[str]]:
    rel = md_path.relative_to(ROOT).as_posix()
    warnings: List[str] = []
    text = read_text(md_path)
    fm, _ = parse_frontmatter_text(text)

    category = md_path.relative_to(PAPER_ANALYSIS_DIR).parts[0]
    title = str(fm.get("title", "")).strip() or md_path.stem
    old_pdf_rel = str(fm.get("pdf_ref", "")).strip() or None

    venue_year = split_venue_year(str(fm.get("venue", "")).strip(), str(fm.get("year", "")).strip())
    chosen_ref: Optional[LogRowRef] = None
    if not venue_year:
        candidates = by_pdf.get(old_pdf_rel or "", []) or by_title.get(title, [])
        chosen_ref = choose_log_ref(candidates, category, old_pdf_rel)
        if chosen_ref:
            venue_year = split_venue_year(chosen_ref.row.get("venue", "").strip(), "")

    if not venue_year:
        warnings.append(f"{rel}: unresolved venue/year")
        return None, warnings

    venue_base, venue_full, year = venue_year
    new_venue_dir = venue_dir_name(venue_base, year)
    new_md_rel = (
        PAPER_ANALYSIS_DIR / category / new_venue_dir / f"{desired_stem(md_path.stem, title, year)}.md"
    ).relative_to(ROOT).as_posix()

    new_pdf_rel = old_pdf_rel
    if old_pdf_rel:
        old_pdf_path = ROOT / old_pdf_rel
        if old_pdf_path.exists():
            new_pdf_rel = (
                PAPER_PDFS_DIR / category / new_venue_dir / f"{desired_stem(old_pdf_path.stem, title, year)}.pdf"
            ).relative_to(ROOT).as_posix()
        elif chosen_ref:
            row_pdf = chosen_ref.row.get("pdf_path", "").strip()
            row_pdf_path = ROOT / row_pdf
            if row_pdf and row_pdf_path.exists():
                new_pdf_rel = (
                    PAPER_PDFS_DIR / category / new_venue_dir / f"{desired_stem(row_pdf_path.stem, title, year)}.pdf"
                ).relative_to(ROOT).as_posix()
            else:
                warnings.append(f"{rel}: pdf_ref missing locally, keeping unresolved PDF path")
                return None, warnings
        else:
            warnings.append(f"{rel}: missing local PDF and no matching log row")
            return None, warnings

    plan = CanonicalPlan(
        category=category,
        title=title,
        venue_base=venue_base,
        venue_full=venue_full,
        year=year,
        source_reason="frontmatter" if split_venue_year(str(fm.get('venue', '')).strip(), str(fm.get('year', '')).strip()) else "matched log row",
        old_pdf_rel=old_pdf_rel,
        new_pdf_rel=new_pdf_rel,
        old_md_rel=rel,
        new_md_rel=new_md_rel,
        log_refs=by_pdf.get(old_pdf_rel or "", []) or by_title.get(title, []),
    )
    return plan, warnings


def resolve_plan_from_log(ref: LogRowRef) -> Optional[CanonicalPlan]:
    pdf_rel = ref.row.get("pdf_path", "").strip()
    if not pdf_rel or not is_unknown_path(pdf_rel):
        return None

    pdf_path = ROOT / pdf_rel
    if not pdf_path.exists():
        return None

    venue_year = split_venue_year(ref.row.get("venue", "").strip(), "")
    if not venue_year:
        return None

    venue_base, venue_full, year = venue_year
    category = ref.row.get("sort", "").strip()
    if not category or category == "Unknown":
        rel_parts = pdf_path.relative_to(PAPER_PDFS_DIR).parts
        if len(rel_parts) >= 2 and rel_parts[0] != "Unknown":
            category = rel_parts[0]
        else:
            return None

    title = ref.row.get("paper_title", "").strip() or pdf_path.stem
    new_venue_dir = venue_dir_name(venue_base, year)
    new_pdf_rel = (
        PAPER_PDFS_DIR / category / new_venue_dir / f"{desired_stem(pdf_path.stem, title, year)}.pdf"
    ).relative_to(ROOT).as_posix()

    return CanonicalPlan(
        category=category,
        title=title,
        venue_base=venue_base,
        venue_full=venue_full,
        year=year,
        source_reason="analysis_log.csv",
        old_pdf_rel=pdf_rel,
        new_pdf_rel=new_pdf_rel,
        log_refs=[ref],
    )


def collect_plans() -> Tuple[Dict[str, CanonicalPlan], Dict[str, CanonicalPlan], List[str]]:
    refs, by_pdf, by_title = load_logs()
    md_plans: Dict[str, CanonicalPlan] = {}
    pdf_only_plans: Dict[str, CanonicalPlan] = {}
    warnings: List[str] = []

    for md_path in PAPER_ANALYSIS_DIR.rglob("*.md"):
        rel = md_path.relative_to(PAPER_ANALYSIS_DIR).as_posix()
        if "processing" in md_path.parts or md_path.name.startswith("quality_report_"):
            continue
        if not is_unknown_path(rel):
            continue
        plan, plan_warnings = resolve_plan_from_md(md_path, by_pdf, by_title)
        warnings.extend(plan_warnings)
        if plan:
            md_plans[plan.old_md_rel] = plan

    for ref in refs:
        plan = resolve_plan_from_log(ref)
        if not plan:
            continue
        # already covered by md plan
        if plan.old_pdf_rel and any(mp.old_pdf_rel == plan.old_pdf_rel for mp in md_plans.values()):
            continue
        pdf_only_plans[plan.old_pdf_rel] = plan

    return md_plans, pdf_only_plans, warnings


def update_log_csv(log_path: Path, updates: Dict[int, CanonicalPlan], apply: bool) -> bool:
    with log_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    changed = False
    for i, row in enumerate(rows, start=2):
        plan = updates.get(i)
        if not plan:
            continue
        if row.get("venue", "") != plan.venue_full:
            row["venue"] = plan.venue_full
            changed = True
        if row.get("pdf_path", "") != (plan.new_pdf_rel or row.get("pdf_path", "")):
            row["pdf_path"] = plan.new_pdf_rel or row.get("pdf_path", "")
            changed = True
        if row.get("sort", "") in {"", "Unknown"} and plan.category:
            row["sort"] = plan.category
            changed = True

    if changed and apply:
        with log_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return changed


def update_derived_cache(path: Path, path_replacements: Dict[str, str], venue_by_title: Dict[str, str], apply: bool) -> bool:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    changed = False
    for row in rows:
        pdf_path = row.get("pdf_path", "")
        if pdf_path in path_replacements and row.get("pdf_path", "") != path_replacements[pdf_path]:
            row["pdf_path"] = path_replacements[pdf_path]
            changed = True
        title = row.get("paper_title", "").strip()
        if title in venue_by_title and row.get("venue", "") != venue_by_title[title]:
            row["venue"] = venue_by_title[title]
            changed = True

    if changed and apply:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return changed


def write_canonical_note(plan: CanonicalPlan, apply: bool) -> Tuple[bool, bool, Optional[str]]:
    old_md = ROOT / (plan.old_md_rel or "")
    new_md = ROOT / (plan.new_md_rel or "")
    if not old_md.exists():
        return False, False, f"missing source note: {plan.old_md_rel}"

    canonical_source = new_md if new_md.exists() and new_md != old_md else old_md
    text = read_text(canonical_source)
    fm, body = parse_frontmatter_text(text)

    fm["title"] = plan.title
    fm["venue"] = plan.venue_base
    fm["year"] = plan.year
    if plan.new_pdf_rel:
        fm["pdf_ref"] = plan.new_pdf_rel

    new_text = dump_frontmatter(fm, body)
    if plan.old_pdf_rel and plan.new_pdf_rel and plan.old_pdf_rel != plan.new_pdf_rel:
        new_text = new_text.replace(plan.old_pdf_rel, plan.new_pdf_rel)
    if plan.old_md_rel and plan.new_md_rel and plan.old_md_rel != plan.new_md_rel:
        new_text = new_text.replace(plan.old_md_rel, plan.new_md_rel)

    content_differs = new_md.exists() and new_md != old_md and read_text(new_md) != read_text(old_md)

    if apply:
        new_md.parent.mkdir(parents=True, exist_ok=True)
        write_text(new_md, new_text)
        if old_md != new_md and old_md.exists():
            old_md.unlink()

    return True, content_differs, None


def apply_pdf_move(old_pdf: Path, new_pdf: Path, apply: bool) -> Tuple[str, Optional[str]]:
    if not old_pdf.exists():
        return "missing", f"missing source PDF: {old_pdf.relative_to(ROOT)}"

    if old_pdf == new_pdf:
        return "noop", None

    if new_pdf.exists():
        if not pdfs_are_duplicates(old_pdf, new_pdf):
            return "conflict", f"non-duplicate PDF collision: {old_pdf.relative_to(ROOT)} -> {new_pdf.relative_to(ROOT)}"
        if apply:
            old_pdf.unlink()
        return "dedup", None

    if apply:
        new_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_pdf), str(new_pdf))
    return "moved", None


def replace_text_refs(path_replacements: Dict[str, str], apply: bool) -> int:
    replacements = sorted(path_replacements.items(), key=lambda kv: len(kv[0]), reverse=True)
    changed = 0
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if "_private/git_backups" in str(path):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = read_text(path)
        except Exception:
            continue

        new_text = text
        for old_rel, new_rel in replacements:
            if old_rel in new_text:
                new_text = new_text.replace(old_rel, new_rel)

        if new_text != text:
            changed += 1
            if apply:
                write_text(path, new_text)
    return changed


def cleanup_empty_dirs(apply: bool) -> int:
    removed = 0
    if not apply:
        return removed

    for base in [PAPER_ANALYSIS_DIR, PAPER_PDFS_DIR]:
        changed = True
        while changed:
            changed = False
            dirs = sorted([p for p in base.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True)
            for d in dirs:
                try:
                    next(d.iterdir())
                except StopIteration:
                    d.rmdir()
                    removed += 1
                    changed = True
    return removed


def rebuild_index(apply: bool) -> Tuple[bool, str]:
    if not apply:
        return False, "skipped rebuild in dry-run"

    builder = ROOT / ".claude/skills/papers-build-collection-index/scripts/build_paper_collection.py"
    out = subprocess.run(["python3", str(builder)], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        return False, out.stderr.strip() or out.stdout.strip()
    return True, out.stdout.strip()


def main() -> int:
    args = parse_args()
    report_lines: List[str] = []
    md_plans, pdf_only_plans, warnings = collect_plans()

    # detect target collisions early
    md_targets = defaultdict(list)
    for plan in md_plans.values():
        md_targets[plan.new_md_rel].append(plan.old_md_rel)
    pdf_targets = defaultdict(list)
    for plan in list(md_plans.values()) + list(pdf_only_plans.values()):
        if plan.old_pdf_rel and plan.new_pdf_rel:
            pdf_targets[plan.new_pdf_rel].append(plan.old_pdf_rel)

    hard_conflicts: List[str] = []
    for target, srcs in md_targets.items():
        if len(srcs) > 1:
            hard_conflicts.append(f"multiple notes map to {target}: {srcs}")
    for target, srcs in pdf_targets.items():
        if len(srcs) > 1:
            hard_conflicts.append(f"multiple PDFs map to {target}: {srcs}")

    if hard_conflicts:
        report_lines.extend(hard_conflicts)
        print("[ERROR] target conflicts detected")
        for line in hard_conflicts:
            print("-", line)
        return 2

    log_updates: Dict[Path, Dict[int, CanonicalPlan]] = defaultdict(dict)
    venue_by_title: Dict[str, str] = {}
    path_replacements: Dict[str, str] = {}

    for plan in list(md_plans.values()) + list(pdf_only_plans.values()):
        if plan.old_pdf_rel and plan.new_pdf_rel and plan.old_pdf_rel != plan.new_pdf_rel:
            path_replacements[plan.old_pdf_rel] = plan.new_pdf_rel
        if plan.old_md_rel and plan.new_md_rel and plan.old_md_rel != plan.new_md_rel:
            path_replacements[plan.old_md_rel] = plan.new_md_rel
        venue_by_title[plan.title] = plan.venue_full
        for ref in plan.log_refs:
            log_updates[ref.log_path][ref.line_no] = plan

    note_written = 0
    note_kept_target = 0
    pdf_moved = 0
    pdf_dedup = 0
    text_ref_updates = 0
    log_files_changed = 0
    derived_changed = 0
    empty_dirs_removed = 0
    unresolved: List[str] = list(warnings)

    # write canonical notes
    for plan in md_plans.values():
        old_md = ROOT / (plan.old_md_rel or "")
        new_md = ROOT / (plan.new_md_rel or "")
        had_target = new_md.exists() and new_md != old_md
        ok, content_differs, error = write_canonical_note(plan, args.apply)
        if error:
            unresolved.append(error)
            continue
        if ok:
            note_written += 1
            if had_target and content_differs:
                note_kept_target += 1
                unresolved.append(
                    f"kept canonical note target for {plan.title}: {plan.new_md_rel} (source note content differed)"
                )

    # move / dedupe PDFs
    seen_pdf = set()
    for plan in list(md_plans.values()) + list(pdf_only_plans.values()):
        if not plan.old_pdf_rel or not plan.new_pdf_rel:
            continue
        if plan.old_pdf_rel in seen_pdf:
            continue
        seen_pdf.add(plan.old_pdf_rel)
        status, error = apply_pdf_move(ROOT / plan.old_pdf_rel, ROOT / plan.new_pdf_rel, args.apply)
        if error:
            unresolved.append(error)
            continue
        if status == "moved":
            pdf_moved += 1
        elif status == "dedup":
            pdf_dedup += 1

    # update analysis logs
    for log_path, updates in log_updates.items():
        if update_log_csv(log_path, updates, args.apply):
            log_files_changed += 1

    # update derived caches
    for extra in [ROOT / "paperAnalysis/processing/unified_paper_index.csv", ROOT / "paperAnalysis/processing/unified_paper_duplicates.csv"]:
        if extra.exists() and update_derived_cache(extra, path_replacements, venue_by_title, args.apply):
            derived_changed += 1

    text_ref_updates = replace_text_refs(path_replacements, args.apply)
    empty_dirs_removed = cleanup_empty_dirs(args.apply)

    rebuild_ok = False
    rebuild_msg = "not requested"
    if args.rebuild_index:
        rebuild_ok, rebuild_msg = rebuild_index(args.apply)
        if not rebuild_ok:
            unresolved.append(f"index rebuild failed: {rebuild_msg}")

    report_lines.extend(
        [
            "# Normalize Unknown Paths Report",
            "",
            f"- generated: {datetime.now().strftime('%Y-%m-%dT%H:%M')}",
            f"- mode: {'apply' if args.apply else 'dry-run'}",
            f"- md plans: {len(md_plans)}",
            f"- pdf-only plans: {len(pdf_only_plans)}",
            f"- notes written: {note_written}",
            f"- canonical-note collisions kept at target: {note_kept_target}",
            f"- PDFs moved: {pdf_moved}",
            f"- duplicate PDFs deleted: {pdf_dedup}",
            f"- analysis_log files changed: {log_files_changed}",
            f"- derived caches changed: {derived_changed}",
            f"- text files with old path replacements: {text_ref_updates}",
            f"- empty dirs removed: {empty_dirs_removed}",
            f"- unresolved / warnings: {len(unresolved)}",
            "",
            "## Unresolved / Warnings",
            "",
        ]
    )

    if not unresolved:
        report_lines.append("- none")
    else:
        report_lines.extend(f"- {item}" for item in unresolved)

    if args.rebuild_index:
        report_lines.extend(["", "## Rebuild", "", f"- success: {rebuild_ok}", f"- detail: {rebuild_msg}"])

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"normalize_unknown_paths_report_{datetime.now().strftime('%Y-%m-%d_%H%M')}.md"
    write_text(report_path, "\n".join(report_lines) + "\n")

    print(f"[OK] mode: {'apply' if args.apply else 'dry-run'}")
    print(f"[OK] md plans: {len(md_plans)}")
    print(f"[OK] pdf-only plans: {len(pdf_only_plans)}")
    print(f"[OK] notes written: {note_written}")
    print(f"[OK] PDFs moved: {pdf_moved}")
    print(f"[OK] duplicate PDFs deleted: {pdf_dedup}")
    print(f"[OK] analysis_log files changed: {log_files_changed}")
    print(f"[OK] derived caches changed: {derived_changed}")
    print(f"[OK] text refs updated: {text_ref_updates}")
    print(f"[OK] empty dirs removed: {empty_dirs_removed}")
    print(f"[OK] unresolved/warnings: {len(unresolved)}")
    print(f"[OK] report: {report_path}")
    if args.rebuild_index:
        print(f"[OK] rebuild requested: {rebuild_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
