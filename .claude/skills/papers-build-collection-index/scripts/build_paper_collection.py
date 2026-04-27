from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


VAULT_ROOT = Path(__file__).resolve().parents[4]
PAPER_ANALYSIS_DIR = VAULT_ROOT / "paperAnalysis"
PAPER_COLLECTION_DIR = VAULT_ROOT / "paperCollection"
TAG_EXPLOSION_THRESHOLD = 500


@dataclass(frozen=True)
class Paper:
    analysis_rel: str  # e.g. paperAnalysis/.../xxx.md (posix)
    title: str
    venue: str
    year: str
    category: str
    tags: Tuple[str, ...]
    core_operator: str
    primary_logic: str
    claims_count: int
    pdf_ref: str  # e.g. paperPDFs/.../xxx.pdf (posix) or ""


FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$")
KEY_LINE = re.compile(r"^([A-Za-z0-9_\-]+):(?:\s*(.*))?$")


def to_posix_rel(path: Path) -> str:
    return path.as_posix()


def sanitize_filename(name: str, max_len: int = 120) -> str:
    # Windows-safe filename; keep readable.
    s = name.strip()
    s = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", s)
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(". ")
    if not s:
        s = "Untitled"
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s


def parse_frontmatter_bounds(lines: List[str]) -> Optional[Tuple[int, int]]:
    if not lines or not FRONTMATTER_BOUNDARY.match(lines[0]):
        return None
    for i in range(1, len(lines)):
        if FRONTMATTER_BOUNDARY.match(lines[i]):
            return (0, i)
    return None


def parse_inline_list(raw: str) -> Optional[List[str]]:
    s = raw.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    inner = s[1:-1].strip()
    if inner == "":
        return []
    try:
        row = next(csv.reader([inner], skipinitialspace=True))
    except Exception:
        return None
    out: List[str] = []
    for it in row:
        x = it.strip()
        if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")):
            x = x[1:-1].strip()
        if x:
            out.append(x)
    return out


def parse_frontmatter(md_text: str) -> Dict[str, object]:
    lines = md_text.splitlines()
    bounds = parse_frontmatter_bounds(lines)
    if not bounds:
        return {}

    _, end_idx = bounds
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

        # list form: key: then indented "- item"
        if rest == "" and i + 1 < len(fm_lines) and fm_lines[i + 1].lstrip().startswith("- "):
            items: List[str] = []
            i += 1
            while i < len(fm_lines):
                li = fm_lines[i]
                if li.lstrip().startswith("- "):
                    items.append(li.lstrip()[2:].strip())
                    i += 1
                else:
                    break
            data[key] = items
            continue

        # multi-line scalar: key: | or key: >
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

        # inline scalar
        val = rest.strip()
        if key == "tags":
            inline_list = parse_inline_list(val)
            if inline_list is not None:
                data[key] = inline_list
                i += 1
                continue
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        data[key] = val
        i += 1

    return data


def extract_frontmatter_and_body(md_text: str) -> Tuple[Dict[str, object], str]:
    lines = md_text.splitlines()
    bounds = parse_frontmatter_bounds(lines)
    if not bounds:
        return {}, md_text
    _, end_idx = bounds
    fm = parse_frontmatter(md_text)
    body = "\n".join(lines[end_idx + 1 :])
    return fm, body


def infer_task(fm: Dict[str, object], analysis_rel: str) -> str:
    # Prefer the folder name under paperAnalysis as the canonical "task",
    # because frontmatter `category` may contain minor naming variants.
    parts = analysis_rel.split("/")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    cat = str(fm.get("category") or "").strip()
    if cat:
        return cat
    return "Uncategorized"


def normalize_tag_value(raw: str, task: str) -> Optional[str]:
    s = str(raw).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if not s:
        return None
    if s.startswith("#"):
        s = s[1:].strip()
    if not s:
        return None
    if s == task:
        return None
    if s.startswith("status/"):
        return None
    return s


def split_scalar_tags(raw: str) -> List[str]:
    s = raw.strip()
    if not s:
        return []
    # Prefer hashtag tokens when present, otherwise comma split fallback.
    hash_tokens = [m.group(0) for m in re.finditer(r"(?<![\w/])#[A-Za-z0-9][A-Za-z0-9_\-/]*", s)]
    if hash_tokens:
        return hash_tokens
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def extract_body_hashtags(body_text: str) -> List[str]:
    out: List[str] = []
    in_fence = False
    for raw_line in body_text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # skip markdown heading lines
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            continue

        # remove inline code spans
        cleaned = re.sub(r"`[^`]*`", " ", line)
        # remove markdown link anchors like ](#section)
        cleaned = re.sub(r"\]\(#.*?\)", " ", cleaned)

        for m in re.finditer(r"(?<![\w/])#([A-Za-z0-9][A-Za-z0-9_\-/]*)", cleaned):
            out.append(m.group(1))
    return out


def infer_technique_tags(fm: Dict[str, object], task: str, body_text: str = "") -> Tuple[str, ...]:
    tags_raw = fm.get("tags")
    candidates: List[str] = []

    if isinstance(tags_raw, list):
        candidates.extend(str(x) for x in tags_raw)
    elif isinstance(tags_raw, str):
        inline_list = parse_inline_list(tags_raw)
        if inline_list is not None:
            candidates.extend(inline_list)
        else:
            candidates.extend(split_scalar_tags(tags_raw))

    if body_text:
        candidates.extend(extract_body_hashtags(body_text))

    out: List[str] = []
    seen = set()
    for c in candidates:
        s = normalize_tag_value(c, task)
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return tuple(out)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def iter_analysis_mds() -> Iterable[Path]:
    yield from PAPER_ANALYSIS_DIR.rglob("*.md")


def load_papers() -> List[Paper]:
    papers: List[Paper] = []
    for md_path in iter_analysis_mds():
        rel = md_path.relative_to(VAULT_ROOT).as_posix()
        text = read_text(md_path)
        fm, body = extract_frontmatter_and_body(text)
        title = str(fm.get("title") or md_path.stem).strip()
        venue = str(fm.get("venue") or "").strip() or "UnknownVenue"
        year = str(fm.get("year") or "").strip() or "UnknownYear"
        core_operator = str(fm.get("core_operator") or "").strip()
        primary_logic = str(fm.get("primary_logic") or "").strip()
        pdf_ref = str(fm.get("pdf_ref") or "").strip()
        claims_raw = fm.get("claims")
        if isinstance(claims_raw, list):
            claims_count = len([str(x).strip() for x in claims_raw if str(x).strip()])
        elif isinstance(claims_raw, str):
            claims_count = 1 if claims_raw.strip() else 0
        else:
            claims_count = 0

        # Heuristic filter: only index notes that clearly correspond to a paper PDF.
        # This avoids pulling in helper docs (e.g., SKILL.md) under paperAnalysis.
        if not (pdf_ref.startswith("paperPDFs/") and pdf_ref.lower().endswith(".pdf")):
            continue

        task = infer_task(fm, rel)
        tech = infer_technique_tags(fm, task, body)

        papers.append(
            Paper(
                analysis_rel=rel,
                title=title,
                venue=venue,
                year=year,
                category=task,
                tags=tech,
                core_operator=core_operator,
                primary_logic=primary_logic,
                claims_count=claims_count,
                pdf_ref=pdf_ref,
            )
        )

    # stable sort: category -> venue -> year desc -> title
    def sort_key(p: Paper) -> Tuple[str, str, int, str]:
        try:
            y = int(re.sub(r"\D+", "", p.year) or "0")
        except ValueError:
            y = 0
        return (p.category.lower(), p.venue.lower(), -y, p.title.lower())

    papers.sort(key=sort_key)
    return papers


def md_link(target_rel: str, alias: Optional[str] = None) -> str:
    # Obsidian supports [[path|alias]]; use posix.
    if alias:
        return f"[[{target_rel}|{alias}]]"
    return f"[[{target_rel}]]"


def format_tech_tags(tags: Tuple[str, ...]) -> str:
    # Render as inline Obsidian tags; avoid spaces.
    rendered: List[str] = []
    for t in tags:
        tag = t.replace(" ", "_")
        rendered.append(f"#{tag}")
    return " ".join(rendered)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def clean_md_dir(dir_path: Path) -> None:
    if not dir_path.exists():
        return
    for p in dir_path.glob("*.md"):
        # keep nothing; we always regenerate indexes and pages
        try:
            p.unlink()
        except OSError:
            pass


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def to_year_value(year: str):
    digits = re.sub(r"\D+", "", str(year))
    if digits:
        try:
            return int(digits)
        except ValueError:
            return year
    return year


def build_agent_index_jsonl(papers: List[Paper]) -> str:
    lines: List[str] = []
    for p in papers:
        row = {
            "path": p.analysis_rel,
            "title": p.title,
            "category": p.category,
            "venue": p.venue,
            "year": to_year_value(p.year),
            "tags": list(p.tags),
            "core_operator": p.core_operator,
            "primary_logic": p.primary_logic,
            "claims_count": p.claims_count,
            "pdf_ref": p.pdf_ref,
        }
        lines.append(json.dumps(row, ensure_ascii=False))
    return ("\n".join(lines) + "\n") if lines else ""


def group_by(items: Iterable[Paper], key_fn) -> Dict[str, List[Paper]]:
    out: Dict[str, List[Paper]] = {}
    for it in items:
        k = key_fn(it)
        out.setdefault(k, []).append(it)
    return out


def build_readme(papers: List[Paper], now: str) -> str:
    tasks = sorted({p.category for p in papers}, key=lambda x: x.lower())
    venues = sorted({p.venue for p in papers}, key=lambda x: x.lower())
    techniques = sorted({t for p in papers for t in p.tags}, key=lambda x: x.lower())

    lines: List[str] = []
    lines.append("---")
    lines.append("type: paper-collection-home")
    lines.append("tags:")
    lines.append("  - paperCollection")
    lines.append(f"generated: {now}")
    lines.append("---")
    lines.append("")
    lines.append("# paperCollection")
    lines.append("")
    lines.append("This directory is auto-generated from `paperAnalysis/` and is used for **agent-side fast filtering**, **statistical overview**, **Obsidian jumps**, and **backlink-friendly browsing**.")
    lines.append("")
    lines.append("## Start here")
    lines.append("")
    lines.append(f"- {md_link('paperCollection/_AllPapers.md', 'All papers (grouped)')}")
    lines.append("- By venue/journal")
    lines.append(f"  - {md_link('paperCollection/by_venue/_Index.md', 'Venue index')}")
    lines.append("")
    lines.append("> Task & technique browsing now lives in Obsidian: search the `task/` and technique tag namespaces, or use the backlink panel on any paper note.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `index.jsonl` is the fast filter layer for agents; the Markdown pages are for human navigation.")
    lines.append("- This index links out by default and does not embed PDFs (to avoid heavy pages).")
    lines.append("- `paperAnalysis` remains the primary evidence layer; `paperCollection` adds generated index and navigation outputs.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- papers: {len(papers)}")
    lines.append(f"- tasks: {len(tasks)}")
    lines.append(f"- venues: {len(venues)}")
    lines.append(f"- technique tags: {len(techniques)}")
    lines.append("")
    return "\n".join(lines)


def build_all_papers(papers: List[Paper], now: str) -> str:
    lines: List[str] = []
    lines.append("---")
    lines.append("type: paper-index")
    lines.append("dimension: all")
    lines.append("tags:")
    lines.append("  - paperCollection")
    lines.append("  - index/all")
    lines.append(f"generated: {now}")
    lines.append("---")
    lines.append("")
    lines.append("# All papers (grouped)")
    lines.append("")

    by_task = group_by(papers, lambda p: p.category)
    for task in sorted(by_task.keys(), key=lambda x: x.lower()):
        lines.append(f"## {task}")
        task_papers = by_task[task]
        # subgroup by venue_year
        by_venue_year: Dict[str, List[Paper]] = {}
        for p in task_papers:
            by_venue_year.setdefault(f"{p.venue} {p.year}", []).append(p)

        for vy in sorted(by_venue_year.keys(), key=lambda x: x.lower()):
            lines.append(f"### {vy}")
            for p in by_venue_year[vy]:
                alias = f"{p.title} ({p.venue} {p.year})"
                ana = md_link(p.analysis_rel, alias)
                if p.pdf_ref:
                    pdf = md_link(p.pdf_ref, "PDF")
                    lines.append(f"- {ana} · {pdf} · techniques: {format_tech_tags(p.tags)}")
                else:
                    lines.append(f"- {ana} · techniques: {format_tech_tags(p.tags)}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_task_page(task: str, papers: List[Paper], now: str) -> str:
    lines: List[str] = []
    lines.append("---")
    lines.append("type: paper-index")
    lines.append("dimension: task")
    lines.append(f"task: {task}")
    lines.append("tags:")
    lines.append("  - paperCollection")
    lines.append("  - index/task")
    lines.append(f"generated: {now}")
    lines.append("---")
    lines.append("")
    lines.append(f"# Task: {task}")
    lines.append("")
    lines.append(f"- Back: {md_link('paperCollection/README.md', 'Home')}")
    lines.append("")

    for p in papers:
        alias = f"{p.title} ({p.venue} {p.year})"
        ana = md_link(p.analysis_rel, alias)
        if p.pdf_ref:
            pdf = md_link(p.pdf_ref, "PDF")
            lines.append(f"- {ana} · {pdf} · techniques: {format_tech_tags(p.tags)}")
        else:
            lines.append(f"- {ana} · techniques: {format_tech_tags(p.tags)}")
    lines.append("")
    return "\n".join(lines)


def build_technique_index(techniques: List[str], now: str) -> str:
    lines: List[str] = []
    lines.append("---")
    lines.append("type: paper-index")
    lines.append("dimension: technique")
    lines.append("tags:")
    lines.append("  - paperCollection")
    lines.append("  - index/technique")
    lines.append(f"generated: {now}")
    lines.append("---")
    lines.append("")
    lines.append("# Technique index")
    lines.append("")
    lines.append(f"- Back: {md_link('paperCollection/README.md', 'Home')}")
    lines.append("")
    for t in techniques:
        lines.append(f"- {md_link(f'paperCollection/by_technique/{sanitize_filename(t)}.md', t)}")
    lines.append("")
    return "\n".join(lines)


def build_technique_page(technique: str, papers: List[Paper], now: str) -> str:
    lines: List[str] = []
    lines.append("---")
    lines.append("type: paper-index")
    lines.append("dimension: technique")
    lines.append(f"technique: {technique}")
    lines.append("tags:")
    lines.append("  - paperCollection")
    lines.append("  - index/technique")
    lines.append(f"generated: {now}")
    lines.append("---")
    lines.append("")
    lines.append(f"# Technique: {technique}")
    lines.append("")
    lines.append(f"- Back: {md_link('paperCollection/by_technique/_Index.md', 'Technique index')}")
    lines.append("")

    for p in papers:
        alias = f"{p.title} ({p.venue} {p.year})"
        ana = md_link(p.analysis_rel, alias)
        task_link = md_link(f'paperCollection/by_task/{sanitize_filename(p.category)}.md', p.category)
        tags_text = format_tech_tags(p.tags)
        if p.pdf_ref:
            pdf = md_link(p.pdf_ref, "PDF")
            lines.append(f"- {ana} · {pdf} · task: {task_link} · techniques: {tags_text}")
        else:
            lines.append(f"- {ana} · task: {task_link} · techniques: {tags_text}")
    lines.append("")
    return "\n".join(lines)


def build_venue_index(venues: List[str], now: str) -> str:
    lines: List[str] = []
    lines.append("---")
    lines.append("type: paper-index")
    lines.append("dimension: venue")
    lines.append("tags:")
    lines.append("  - paperCollection")
    lines.append("  - index/venue")
    lines.append(f"generated: {now}")
    lines.append("---")
    lines.append("")
    lines.append("# Venue index")
    lines.append("")
    lines.append(f"- Back: {md_link('paperCollection/README.md', 'Home')}")
    lines.append("")
    for v in venues:
        lines.append(f"- {md_link(f'paperCollection/by_venue/{sanitize_filename(v)}.md', v)}")
    lines.append("")
    return "\n".join(lines)


def build_venue_page(venue: str, papers: List[Paper], now: str) -> str:
    lines: List[str] = []
    lines.append("---")
    lines.append("type: paper-index")
    lines.append("dimension: venue")
    lines.append(f"venue: {venue}")
    lines.append("tags:")
    lines.append("  - paperCollection")
    lines.append("  - index/venue")
    lines.append(f"generated: {now}")
    lines.append("---")
    lines.append("")
    lines.append(f"# Venue: {venue}")
    lines.append("")
    lines.append(f"- Back: {md_link('paperCollection/by_venue/_Index.md', 'Venue index')}")
    lines.append("")

    # group by year desc
    by_year = group_by(papers, lambda p: p.year)
    def year_sort_key(y: str) -> Tuple[int, int]:
        # Sort numeric years desc; non-numeric (e.g., UnknownYear) goes last.
        digits = re.sub(r"\D+", "", str(y))
        if digits:
            return (0, -int(digits))
        return (1, 0)

    years_sorted = sorted(by_year.keys(), key=year_sort_key)
    for y in years_sorted:
        lines.append(f"## {y}")
        for p in by_year[y]:
            alias = f"{p.title} ({p.venue} {p.year})"
            ana = md_link(p.analysis_rel, alias)
            if p.pdf_ref:
                pdf = md_link(p.pdf_ref, "PDF")
                lines.append(f"- {ana} · {pdf} · task: {md_link(f'paperCollection/by_task/{sanitize_filename(p.category)}.md', p.category)} · techniques: {format_tech_tags(p.tags)}")
            else:
                lines.append(f"- {ana} · task: {md_link(f'paperCollection/by_task/{sanitize_filename(p.category)}.md', p.category)} · techniques: {format_tech_tags(p.tags)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_optional_tag_maintenance(passes: int = 1) -> None:
    """Run repo-wide tag normalization/audit when technique-tag count explodes."""
    normalize_script = VAULT_ROOT / ".claude" / "skills" / "papers-build-collection-index" / "scripts" / "normalize_paper_tags.py"
    audit_script = VAULT_ROOT / ".claude" / "skills" / "papers-build-collection-index" / "scripts" / "audit_paper_tags.py"

    for _ in range(max(1, passes)):
        if normalize_script.exists():
            subprocess.run([sys.executable, str(normalize_script)], check=False, cwd=str(VAULT_ROOT))

    if audit_script.exists():
        subprocess.run([sys.executable, str(audit_script)], check=False, cwd=str(VAULT_ROOT))


def main() -> int:
    if not PAPER_ANALYSIS_DIR.exists():
        raise RuntimeError(f"Missing folder: {PAPER_ANALYSIS_DIR}")

    now = datetime.now().strftime("%Y-%m-%dT%H:%M")
    papers = load_papers()

    # Pre-check: count raw technique tags before writing outputs.
    pre_tech_to_papers: Dict[str, List[Paper]] = {}
    for p in papers:
        for t in p.tags:
            pre_tech_to_papers.setdefault(t, []).append(p)

    if len(pre_tech_to_papers) > TAG_EXPLOSION_THRESHOLD:
        print(
            f"[WARN] technique tags exceeded threshold ({len(pre_tech_to_papers)} > {TAG_EXPLOSION_THRESHOLD}); running tag normalization/audit first"
        )
        run_optional_tag_maintenance(passes=1)
        papers = load_papers()

    # Prepare output dirs.
    # by_task / by_technique were retired — task & technique browsing now goes
    # through Obsidian tag/backlink navigation on `paperAnalysis/` frontmatter.
    # Only by_venue remains because venue grouping has no tag equivalent.
    ensure_dir(PAPER_COLLECTION_DIR)
    ensure_dir(PAPER_COLLECTION_DIR / "by_venue")

    clean_md_dir(PAPER_COLLECTION_DIR / "by_venue")

    # Remove any stale by_task / by_technique directories left over from
    # previous runs so the vault doesn't keep dead navigation pages.
    import shutil as _shutil
    for dead in ("by_task", "by_technique"):
        dead_dir = PAPER_COLLECTION_DIR / dead
        if dead_dir.exists():
            _shutil.rmtree(dead_dir)

    # Agent index + home + all
    write_text(PAPER_COLLECTION_DIR / "index.jsonl", build_agent_index_jsonl(papers))
    write_text(PAPER_COLLECTION_DIR / "README.md", build_readme(papers, now))
    write_text(PAPER_COLLECTION_DIR / "_AllPapers.md", build_all_papers(papers, now))

    # Track tasks/techniques only for stats (no per-tag page emission)
    by_task = group_by(papers, lambda p: p.category)
    tech_to_papers: Dict[str, List[Paper]] = {}
    for p in papers:
        for t in p.tags:
            tech_to_papers.setdefault(t, []).append(p)

    if len(tech_to_papers) > TAG_EXPLOSION_THRESHOLD:
        print(
            f"[WARN] technique tags still high after normalization ({len(tech_to_papers)} > {TAG_EXPLOSION_THRESHOLD}); emitted tag audit report for manual consolidation"
        )
        run_optional_tag_maintenance(passes=0)

    # By venue (sole surviving navigation dimension)
    by_venue = group_by(papers, lambda p: p.venue)
    venues = sorted(by_venue.keys(), key=lambda x: x.lower())
    write_text(PAPER_COLLECTION_DIR / "by_venue" / "_Index.md", build_venue_index(venues, now))
    for v, v_papers in by_venue.items():
        out = PAPER_COLLECTION_DIR / "by_venue" / f"{sanitize_filename(v)}.md"
        write_text(out, build_venue_page(v, v_papers, now))

    print(f"[OK] papers: {len(papers)}")
    print(f"[OK] tasks (stats only): {len(by_task)}")
    print(f"[OK] technique tags (stats only): {len(tech_to_papers)}")
    print(f"[OK] venues: {len(by_venue)}")
    print(f"[OK] output: {PAPER_COLLECTION_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
