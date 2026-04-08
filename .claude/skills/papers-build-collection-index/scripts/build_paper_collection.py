from __future__ import annotations

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


def parse_frontmatter(md_text: str) -> Dict[str, object]:
    lines = md_text.splitlines()
    if not lines or not FRONTMATTER_BOUNDARY.match(lines[0]):
        return {}

    # locate closing boundary
    end_idx = None
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
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        data[key] = val
        i += 1

    return data


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


def infer_technique_tags(fm: Dict[str, object], task: str) -> Tuple[str, ...]:
    tags = fm.get("tags")
    if not isinstance(tags, list):
        return tuple()
    out: List[str] = []
    for t in tags:
        s = str(t).strip()
        # Normalize common YAML scalar quoting in list items.
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        # YAML tags should not include leading '#', but some notes may.
        # Normalize so technique tags are consistent with Obsidian-style rendering.
        if s.startswith("#"):
            s = s[1:].strip()
        if not s or s == task:
            continue
        if s.startswith("status/"):
            continue
        out.append(s)
    # keep original order but unique
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return tuple(uniq)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def iter_analysis_mds() -> Iterable[Path]:
    yield from PAPER_ANALYSIS_DIR.rglob("*.md")


def load_papers() -> List[Paper]:
    papers: List[Paper] = []
    for md_path in iter_analysis_mds():
        rel = md_path.relative_to(VAULT_ROOT).as_posix()
        text = read_text(md_path)
        fm = parse_frontmatter(text)
        title = str(fm.get("title") or md_path.stem).strip()
        venue = str(fm.get("venue") or "").strip() or "UnknownVenue"
        year = str(fm.get("year") or "").strip() or "UnknownYear"
        pdf_ref = str(fm.get("pdf_ref") or "").strip()

        # Heuristic filter: only index notes that clearly correspond to a paper PDF.
        # This avoids pulling in helper docs (e.g., SKILL.md) under paperAnalysis.
        if not (pdf_ref.startswith("paperPDFs/") and pdf_ref.lower().endswith(".pdf")):
            continue

        task = infer_task(fm, rel)
        tech = infer_technique_tags(fm, task)

        papers.append(
            Paper(
                analysis_rel=rel,
                title=title,
                venue=venue,
                year=year,
                category=task,
                tags=tech,
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
    lines.append("这个目录由脚本自动从 `paperAnalysis/` 生成索引页，用于 **分类阅读** 与 **知识库入口层**。")
    lines.append("")
    lines.append("## Start here")
    lines.append("")
    lines.append(f"- {md_link('paperCollection/_AllPapers.md', 'All papers (grouped)')}")
    lines.append("- 按任务（Task）")
    for t in tasks:
        lines.append(f"  - {md_link(f'paperCollection/by_task/{sanitize_filename(t)}.md', t)}")
    lines.append("- 按技术（Technique tags）")
    lines.append(f"  - {md_link('paperCollection/by_technique/_Index.md', 'Technique index')}")
    lines.append("- 按会议/期刊（Venue）")
    lines.append(f"  - {md_link('paperCollection/by_venue/_Index.md', 'Venue index')}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- 本索引默认只链接，不嵌入 PDF（避免页面过重）。")
    lines.append("- `paperAnalysis` 才是主内容；`paperCollection` 是便于检索与聚合的入口层。")
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
        if p.pdf_ref:
            pdf = md_link(p.pdf_ref, "PDF")
            lines.append(f"- {ana} · {pdf} · task: {md_link(f'paperCollection/by_task/{sanitize_filename(p.category)}.md', p.category)}")
        else:
            lines.append(f"- {ana} · task: {md_link(f'paperCollection/by_task/{sanitize_filename(p.category)}.md', p.category)}")
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

    # Prepare output dirs
    ensure_dir(PAPER_COLLECTION_DIR)
    ensure_dir(PAPER_COLLECTION_DIR / "by_task")
    ensure_dir(PAPER_COLLECTION_DIR / "by_technique")
    ensure_dir(PAPER_COLLECTION_DIR / "by_venue")

    # Clean stale generated pages (avoid leaving old tag pages around).
    clean_md_dir(PAPER_COLLECTION_DIR / "by_task")
    clean_md_dir(PAPER_COLLECTION_DIR / "by_technique")
    clean_md_dir(PAPER_COLLECTION_DIR / "by_venue")

    # Home + all
    write_text(PAPER_COLLECTION_DIR / "README.md", build_readme(papers, now))
    write_text(PAPER_COLLECTION_DIR / "_AllPapers.md", build_all_papers(papers, now))

    # By task
    by_task = group_by(papers, lambda p: p.category)
    for task, task_papers in by_task.items():
        out = PAPER_COLLECTION_DIR / "by_task" / f"{sanitize_filename(task)}.md"
        write_text(out, build_task_page(task, task_papers, now))

    # By technique
    tech_to_papers: Dict[str, List[Paper]] = {}
    for p in papers:
        for t in p.tags:
            tech_to_papers.setdefault(t, []).append(p)
    techniques = sorted(tech_to_papers.keys(), key=lambda x: x.lower())
    write_text(PAPER_COLLECTION_DIR / "by_technique" / "_Index.md", build_technique_index(techniques, now))
    for t, t_papers in tech_to_papers.items():
        out = PAPER_COLLECTION_DIR / "by_technique" / f"{sanitize_filename(t)}.md"
        write_text(out, build_technique_page(t, t_papers, now))

    # If still above threshold after normalization, emit stronger warning + refresh audit once more.
    if len(tech_to_papers) > TAG_EXPLOSION_THRESHOLD:
        print(
            f"[WARN] technique tags still high after normalization ({len(tech_to_papers)} > {TAG_EXPLOSION_THRESHOLD}); emitted tag audit report for manual consolidation"
        )
        run_optional_tag_maintenance(passes=0)

    # By venue
    by_venue = group_by(papers, lambda p: p.venue)
    venues = sorted(by_venue.keys(), key=lambda x: x.lower())
    write_text(PAPER_COLLECTION_DIR / "by_venue" / "_Index.md", build_venue_index(venues, now))
    for v, v_papers in by_venue.items():
        out = PAPER_COLLECTION_DIR / "by_venue" / f"{sanitize_filename(v)}.md"
        write_text(out, build_venue_page(v, v_papers, now))

    print(f"[OK] papers: {len(papers)}")
    print(f"[OK] tasks: {len(by_task)}")
    print(f"[OK] technique tags: {len(tech_to_papers)}")
    print(f"[OK] venues: {len(by_venue)}")
    print(f"[OK] output: {PAPER_COLLECTION_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

