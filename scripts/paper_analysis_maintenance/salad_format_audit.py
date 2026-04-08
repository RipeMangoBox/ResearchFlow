import argparse
import csv
import sys
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:  # pragma: no cover
        PdfReader = None  # type: ignore


@dataclass
class SaladCheckResult:
    ok: bool
    reasons: List[str]
    md_path: Path
    status: str


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

REQUIRED_PART_MARKERS = [
    "Part I: The \"Skill\" Signature",
    "Part II: High-Dimensional Insight",
    "Part III: Technical Deep Dive",
]

REQUIRED_BULLET_MARKERS = [
    "Atomic Capability",
    "Data Interface",
    "Operational Logic",
    "Boundary Conditions",
]

REQUIRED_QUICKLINK_MARKERS = [
    "Quick Links & TL;DR",
    "Links",
    "Summary",
    "Key Performance",
]

REQUIRED_SECTION_MARKERS = [
    "The Design Philosophy",
    "The \"Aha!\" Moment",
    "Methodological Pipeline",
    "Empirical Evidence",
    "Implementation Constraints",
    "Local Reading",
]


def split_frontmatter_and_body(text: str) -> Tuple[str, str]:
    """
    Very lightweight YAML frontmatter splitter.
    Assumes Obsidian-style:
    ---
    key: value
    ---
    body...
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", text

    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :])
            return fm, body
    return "", text


def parse_frontmatter_keys(frontmatter: str) -> List[str]:
    keys = []
    for line in frontmatter.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key = line.split(":", 1)[0].strip()
            if key:
                keys.append(key)
    return keys


def get_section_text(body: str, header: str) -> str:
    lines = body.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("#") and header in line:
            start_idx = i + 1
            break
    if start_idx is None:
        return ""
    for j in range(start_idx, len(lines)):
        if lines[j].strip().startswith("#"):
            return "\n".join(lines[start_idx:j]).strip()
    return "\n".join(lines[start_idx:]).strip()


def has_numeric_signal(text: str) -> bool:
    for token in text.replace(",", " ").split():
        if any(ch.isdigit() for ch in token):
            return True
    return False


def check_salad_md(md_path: Path) -> Tuple[bool, List[str]]:
    text = md_path.read_text(encoding="utf-8")
    reasons: List[str] = []

    frontmatter, body = split_frontmatter_and_body(text)
    if not frontmatter:
        reasons.append("missing YAML frontmatter block")
    else:
        keys = set(parse_frontmatter_keys(frontmatter))
        missing_keys = sorted(REQUIRED_FRONTMATTER_KEYS - keys)
        if missing_keys:
            reasons.append(f"missing frontmatter keys: {', '.join(missing_keys)}")

    if frontmatter:
        title_line = next((l for l in frontmatter.splitlines() if l.strip().startswith("title:")), "")
        title_value = title_line.split(":", 1)[1].strip().strip('"') if ":" in title_line else ""
    else:
        title_value = ""

    first_h1 = next((l for l in body.splitlines() if l.startswith("# ")), "")
    if not first_h1:
        reasons.append("missing level-1 title heading in body")
    elif title_value and title_value not in first_h1:
        reasons.append("H1 title heading does not contain frontmatter title")

    for marker in REQUIRED_QUICKLINK_MARKERS:
        if marker not in body:
            reasons.append(f"missing quick-link marker: {marker}")

    for marker in REQUIRED_PART_MARKERS:
        if marker not in body:
            reasons.append(f"missing section marker: {marker}")

    for marker in REQUIRED_SECTION_MARKERS:
        if marker not in body:
            reasons.append(f"missing section marker: {marker}")

    for marker in REQUIRED_BULLET_MARKERS:
        if marker not in body:
            reasons.append(f"missing bullet/field: {marker}")

    data_interface = get_section_text(body, "Data Interface")
    if "Input" not in data_interface or "Output" not in data_interface:
        reasons.append("missing Input/Output in Data Interface")

    operational_logic = get_section_text(body, "Operational Logic")
    if not any(x in operational_logic for x in ["1)", "1.", "Step", "->"]):
        reasons.append("Operational Logic lacks stepwise flow")

    empirical = get_section_text(body, "Empirical Evidence")
    if not has_numeric_signal(empirical):
        reasons.append("Empirical Evidence lacks numeric results")

    return (not reasons), reasons


def find_md_for_title(root: Path, title: str) -> Optional[Path]:
    for md_path in root.rglob("*.md"):
        try:
            text = md_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        frontmatter, _ = split_frontmatter_and_body(text)
        if not frontmatter:
            continue
        for line in frontmatter.splitlines():
            line = line.strip()
            if line.startswith("title:"):
                value = line.split(":", 1)[1].strip().strip('"')
                if value == title:
                    return md_path
                break
    return None


def iter_wait_entries(rows: List[List[str]]):
    data_start = 1 if rows and rows[0] and rows[0][0] == "state" else 0
    for idx in range(data_start, len(rows)):
        row = (rows[idx] + [""] * 8)[:8]
        status = row[0]
        if status != "Wait":
            continue
        title = row[2]
        venue = row[3]
        github = row[4]
        pdf_url = row[5]
        category = row[6]
        yield idx, row, title, venue, github, pdf_url, category


def update_log_rows(rows: List[List[str]], index: int, new_status: str) -> None:
    row = (rows[index] + [""] * 8)[:8]
    row[0] = new_status
    rows[index] = row


def extract_pdf_text(pdf_path: Path, max_pages: int = 3) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return ""
    text_chunks: List[str] = []
    for i, page in enumerate(reader.pages[:max_pages]):
        try:
            text_chunks.append(page.extract_text() or "")
        except Exception:
            continue
        if i >= max_pages - 1:
            break
    return "\n".join(text_chunks)


def simple_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_tokens = {t.lower() for t in a.split() if t.isalpha() or t.isalnum()}
    b_tokens = {t.lower() for t in b.split() if t.isalpha() or t.isalnum()}
    if not a_tokens or not b_tokens:
        return 0.0
    inter = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(inter) / max(len(union), 1)


def find_pdf_for_md(md_path: Path, repo_root: Path) -> Optional[Path]:
    try:
        rel = md_path.relative_to(repo_root / "paperAnalysis")
    except Exception:
        return None
    pdf_path = repo_root / "paperPDFs" / rel.with_suffix(".pdf")
    if pdf_path.exists():
        return pdf_path
    alt_name = pdf_path.name.replace("-", " ")
    alt_path = pdf_path.with_name(alt_name)
    if alt_path.exists():
        return alt_path
    return None


def pdf_mismatch_check(md_path: Path, pdf_path: Path) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    text = md_path.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter_and_body(text)
    title_value = ""
    if frontmatter:
        for line in frontmatter.splitlines():
            if line.strip().startswith("title:"):
                title_value = line.split(":", 1)[1].strip().strip('"')
                break

    pdf_text = extract_pdf_text(pdf_path)
    if not pdf_text:
        reasons.append("unable to extract PDF text for mismatch check")
        return False, reasons

    first_lines = " ".join(pdf_text.splitlines()[:5])
    if title_value:
        sim = simple_similarity(title_value, first_lines)
        if sim < 0.2:
            reasons.append("PDF title text does not match MD title")

    abstract_text = get_section_text(body, "Abstract")
    if not abstract_text:
        abstract_text = "\n".join(line for line in body.splitlines() if "Summary" in line or "abstract" in line.lower())
    if abstract_text:
        sim = simple_similarity(abstract_text, pdf_text)
        if sim < 0.05:
            reasons.append("MD abstract/summary does not overlap with PDF text")

    return bool(reasons), reasons


def normalize_category_dir(category: str) -> str:
    return category.replace(" ", "_").replace("-", "_")


def sanitize_title_for_filename(title: str) -> str:
    out = []
    prev_us = False
    for ch in title:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    return "".join(out).strip("_")


def normalize_venue_dir(venue: str) -> str:
    return venue.replace(" ", "_")


def build_md_path(repo_root: Path, category: str, venue: str, year: str, title: str) -> Path:
    category_dir = normalize_category_dir(category)
    venue_dir = normalize_venue_dir(venue)
    safe_title = sanitize_title_for_filename(title)
    return repo_root / "paperAnalysis" / category_dir / venue_dir / f"{year}_{safe_title}.md"


def build_pdf_path(repo_root: Path, category: str, venue: str, year: str, title: str) -> Optional[Path]:
    category_dir = normalize_category_dir(category)
    venue_dir = normalize_venue_dir(venue)
    safe_title = sanitize_title_for_filename(title)
    candidate = repo_root / "paperPDFs" / category_dir / venue_dir / f"{year}_{safe_title}.pdf"
    if candidate.exists():
        return candidate
    alt = candidate.with_name(candidate.name.replace("-", " "))
    if alt.exists():
        return alt
    return None


def build_pdf_path_strict(repo_root: Path, category: str, venue: str, title: str) -> Optional[Path]:
    year = "".join([ch for ch in venue if ch.isdigit()]) or "Unknown"
    category_dir = normalize_category_dir(category)
    venue_dir = normalize_venue_dir(venue)
    safe_title = sanitize_title_for_filename(title)

    pdf_dir = repo_root / "paperPDFs" / category_dir / venue_dir
    candidate = pdf_dir / f"{year}_{safe_title}.pdf"
    if candidate.exists():
        return candidate
    alt = candidate.with_name(candidate.name.replace("-", " "))
    if alt.exists():
        return alt

    if not pdf_dir.exists():
        return None

    def norm(s: str) -> str:
        x = sanitize_title_for_filename(s).lower()
        x = x.replace("modelling", "modeling")
        return x

    expected = norm(f"{year}_{title}")
    prefix_hits: List[Path] = []
    for pdf in pdf_dir.glob("*.pdf"):
        stem_norm = norm(pdf.stem)
        if stem_norm == expected:
            return pdf
        if stem_norm.startswith(expected) or expected.startswith(stem_norm):
            prefix_hits.append(pdf)

    if len(prefix_hits) == 1:
        return prefix_hits[0]
    return None


def md_path_from_pdf_path(repo_root: Path, pdf_path: Path) -> Optional[Path]:
    try:
        rel = pdf_path.relative_to(repo_root / "paperPDFs")
    except Exception:
        return None
    return (repo_root / "paperAnalysis" / rel).with_suffix(".md")


def extract_abstract(pdf_text: str) -> str:
    if not pdf_text:
        return ""
    lower = pdf_text.lower()
    idx = lower.find("abstract")
    if idx == -1:
        return ""
    snippet = pdf_text[idx : idx + 2000]
    lines = [l.strip() for l in snippet.splitlines() if l.strip()]
    if not lines:
        return ""
    if lines[0].lower().startswith("abstract"):
        lines = lines[1:]
    abstract = " ".join(lines[:6])
    return abstract.strip()


def extract_numbers(pdf_text: str, limit: int = 3) -> list[str]:
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", pdf_text or "")
    return nums[:limit]


def generate_md_from_pdf(
    md_path: Path,
    pdf_path: Path,
    repo_root: Path,
    title: str,
    venue: str,
    year: str,
    github: str,
    pdf_url: str,
    category: str,
) -> None:
    pdf_text = extract_pdf_text(pdf_path, max_pages=3)
    abstract = extract_abstract(pdf_text)
    numbers = extract_numbers(pdf_text)
    created = datetime.now().strftime("%Y-%m-%dT%H:%M")
    tags = [normalize_category_dir(category), "status/analyzed"]

    md_path.parent.mkdir(parents=True, exist_ok=True)

    summary = abstract if abstract else "[TODO: add 1-2 sentence summary]"
    key_perf = f"Example numeric signals: {', '.join(numbers)}" if numbers else "Example numeric signals: 0"

    metric = numbers[0] if len(numbers) > 0 else "0"
    baseline = numbers[1] if len(numbers) > 1 else "0"
    delta = numbers[2] if len(numbers) > 2 else "0"

    pdf_ref = pdf_path.relative_to(repo_root).as_posix()
    content = f"""---
created: {created}
updated: {created}
title: "{title}"
venue: {venue}
year: {year}
tags:
  - {tags[0]}
  - status/analyzed
core_operator: "[TODO]"
primary_logic: "[TODO]"
pdf_ref: "{pdf_ref}"
category: {tags[0]}
---

# {title}

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv]({pdf_url if pdf_url else "N/A"}) | [GitHub]({github if github else "N/A"}) | [Project](N/A)
> - **Summary**: {summary}
> - **Key Performance**: {key_perf}

---

## Part I: The "Skill" Signature (AI-Readiness)
*For AI skill indexing and retrieval.*

### Atomic Capability
{summary}

### Data Interface
- **Input**: [TODO: from PDF]
- **Output**: [TODO: from PDF]

### Operational Logic
1) [TODO] -> 2) [TODO] -> 3) [TODO]

### Boundary Conditions
[TODO]

---

## Part II: High-Dimensional Insight (Human-Centric)

### 1. The Design Philosophy
* **Conventional Wisdom**: [TODO]
* **The Paradigm Shift**: [TODO]
* **Why it Matters**: [TODO]

### 2. The "Aha!" Moment
* **Core Intuition**: [TODO]
* **Strategic Trade-off**: [TODO]

---

## Part III: Technical Deep Dive

### 1. Methodological Pipeline
1. **[Step A]**: [TODO]
2. **[Step B]**: [TODO]
3. **[Step C]**: [TODO]

### 2. Empirical Evidence
| Metrics | Baseline | **Ours** | Delta |
| :--- | :--- | :--- | :--- |
| [Metric Name] | {baseline} | **{metric}** | {delta} |

### 3. Implementation Constraints
* **Resources**: [TODO]
* **Dependency**: [TODO]

---

## emergentmind-paper-analysis (Required Core Sections)

### Abstract
{abstract if abstract else "[TODO: abstract]"}

### Technical Innovations
- [TODO: key contributions from PDF]

### Empirical Evaluation
- {key_perf}

---

## Local Reading
![[{pdf_ref}]]
"""
    md_path.write_text(content, encoding="utf-8")


def run_batch(repo_root: Path, batch_size: int) -> Tuple[List[SaladCheckResult], int]:
    paper_root = repo_root / "paperAnalysis"
    log_path = paper_root / "analysis_log.csv"

    with log_path.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f)]

    batch_indices: List[Tuple[int, list[str], str, str, str, str, str]] = []
    for idx, row, title, venue, github, pdf_url, category in iter_wait_entries(rows):
        batch_indices.append((idx, row, title, venue, github, pdf_url, category))
        if len(batch_indices) >= batch_size:
            break

    results: List[SaladCheckResult] = []
    updated = 0

    for idx, _row, title, venue, github, pdf_url, category in batch_indices:
        pdf_path = build_pdf_path_strict(repo_root, category, venue, title)
        if pdf_path is None or not pdf_path.exists():
            results.append(
                SaladCheckResult(
                    ok=False,
                    reasons=["PDF not found for this Wait entry (strict path mapping)"],
                    md_path=build_md_path(
                        repo_root,
                        category,
                        venue,
                        "".join([ch for ch in venue if ch.isdigit()]) or "Unknown",
                        title,
                    ),
                    status="Wait",
                )
            )
            continue

        md_path = md_path_from_pdf_path(repo_root, pdf_path)
        if md_path is None:
            results.append(
                SaladCheckResult(
                    ok=False,
                    reasons=["unable to map PDF path to MD path (paperPDFs -> paperAnalysis)"],
                    md_path=pdf_path.with_suffix(".md"),
                    status="Wait",
                )
            )
            continue

        year = "".join([ch for ch in venue if ch.isdigit()]) or "Unknown"

        generate_md_from_pdf(
            md_path=md_path,
            pdf_path=pdf_path,
            repo_root=repo_root,
            title=title,
            venue=venue,
            year=year,
            github=github,
            pdf_url=pdf_url,
            category=category,
        )

        ok, reasons = check_salad_md(md_path)
        status = "checked" if ok else "analysis_mismatch"
        results.append(SaladCheckResult(ok=ok, reasons=reasons, md_path=md_path, status=status))
        update_log_rows(rows, idx, status)
        updated += 1

    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return results, updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch SALAD-format audit for paperAnalysis markdowns.")
    parser.add_argument("--root", type=str, default=".", help="Repository root (default: current directory)")
    parser.add_argument("--batch-size", type=int, default=6, help="Number of Wait entries to process in this batch.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    batch_count = 0

    def safe_text(value: object) -> str:
        text = str(value)
        try:
            text.encode(sys.stdout.encoding or "utf-8")
            return text
        except Exception:
            return text.encode("utf-8", errors="backslashreplace").decode("utf-8")

    while True:
        results, updated = run_batch(root, batch_size=args.batch_size)
        if not results:
            if batch_count == 0:
                print("No Wait entries found in this batch.")
            break
        batch_count += 1
        print(f"SALAD audit batch summary #{batch_count}:")
        for r in results:
            status = "OK" if r.ok else "NG"
            print(f"- [{status}] ({r.status}) {safe_text(r.md_path)}")
            if r.reasons:
                for reason in r.reasons:
                    print(f"    - {safe_text(reason)}")

        if updated == 0:
            print("No log updates applied in this batch (likely missing MDs). Stopping.")
            break


if __name__ == "__main__":
    main()

