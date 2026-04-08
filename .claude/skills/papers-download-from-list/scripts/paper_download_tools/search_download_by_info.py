#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
import difflib
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


SCRIPT_DIR = Path(__file__).absolute().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_LOG_PATH = REPO_ROOT / "paperAnalysis" / "analysis_log.csv"
PAPER_ROOT = REPO_ROOT / "paperPDFs"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_ID_RE = re.compile(r"(?P<id>\d{4}\.\d{4,5})(?:v\d+)?")


@dataclass
class PaperSpec:
    query: str
    title: str
    alias: str
    arxiv_id: str
    paper_link: str
    project_link: str
    importance: str
    category: str
    sort: str
    venue: str
    require_keywords: list[str]


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    abs_url: str
    pdf_url: str
    updated: str
    published: str


@dataclass
class DownloadResult:
    matched: bool
    reason: str
    spec: PaperSpec
    paper: Optional[ArxivPaper]
    pdf_path: Optional[Path]
    log_line: Optional[str]


LOG_HEADER = [
    "state",
    "importance",
    "paper_title",
    "venue",
    "project_link_or_github_link",
    "paper_link",
    "sort",
    "pdf_path",
]


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sanitize_filename(text: str, max_len: int = 180) -> str:
    text = (text or "").strip()
    text = text.replace("–", "-").replace("—", "-").replace("&", " and ")
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ", "_").replace("-", "_")
    text = re.sub(r"[<>:\"/\\|?*\x00-\x1F,]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_. ")
    if not text:
        text = "Untitled"
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text


def parse_spec_file(path: Path) -> list[PaperSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("spec file must be a JSON array")

    specs: list[PaperSpec] = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"spec #{idx} must be an object")
        category = str(item.get("category") or "").strip()
        if not category:
            raise ValueError(f"spec #{idx} missing required field: category")
        require_keywords = item.get("require_keywords") or []
        if not isinstance(require_keywords, list):
            raise ValueError(f"spec #{idx} field require_keywords must be a list")
        specs.append(
            PaperSpec(
                query=str(item.get("query") or "").strip(),
                title=str(item.get("title") or "").strip(),
                alias=str(item.get("alias") or "").strip(),
                arxiv_id=extract_arxiv_id(str(item.get("arxiv_id") or "").strip()),
                paper_link=str(item.get("paper_link") or "").strip(),
                project_link=str(item.get("project_link") or "").strip() or "no",
                importance=str(item.get("importance") or "B").strip() or "B",
                category=category,
                sort=str(item.get("sort") or category).strip() or category,
                venue=str(item.get("venue") or "").strip(),
                require_keywords=[str(x).strip() for x in require_keywords if str(x).strip()],
            )
        )
    return specs


def extract_arxiv_id(text: str) -> str:
    if not text:
        return ""
    m = ARXIV_ID_RE.search(text)
    return m.group("id") if m else ""


def fetch_url(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_arxiv_entry_by_id(arxiv_id: str) -> Optional[ArxivPaper]:
    if not arxiv_id:
        return None
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    root = ET.fromstring(fetch_url(url))
    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        return None
    return parse_arxiv_entry(entry)


def search_arxiv_entries(query: str, max_results: int = 5) -> list[ArxivPaper]:
    if not query.strip():
        return []
    encoded = urllib.parse.quote(query)
    url = (
        "https://export.arxiv.org/api/query?search_query="
        f"all:{encoded}&start=0&max_results={max_results}"
    )
    root = ET.fromstring(fetch_url(url))
    out: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        parsed = parse_arxiv_entry(entry)
        if parsed:
            out.append(parsed)
    return out


def parse_arxiv_entry(entry: ET.Element) -> Optional[ArxivPaper]:
    title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
    summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
    entry_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
    updated = (entry.findtext("atom:updated", default="", namespaces=ATOM_NS) or "").strip()
    published = (entry.findtext("atom:published", default="", namespaces=ATOM_NS) or "").strip()
    arxiv_id = extract_arxiv_id(entry_id)
    if not arxiv_id:
        return None
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=re.sub(r"\s+", " ", title),
        summary=re.sub(r"\s+", " ", summary),
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        updated=updated,
        published=published,
    )


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def contains_keywords(text: str, keywords: Iterable[str]) -> bool:
    norm = normalize_text(text)
    for kw in keywords:
        k = normalize_text(kw)
        if not k:
            continue
        if k not in norm:
            return False
    return True


def choose_best_match(spec: PaperSpec) -> tuple[Optional[ArxivPaper], str]:
    arxiv_id = spec.arxiv_id or extract_arxiv_id(spec.paper_link)
    if arxiv_id:
        paper = fetch_arxiv_entry_by_id(arxiv_id)
        if paper is None:
            return None, f"arXiv id not found: {arxiv_id}"
        return paper, "matched by arXiv id"

    queries = [q for q in [spec.title, spec.alias, spec.query] if q]
    candidates: list[ArxivPaper] = []
    for q in queries:
        candidates.extend(search_arxiv_entries(q, max_results=5))

    dedup: dict[str, ArxivPaper] = {paper.arxiv_id: paper for paper in candidates}
    scored: list[tuple[float, ArxivPaper]] = []
    for paper in dedup.values():
        score = 0.0
        if spec.title:
            score = max(score, similarity(spec.title, paper.title))
        if spec.alias:
            alias_norm = normalize_text(spec.alias)
            if alias_norm and alias_norm in normalize_text(paper.title + " " + paper.summary):
                score = max(score, 0.88)
        if spec.query:
            score = max(score, similarity(spec.query, paper.title))
        if spec.require_keywords and contains_keywords(paper.title + " " + paper.summary, spec.require_keywords):
            score = max(score, 0.92)
        scored.append((score, paper))

    if not scored:
        return None, "no arXiv candidates found"

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_paper = scored[0]
    if best_score < 0.72:
        return None, f"best candidate below threshold ({best_score:.2f})"
    return best_paper, f"matched by search (score={best_score:.2f})"


def year_from_paper(paper: ArxivPaper) -> str:
    source = paper.published or paper.updated
    m = re.search(r"(19|20)\d{2}", source)
    return m.group(0) if m else "UnknownYear"


def build_pdf_path(spec: PaperSpec, paper: ArxivPaper) -> Path:
    year = year_from_paper(paper)
    venue = spec.venue or f"arXiv {year}"
    venue_dir = sanitize_filename(venue)
    title_for_file = sanitize_filename(paper.title)
    filename = f"{year}_{title_for_file}.pdf"
    return PAPER_ROOT / spec.category / venue_dir / filename


def display_title(spec: PaperSpec, paper: ArxivPaper) -> str:
    alias = spec.alias.strip()
    if not alias:
        return paper.title

    alias_norm = normalize_text(alias)
    title_norm = normalize_text(paper.title)
    if not alias_norm or not title_norm:
        return paper.title
    if alias_norm in title_norm or title_norm in alias_norm:
        return paper.title
    if "/" in alias or "|" in alias:
        return alias
    return f"{alias}: {paper.title}"


def load_existing_log_rows(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [row for row in csv.reader(f)]


def line_already_exists(existing_rows: list[list[str]], paper: ArxivPaper, pdf_path: Path) -> bool:
    abs_url = paper.abs_url
    pdf_ref = to_pdf_ref(pdf_path)
    title_norm = normalize_text(paper.title)
    for row in existing_rows:
        if not row:
            continue
        paper_link = row[5].strip() if len(row) > 5 else ""
        row_pdf_ref = row[7].strip() if len(row) > 7 else ""
        row_title = row[2].strip() if len(row) > 2 else ""
        if abs_url and abs_url == paper_link:
            return True
        if pdf_ref and pdf_ref == row_pdf_ref:
            return True
        if title_norm and title_norm in normalize_text(row_title):
            return True
    return False


def to_pdf_ref(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def make_log_row(spec: PaperSpec, paper: ArxivPaper, pdf_path: Path) -> list[str]:
    year = year_from_paper(paper)
    venue = spec.venue or f"arXiv {year}"
    return [
        "checked",
        spec.importance,
        display_title(spec, paper),
        venue,
        spec.project_link or "no",
        paper.abs_url,
        spec.sort,
        to_pdf_ref(pdf_path),
    ]


def append_log_rows(path: Path, rows_to_add: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = load_existing_log_rows(path)
    has_header = bool(existing_rows and existing_rows[0][:8] == LOG_HEADER)
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not has_header and mode == "w":
            writer.writerow(LOG_HEADER)
        for row in rows_to_add:
            writer.writerow(row)


def download_pdf(paper: ArxivPaper, dest: Path, timeout: int = 180) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(paper.pdf_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not data.startswith(b"%PDF"):
        raise RuntimeError(f"downloaded file is not a PDF: {paper.pdf_url}")
    if len(data) < 5000:
        raise RuntimeError(f"downloaded PDF too small ({len(data)} bytes): {paper.pdf_url}")
    dest.write_bytes(data)


def process_spec(spec: PaperSpec, log_path: Path, dry_run: bool = False) -> DownloadResult:
    paper, match_reason = choose_best_match(spec)
    if paper is None:
        return DownloadResult(False, match_reason, spec, None, None, None)

    if spec.title and similarity(spec.title, paper.title) < 0.62 and not spec.arxiv_id:
        return DownloadResult(False, "title mismatch after search", spec, paper, None, None)

    combined = f"{paper.title} {paper.summary}"
    if spec.require_keywords and not contains_keywords(combined, spec.require_keywords):
        return DownloadResult(False, "keyword validation failed", spec, paper, None, None)

    pdf_path = build_pdf_path(spec, paper)
    log_row = make_log_row(spec, paper, pdf_path)
    log_line = ",".join(log_row)
    if dry_run:
        return DownloadResult(True, match_reason + " (dry-run)", spec, paper, pdf_path, log_line)

    if not (pdf_path.is_file() and pdf_path.stat().st_size > 5000):
        download_pdf(paper, pdf_path)

    existing_rows = load_existing_log_rows(log_path)
    if not line_already_exists(existing_rows, paper, pdf_path):
        append_log_rows(log_path, [log_row])
        reason = match_reason + "; downloaded and logged"
    else:
        reason = match_reason + "; downloaded/already present, log already existed"
    return DownloadResult(True, reason, spec, paper, pdf_path, log_line)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search arXiv papers from user-provided info, download PDFs, and append analysis log entries."
    )
    parser.add_argument("--spec-file", required=True, help="JSON array of paper specs")
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH), help="analysis log file to append")
    parser.add_argument("--dry-run", action="store_true", help="validate matches without downloading or editing logs")
    args = parser.parse_args(argv)

    spec_path = Path(args.spec_file).expanduser().resolve()
    log_path = Path(args.log_path).expanduser().resolve()
    specs = parse_spec_file(spec_path)

    results: list[dict[str, Any]] = []
    exit_code = 0
    for spec in specs:
        try:
            result = process_spec(spec, log_path=log_path, dry_run=args.dry_run)
        except Exception as exc:
            result = DownloadResult(False, str(exc), spec, None, None, None)

        payload = {
            "query": spec.query,
            "alias": spec.alias,
            "requested_title": spec.title,
            "arxiv_id": spec.arxiv_id,
            "matched": result.matched,
            "reason": result.reason,
            "matched_title": result.paper.title if result.paper else "",
            "abs_url": result.paper.abs_url if result.paper else "",
            "pdf_path": str(result.pdf_path) if result.pdf_path else "",
            "log_line": result.log_line or "",
        }
        print(json.dumps(payload, ensure_ascii=False))
        results.append(payload)
        if not result.matched:
            exit_code = 1

    matched = sum(1 for r in results if r["matched"])
    print(json.dumps({"summary": {"total": len(results), "matched": matched, "failed": len(results) - matched}}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
