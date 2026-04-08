from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from pypdf import PdfReader
except ImportError:  # graceful fallback if dependency not installed yet
    PdfReader = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
PAPER_ROOT = REPO_ROOT / "paperPDFs"
LOG_PATH = PAPER_ROOT / "download_log_updated.txt"
WRONG_LOG_PATH = PAPER_ROOT / "wrong_download.txt"


@dataclass
class LogEntry:
    """Single line parsed from download_log_updated.txt."""

    index: int  # 1-based line number
    raw_line: str
    status: str
    title: str
    venue: str
    project_url: str
    pdf_url: str
    category: str
    local_pdf: Optional[Path] = None
    filename_similarity: Optional[float] = None
    pdf_title: Optional[str] = None
    title_similarity: Optional[float] = None


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching."""
    import re

    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = " ".join(s.split())
    return s


def parse_log(path: Path = LOG_PATH) -> List[LogEntry]:
    """
    Parse download_log_updated.txt into structured entries.

    Expected columns (6):
      status | title | venue | project_url | pdf_url | category
    """
    entries: List[LogEntry] = []
    if not path.is_file():
        raise FileNotFoundError(f"Log file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(" | ")]

            if len(parts) < 5:
                continue

            status = parts[0]
            title = parts[1] if len(parts) > 1 else ""
            venue = parts[2] if len(parts) > 2 else ""
            project_url = parts[3] if len(parts) > 3 else ""
            pdf_url = parts[4] if len(parts) > 4 else ""
            category = parts[5] if len(parts) > 5 else ""

            entries.append(
                LogEntry(
                    index=idx,
                    raw_line=line,
                    status=status,
                    title=title,
                    venue=venue,
                    project_url=project_url,
                    pdf_url=pdf_url,
                    category=category,
                )
            )

    return entries


def scan_pdfs(root: Path = PAPER_ROOT) -> List[Path]:
    """Scan paperPDFs/** for all PDF files."""
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.pdf"))


def _category_matches(path: Path, entry: LogEntry) -> bool:
    """Roughly check if the top-level category folder matches the log category."""
    rel = path.relative_to(PAPER_ROOT)
    parts = rel.parts
    if not parts:
        return False
    folder_cat = parts[0]

    cat_norm = normalize_text(entry.category.replace("_", " "))
    folder_norm = normalize_text(folder_cat.replace("_", " "))

    if not cat_norm or not folder_norm:
        return False

    ratio = difflib.SequenceMatcher(None, cat_norm, folder_norm).ratio()
    return ratio >= 0.6


def match_entries_to_pdfs(entries: List[LogEntry], pdf_paths: List[Path]) -> None:
    """
    For each log entry, try to find the most likely local PDF file, based on:
    - rough category folder match
    - fuzzy match between title and filename stem
    """
    if not pdf_paths:
        return

    for entry in entries:
        if entry.status not in {"Downloaded", "WAIT"}:
            continue

        title_norm = normalize_text(entry.title)
        if not title_norm:
            continue

        candidates: List[Path] = [p for p in pdf_paths if _category_matches(p, entry)] or pdf_paths

        best_score = 0.0
        best_path: Optional[Path] = None

        for pdf in candidates:
            stem_norm = normalize_text(pdf.stem)
            if not stem_norm:
                continue
            score = difflib.SequenceMatcher(None, title_norm, stem_norm).ratio()
            if score > best_score:
                best_score = score
                best_path = pdf

        if best_path is not None and best_score >= 0.5:
            entry.local_pdf = best_path
            entry.filename_similarity = best_score


def extract_title_from_pdf(pdf_path: Path, log_title: str) -> Tuple[Optional[str], Optional[float]]:
    if PdfReader is None:
        return None, None

    try:
        reader = PdfReader(str(pdf_path))
        if not reader.pages:
            return None, None
        first_page = reader.pages[0]
        text = first_page.extract_text() or ""
    except Exception:
        return None, None

    lines = [ln.strip() for ln in text.splitlines()]
    candidate_lines = [ln for ln in lines if len(ln) >= 10][:80]
    if not candidate_lines:
        return None, None

    log_norm = normalize_text(log_title)
    if not log_norm:
        return None, None

    best_line: Optional[str] = None
    best_score: float = 0.0

    for ln in candidate_lines:
        cand_norm = normalize_text(ln)
        if not cand_norm:
            continue
        score = difflib.SequenceMatcher(None, log_norm, cand_norm).ratio()
        if score > best_score:
            best_score = score
            best_line = ln

    if best_line is None:
        return None, None

    return best_line, best_score


def enrich_entries_with_pdf_titles(entries: List[LogEntry]) -> None:
    if PdfReader is None:
        return

    for entry in entries:
        if entry.local_pdf is None:
            continue
        if entry.title_similarity is not None:
            continue

        pdf_title, score = extract_title_from_pdf(entry.local_pdf, entry.title)
        if not pdf_title or score is None:
            continue

        entry.pdf_title = pdf_title
        entry.title_similarity = score


def detect_wrong_downloads(entries: List[LogEntry]) -> List[Tuple[LogEntry, str]]:
    wrong: List[Tuple[LogEntry, str]] = []

    bad_url_substrings = [
        "nerfies/videos/nerfies_paper.pdf",
        "nerfies_paper.pdf",
        "MotionCLIP.pdf",
        "tmr_poster.pdf",
        "Parameterized_Quasi_Physical_Simulators_for_Dexterous_Manipulations_Transfer.pdf",
        "<ARXIV PAPER ID>",
    ]

    for entry in entries:
        issues: List[str] = []

        pdf_url_lower = entry.pdf_url.lower()
        for pat in bad_url_substrings:
            if pat.lower() in pdf_url_lower:
                issues.append(f"PDF URL contains placeholder/wrong file pattern: {pat}")
                break

        if entry.local_pdf is not None and entry.filename_similarity is not None:
            if entry.filename_similarity < 0.5:
                issues.append(
                    f"Local filename '{entry.local_pdf.name}' is poorly matched to title "
                    f"(filename_similarity={entry.filename_similarity:.2f})"
                )

        if entry.pdf_title and entry.title_similarity is not None:
            if entry.title_similarity < 0.6:
                issues.append(
                    f"PDF first-page title '{entry.pdf_title}' is poorly matched to log title "
                    f"(title_similarity={entry.title_similarity:.2f})"
                )

        if issues:
            wrong.append((entry, "; ".join(issues)))

    return wrong


def write_wrong_download_log(wrong_entries: List[Tuple[LogEntry, str]], path: Path = WRONG_LOG_PATH) -> None:
    if not wrong_entries:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("# No suspicious downloads detected based on current heuristics.\n")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry, issue in wrong_entries:
            local_path_str = str(entry.local_pdf) if entry.local_pdf else "-"
            line = f"{entry.title} | {local_path_str} | {entry.pdf_url} | {issue}\n"
            f.write(line)


def main() -> None:
    print(f"Using log file: {LOG_PATH}")
    print(f"Scanning PDFs under: {PAPER_ROOT}")

    entries = parse_log(LOG_PATH)
    pdf_paths = scan_pdfs(PAPER_ROOT)
    match_entries_to_pdfs(entries, pdf_paths)
    enrich_entries_with_pdf_titles(entries)
    wrong_entries = detect_wrong_downloads(entries)
    write_wrong_download_log(wrong_entries, WRONG_LOG_PATH)

    print(f"Total log entries parsed: {len(entries)}")
    print(f"Total PDFs found: {len(pdf_paths)}")
    print(f"Suspicious / wrong downloads detected: {len(wrong_entries)}")
    print(f"Wrong download log written to: {WRONG_LOG_PATH}")


if __name__ == "__main__":
    main()

