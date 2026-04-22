#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = DEFAULT_REPO_ROOT
PAPER_ANALYSIS_DIR = REPO_ROOT / "paperAnalysis"
PAPER_PDFS_DIR = REPO_ROOT / "paperPDFs"
ANALYSIS_LOG_PATH = PAPER_ANALYSIS_DIR / "analysis_log.csv"
PROCESSING_DIR = PAPER_ANALYSIS_DIR / "processing" / "zotero"
MANIFEST_PATH = PROCESSING_DIR / "manifest.jsonl"
SYNC_STATE_PATH = PROCESSING_DIR / "sync_state.json"
CATEGORY_MAP_PATH = PROCESSING_DIR / "category_map.json"
SYNC_ERRORS_PATH = PROCESSING_DIR / "sync_errors.jsonl"
ANALYSIS_LOG_HEADER = [
    "state",
    "importance",
    "paper_title",
    "venue",
    "project_link_or_github_link",
    "paper_link",
    "sort",
    "pdf_path",
]
DEFAULT_ALLOWED_ITEM_TYPES = ("conferencePaper", "journalArticle", "preprint")
DEFAULT_CATEGORY_MAP = {
    "zotero_collection_to_rf_category": {
        "motion generation": "Motion_Generation_Text_Speech_Music_Driven",
        "video generation": "Video_Generation",
        "3d gaussian splatting": "3D_Gaussian_Splatting",
    },
    "zotero_tag_to_rf_category": {
        "motion-gen": "Motion_Generation_Text_Speech_Music_Driven",
        "motion generation": "Motion_Generation_Text_Speech_Music_Driven",
        "video-diffusion": "Video_Generation",
        "video generation": "Video_Generation",
    },
    "default_category": "Uncategorized",
}
ZH_ANNOTATION_HEADING = "## 附：Zotero 高亮与笔记"
EN_ANNOTATION_HEADING = "## Appendix: Zotero Highlights & Notes"
ZH_PDF_HEADING = "## 本地 PDF 引用"
EN_PDF_HEADING = "## Local PDF reference"


def set_repo_root(path: str | Path) -> None:
    global REPO_ROOT
    global PAPER_ANALYSIS_DIR
    global PAPER_PDFS_DIR
    global ANALYSIS_LOG_PATH
    global PROCESSING_DIR
    global MANIFEST_PATH
    global SYNC_STATE_PATH
    global CATEGORY_MAP_PATH
    global SYNC_ERRORS_PATH

    REPO_ROOT = Path(path).expanduser().resolve()
    PAPER_ANALYSIS_DIR = REPO_ROOT / "paperAnalysis"
    PAPER_PDFS_DIR = REPO_ROOT / "paperPDFs"
    ANALYSIS_LOG_PATH = PAPER_ANALYSIS_DIR / "analysis_log.csv"
    PROCESSING_DIR = PAPER_ANALYSIS_DIR / "processing" / "zotero"
    MANIFEST_PATH = PROCESSING_DIR / "manifest.jsonl"
    SYNC_STATE_PATH = PROCESSING_DIR / "sync_state.json"
    CATEGORY_MAP_PATH = PROCESSING_DIR / "category_map.json"
    SYNC_ERRORS_PATH = PROCESSING_DIR / "sync_errors.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def err(msg: str) -> None:
    print(f"[ERR] {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def json_load(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def normalize_text(text: str) -> str:
    s = (text or "").lower()
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_set(text: str) -> set[str]:
    return set(normalize_text(text).split())


def sanitize_venue(venue: str) -> str:
    s = (venue or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    return s or "unknown"


def sanitize_year(year: str) -> str:
    m = re.search(r"(19|20)\d{2}", str(year or ""))
    return m.group(0) if m else "UnknownYear"


def sanitize_title_for_filename(title: str, max_len: int = 140) -> str:
    s = (title or "").strip()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"[<>:\"/\\|?*\x00-\x1F,]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_. ")
    if not s:
        s = "Untitled"
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def extract_year(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        m = re.search(r"(19|20)\d{2}", str(value))
        if m:
            return m.group(0)
    return "UnknownYear"


def strip_embedded_year(venue: str, year: str) -> str:
    if not venue or not year or year == "UnknownYear":
        return venue or "unknown"
    s = re.sub(rf"[\s_\-:,/]*{re.escape(year)}\b", "", venue).strip(" _-:,/")
    return s or venue


def html_to_text(fragment: str) -> str:
    if not fragment:
        return ""
    s = fragment.replace("\r", "")
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</(p|div|blockquote|h[1-6]|tr)>", "\n", s)
    s = re.sub(r"(?i)<li[^>]*>", "- ", s)
    s = re.sub(r"(?i)</li>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = s.replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in s.splitlines()]
    cleaned = "\n".join(ln for ln in lines if ln)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def ensure_processing_dir() -> None:
    PROCESSING_DIR.mkdir(parents=True, exist_ok=True)


def ensure_analysis_log() -> None:
    if ANALYSIS_LOG_PATH.exists():
        return
    ANALYSIS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ANALYSIS_LOG_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANALYSIS_LOG_HEADER)
        writer.writeheader()


def load_category_map(path: Path) -> Dict[str, object]:
    if not path.exists():
        return DEFAULT_CATEGORY_MAP
    loaded = json_load(path, {})
    if not isinstance(loaded, dict):
        warn(f"Category map at {path} is not an object. Falling back to defaults.")
        return DEFAULT_CATEGORY_MAP
    merged = json.loads(json.dumps(DEFAULT_CATEGORY_MAP))
    for key in ("zotero_collection_to_rf_category", "zotero_tag_to_rf_category"):
        if isinstance(loaded.get(key), dict):
            merged[key].update(
                {normalize_text(str(k)): str(v) for k, v in loaded[key].items() if str(v).strip()}
            )
    if str(loaded.get("default_category") or "").strip():
        merged["default_category"] = str(loaded["default_category"]).strip()
    return merged


def list_existing_categories() -> List[str]:
    categories = set()
    for root in (PAPER_ANALYSIS_DIR, PAPER_PDFS_DIR):
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name != "processing":
                categories.add(child.name)
    return sorted(categories)


def best_existing_category(label: str, categories: Sequence[str]) -> Optional[str]:
    if not label:
        return None
    normalized = normalize_text(label)
    if not normalized:
        return None
    for category in categories:
        if normalize_text(category) == normalized:
            return category
    label_tokens = token_set(label)
    best: Optional[Tuple[float, str]] = None
    for category in categories:
        cat_tokens = token_set(category.replace("_", " "))
        if not cat_tokens:
            continue
        overlap = len(label_tokens & cat_tokens)
        if overlap == 0:
            continue
        score = overlap / max(1, min(len(label_tokens), len(cat_tokens)))
        if best is None or score > best[0]:
            best = (score, category)
    if best and best[0] >= 0.6:
        return best[1]
    return None


def choose_rf_category(
    category_hint: str,
    item_title: str,
    collection_names: Sequence[str],
    tag_names: Sequence[str],
    category_map: Dict[str, object],
    existing_categories: Sequence[str],
) -> str:
    if category_hint:
        return category_hint.strip()

    coll_map = category_map.get("zotero_collection_to_rf_category", {})
    tag_map = category_map.get("zotero_tag_to_rf_category", {})
    default_category = str(category_map.get("default_category") or "Uncategorized")

    for name in collection_names:
        mapped = coll_map.get(normalize_text(name))
        if mapped:
            return str(mapped)
        matched = best_existing_category(name, existing_categories)
        if matched:
            return matched

    for tag in tag_names:
        mapped = tag_map.get(normalize_text(tag))
        if mapped:
            return str(mapped)
        matched = best_existing_category(tag, existing_categories)
        if matched:
            return matched

    matched = best_existing_category(item_title, existing_categories)
    if matched:
        return matched

    return default_category


def compute_rf_paths(category: str, venue: str, year: str, title: str) -> Dict[str, str]:
    cat = category.strip() or "Uncategorized"
    clean_venue = sanitize_venue(venue)
    clean_year = sanitize_year(year)
    clean_title = sanitize_title_for_filename(title)

    venue_year = f"{clean_venue}_{clean_year}"
    pdf_name = f"{clean_year}_{clean_title}.pdf"
    md_name = f"{clean_year}_{clean_title}.md"

    pdf_ref = f"paperPDFs/{cat}/{venue_year}/{pdf_name}"
    analysis_ref = f"paperAnalysis/{cat}/{venue_year}/{md_name}"
    return {
        "rf_category": cat,
        "rf_sort": cat,
        "rf_venue": venue.strip() or "unknown",
        "rf_year": clean_year,
        "rf_pdf_path": pdf_ref,
        "rf_analysis_path": analysis_ref,
        "rf_pdf_abs": str(REPO_ROOT / pdf_ref),
        "rf_analysis_abs": str(REPO_ROOT / analysis_ref),
    }


def extract_citekey(extra: str) -> str:
    if not extra:
        return ""
    patterns = (
        r"(?im)^\s*(?:citation key|bbt citation key)\s*:\s*(.+?)\s*$",
        r"(?im)^\s*citekey\s*:\s*(.+?)\s*$",
    )
    for pattern in patterns:
        m = re.search(pattern, extra)
        if m:
            return m.group(1).strip()
    return ""


def extract_arxiv_id(*values: object) -> str:
    patterns = (
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?",
        r"\barxiv[:\s]+([0-9]{4}\.[0-9]{4,5})(?:v\d+)?",
    )
    for value in values:
        text = str(value or "")
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return m.group(1)
    return ""


def extract_doi(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    return text.strip()


def choose_paper_link(doi: str, url: str, extra: str) -> str:
    doi = extract_doi(doi)
    if doi:
        return f"https://doi.org/{doi}"
    arxiv_id = extract_arxiv_id(url, extra)
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return str(url or "").strip()


def looks_like_pdf_url(url: str) -> bool:
    s = str(url or "").strip().lower()
    if not s.startswith(("http://", "https://")):
        return False
    return s.endswith(".pdf") or "/pdf/" in s or "openaccess.thecvf.com/content" in s


def creator_name(creator: Dict[str, object]) -> str:
    first = str(creator.get("firstName") or "").strip()
    last = str(creator.get("lastName") or "").strip()
    name = str(creator.get("name") or "").strip()
    if name:
        return name
    if first and last:
        return f"{last}, {first}"
    return last or first


def infer_venue(data: Dict[str, object], year: str) -> str:
    candidates = [
        data.get("conferenceName"),
        data.get("proceedingsTitle"),
        data.get("publicationTitle"),
        data.get("seriesTitle"),
        data.get("repository"),
        data.get("archive"),
        data.get("libraryCatalog"),
    ]
    url = str(data.get("url") or "")
    archive = str(data.get("archive") or "")
    library_catalog = str(data.get("libraryCatalog") or "")
    if any("arxiv" in normalize_text(v) for v in (url, archive, library_catalog)):
        return "arXiv"
    for candidate in candidates:
        s = str(candidate or "").strip()
        if s:
            return strip_embedded_year(s, year)
    if str(data.get("itemType") or "") == "preprint":
        return "Preprint"
    return "unknown"


def format_log_venue(venue: str, year: str) -> str:
    venue = str(venue or "").strip()
    year = sanitize_year(year)
    if not venue:
        return year
    if year != "UnknownYear" and re.search(rf"\b{re.escape(year)}\b", venue):
        return venue
    if year == "UnknownYear":
        return venue
    return f"{venue} {year}"


def zotero_scope(library_type: str, library_id: str) -> str:
    if library_type == "group":
        return f"groups/{library_id}"
    return "library"


def build_open_pdf_uri(
    library_type: str,
    library_id: str,
    attachment_key: str,
    annotation_key: str = "",
    page_label: str = "",
) -> str:
    if not attachment_key:
        return ""
    scope = zotero_scope(library_type, library_id)
    uri = f"zotero://open-pdf/{scope}/items/{attachment_key}"
    query = []
    if page_label:
        query.append(f"page={page_label}")
    if annotation_key:
        query.append(f"annotation={annotation_key}")
    if query:
        uri = f"{uri}?{'&'.join(query)}"
    return uri


def build_select_uri(library_type: str, library_id: str, item_key: str) -> str:
    if not item_key:
        return ""
    return f"zotero://select/{zotero_scope(library_type, library_id)}/items/{item_key}"


def load_analysis_log_rows() -> List[Dict[str, str]]:
    ensure_analysis_log()
    with ANALYSIS_LOG_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({key: str(row.get(key) or "") for key in ANALYSIS_LOG_HEADER})
        return rows


def write_analysis_log_rows(rows: Sequence[Dict[str, str]]) -> None:
    ensure_analysis_log()
    with ANALYSIS_LOG_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANALYSIS_LOG_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: str(row.get(key) or "") for key in ANALYSIS_LOG_HEADER})


def load_manifest_records() -> List[Dict[str, object]]:
    if not MANIFEST_PATH.exists():
        return []
    records = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL in {MANIFEST_PATH}:{line_no}: {exc}") from exc
    return records


def write_manifest_records(records: Sequence[Dict[str, object]]) -> None:
    ensure_processing_dir()
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def write_error_records(records: Sequence[Dict[str, object]]) -> None:
    ensure_processing_dir()
    with SYNC_ERRORS_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def make_sync_target_id(args: argparse.Namespace) -> str:
    tags = ",".join(sorted(args.tag or [])) or "*"
    collection = args.collection or "*"
    if args.local_api:
        endpoint = "local"
    elif args.base_url:
        endpoint = args.base_url.rstrip("/")
    else:
        endpoint = "https://api.zotero.org"
    return f"{endpoint}|{args.library_type}|{args.library_id}|{collection}|{tags}"


def load_sync_state() -> Dict[str, object]:
    state = json_load(SYNC_STATE_PATH, {"targets": {}})
    if not isinstance(state, dict):
        return {"targets": {}}
    state.setdefault("targets", {})
    return state


def write_sync_state(state: Dict[str, object]) -> None:
    ensure_processing_dir()
    SYNC_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_raw_item(raw: Dict[str, object]) -> Dict[str, object]:
    data = raw.get("data")
    if isinstance(data, dict):
        merged = dict(data)
        merged["_raw_key"] = raw.get("key") or data.get("key") or ""
        merged["_raw_version"] = raw.get("version") or data.get("version") or 0
        return merged
    merged = dict(raw)
    merged["_raw_key"] = raw.get("key") or ""
    merged["_raw_version"] = raw.get("version") or 0
    return merged


def child_sort_key(raw: Dict[str, object]) -> Tuple[str, str]:
    data = parse_raw_item(raw)
    return (str(data.get("itemType") or ""), str(data.get("title") or data.get("filename") or ""))


def choose_pdf_attachment(children: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    pdfs = []
    for raw in children:
        data = parse_raw_item(raw)
        if str(data.get("itemType") or "") != "attachment":
            continue
        if str(data.get("contentType") or "").lower() != "application/pdf":
            continue
        pdfs.append(data)
    if not pdfs:
        return None
    link_mode_priority = {
        "imported_file": 0,
        "imported_url": 1,
        "linked_file": 2,
        "linked_url": 3,
    }
    pdfs.sort(key=lambda item: (link_mode_priority.get(str(item.get("linkMode") or ""), 99), str(item.get("filename") or "")))
    return pdfs[0]


def parse_annotation(
    raw: Dict[str, object],
    library_type: str,
    library_id: str,
    attachment_key: str,
) -> Optional[Dict[str, object]]:
    data = parse_raw_item(raw)
    if str(data.get("itemType") or "") != "annotation":
        return None
    key = str(data.get("_raw_key") or data.get("key") or "")
    text = html_to_text(str(data.get("annotationText") or ""))
    comment = html_to_text(str(data.get("annotationComment") or ""))
    if not text and not comment:
        return None
    page_label = str(data.get("pageLabel") or "").strip()
    return {
        "annotation_key": key,
        "annotation_type": str(data.get("annotationType") or "highlight"),
        "annotation_color": str(data.get("annotationColor") or "").strip(),
        "annotation_text": text,
        "annotation_comment": comment,
        "page_label": page_label,
        "sort_index": str(data.get("sortIndex") or "").strip(),
        "position": str(data.get("position") or "").strip(),
        "attachment_key": attachment_key,
        "open_pdf_uri": build_open_pdf_uri(
            library_type=library_type,
            library_id=library_id,
            attachment_key=attachment_key,
            annotation_key=key,
            page_label=page_label,
        ),
    }


def parse_child_note(raw: Dict[str, object], library_type: str, library_id: str) -> Optional[Dict[str, object]]:
    data = parse_raw_item(raw)
    if str(data.get("itemType") or "") != "note":
        return None
    body = html_to_text(str(data.get("note") or ""))
    if not body:
        return None
    key = str(data.get("_raw_key") or data.get("key") or "")
    return {
        "note_key": key,
        "title": str(data.get("title") or "").strip(),
        "body": body,
        "select_uri": build_select_uri(library_type, library_id, key),
    }


def parse_identifiers(item: Dict[str, object], paper_link: str) -> Dict[str, str]:
    extra = str(item.get("extra") or "")
    doi = extract_doi(item.get("DOI"))
    arxiv_id = extract_arxiv_id(item.get("url"), extra, paper_link)
    return {
        "doi": doi,
        "arxiv_id": arxiv_id,
    }


def index_tokens(record_title: str, year: str) -> str:
    return f"{sanitize_year(year)}::{normalize_text(record_title)}"


def manifest_match(record: Dict[str, object], candidate: Dict[str, object]) -> bool:
    if str(record.get("source_signature") or "") and str(record.get("source_signature")) == str(candidate.get("source_signature") or ""):
        return True
    if str(record.get("rf_pdf_path") or "") and str(record.get("rf_pdf_path")) == str(candidate.get("rf_pdf_path") or ""):
        return True
    if str(record.get("doi") or "") and str(record.get("doi")) == str(candidate.get("doi") or ""):
        return True
    if str(record.get("arxiv_id") or "") and str(record.get("arxiv_id")) == str(candidate.get("arxiv_id") or ""):
        return True
    title_key = index_tokens(str(record.get("title") or ""), str(record.get("year") or ""))
    candidate_key = index_tokens(str(candidate.get("title") or ""), str(candidate.get("year") or ""))
    return bool(title_key and title_key == candidate_key)


def log_row_match(row: Dict[str, str], candidate: Dict[str, object]) -> bool:
    if row.get("pdf_path") and row["pdf_path"] == candidate.get("rf_pdf_path"):
        return True
    row_link = row.get("paper_link") or ""
    if candidate.get("doi") and extract_doi(row_link) == candidate.get("doi"):
        return True
    if candidate.get("arxiv_id") and extract_arxiv_id(row_link) == candidate.get("arxiv_id"):
        return True
    row_title = row.get("paper_title") or ""
    if not row_title:
        return False
    return index_tokens(row_title, row.get("venue") or "") == index_tokens(
        str(candidate.get("title") or ""),
        str(candidate.get("year") or ""),
    )


def upsert_manifest(records: List[Dict[str, object]], candidate: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
    for idx, record in enumerate(records):
        if manifest_match(record, candidate):
            merged = dict(record)
            for key, value in candidate.items():
                if value not in ("", [], {}, None):
                    merged[key] = value
            records[idx] = merged
            return "updated", merged
    records.append(candidate)
    return "created", candidate


def upsert_log_row(rows: List[Dict[str, str]], candidate: Dict[str, object]) -> str:
    new_row = {
        "state": str(candidate.get("log_state") or "Downloaded"),
        "importance": "",
        "paper_title": str(candidate.get("title") or ""),
        "venue": str(candidate.get("log_venue") or ""),
        "project_link_or_github_link": "",
        "paper_link": str(candidate.get("paper_link") or ""),
        "sort": str(candidate.get("rf_sort") or ""),
        "pdf_path": str(candidate.get("rf_pdf_path") or ""),
    }
    for row in rows:
        if not log_row_match(row, candidate):
            continue
        changed = False
        for key, value in new_row.items():
            if key in {"state"}:
                if row.get(key) in {"", "Wait", "Downloaded"} and value:
                    if row.get(key) != value:
                        row[key] = value
                        changed = True
                continue
            if not row.get(key) and value:
                row[key] = value
                changed = True
        return "updated" if changed else "duplicate"
    rows.append(new_row)
    return "created"


def note_language(md_text: str) -> str:
    if "## Part I：" in md_text or "## Part II：" in md_text or "## Part III：" in md_text:
        return "zh"
    return "en"


def build_annotation_section(record: Dict[str, object], language: str) -> str:
    annotations = list(record.get("annotations") or [])
    notes = list(record.get("zotero_notes") or [])
    if not annotations and not notes:
        return ""

    lines: List[str] = []
    if language == "zh":
        lines.append(ZH_ANNOTATION_HEADING)
        lines.append("")
        lines.append("> [!note] 本段来自 Zotero 的高亮与批注同步，用于补充你自己的阅读痕迹，不替代上面的结构化分析。")
        if notes:
            lines.append("")
            lines.append("### Zotero 笔记")
            lines.append("")
            for note in notes:
                body = str(note.get("body") or "").strip()
                if not body:
                    continue
                title = str(note.get("title") or "").strip()
                label = f"{title}：" if title else ""
                lines.append(f"- {label}{body}")
                if note.get("select_uri"):
                    lines.append(f"  Zotero：{note['select_uri']}")
        if annotations:
            lines.append("")
            lines.append("### Zotero 高亮与批注")
            lines.append("")
            for ann in sorted(
                annotations,
                key=lambda item: (str(item.get("page_label") or ""), str(item.get("sort_index") or "")),
            ):
                page = str(ann.get("page_label") or "?")
                ann_type = str(ann.get("annotation_type") or "highlight")
                color = str(ann.get("annotation_color") or "")
                prefix = f"- p.{page} | {ann_type}"
                if color:
                    prefix += f" | {color}"
                lines.append(prefix)
                if ann.get("annotation_text"):
                    lines.append(f"  高亮：{ann['annotation_text']}")
                if ann.get("annotation_comment"):
                    lines.append(f"  批注：{ann['annotation_comment']}")
                if ann.get("open_pdf_uri"):
                    lines.append(f"  Zotero：{ann['open_pdf_uri']}")
    else:
        lines.append(EN_ANNOTATION_HEADING)
        lines.append("")
        lines.append("> [!note] This section is synced from Zotero highlights and notes to preserve your reading trace beside the structured analysis above.")
        if notes:
            lines.append("")
            lines.append("### Zotero Notes")
            lines.append("")
            for note in notes:
                body = str(note.get("body") or "").strip()
                if not body:
                    continue
                title = str(note.get("title") or "").strip()
                label = f"{title}: " if title else ""
                lines.append(f"- {label}{body}")
                if note.get("select_uri"):
                    lines.append(f"  Zotero: {note['select_uri']}")
        if annotations:
            lines.append("")
            lines.append("### Zotero Highlights & Comments")
            lines.append("")
            for ann in sorted(
                annotations,
                key=lambda item: (str(item.get("page_label") or ""), str(item.get("sort_index") or "")),
            ):
                page = str(ann.get("page_label") or "?")
                ann_type = str(ann.get("annotation_type") or "highlight")
                color = str(ann.get("annotation_color") or "")
                prefix = f"- p.{page} | {ann_type}"
                if color:
                    prefix += f" | {color}"
                lines.append(prefix)
                if ann.get("annotation_text"):
                    lines.append(f"  Highlight: {ann['annotation_text']}")
                if ann.get("annotation_comment"):
                    lines.append(f"  Comment: {ann['annotation_comment']}")
                if ann.get("open_pdf_uri"):
                    lines.append(f"  Zotero: {ann['open_pdf_uri']}")
    return "\n".join(lines).strip() + "\n"


def replace_or_insert_annotation_section(md_text: str, section_text: str) -> Tuple[str, bool]:
    if not section_text:
        return md_text, False

    existing_patterns = [
        re.compile(rf"(?ms)^{re.escape(ZH_ANNOTATION_HEADING)}\n.*?(?=^{re.escape(ZH_PDF_HEADING)}\n|^{re.escape(EN_PDF_HEADING)}\n|\Z)"),
        re.compile(rf"(?ms)^{re.escape(EN_ANNOTATION_HEADING)}\n.*?(?=^{re.escape(ZH_PDF_HEADING)}\n|^{re.escape(EN_PDF_HEADING)}\n|\Z)"),
    ]
    for pattern in existing_patterns:
        if pattern.search(md_text):
            new_text = pattern.sub(section_text + "\n", md_text, count=1)
            return new_text, new_text != md_text

    insert_before = None
    for heading in (ZH_PDF_HEADING, EN_PDF_HEADING):
        needle = f"\n{heading}\n"
        pos = md_text.find(needle)
        if pos != -1:
            insert_before = pos + 1
            break
    if insert_before is None:
        embed_pos = md_text.rfind("\n![[")
        insert_before = embed_pos + 1 if embed_pos != -1 else len(md_text)

    prefix = md_text[:insert_before].rstrip()
    suffix = md_text[insert_before:].lstrip("\n")
    new_text = f"{prefix}\n\n{section_text.strip()}\n\n{suffix}".rstrip() + "\n"
    return new_text, new_text != md_text


def append_annotations_to_notes(
    manifest_records: Sequence[Dict[str, object]],
    analysis_md: str = "",
    pdf_ref: str = "",
    dry_run: bool = False,
) -> Dict[str, int]:
    stats = {"written": 0, "skipped": 0, "missing_note": 0}
    for record in manifest_records:
        record_analysis_path = str(record.get("rf_analysis_path") or "")
        record_pdf_ref = str(record.get("rf_pdf_path") or "")
        if analysis_md and record_analysis_path != analysis_md:
            continue
        if pdf_ref and record_pdf_ref != pdf_ref:
            continue
        md_abs = REPO_ROOT / record_analysis_path if record_analysis_path else None
        if not md_abs or not md_abs.exists():
            stats["missing_note"] += 1
            continue
        text = md_abs.read_text(encoding="utf-8")
        language = note_language(text)
        section = build_annotation_section(record, language)
        if not section:
            stats["skipped"] += 1
            continue
        new_text, changed = replace_or_insert_annotation_section(text, section)
        if not changed:
            stats["skipped"] += 1
            continue
        if not dry_run:
            md_abs.write_text(new_text, encoding="utf-8")
        stats["written"] += 1
    return stats


class ZoteroAPIClient:
    def __init__(
        self,
        library_type: str,
        library_id: str,
        api_key: str = "",
        base_url: str = "https://api.zotero.org",
        timeout: int = 60,
    ) -> None:
        if not library_type:
            raise ValueError("library_type is required")
        if library_type not in {"user", "group"}:
            raise ValueError("library_type must be 'user' or 'group'")
        if not library_id:
            raise ValueError("library_id is required")
        self.library_type = library_type
        self.library_id = library_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "ResearchFlow Zotero Sync",
                "Accept": "application/json",
            }
        )
        if api_key:
            self.session.headers["Zotero-API-Key"] = api_key

    @property
    def prefix(self) -> str:
        return f"{self.base_url}/{self.library_type}s/{self.library_id}"

    def _get(self, url: str, params: Optional[Dict[str, object]] = None, stream: bool = False) -> requests.Response:
        try:
            resp = self.session.get(url, params=params or {}, timeout=self.timeout, stream=stream)
            resp.raise_for_status()
            return resp
        except requests.HTTPError as exc:
            raise SystemExit(f"Zotero request failed: {exc.response.status_code} {exc.response.url}") from exc
        except requests.RequestException as exc:
            raise SystemExit(f"Zotero request failed: {exc}") from exc

    def fetch_paginated(self, path: str, params: Optional[Dict[str, object]] = None) -> Tuple[List[Dict[str, object]], int]:
        all_items: List[Dict[str, object]] = []
        start = 0
        limit = 100
        max_version = 0
        while True:
            query = dict(params or {})
            query["format"] = "json"
            query["limit"] = limit
            query["start"] = start
            url = f"{self.prefix}/{path.lstrip('/')}"
            resp = self._get(url, params=query)
            try:
                batch = resp.json()
            except ValueError as exc:
                raise SystemExit(f"Failed to decode Zotero JSON from {resp.url}") from exc
            if not isinstance(batch, list):
                raise SystemExit(f"Unexpected Zotero response shape from {resp.url}: expected list")
            all_items.extend(batch)
            header_version = resp.headers.get("Last-Modified-Version")
            if header_version and header_version.isdigit():
                max_version = max(max_version, int(header_version))
            for item in batch:
                raw_version = parse_raw_item(item).get("_raw_version")
                if isinstance(raw_version, int):
                    max_version = max(max_version, raw_version)
                elif str(raw_version).isdigit():
                    max_version = max(max_version, int(str(raw_version)))
            if len(batch) < limit:
                break
            start += len(batch)
        return all_items, max_version

    def fetch_children(self, item_key: str) -> Tuple[List[Dict[str, object]], int]:
        return self.fetch_paginated(f"items/{item_key}/children", {})

    def fetch_collections(self) -> List[Dict[str, object]]:
        collections, _ = self.fetch_paginated("collections", {})
        return collections

    def download_attachment(self, attachment_key: str, dest: Path, dry_run: bool = False) -> str:
        url = f"{self.prefix}/items/{attachment_key}/file"
        if dry_run:
            return str(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._get(url, stream=True) as resp:
            with dest.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return str(dest)


def collection_key_to_name(raw_collections: Sequence[Dict[str, object]]) -> Dict[str, str]:
    by_key = {}
    data_by_key = {}
    for raw in raw_collections:
        data = parse_raw_item(raw)
        key = str(data.get("_raw_key") or data.get("key") or "")
        if key:
            data_by_key[key] = data
    def full_name(key: str) -> str:
        data = data_by_key.get(key, {})
        name = str(data.get("name") or "").strip()
        parent = str(data.get("parentCollection") or "").strip()
        if parent and parent in data_by_key:
            parent_name = full_name(parent)
            return f"{parent_name}/{name}" if parent_name else name
        return name
    for key in data_by_key:
        by_key[key] = full_name(key)
    return by_key


def resolve_collection_filter(raw_collections: Sequence[Dict[str, object]], wanted: str) -> Optional[str]:
    if not wanted:
        return None
    wanted_norm = normalize_text(wanted)
    by_key = collection_key_to_name(raw_collections)
    for key, name in by_key.items():
        if key == wanted or normalize_text(name) == wanted_norm or normalize_text(name.split("/")[-1]) == wanted_norm:
            return key
    raise SystemExit(f"Collection filter not found in Zotero library: {wanted}")


def item_type_allowed(item_type: str, allowed_types: Sequence[str]) -> bool:
    return item_type in set(allowed_types)


def build_candidate_record(
    item: Dict[str, object],
    child_items: Sequence[Dict[str, object]],
    attachment: Dict[str, object],
    annotation_children: Sequence[Dict[str, object]],
    collection_name_map: Dict[str, str],
    category_map: Dict[str, object],
    existing_categories: Sequence[str],
    category_hint: str,
    library_type: str,
    library_id: str,
    sync_version: int,
) -> Optional[Dict[str, object]]:
    title = str(item.get("title") or "").strip()
    if not title:
        return None
    year = extract_year(item.get("year"), item.get("date"), item.get("filingDate"), item.get("accessDate"))
    venue = infer_venue(item, year)
    collection_names = [collection_name_map.get(str(key), str(key)) for key in item.get("collections", []) or []]
    tag_names = [str(tag.get("tag") or "").strip() for tag in item.get("tags", []) or [] if str(tag.get("tag") or "").strip()]
    rf_category = choose_rf_category(
        category_hint=category_hint,
        item_title=title,
        collection_names=collection_names,
        tag_names=tag_names,
        category_map=category_map,
        existing_categories=existing_categories,
    )
    rf_paths = compute_rf_paths(rf_category, venue, year, title)
    paper_link = choose_paper_link(item.get("DOI"), item.get("url"), item.get("extra"))
    identifiers = parse_identifiers(item, paper_link)
    authors = [creator_name(c) for c in item.get("creators", []) or [] if creator_name(c)]
    top_level_notes = [n for n in (parse_child_note(raw, library_type, library_id) for raw in child_items) if n]

    annotations = []
    attachment_key = str(attachment.get("_raw_key") or attachment.get("key") or "")
    for raw in annotation_children:
        ann = parse_annotation(raw, library_type, library_id, attachment_key)
        if ann:
            annotations.append(ann)
    annotations = list({ann["annotation_key"]: ann for ann in annotations}.values())

    citekey = extract_citekey(str(item.get("extra") or ""))
    zotero_key = str(item.get("_raw_key") or item.get("key") or "")
    zotero_version = item.get("_raw_version") or item.get("version") or 0
    source_signature = f"{library_type}:{library_id}:{zotero_key}"

    candidate = {
        "source_signature": source_signature,
        "zotero_key": zotero_key,
        "zotero_library_type": library_type,
        "zotero_library_id": library_id,
        "zotero_version": int(zotero_version) if str(zotero_version).isdigit() else 0,
        "item_type": str(item.get("itemType") or ""),
        "title": title,
        "authors": authors,
        "year": sanitize_year(year),
        "venue": venue,
        "log_venue": format_log_venue(venue, year),
        "doi": identifiers["doi"],
        "arxiv_id": identifiers["arxiv_id"],
        "paper_link": paper_link,
        "url": str(item.get("url") or "").strip(),
        "abstract": str(item.get("abstractNote") or "").strip(),
        "citekey": citekey,
        "zotero_tags": tag_names,
        "zotero_collections": collection_names,
        "zotero_select_uri": build_select_uri(library_type, library_id, zotero_key),
        "pdf_attachment_key": attachment_key,
        "pdf_attachment_filename": str(attachment.get("filename") or "").strip(),
        "pdf_attachment_link_mode": str(attachment.get("linkMode") or "").strip(),
        "pdf_source_path": str(attachment.get("path") or attachment.get("localPath") or "").strip(),
        "pdf_attachment_url": str(attachment.get("url") or "").strip(),
        "annotations": annotations,
        "annotation_count": len(annotations),
        "zotero_notes": top_level_notes,
        "zotero_note_count": len(top_level_notes),
        "synced_at": now_iso(),
        "sync_version": sync_version,
        **rf_paths,
    }
    candidate["log_state"] = "checked" if (REPO_ROOT / candidate["rf_analysis_path"]).exists() else "Downloaded"
    return candidate


def download_pdf_from_url(session: requests.Session, url: str, dest: Path, timeout: int, dry_run: bool) -> Tuple[bool, str]:
    if not url:
        return False, "missing attachment url for fallback download"
    if dry_run:
        return True, str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        head = tmp.read_bytes()[:5]
        if head != b"%PDF-":
            tmp.unlink(missing_ok=True)
            return False, f"url fallback did not yield a PDF: {url}"
        tmp.replace(dest)
        return True, str(dest)
    except requests.RequestException as exc:
        tmp.unlink(missing_ok=True)
        return False, f"url fallback failed: {exc}"


def maybe_copy_pdf(
    client: ZoteroAPIClient,
    candidate: Dict[str, object],
    dry_run: bool,
) -> Tuple[bool, str]:
    dest = Path(str(candidate["rf_pdf_abs"]))
    if dest.exists():
        candidate["pdf_import_source"] = "existing"
        return True, str(dest)
    attachment_key = str(candidate.get("pdf_attachment_key") or "")
    if not attachment_key:
        return False, "missing pdf attachment key"
    try:
        client.download_attachment(attachment_key, dest, dry_run=dry_run)
        candidate["pdf_import_source"] = "zotero_file_api"
    except SystemExit as exc:
        first_error = str(exc)
        attachment_url = str(candidate.get("pdf_attachment_url") or "")
        if looks_like_pdf_url(attachment_url):
            ok, detail = download_pdf_from_url(client.session, attachment_url, dest, client.timeout, dry_run)
            if ok:
                candidate["pdf_import_source"] = "attachment_url_fallback"
                return True, detail
            return False, f"{first_error}; {detail}"
        return False, first_error
    return True, str(dest)


def sync_from_api(args: argparse.Namespace) -> int:
    ensure_processing_dir()
    ensure_analysis_log()
    category_map = load_category_map(Path(args.category_map).expanduser() if args.category_map else CATEGORY_MAP_PATH)
    existing_categories = list_existing_categories()
    manifest_records = load_manifest_records()
    log_rows = load_analysis_log_rows()
    sync_state = load_sync_state()
    sync_target = make_sync_target_id(args)
    prior_state = sync_state["targets"].get(sync_target, {})
    since_version = int(args.since_version or 0)
    if not since_version and not args.full:
        raw_last = prior_state.get("last_version")
        if isinstance(raw_last, int):
            since_version = raw_last
        elif str(raw_last).isdigit():
            since_version = int(str(raw_last))

    base_url = "http://localhost:23119/api" if args.local_api else (args.base_url or "https://api.zotero.org")
    library_id = str(args.library_id or ("0" if args.local_api and args.library_type == "user" else ""))
    client = ZoteroAPIClient(
        library_type=args.library_type,
        library_id=library_id,
        api_key=args.api_key or "",
        base_url=base_url,
        timeout=args.timeout,
    )

    raw_collections = client.fetch_collections()
    collection_filter_key = resolve_collection_filter(raw_collections, args.collection)
    collection_names = collection_key_to_name(raw_collections)

    item_path = f"collections/{collection_filter_key}/items/top" if collection_filter_key else "items/top"
    item_params: Dict[str, object] = {"includeTrashed": 0}
    if since_version:
        item_params["since"] = since_version
    if args.tag:
        item_params["tag"] = list(args.tag)

    raw_items, item_max_version = client.fetch_paginated(item_path, item_params)
    if args.max_items:
        raw_items = raw_items[: args.max_items]
    sync_version = 1 + max([0] + [int(r.get("sync_version") or 0) for r in manifest_records if str(r.get("sync_version") or "").isdigit()])

    stats = defaultdict(int)
    touched_records: List[Dict[str, object]] = []
    error_records: List[Dict[str, object]] = []
    last_version_seen = item_max_version

    for raw in raw_items:
        item = parse_raw_item(raw)
        item_key = str(item.get("_raw_key") or item.get("key") or "")
        item_title = str(item.get("title") or "").strip()
        item_type = str(item.get("itemType") or "")
        if not item_type_allowed(item_type, args.allowed_item_type):
            stats["skipped_nonpaper"] += 1
            continue

        try:
            child_items, child_max_version = client.fetch_children(item_key)
        except SystemExit as exc:
            error_records.append(
                {
                    "stage": "fetch_item_children",
                    "item_key": item_key,
                    "title": item_title,
                    "item_type": item_type,
                    "error": str(exc),
                    "recorded_at": now_iso(),
                }
            )
            stats["errors"] += 1
            continue
        last_version_seen = max(last_version_seen, child_max_version)
        child_items = sorted(child_items, key=child_sort_key)
        attachment = choose_pdf_attachment(child_items)
        if attachment is None:
            stats["skipped_missing_pdf"] += 1
            error_records.append(
                {
                    "stage": "select_attachment",
                    "item_key": item_key,
                    "title": item_title,
                    "item_type": item_type,
                    "error": "no downloadable PDF attachment found",
                    "recorded_at": now_iso(),
                }
            )
            stats["errors"] += 1
            continue
        attachment_key = str(attachment.get("_raw_key") or attachment.get("key") or "")
        try:
            annotation_children, annotation_max_version = client.fetch_children(attachment_key)
        except SystemExit as exc:
            error_records.append(
                {
                    "stage": "fetch_attachment_children",
                    "item_key": item_key,
                    "title": item_title,
                    "item_type": item_type,
                    "attachment_key": attachment_key,
                    "error": str(exc),
                    "recorded_at": now_iso(),
                }
            )
            stats["errors"] += 1
            continue
        last_version_seen = max(last_version_seen, annotation_max_version)

        candidate = build_candidate_record(
            item=item,
            child_items=child_items,
            attachment=attachment,
            annotation_children=annotation_children,
            collection_name_map=collection_names,
            category_map=category_map,
            existing_categories=existing_categories,
            category_hint=args.category_hint or "",
            library_type=args.library_type,
            library_id=library_id,
            sync_version=sync_version,
        )
        dest_path = Path(str(candidate["rf_pdf_abs"]))
        if dest_path.exists():
            existing_record = any(manifest_match(record, candidate) for record in manifest_records)
            existing_log = any(log_row_match(row, candidate) for row in log_rows)
            if not existing_record and not existing_log:
                warn(f"Target PDF path already exists without a known duplicate record, skipping: {candidate['rf_pdf_path']}")
                stats["skipped_collision"] += 1
                error_records.append(
                    {
                        "stage": "path_collision",
                        "item_key": item_key,
                        "title": item_title,
                        "item_type": item_type,
                        "attachment_key": attachment_key,
                        "rf_pdf_path": candidate["rf_pdf_path"],
                        "error": "target PDF path exists without matching manifest/log record",
                        "recorded_at": now_iso(),
                    }
                )
                stats["errors"] += 1
                continue

        ok, detail = maybe_copy_pdf(client, candidate, dry_run=args.dry_run)
        if not ok:
            warn(f"Failed to copy PDF for {candidate['title']}: {detail}")
            stats["skipped_missing_pdf"] += 1
            error_records.append(
                {
                    "stage": "download_attachment",
                    "item_key": item_key,
                    "title": item_title,
                    "item_type": item_type,
                    "attachment_key": attachment_key,
                    "rf_pdf_path": candidate["rf_pdf_path"],
                    "error": detail,
                    "recorded_at": now_iso(),
                }
            )
            stats["errors"] += 1
            continue
        candidate["rf_pdf_abs"] = detail

        manifest_status, record = upsert_manifest(manifest_records, candidate)
        log_status = upsert_log_row(log_rows, candidate)
        touched_records.append(record)
        stats[f"manifest_{manifest_status}"] += 1
        stats[f"log_{log_status}"] += 1
        stats["papers_processed"] += 1

    if not args.dry_run:
        write_manifest_records(manifest_records)
        write_error_records(error_records)
        write_analysis_log_rows(log_rows)
        sync_state["targets"][sync_target] = {
            "last_version": last_version_seen,
            "last_synced": now_iso(),
            "library_type": args.library_type,
            "library_id": library_id,
            "collection": args.collection or "",
            "tags": list(args.tag or []),
        }
        write_sync_state(sync_state)

    annotation_stats = {"written": 0, "skipped": 0, "missing_note": 0}
    if args.append_annotations_to_md:
        annotation_stats = append_annotations_to_notes(touched_records, dry_run=args.dry_run)

    info(f"sync target: {sync_target}")
    info(
        "processed={processed} nonpaper={nonpaper} missing_pdf={missing} collision={collision}".format(
            processed=stats["papers_processed"],
            nonpaper=stats["skipped_nonpaper"],
            missing=stats["skipped_missing_pdf"],
            collision=stats["skipped_collision"],
        )
    )
    info(
        "manifest created={created} updated={updated} | log created={log_created} updated={log_updated} duplicate={log_dup}".format(
            created=stats["manifest_created"],
            updated=stats["manifest_updated"],
            log_created=stats["log_created"],
            log_updated=stats["log_updated"],
            log_dup=stats["log_duplicate"],
        )
    )
    info(f"errors recorded={stats['errors']}")
    if args.append_annotations_to_md:
        info(
            "annotation sections written={written} skipped={skipped} missing_note={missing}".format(
                written=annotation_stats["written"],
                skipped=annotation_stats["skipped"],
                missing=annotation_stats["missing_note"],
            )
        )
    return 0


def append_annotations_command(args: argparse.Namespace) -> int:
    records = load_manifest_records()
    if not records:
        raise SystemExit(f"No manifest records found at {MANIFEST_PATH}")
    stats = append_annotations_to_notes(
        manifest_records=records,
        analysis_md=args.analysis_md or "",
        pdf_ref=args.pdf_ref or "",
        dry_run=args.dry_run,
    )
    info(
        "annotation sections written={written} skipped={skipped} missing_note={missing}".format(
            written=stats["written"],
            skipped=stats["skipped"],
            missing=stats["missing_note"],
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync Zotero papers into ResearchFlow and optionally append Zotero annotations into analysis notes."
    )
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_REPO_ROOT),
        help="Target RF vault root. Defaults to the repository that contains this script.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Sync papers from Zotero API into RF")
    sync_parser.add_argument("--library-type", choices=("user", "group"), required=True)
    sync_parser.add_argument("--library-id", help="Zotero library ID; for local Zotero user library use 0")
    sync_parser.add_argument("--api-key", default="", help="Read-only Zotero API key; omit for local API if unnecessary")
    sync_parser.add_argument("--local-api", action="store_true", help="Use local Zotero API at http://localhost:23119/api")
    sync_parser.add_argument("--base-url", default="", help="Override API base URL")
    sync_parser.add_argument("--collection", default="", help="Collection key or name filter")
    sync_parser.add_argument("--tag", action="append", default=[], help="Repeatable tag filter")
    sync_parser.add_argument(
        "--allowed-item-type",
        action="append",
        default=list(DEFAULT_ALLOWED_ITEM_TYPES),
        help="Repeatable Zotero item types to keep. Defaults to conferencePaper/journalArticle/preprint.",
    )
    sync_parser.add_argument("--category-hint", default="", help="Force one RF category for all synced items")
    sync_parser.add_argument("--category-map", default="", help="Path to category_map.json override")
    sync_parser.add_argument("--since-version", type=int, default=0, help="Override incremental sync version")
    sync_parser.add_argument("--full", action="store_true", help="Ignore sync_state.json and rescan matching items")
    sync_parser.add_argument("--append-annotations-to-md", action="store_true", help="Append Zotero notes/highlights into existing analysis notes")
    sync_parser.add_argument("--max-items", type=int, default=0, help="Limit fetched items for testing")
    sync_parser.add_argument("--dry-run", action="store_true")
    sync_parser.add_argument("--timeout", type=int, default=60)
    sync_parser.set_defaults(func=sync_from_api)

    ann_parser = subparsers.add_parser("append-annotations", help="Append Zotero annotations from manifest into analysis notes")
    ann_parser.add_argument("--analysis-md", default="", help="Restrict to one analysis path such as paperAnalysis/.../foo.md")
    ann_parser.add_argument("--pdf-ref", default="", help="Restrict to one pdf_ref such as paperPDFs/.../foo.pdf")
    ann_parser.add_argument("--dry-run", action="store_true")
    ann_parser.set_defaults(func=append_annotations_command)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    set_repo_root(args.repo_root)
    if args.command == "sync" and args.local_api and not args.library_id:
        args.library_id = "0" if args.library_type == "user" else args.library_id
    if args.command == "sync" and not args.library_id:
        raise SystemExit("--library-id is required")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
