#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
CSV_COLUMNS = [
    "state",
    "importance",
    "paper_title",
    "venue",
    "project_link_or_github_link",
    "paper_link",
    "sort",
    "pdf_path",
]
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "researchflow" / "papers_collect_from_web.json"
DEFAULT_SOURCE_PREFERENCE = ["openreview", "arxiv", "semantic_scholar", "html"]
ARXIV_API_PAGE_SIZE = 100
OPENREVIEW_API_PAGE_SIZE = 50
SEMANTIC_SCHOLAR_API_PAGE_SIZE = 100
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}
SEMANTIC_SCHOLAR_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "abstract",
        "year",
        "url",
        "venue",
        "authors",
        "externalIds",
        "openAccessPdf",
    ]
)
PRESENTATION_TYPE_RE = re.compile(
    r"\b(oral|poster|spotlight|notable|award|accepted|workshop)\b",
    re.I,
)


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "run"


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _write_text(p: Path, text: str) -> None:
    _ensure_parent(p)
    p.write_text(text, encoding="utf-8")


def _write_jsonl(p: Path, rows: Iterable[dict[str, object]]) -> None:
    _ensure_parent(p)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha1_short(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:10]


def _decode_html(data: bytes, content_type: str) -> str:
    m = re.search(r"charset\s*=\s*([^\s;]+)", content_type, flags=re.I)
    if m:
        enc = m.group(1).strip("\"'").lower()
        try:
            return data.decode(enc, errors="replace")
        except Exception:
            pass
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc, errors="replace")
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _normalize_title(s: str) -> str:
    s = _normalize_space(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _split_keywords(s: str) -> list[str]:
    if not s:
        return []
    parts = re.split(r"[;,]\s*|\s+\|\s+|\n+", s.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p.lower())
    return out


def _split_names(s: str) -> list[str]:
    if not s:
        return []
    if isinstance(s, list):  # type: ignore[unreachable]
        return [_normalize_space(str(x)) for x in s if _normalize_space(str(x))]
    raw = str(s).strip()
    if not raw:
        return []
    parts = re.split(r"[;\n]|,\s*(?=[A-Z][a-z])", raw)
    out: list[str] = []
    for p in parts:
        p = _normalize_space(p)
        if p:
            out.append(p)
    return out


def _split_query_terms(raw: str) -> list[str]:
    return [term.strip() for term in re.findall(r'"[^"]+"|\S+', raw) if term.strip()]


def _parse_preferred_sources(raw: str) -> list[str]:
    if not raw.strip():
        return []
    out: list[str] = []
    for part in re.split(r"[\s,;]+", raw.strip()):
        val = part.strip().lower()
        if not val:
            continue
        if val not in {"openreview", "arxiv", "semantic_scholar", "html"}:
            raise ValueError(f"unknown source preference: {val}")
        if val not in out:
            out.append(val)
    return out


def _matches_keywords(text: str, include: list[str], exclude: list[str]) -> bool:
    t = text.lower()

    def hit(k: str) -> bool:
        kk = k.lower().strip()
        if not kk:
            return False
        if re.fullmatch(r"[a-z0-9_]+", kk):
            return re.search(rf"\b{re.escape(kk)}\b", t) is not None
        return kk in t

    if include and not all(hit(k) for k in include):
        return False
    if exclude and any(hit(k) for k in exclude):
        return False
    return True


ARXIV_ABS_RE = re.compile(r"^https?://arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})(v[0-9]+)?/?$", re.I)
ARXIV_PDF_RE = re.compile(r"^https?://arxiv\.org/pdf/([0-9]{4}\.[0-9]{4,5})(v[0-9]+)?(\.pdf)?$", re.I)
OPENREVIEW_FORUM_RE = re.compile(r"^https?://openreview\.net/forum\?(.*&)?id=([A-Za-z0-9_-]+)(&.*)?$", re.I)
OPENREVIEW_PDF_RE = re.compile(r"^https?://openreview\.net/pdf\?(.*&)?id=([A-Za-z0-9_-]+)(&.*)?$", re.I)
SEMANTIC_SCHOLAR_PAPER_ID_RE = re.compile(r"/paper/[^/]+/([0-9a-f]{40}|[A-Za-z0-9_-]{20,})/?$", re.I)


def _expand_path(raw: str) -> Path:
    return Path(raw).expanduser()


def _extract_arxiv_id(text: str) -> str:
    for pattern in (ARXIV_ABS_RE, ARXIV_PDF_RE):
        m = pattern.match(text.strip())
        if m:
            return m.group(1)
    m = re.search(r"\b([0-9]{4}\.[0-9]{4,5})(v[0-9]+)?\b", text)
    return m.group(1) if m else ""


def _extract_openreview_id(text: str) -> str:
    for pattern in (OPENREVIEW_FORUM_RE, OPENREVIEW_PDF_RE):
        m = pattern.match(text.strip())
        if m:
            return m.group(2)
    m = re.search(r"\bid=([A-Za-z0-9_-]+)\b", text)
    return m.group(1) if m else ""


def normalize_paper_link(url: str) -> str:
    url = url.strip()
    url = url.strip("()[]{}<>\"'.,;")
    if not url:
        return ""

    u = urllib.parse.urlsplit(url)
    url = urllib.parse.urlunsplit((u.scheme, u.netloc, u.path, u.query, ""))

    m = ARXIV_PDF_RE.match(url)
    if m:
        return f"https://arxiv.org/abs/{m.group(1)}"
    m = ARXIV_ABS_RE.match(url)
    if m:
        return f"https://arxiv.org/abs/{m.group(1)}"
    m = OPENREVIEW_FORUM_RE.match(url)
    if m:
        return f"https://openreview.net/forum?id={m.group(2)}"
    m = OPENREVIEW_PDF_RE.match(url)
    if m:
        return f"https://openreview.net/forum?id={m.group(2)}"
    return url


def is_likely_paper_link(url: str) -> bool:
    u = normalize_paper_link(url)
    if not u:
        return False
    if "arxiv.org/abs/" in u:
        return True
    if u.startswith("https://openreview.net/forum?id=") or u.startswith("http://openreview.net/forum?id="):
        return True
    if "semanticscholar.org/paper/" in u.lower():
        return True
    if re.search(r"\.pdf(\?|$)", u, flags=re.I):
        return True
    return False


def _config_template() -> dict[str, object]:
    return {
        "preferred_sources": DEFAULT_SOURCE_PREFERENCE[:],
        "api_keys": {},
        "updated_at": "",
    }


def _load_persistent_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return _config_template()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _config_template()
    if not isinstance(data, dict):
        return _config_template()
    config = _config_template()
    config.update(data)
    api_keys = config.get("api_keys")
    if not isinstance(api_keys, dict):
        config["api_keys"] = {}
    preferred = config.get("preferred_sources")
    if not isinstance(preferred, list):
        config["preferred_sources"] = DEFAULT_SOURCE_PREFERENCE[:]
    return config


def _save_persistent_config(
    path: Path,
    *,
    preferred_sources: Optional[list[str]],
    semantic_scholar_api_key: Optional[str],
) -> None:
    cfg = _load_persistent_config(path)
    if preferred_sources:
        cfg["preferred_sources"] = preferred_sources
    api_keys = cfg.get("api_keys")
    if not isinstance(api_keys, dict):
        api_keys = {}
    if semantic_scholar_api_key:
        api_keys["semantic_scholar"] = semantic_scholar_api_key
    cfg["api_keys"] = api_keys
    cfg["updated_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    _ensure_parent(path)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_semantic_scholar_api_key(ns: argparse.Namespace, cfg: dict[str, object]) -> str:
    if ns.semantic_scholar_api_key:
        return ns.semantic_scholar_api_key
    env_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if env_key:
        return env_key
    api_keys = cfg.get("api_keys")
    if isinstance(api_keys, dict):
        key = api_keys.get("semantic_scholar")
        if isinstance(key, str):
            return key.strip()
    return ""


def _resolve_preferred_sources(ns: argparse.Namespace, cfg: dict[str, object]) -> list[str]:
    if ns.preferred_sources:
        return _parse_preferred_sources(ns.preferred_sources)
    stored = cfg.get("preferred_sources")
    if isinstance(stored, list):
        out = []
        for item in stored:
            if isinstance(item, str) and item in {"openreview", "arxiv", "semantic_scholar", "html"}:
                if item not in out:
                    out.append(item)
        if out:
            return out
    return DEFAULT_SOURCE_PREFERENCE[:]


def _request_headers(user_agent: str, extra_headers: Optional[dict[str, str]] = None) -> dict[str, str]:
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _fetch_url(
    url: str,
    timeout_s: int,
    user_agent: str,
    *,
    extra_headers: Optional[dict[str, str]] = None,
    retries: int = 3,
) -> tuple[int, str, bytes]:
    req = urllib.request.Request(
        url,
        headers=_request_headers(user_agent, extra_headers=extra_headers),
        method="GET",
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                status = getattr(resp, "status", 200)
                content_type = resp.headers.get("Content-Type", "")
                return status, content_type, resp.read()
        except urllib.error.HTTPError as e:
            status = int(getattr(e, "code", 0) or 0)
            data = e.read() if hasattr(e, "read") else b""
            if status == 429 and attempt < retries:
                time.sleep(attempt)
                continue
            return status, e.headers.get("Content-Type", "") if getattr(e, "headers", None) else "", data
        except Exception as e:
            if attempt >= retries:
                raise RuntimeError(f"fetch failed: {url} ({e})") from e
            time.sleep(attempt)
    raise RuntimeError(f"fetch failed after retries: {url}")


def _fetch_json(
    url: str,
    timeout_s: int,
    user_agent: str,
    *,
    extra_headers: Optional[dict[str, str]] = None,
) -> tuple[dict[str, object], bytes]:
    status, ctype, data = _fetch_url(
        url,
        timeout_s=timeout_s,
        user_agent=user_agent,
        extra_headers=extra_headers,
    )
    if status >= 400:
        body = _decode_html(data, ctype)[:400]
        raise RuntimeError(f"request failed ({status}) for {url}: {body}")
    text = _decode_html(data, ctype)
    try:
        return json.loads(text), data
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid json from {url}: {e}") from e


def _save_source_payload(
    sources_root: Path,
    *,
    prefix: str,
    suffix: str,
    data: bytes,
    content_type: str = "",
) -> None:
    payload = data if isinstance(data, bytes) else str(data).encode("utf-8", errors="replace")
    ext = suffix
    fname = f"{_slug(prefix)}_{_sha1_short(payload)}{ext}"
    text = _decode_html(payload, content_type) if ext in {".html", ".xml", ".json"} else payload.decode("utf-8", errors="replace")
    _write_text(sources_root / fname, text)


@dataclass
class PaperRecord:
    state: str = "Wait"
    importance: str = ""
    paper_title: str = ""
    venue: str = ""
    project_link_or_github_link: str = ""
    paper_link: str = ""
    sort: str = ""
    pdf_path: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    source: str = ""
    source_id: str = ""
    arxiv_id: str = ""
    openreview_id: str = ""
    semantic_scholar_id: str = ""
    evidence: str = ""
    matched_live: bool = False
    provenance: list[str] = field(default_factory=list)
    source_url: str = ""
    preset_origin: str = ""

    def to_csv_row(self) -> list[str]:
        return [
            self.state.strip(),
            self.importance.strip(),
            self.paper_title.strip(),
            self.venue.strip(),
            self.project_link_or_github_link.strip(),
            self.paper_link.strip(),
            self.sort.strip(),
            self.pdf_path.strip(),
        ]

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["paper_link"] = normalize_paper_link(self.paper_link)
        return data


def _content_value(content: dict[str, object], key: str) -> object:
    raw = content.get(key)
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def _paper_record_keys(rec: PaperRecord) -> list[str]:
    keys: list[str] = []

    def add(key: str) -> None:
        if key and key not in keys:
            keys.append(key)

    if rec.openreview_id:
        add(f"openreview:{rec.openreview_id}")
    extracted_openreview = _extract_openreview_id(rec.paper_link)
    if extracted_openreview:
        add(f"openreview:{extracted_openreview}")
    if rec.arxiv_id:
        add(f"arxiv:{rec.arxiv_id}")
    extracted_arxiv = _extract_arxiv_id(rec.paper_link)
    if extracted_arxiv:
        add(f"arxiv:{extracted_arxiv}")
    if rec.paper_link:
        add(f"url:{normalize_paper_link(rec.paper_link)}")
    if rec.semantic_scholar_id:
        add(f"s2:{rec.semantic_scholar_id}")
    norm_title = _normalize_title(rec.paper_title)
    if norm_title:
        add(f"title:{norm_title}")
    return keys


def _record_evidence(rec: PaperRecord) -> str:
    return " ".join(
        [
            rec.paper_title,
            rec.abstract,
            rec.venue,
            rec.paper_link,
            rec.project_link_or_github_link,
            " ".join(rec.authors),
            rec.source,
        ]
    ).strip()


def _record_matches_filters(rec: PaperRecord, include: list[str], exclude: list[str]) -> bool:
    return _matches_keywords(_record_evidence(rec), include=include, exclude=exclude)


def _venue_has_presentation_type(venue: str) -> bool:
    return bool(PRESENTATION_TYPE_RE.search(venue or ""))


def _merge_venue(base_venue: str, incoming_venue: str) -> str:
    base_venue = base_venue.strip()
    incoming_venue = incoming_venue.strip()
    if not base_venue:
        return incoming_venue
    if not incoming_venue:
        return base_venue
    if _venue_has_presentation_type(incoming_venue) and not _venue_has_presentation_type(base_venue):
        return incoming_venue
    return base_venue


def _should_apply_venue_time(rec: PaperRecord) -> bool:
    if rec.venue:
        return False
    text = " ".join([rec.paper_link, rec.source_url]).lower()
    if rec.source in {"openreview", "arxiv", "semantic_scholar"}:
        return False
    if rec.openreview_id or rec.arxiv_id or rec.semantic_scholar_id:
        return False
    if any(host in text for host in ("openreview.net", "arxiv.org", "semanticscholar.org")):
        return False
    return True


def _merge_record(base: PaperRecord, incoming: PaperRecord, *, prefer_live_link: bool = False) -> None:
    if not base.paper_title and incoming.paper_title:
        base.paper_title = incoming.paper_title
    if (not base.paper_link or prefer_live_link) and incoming.paper_link:
        base.paper_link = incoming.paper_link
    if not base.project_link_or_github_link and incoming.project_link_or_github_link:
        base.project_link_or_github_link = incoming.project_link_or_github_link
    base.venue = _merge_venue(base.venue, incoming.venue)
    if not base.abstract and incoming.abstract:
        base.abstract = incoming.abstract
    if not base.authors and incoming.authors:
        base.authors = incoming.authors[:]
    if not base.arxiv_id and incoming.arxiv_id:
        base.arxiv_id = incoming.arxiv_id
    if not base.openreview_id and incoming.openreview_id:
        base.openreview_id = incoming.openreview_id
    if not base.semantic_scholar_id and incoming.semantic_scholar_id:
        base.semantic_scholar_id = incoming.semantic_scholar_id
    if incoming.source and incoming.source not in base.provenance:
        base.provenance.append(incoming.source)
    if not base.source and incoming.source:
        base.source = incoming.source
    if not base.source_id and incoming.source_id:
        base.source_id = incoming.source_id
    base.evidence = _record_evidence(base)


def _source_rank(source: str, preferred_sources: list[str]) -> int:
    try:
        return preferred_sources.index(source)
    except ValueError:
        return len(preferred_sources) + 1


def _candidate_quality_key(rec: PaperRecord, preferred_sources: list[str]) -> tuple[int, int, int, int]:
    return (
        _source_rank(rec.source, preferred_sources),
        -int(bool(rec.abstract)),
        -len(rec.abstract or ""),
        -int(bool(rec.project_link_or_github_link)),
    )


def _merge_duplicate_records(records: list[PaperRecord], preferred_sources: list[str]) -> list[PaperRecord]:
    by_link: dict[str, list[PaperRecord]] = {}
    no_link: list[PaperRecord] = []
    for rec in records:
        key = normalize_paper_link(rec.paper_link)
        if key:
            by_link.setdefault(key, []).append(rec)
        else:
            no_link.append(rec)

    collapsed: list[PaperRecord] = []
    for group in by_link.values():
        best = sorted(group, key=lambda r: _candidate_quality_key(r, preferred_sources))[0]
        merged = PaperRecord(**best.to_json_dict())
        merged.provenance = []
        for item in group:
            _merge_record(merged, item, prefer_live_link=False)
        collapsed.append(merged)
    collapsed.extend(no_link)

    by_title: dict[str, list[PaperRecord]] = {}
    without_title: list[PaperRecord] = []
    for rec in collapsed:
        key = _normalize_title(rec.paper_title)
        if key:
            by_title.setdefault(key, []).append(rec)
        else:
            without_title.append(rec)

    merged_records: list[PaperRecord] = []
    for group in by_title.values():
        if len(group) == 1:
            merged_records.append(group[0])
            continue
        best = sorted(group, key=lambda r: _candidate_quality_key(r, preferred_sources))[0]
        merged = PaperRecord(**best.to_json_dict())
        merged.provenance = []
        for item in group:
            _merge_record(merged, item, prefer_live_link=False)
        merged_records.append(merged)
    merged_records.extend(without_title)
    return merged_records


def _load_existing_output_keys(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    try:
        with out_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return set()
    keys: set[str] = set()
    for row in rows:
        rec = PaperRecord(
            state=str(row.get("state", "")),
            importance=str(row.get("importance", "")),
            paper_title=str(row.get("paper_title", "")),
            venue=str(row.get("venue", "")),
            project_link_or_github_link=str(row.get("project_link_or_github_link", "")),
            paper_link=str(row.get("paper_link", "")),
            sort=str(row.get("sort", "")),
            pdf_path=str(row.get("pdf_path", "")),
        )
        keys.update(_paper_record_keys(rec))
    return keys


def _read_preset_rows(preset_path: Path) -> list[dict[str, str]]:
    if preset_path.suffix.lower() == ".jsonl":
        rows: list[dict[str, str]] = []
        with preset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if isinstance(data, dict):
                    rows.append({str(k): "" if v is None else str(v) for k, v in data.items()})
        return rows

    raw = preset_path.read_text(encoding="utf-8", errors="replace")
    sample = "\n".join(raw.splitlines()[:5])
    delimiter = ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
        delimiter = dialect.delimiter
    except Exception:
        pass

    with preset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        return []

    normalized_header = [_normalize_title(x) for x in rows[0]]
    known_headers = {
        "state",
        "status",
        "importance",
        "paper title",
        "paper_title",
        "title",
        "venue",
        "paper link",
        "paper_link",
        "url",
        "abstract",
    }
    has_header = any(h in known_headers for h in normalized_header)
    if not has_header:
        if len(rows[0]) >= 8:
            header = CSV_COLUMNS[: len(rows[0])]
        elif len(rows[0]) == 6:
            header = ["state", "paper_title", "venue", "paper_link", "project_link_or_github_link", "sort"]
        else:
            raise RuntimeError(f"preset list has no recognized header: {preset_path}")
        data_rows = rows
    else:
        header = rows[0]
        data_rows = rows[1:]

    out: list[dict[str, str]] = []
    for row in data_rows:
        if not any(cell.strip() for cell in row):
            continue
        padded = row + [""] * max(0, len(header) - len(row))
        out.append({str(header[i]): padded[i].strip() for i in range(len(header))})
    return out


def _pick_alias(row: dict[str, str], aliases: list[str]) -> str:
    lowered = {_normalize_title(k).replace(" ", "_"): v for k, v in row.items()}
    for alias in aliases:
        key = alias.lower()
        if key in lowered and lowered[key].strip():
            return lowered[key].strip()
    return ""


def _load_preset_records(preset_path: Path) -> list[PaperRecord]:
    rows = _read_preset_rows(preset_path)
    out: list[PaperRecord] = []
    for row in rows:
        rec = PaperRecord(
            state=_pick_alias(row, ["state", "status"]) or "Wait",
            importance=_pick_alias(row, ["importance"]),
            paper_title=_pick_alias(row, ["paper_title", "title", "name"]),
            venue=_pick_alias(row, ["venue", "venue_time", "conference"]),
            project_link_or_github_link=_pick_alias(
                row,
                ["project_link_or_github_link", "project_link", "github_link", "code_link"],
            ),
            paper_link=_pick_alias(row, ["paper_link", "url", "abs_url", "pdf_url"]),
            sort=_pick_alias(row, ["sort", "category", "tag"]),
            pdf_path=_pick_alias(row, ["pdf_path", "local_pdf"]),
            abstract=_pick_alias(row, ["abstract", "summary"]),
            authors=_split_names(_pick_alias(row, ["authors", "author_list"])),
            arxiv_id=_pick_alias(row, ["arxiv_id"]) or _extract_arxiv_id(_pick_alias(row, ["paper_link", "url", "abs_url", "pdf_url"])),
            openreview_id=_pick_alias(row, ["openreview_id", "forum_id"]) or _extract_openreview_id(_pick_alias(row, ["paper_link", "url"])),
            semantic_scholar_id=_pick_alias(row, ["semantic_scholar_id", "s2_paper_id", "paperid"]),
            preset_origin=str(preset_path),
        )
        rec.paper_link = normalize_paper_link(rec.paper_link) if rec.paper_link else ""
        rec.evidence = _record_evidence(rec)
        out.append(rec)
    return out


def _html_to_records(html_text: str, base_url: str) -> list[PaperRecord]:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html_text, "html.parser")
        records: list[PaperRecord] = []

        for a in soup.find_all("a", href=True):
            href = urllib.parse.urljoin(base_url, a.get("href", "").strip())
            text = _normalize_space(a.get_text(" ", strip=True))
            if not href or not is_likely_paper_link(href):
                continue

            paper_link = normalize_paper_link(href)
            project = ""
            parent = a.parent
            if parent is not None:
                for b in parent.find_all("a", href=True):
                    bhref = urllib.parse.urljoin(base_url, b.get("href", "").strip())
                    if not bhref or bhref == href:
                        continue
                    bl = bhref.lower()
                    bt = _normalize_space(b.get_text(" ", strip=True)).lower()
                    if ("github.com" in bl) or ("github.io" in bl) or ("project" in bt) or ("code" in bt):
                        project = bhref.strip()
                        break

            rec = PaperRecord(
                paper_title=text,
                paper_link=paper_link,
                project_link_or_github_link=project,
                source="html",
                source_url=base_url,
            )
            rec.arxiv_id = _extract_arxiv_id(paper_link)
            rec.openreview_id = _extract_openreview_id(paper_link)
            rec.evidence = _record_evidence(rec)
            records.append(rec)
        return records
    except Exception:
        pass

    link_re = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.I | re.S)
    tag_re = re.compile(r"<[^>]+>")
    records: list[PaperRecord] = []
    for m in link_re.finditer(html_text):
        href = urllib.parse.urljoin(base_url, html.unescape(m.group(1).strip()))
        if not is_likely_paper_link(href):
            continue
        text = _normalize_space(html.unescape(tag_re.sub(" ", m.group(2))))
        rec = PaperRecord(
            paper_title=text,
            paper_link=normalize_paper_link(href),
            source="html",
            source_url=base_url,
        )
        rec.arxiv_id = _extract_arxiv_id(rec.paper_link)
        rec.openreview_id = _extract_openreview_id(rec.paper_link)
        rec.evidence = _record_evidence(rec)
        records.append(rec)
    return records


def _arxiv_html_search_records(html_text: str, base_url: str) -> list[PaperRecord]:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        return _html_to_records(html_text, base_url)

    soup = BeautifulSoup(html_text, "html.parser")
    out: list[PaperRecord] = []
    for item in soup.select("li.arxiv-result"):
        title_el = item.select_one("p.title")
        abstract_el = item.select_one("span.abstract-full")
        abs_href = ""
        for a in item.select("p.list-title a[href]"):
            href = urllib.parse.urljoin(base_url, a.get("href", "").strip())
            if "/abs/" in href:
                abs_href = href
                break
        if not abs_href:
            continue
        title = _normalize_space(title_el.get_text(" ", strip=True)) if title_el else ""
        abstract = _normalize_space(abstract_el.get_text(" ", strip=True)) if abstract_el else ""
        rec = PaperRecord(
            paper_title=title,
            paper_link=normalize_paper_link(abs_href),
            abstract=abstract,
            source="html",
            source_url=base_url,
        )
        rec.arxiv_id = _extract_arxiv_id(rec.paper_link)
        rec.provenance.append("arxiv_html_search")
        rec.evidence = _record_evidence(rec)
        out.append(rec)
    return out


def _detect_source(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    host = parsed.netloc.lower()
    if "openreview.net" in host:
        return "openreview"
    if "arxiv.org" in host:
        return "arxiv"
    if "semanticscholar.org" in host or "api.semanticscholar.org" in host:
        return "semantic_scholar"
    return "html"


def _arxiv_search_queries(raw_query: str, searchtype: str) -> list[str]:
    if searchtype == "title":
        return [f'ti:"{raw_query}"']
    if searchtype == "abstract":
        return [f'abs:"{raw_query}"']
    if searchtype == "author":
        return [f'au:"{raw_query}"']

    terms = _split_query_terms(raw_query)
    if len(terms) <= 1:
        return [f'all:"{raw_query}"']

    parts = []
    for term in terms:
        clean = term.strip().strip('"')
        if not clean:
            continue
        parts.append(f'all:"{clean}"' if " " in clean else f"all:{clean}")

    queries: list[str] = []
    if parts:
        queries.append(" AND ".join(parts))
    quoted = f'all:"{raw_query}"'
    if quoted not in queries:
        queries.append(quoted)
    return queries


def _collect_arxiv_records(url: str, ns: argparse.Namespace, sources_root: Path) -> list[PaperRecord]:
    parsed = urllib.parse.urlsplit(url)
    arxiv_id = _extract_arxiv_id(url)
    api_urls: list[tuple[str, str]] = []
    if arxiv_id:
        api_urls.append(("id_lookup", f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"))
    else:
        qs = urllib.parse.parse_qs(parsed.query)
        raw_query = (qs.get("query") or qs.get("search_query") or qs.get("q") or [""])[0].strip()
        if raw_query:
            searchtype = (qs.get("searchtype") or ["all"])[0].strip().lower()
            search_queries = _arxiv_search_queries(raw_query, searchtype)

            for query_idx, search_query in enumerate(search_queries):
                start = 0
                page_count = 0
                while start < ns.max_per_url and page_count < ns.max_source_pages:
                    max_results = min(ARXIV_API_PAGE_SIZE, ns.max_per_url - start)
                    api_url = (
                        "https://export.arxiv.org/api/query?"
                        + urllib.parse.urlencode(
                            {
                                "search_query": search_query,
                                "start": start,
                                "max_results": max_results,
                            }
                        )
                    )
                    api_urls.append((f"query_{query_idx}_page_{page_count}", api_url))
                    start += max_results
                    page_count += 1

    out: list[PaperRecord] = []
    seen_links: set[str] = set()

    def append_record(rec: PaperRecord) -> bool:
        key = normalize_paper_link(rec.paper_link) or _normalize_title(rec.paper_title)
        if not key or key in seen_links:
            return False
        seen_links.add(key)
        out.append(rec)
        return len(out) >= ns.max_per_url

    for idx, (label, api_url) in enumerate(api_urls):
        try:
            status, ctype, data = _fetch_url(api_url, timeout_s=ns.timeout, user_agent=ns.user_agent)
        except Exception:
            if parsed.path.startswith("/search"):
                break
            raise
        if status >= 400:
            _save_source_payload(
                sources_root,
                prefix=f"arxiv_api_{label}_{idx}_{status}",
                suffix=".xml",
                data=data,
                content_type=ctype,
            )
            if not parsed.path.startswith("/search"):
                body = _decode_html(data, ctype)[:300]
                raise RuntimeError(f"arXiv API request failed ({status}) for {api_url}: {body}")
            break
        _save_source_payload(
            sources_root,
            prefix=f"arxiv_api_{label}_{idx}",
            suffix=".xml",
            data=data,
            content_type=ctype,
        )
        root = ET.fromstring(data)
        entries = root.findall("atom:entry", ATOM_NS)
        if not entries:
            break
        for entry in entries:
            title = _normalize_space(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
            abstract = _normalize_space(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
            abs_url = _normalize_space(entry.findtext("atom:id", default="", namespaces=ATOM_NS))
            published = _normalize_space(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
            authors = [
                _normalize_space(node.text or "")
                for node in entry.findall("atom:author/atom:name", ATOM_NS)
                if _normalize_space(node.text or "")
            ]
            rec = PaperRecord(
                paper_title=title,
                paper_link=normalize_paper_link(abs_url),
                abstract=abstract,
                authors=authors,
                venue=f"arXiv {published[:4]}" if len(published) >= 4 else "arXiv",
                source="arxiv",
                source_id=_extract_arxiv_id(abs_url),
                arxiv_id=_extract_arxiv_id(abs_url),
                source_url=url,
            )
            rec.evidence = _record_evidence(rec)
            if append_record(rec):
                return out
        if len(entries) < ARXIV_API_PAGE_SIZE:
            break

    if parsed.path.startswith("/search") and len(out) < ns.max_per_url:
        page_status, page_ctype, page_data = _fetch_url(url, timeout_s=ns.timeout, user_agent=ns.user_agent)
        page_text = _decode_html(page_data, page_ctype)
        _save_source_payload(
            sources_root,
            prefix=f"arxiv_source_{page_status}",
            suffix=".html",
            data=page_data,
            content_type=page_ctype,
        )
        for rec in _arxiv_html_search_records(page_text, base_url=url):
            if append_record(rec):
                return out
    return out


def _collect_openreview_group_records(url: str, ns: argparse.Namespace, sources_root: Path) -> list[PaperRecord]:
    parsed = urllib.parse.urlsplit(url)
    group_id = (urllib.parse.parse_qs(parsed.query).get("id") or [""])[0].strip()
    if not group_id:
        return []

    group_api_url = "https://api2.openreview.net/groups?" + urllib.parse.urlencode({"id": group_id})
    group_data, group_raw = _fetch_json(group_api_url, timeout_s=ns.timeout, user_agent=ns.user_agent)
    _save_source_payload(sources_root, prefix="openreview_group_api", suffix=".json", data=group_raw, content_type="application/json")

    groups = group_data.get("groups")
    if not isinstance(groups, list) or not groups:
        return []
    group = groups[0] if isinstance(groups[0], dict) else {}
    content = group.get("content") if isinstance(group, dict) else {}
    if not isinstance(content, dict):
        return []

    submission_id = str(_content_value(content, "submission_id") or "").strip()
    accept_options_raw = _content_value(content, "accept_decision_options")
    decision_heading_map_raw = _content_value(content, "decision_heading_map")
    accept_options = accept_options_raw if isinstance(accept_options_raw, list) else []
    decision_heading_map = decision_heading_map_raw if isinstance(decision_heading_map_raw, dict) else {}
    accepted_venues = [
        venue
        for venue, heading in decision_heading_map.items()
        if isinstance(venue, str) and isinstance(heading, str) and heading in accept_options
    ]
    if not accepted_venues and accept_options:
        accepted_venues = [str(x) for x in accept_options if isinstance(x, str)]
    if not submission_id or not accepted_venues:
        return []

    out: list[PaperRecord] = []
    for venue_label in accepted_venues:
        offset = 0
        page = 0
        venue_collected = 0
        while venue_collected < ns.max_per_url and page < ns.max_source_pages:
            params = {
                "invitation": submission_id,
                "content.venue": venue_label,
                "limit": min(OPENREVIEW_API_PAGE_SIZE, ns.max_per_url - venue_collected),
                "offset": offset,
            }
            api_url = "https://api2.openreview.net/notes?" + urllib.parse.urlencode(params)
            note_data, note_raw = _fetch_json(api_url, timeout_s=ns.timeout, user_agent=ns.user_agent)
            _save_source_payload(
                sources_root,
                prefix=f"openreview_notes_{_slug(venue_label)}_{page}",
                suffix=".json",
                data=note_raw,
                content_type="application/json",
            )
            notes = note_data.get("notes")
            if not isinstance(notes, list) or not notes:
                break
            for note in notes:
                if not isinstance(note, dict):
                    continue
                note_content = note.get("content")
                if not isinstance(note_content, dict):
                    continue
                title = str(_content_value(note_content, "title") or "").strip()
                abstract = str(_content_value(note_content, "abstract") or "").strip()
                authors_val = _content_value(note_content, "authors")
                authors = [str(x).strip() for x in authors_val if str(x).strip()] if isinstance(authors_val, list) else []
                forum_id = str(note.get("forum") or note.get("id") or "").strip()
                rec = PaperRecord(
                    paper_title=title,
                    paper_link=f"https://openreview.net/forum?id={forum_id}" if forum_id else "",
                    abstract=abstract,
                    authors=authors,
                    venue=str(_content_value(note_content, "venue") or venue_label),
                    source="openreview",
                    source_id=forum_id,
                    openreview_id=forum_id,
                    source_url=url,
                )
                rec.evidence = _record_evidence(rec)
                out.append(rec)
                venue_collected += 1
            if len(notes) < params["limit"]:
                break
            offset += len(notes)
            page += 1
    return out


def _collect_openreview_forum_record(url: str, ns: argparse.Namespace, sources_root: Path) -> list[PaperRecord]:
    forum_id = _extract_openreview_id(url)
    if not forum_id:
        return []
    api_url = "https://api2.openreview.net/notes?" + urllib.parse.urlencode({"id": forum_id})
    note_data, note_raw = _fetch_json(api_url, timeout_s=ns.timeout, user_agent=ns.user_agent)
    _save_source_payload(sources_root, prefix=f"openreview_note_{forum_id}", suffix=".json", data=note_raw, content_type="application/json")
    notes = note_data.get("notes")
    if not isinstance(notes, list) or not notes:
        return []
    note = notes[0]
    if not isinstance(note, dict):
        return []
    content = note.get("content")
    if not isinstance(content, dict):
        return []
    authors_val = _content_value(content, "authors")
    authors = [str(x).strip() for x in authors_val if str(x).strip()] if isinstance(authors_val, list) else []
    rec = PaperRecord(
        paper_title=str(_content_value(content, "title") or "").strip(),
        paper_link=f"https://openreview.net/forum?id={forum_id}",
        abstract=str(_content_value(content, "abstract") or "").strip(),
        authors=authors,
        venue=str(_content_value(content, "venue") or ""),
        source="openreview",
        source_id=forum_id,
        openreview_id=forum_id,
        source_url=url,
    )
    rec.evidence = _record_evidence(rec)
    return [rec]


def _collect_openreview_records(url: str, ns: argparse.Namespace, sources_root: Path) -> list[PaperRecord]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.path.startswith("/group"):
        return _collect_openreview_group_records(url, ns, sources_root)
    if parsed.path.startswith("/forum") or parsed.path.startswith("/pdf"):
        return _collect_openreview_forum_record(url, ns, sources_root)
    return []


def _semantic_scholar_headers(api_key: str) -> dict[str, str]:
    if not api_key:
        return {}
    return {"x-api-key": api_key}


def _semantic_scholar_record_from_paper(
    paper: dict[str, object],
    *,
    source_url: str,
    source_name: str,
) -> PaperRecord:
    link, arxiv_id = _semantic_scholar_candidate_link(paper)
    authors = []
    raw_authors = paper.get("authors")
    if isinstance(raw_authors, list):
        for author in raw_authors:
            if isinstance(author, dict):
                name = str(author.get("name") or "").strip()
                if name:
                    authors.append(name)
    rec = PaperRecord(
        paper_title=str(paper.get("title") or "").strip(),
        paper_link=link,
        abstract=str(paper.get("abstract") or "").strip(),
        authors=authors,
        venue=str(paper.get("venue") or ""),
        source="semantic_scholar",
        source_id=str(paper.get("paperId") or "").strip(),
        semantic_scholar_id=str(paper.get("paperId") or "").strip(),
        arxiv_id=arxiv_id,
        source_url=source_url,
    )
    rec.provenance.append(source_name)
    rec.evidence = _record_evidence(rec)
    return rec


def _semantic_scholar_candidate_link(paper: dict[str, object]) -> tuple[str, str]:
    external_ids = paper.get("externalIds")
    if isinstance(external_ids, dict):
        arxiv_id = external_ids.get("ArXiv")
        if isinstance(arxiv_id, str) and arxiv_id.strip():
            return normalize_paper_link(f"https://arxiv.org/abs/{arxiv_id.strip()}"), arxiv_id.strip()
    open_access_pdf = paper.get("openAccessPdf")
    if isinstance(open_access_pdf, dict):
        pdf_url = open_access_pdf.get("url")
        if isinstance(pdf_url, str) and pdf_url.strip():
            normalized = normalize_paper_link(pdf_url.strip())
            return normalized, _extract_arxiv_id(normalized)
    url = paper.get("url")
    if isinstance(url, str) and url.strip():
        normalized = normalize_paper_link(url.strip())
        return normalized, _extract_arxiv_id(normalized)
    return "", ""


def _collect_semantic_scholar_records(
    url: str,
    ns: argparse.Namespace,
    sources_root: Path,
    *,
    api_key: str,
) -> list[PaperRecord]:
    parsed = urllib.parse.urlsplit(url)
    query = ""
    if "api.semanticscholar.org" in parsed.netloc.lower():
        qs = urllib.parse.parse_qs(parsed.query)
        query = (qs.get("query") or [""])[0].strip()
    else:
        qs = urllib.parse.parse_qs(parsed.query)
        query = (qs.get("q") or qs.get("query") or [""])[0].strip()
    if not query:
        return []

    out: list[PaperRecord] = []
    offset = 0
    page = 0
    while offset < ns.max_per_url and page < ns.max_source_pages:
        params = {
            "query": query,
            "offset": offset,
            "limit": min(SEMANTIC_SCHOLAR_API_PAGE_SIZE, ns.max_per_url - offset),
            "fields": SEMANTIC_SCHOLAR_FIELDS,
        }
        api_url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)
        data, raw = _fetch_json(
            api_url,
            timeout_s=ns.timeout,
            user_agent=ns.user_agent,
            extra_headers=_semantic_scholar_headers(api_key),
        )
        _save_source_payload(
            sources_root,
            prefix=f"semantic_scholar_page_{page}",
            suffix=".json",
            data=raw,
            content_type="application/json",
        )
        papers = data.get("data")
        if not isinstance(papers, list) or not papers:
            break
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            rec = _semantic_scholar_record_from_paper(
                paper,
                source_url=url,
                source_name="semantic_scholar_search",
            )
            out.append(rec)
            if len(out) >= ns.max_per_url:
                return out
        if len(papers) < params["limit"]:
            break
        offset += len(papers)
        page += 1
    return out


def _collect_html_records(url: str, ns: argparse.Namespace, sources_root: Path) -> list[PaperRecord]:
    status, ctype, data = _fetch_url(url, timeout_s=ns.timeout, user_agent=ns.user_agent)
    text = _decode_html(data, ctype)
    _save_source_payload(sources_root, prefix="html_source", suffix=".html", data=data, content_type=ctype)
    return _html_to_records(text, base_url=url)


def _collect_records_from_url(
    url: str,
    ns: argparse.Namespace,
    sources_root: Path,
    *,
    semantic_scholar_api_key: str,
) -> list[PaperRecord]:
    source = _detect_source(url)
    if source == "arxiv":
        return _collect_arxiv_records(url, ns, sources_root)
    if source == "openreview":
        return _collect_openreview_records(url, ns, sources_root)
    if source == "semantic_scholar":
        return _collect_semantic_scholar_records(
            url,
            ns,
            sources_root,
            api_key=semantic_scholar_api_key,
    )
    return _collect_html_records(url, ns, sources_root)


def _direct_lookup_urls_for_preset(preset_records: list[PaperRecord], source_kinds: set[str]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for rec in preset_records:
        if "openreview" in source_kinds and rec.openreview_id:
            url = f"https://openreview.net/forum?id={rec.openreview_id}"
            if url not in seen:
                seen.add(url)
                urls.append(url)
        if "arxiv" in source_kinds:
            arxiv_id = rec.arxiv_id or _extract_arxiv_id(rec.paper_link)
            if arxiv_id:
                url = f"https://arxiv.org/abs/{arxiv_id}"
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
    return urls


def _merge_live_records_into_preset(
    preset_records: list[PaperRecord],
    live_records: list[PaperRecord],
) -> tuple[list[PaperRecord], list[PaperRecord]]:
    key_to_index: dict[str, int] = {}
    for idx, rec in enumerate(preset_records):
        for key in _paper_record_keys(rec):
            key_to_index.setdefault(key, idx)

    source_only: list[PaperRecord] = []
    for live in live_records:
        match_idx: Optional[int] = None
        for key in _paper_record_keys(live):
            if key in key_to_index:
                match_idx = key_to_index[key]
                break
        if match_idx is None:
            source_only.append(live)
            continue
        target = preset_records[match_idx]
        _merge_record(target, live, prefer_live_link=False)
        target.matched_live = True
        for key in _paper_record_keys(target):
            key_to_index.setdefault(key, match_idx)
    return preset_records, source_only


def _write_output_csv(out_path: Path, rows: list[PaperRecord], *, append: bool) -> int:
    existing_keys = _load_existing_output_keys(out_path) if append else set()
    selected: list[PaperRecord] = []
    for row in rows:
        keys = _paper_record_keys(row)
        if existing_keys and keys and any(k in existing_keys for k in keys):
            continue
        selected.append(row)
        existing_keys.update(keys)

    should_write_header = not append or not out_path.exists() or out_path.stat().st_size == 0
    _ensure_parent(out_path)
    mode = "a" if append else "w"
    with out_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if should_write_header:
            writer.writerow(CSV_COLUMNS)
        for row in selected:
            writer.writerow(row.to_csv_row())
    return len(selected)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch web pages or supported APIs, merge against an optional preset paper list, "
            "and write an analysis_log.csv-compatible candidate file."
        )
    )
    ap.add_argument("--urls", nargs="*", default=[], help="One or more URLs to fetch.")
    ap.add_argument("--preset-list", default="", help="Optional preset paper list (CSV/TSV/pipe/JSONL).")
    ap.add_argument("--include", default="", help="Include keywords (all must match). Separator: ';' or ','.")
    ap.add_argument("--exclude", default="", help="Exclude keywords (any match filters out). Separator: ';' or ','.")
    ap.add_argument("--venue-time", default="", help='Venue/time label, e.g. "ICLR 2026". Used for fallback rows and run_id.')
    ap.add_argument("--out", default="paperAnalysis/analysis_log.csv", help='Output CSV path (default: "paperAnalysis/analysis_log.csv").')
    ap.add_argument("--append", action="store_true", help="Append to existing output file instead of overwriting.")
    ap.add_argument("--status", default="Wait", help='Default state for newly discovered source-only entries (default: "Wait").')
    ap.add_argument("--sources-dir", default="paperSources", help='Where to store fetched source payloads (default: "paperSources").')
    ap.add_argument("--timeout", type=int, default=20, help="Fetch timeout seconds (default: 20).")
    ap.add_argument("--user-agent", default="Mozilla/5.0 (X11; Linux) PaperCollector/2.0", help="HTTP User-Agent header.")
    ap.add_argument("--max-per-url", type=int, default=500, help="Maximum records retained per source URL (default: 500).")
    ap.add_argument("--max-source-pages", type=int, default=20, help="Maximum paginated API pages fetched per source URL (default: 20).")
    ap.add_argument("--api-config", default=str(DEFAULT_CONFIG_PATH), help=f"Persistent config path (default: {DEFAULT_CONFIG_PATH}).")
    ap.add_argument("--preferred-sources", default="", help="Comma-separated source preference order for duplicate resolution, e.g. openreview,arxiv,semantic_scholar,html.")
    ap.add_argument("--semantic-scholar-api-key", default="", help="Optional Semantic Scholar API key. If omitted, the script uses $SEMANTIC_SCHOLAR_API_KEY or the persistent config file; unauthenticated requests may be rate-limited.")
    ap.add_argument("--remember-config", action="store_true", help="Persist the current preferred sources and provided API keys into --api-config.")
    ap.add_argument("--configure-only", action="store_true", help="Update persistent config and exit without collecting papers.")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    ap = _build_arg_parser()
    ns = ap.parse_args(argv)

    config_path = _expand_path(ns.api_config)
    stored_cfg = _load_persistent_config(config_path)
    preferred_sources = _resolve_preferred_sources(ns, stored_cfg)
    semantic_scholar_api_key = _resolve_semantic_scholar_api_key(ns, stored_cfg)

    if ns.remember_config or ns.configure_only:
        _save_persistent_config(
            config_path,
            preferred_sources=preferred_sources,
            semantic_scholar_api_key=semantic_scholar_api_key or None,
        )
        if ns.configure_only:
            sys.stderr.write(f"[paper-collector-online] saved config to {config_path}\n")
            return 0

    if not ns.urls and not ns.preset_list:
        ap.error("at least one --urls entry or --preset-list is required unless --configure-only is used")

    include = _split_keywords(ns.include)
    exclude = _split_keywords(ns.exclude)

    run_label = ns.venue_time or (Path(ns.preset_list).stem if ns.preset_list else "paper_collection")
    run_id = f"{_slug(run_label)}_{_now_stamp()}"

    out_path = _expand_path(ns.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    sources_root = _expand_path(ns.sources_dir)
    if not sources_root.is_absolute():
        sources_root = REPO_ROOT / sources_root
    sources_root = sources_root / run_id
    sources_root.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    preset_records: list[PaperRecord] = []
    if ns.preset_list:
        preset_path = _expand_path(ns.preset_list)
        if not preset_path.is_absolute():
            preset_path = REPO_ROOT / preset_path
        preset_records = _load_preset_records(preset_path)
        _write_jsonl(sources_root / "preset_records.jsonl", (r.to_json_dict() for r in preset_records))

    live_records: list[PaperRecord] = []
    source_kinds = {_detect_source(url) for url in ns.urls}
    collection_urls = list(ns.urls)
    if preset_records:
        collection_urls.extend(_direct_lookup_urls_for_preset(preset_records, source_kinds))

    for url in collection_urls:
        try:
            collected = _collect_records_from_url(
                url,
                ns,
                sources_root,
                semantic_scholar_api_key=semantic_scholar_api_key,
            )
        except Exception as e:
            warnings.append(f"{url}: {e}")
            continue
        live_records.extend(collected)

    for rec in live_records:
        if ns.venue_time and _should_apply_venue_time(rec):
            rec.venue = ns.venue_time
        if not rec.state:
            rec.state = ns.status
        rec.evidence = _record_evidence(rec)

    deduped_live = _merge_duplicate_records(live_records, preferred_sources=preferred_sources)
    _write_jsonl(sources_root / "discovered_live_records.jsonl", (r.to_json_dict() for r in deduped_live))

    if preset_records:
        merged_preset, source_only_records = _merge_live_records_into_preset(preset_records, deduped_live)
        filtered_rows = [r for r in merged_preset if _record_matches_filters(r, include, exclude)]
        source_only_filtered = [r for r in source_only_records if _record_matches_filters(r, include, exclude)]
        for rec in source_only_filtered:
            if not rec.state:
                rec.state = ns.status
            if ns.venue_time and _should_apply_venue_time(rec):
                rec.venue = ns.venue_time
        output_rows = filtered_rows + source_only_filtered
        _write_jsonl(sources_root / "merged_records.jsonl", (r.to_json_dict() for r in output_rows))
    else:
        output_rows = [r for r in deduped_live if _record_matches_filters(r, include, exclude)]
        for rec in output_rows:
            rec.state = rec.state or ns.status
            if ns.venue_time and _should_apply_venue_time(rec):
                rec.venue = ns.venue_time
        _write_jsonl(sources_root / "merged_records.jsonl", (r.to_json_dict() for r in output_rows))

    added = _write_output_csv(out_path, output_rows, append=ns.append)
    sys.stderr.write(
        f"[paper-collector-online] run_id={run_id} urls={len(ns.urls)} "
        f"preset={len(preset_records)} live={len(deduped_live)} out_rows={len(output_rows)} "
        f"added={added} out={out_path}\n"
    )
    sys.stderr.write(f"[paper-collector-online] sources_saved_under={sources_root}\n")
    if warnings:
        for warning in warnings:
            sys.stderr.write(f"[paper-collector-online][warning] {warning}\n")
    if semantic_scholar_api_key:
        sys.stderr.write(f"[paper-collector-online] semantic_scholar_api=configured via {config_path if ns.remember_config else 'runtime input/env'}\n")
    elif any(_detect_source(url) == "semantic_scholar" for url in ns.urls):
        sys.stderr.write(
            "[paper-collector-online] semantic_scholar_api=not_configured; using unauthenticated "
            "requests, which may be rate-limited\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
