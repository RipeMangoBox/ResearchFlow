#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import html
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


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


def _fetch_url(url: str, timeout_s: int, user_agent: str) -> tuple[int, str, bytes]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
            return status, content_type, data
    except urllib.error.HTTPError as e:
        return int(getattr(e, "code", 0) or 0), "", e.read() if hasattr(e, "read") else b""
    except Exception as e:
        raise RuntimeError(f"fetch failed: {url} ({e})") from e


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


def _matches_keywords(text: str, include: list[str], exclude: list[str]) -> bool:
    t = text.lower()
    def hit(k: str) -> bool:
        kk = k.lower().strip()
        if not kk:
            return False
        # For pure word-like keywords (letters/digits/_), require word-boundary match
        # to avoid substring false positives like "proMOTION".
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
ICLR_VIRTUAL_POSTER_RE = re.compile(r"^https?://iclr\.cc/virtual/(\d{4})/poster/(\d+)(/)?$", re.I)


def normalize_paper_link(url: str) -> str:
    url = url.strip()
    url = url.strip("()[]{}<>\"'.,;")
    if not url:
        return ""

    u = urllib.parse.urlsplit(url)
    url = urllib.parse.urlunsplit((u.scheme, u.netloc, u.path, u.query, ""))

    m = ARXIV_PDF_RE.match(url)
    if m:
        pid = m.group(1)
        return f"https://arxiv.org/abs/{pid}"
    m = ARXIV_ABS_RE.match(url)
    if m:
        pid = m.group(1)
        return f"https://arxiv.org/abs/{pid}"

    m = OPENREVIEW_FORUM_RE.match(url)
    if m:
        oid = m.group(2)
        return f"https://openreview.net/forum?id={oid}"

    return url


def is_likely_paper_link(url: str) -> bool:
    u = normalize_paper_link(url)
    if not u:
        return False
    if "arxiv.org/abs/" in u:
        return True
    if u.startswith("https://openreview.net/forum?id=") or u.startswith("http://openreview.net/forum?id="):
        return True
    if ICLR_VIRTUAL_POSTER_RE.match(u):
        return True
    if re.search(r"\.pdf(\?|$)", u, flags=re.I):
        return True
    return False


@dataclass(frozen=True)
class Candidate:
    title: str
    paper_link: str
    project_link: str
    evidence: str


def _html_to_candidates(html_text: str, base_url: str) -> list[Candidate]:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html_text, "html.parser")
        candidates: list[Candidate] = []

        for a in soup.find_all("a", href=True):
            href = urllib.parse.urljoin(base_url, a.get("href", "").strip())
            text = _normalize_space(a.get_text(" ", strip=True))
            if not href:
                continue
            if not is_likely_paper_link(href):
                continue

            paper_link = normalize_paper_link(href)
            title = text or ""

            project = ""
            parent = a.parent
            if parent is not None:
                for b in parent.find_all("a", href=True):
                    bhref = urllib.parse.urljoin(base_url, b.get("href", "").strip())
                    if not bhref:
                        continue
                    if bhref == href:
                        continue
                    bl = bhref.lower()
                    bt = _normalize_space(b.get_text(" ", strip=True)).lower()
                    if ("github.com" in bl) or ("github.io" in bl) or ("project" in bt) or ("code" in bt):
                        project = bhref.strip()
                        break

            evidence = f"{title} | {paper_link}"
            candidates.append(Candidate(title=title, paper_link=paper_link, project_link=project, evidence=evidence))

        return candidates
    except Exception:
        pass

    link_re = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.I | re.S)
    tag_re = re.compile(r"<[^>]+>")
    candidates: list[Candidate] = []
    for m in link_re.finditer(html_text):
        href = urllib.parse.urljoin(base_url, html.unescape(m.group(1).strip()))
        inner = m.group(2)
        text = _normalize_space(html.unescape(tag_re.sub(" ", inner)))
        if not is_likely_paper_link(href):
            continue
        paper_link = normalize_paper_link(href)
        candidates.append(Candidate(title=text, paper_link=paper_link, project_link="", evidence=f"{text} | {paper_link}"))
    return candidates


def _dedup_by_paper_link(items: Iterable[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for it in items:
        key = normalize_paper_link(it.paper_link)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(Candidate(title=it.title, paper_link=key, project_link=it.project_link, evidence=it.evidence))
    return out


def _choose_title(c: Candidate) -> str:
    t = _normalize_space(c.title)
    if not t or t.lower() in {"pdf", "arxiv", "paper", "abs", "link"}:
        t = ""
    return t


def _load_existing_paper_links(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    text = _read_text(out_path)
    links: set[str] = set()
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:
            continue
        paper_link = normalize_paper_link(parts[3])
        if paper_link:
            links.add(paper_link)
    return links


def _format_line(
    status: str,
    title: str,
    venue_time: str,
    paper_link: str,
    project_link: str,
    category: str,
) -> str:
    return " | ".join(
        [
            status.strip(),
            title.strip(),
            venue_time.strip(),
            paper_link.strip(),
            project_link.strip(),
            category.strip(),
        ]
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fetch web pages, store HTML locally, extract paper candidates, and generate a triage list."
    )
    ap.add_argument("--urls", nargs="+", required=True, help="One or more URLs to fetch (static HTML).")
    ap.add_argument("--include", default="", help="Include keywords (all must match). Separator: ';' or ','.")
    ap.add_argument("--exclude", default="", help="Exclude keywords (any match filters out). Separator: ';' or ','.")
    ap.add_argument("--venue-time", required=True, help='Venue/time label, e.g. "ICLR 2026".')
    ap.add_argument("--out", required=True, help='Output list path, e.g. "paperAnalysis/ICLR_2026.txt".')
    ap.add_argument("--append", action="store_true", help="Append to existing output file instead of overwriting.")
    ap.add_argument("--status", default="Wait", help='Default status value for new entries (default: "Wait").')
    ap.add_argument(
        "--sources-dir",
        default="paperSources",
        help='Where to store fetched HTML (default: "paperSources").',
    )
    ap.add_argument("--timeout", type=int, default=20, help="Fetch timeout seconds (default: 20).")
    ap.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (X11; Linux) PaperCollector/1.0",
        help="HTTP User-Agent header.",
    )
    ap.add_argument(
        "--max-per-url",
        type=int,
        default=500,
        help="Max candidates retained per URL before keyword filter (default: 500).",
    )

    ns = ap.parse_args(argv)

    include = _split_keywords(ns.include)
    exclude = _split_keywords(ns.exclude)

    out_path = Path(ns.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    if ns.append:
        existing = _load_existing_paper_links(out_path)
    else:
        existing = set()

    run_id = f"{_slug(ns.venue_time)}_{_now_stamp()}"
    sources_root = Path(ns.sources_dir)
    if not sources_root.is_absolute():
        sources_root = REPO_ROOT / sources_root
    sources_root = sources_root / run_id
    sources_root.mkdir(parents=True, exist_ok=True)

    all_candidates: list[Candidate] = []
    for url in ns.urls:
        status, ctype, data = _fetch_url(url, timeout_s=ns.timeout, user_agent=ns.user_agent)
        html_text = _decode_html(data, ctype)

        us = urllib.parse.urlsplit(url)
        url_slug = _slug(us.netloc + "_" + us.path)
        if len(url_slug) > 80:
            url_slug = url_slug[:80]
        fname = f"{url_slug}_{status}_{_sha1_short(data)}.html"
        _write_text(sources_root / fname, html_text)

        cands = _html_to_candidates(html_text, base_url=url)
        if ns.max_per_url and len(cands) > ns.max_per_url:
            cands = cands[: ns.max_per_url]
        all_candidates.extend(cands)

    deduped = _dedup_by_paper_link(all_candidates)

    lines: list[str] = []
    added = 0
    for c in deduped:
        paper_link = normalize_paper_link(c.paper_link)
        if not paper_link or paper_link in existing:
            continue

        title = _choose_title(c)
        evidence = (title or "") + " " + (paper_link or "") + " " + (c.project_link or "")
        if not _matches_keywords(evidence, include=include, exclude=exclude):
            continue

        line = _format_line(
            status=ns.status,
            title=title,
            venue_time=ns.venue_time,
            paper_link=paper_link,
            project_link=c.project_link or "",
            category="",
        )
        lines.append(line)
        existing.add(paper_link)
        added += 1

    if not ns.append:
        _ensure_parent(out_path)
        out_path.write_text("", encoding="utf-8")

    if lines:
        with out_path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line.rstrip() + "\n")

    sys.stderr.write(
        f"[paper-collector-online] run_id={run_id} fetched={len(ns.urls)} "
        f"candidates={len(deduped)} added={added} out={out_path}\n"
    )
    sys.stderr.write(f"[paper-collector-online] sources_saved_under={sources_root}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

