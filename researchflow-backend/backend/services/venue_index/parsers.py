from __future__ import annotations

import html
import re
from typing import Callable

from .models import AcceptedPaperRecord, ConferenceYearConfig, SourceConfig
from .normalize import (
    extract_arxiv_id,
    extract_openreview_forum_id,
    normalize_link,
    normalize_whitespace,
)


def parse_openreview_notes_json(payload: dict, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    records: list[AcceptedPaperRecord] = []
    for item in payload.get("notes", []):
        content = item.get("content", {})
        title = normalize_whitespace(str(content.get("title", "")))
        authors = content.get("authors", [])
        if isinstance(authors, list):
            authors_text = "; ".join(normalize_whitespace(str(x)) for x in authors if normalize_whitespace(str(x)))
        else:
            authors_text = normalize_whitespace(str(authors))
        pdf_path = normalize_whitespace(str(content.get("pdf", "")))
        paper_link = ""
        if pdf_path:
            if pdf_path.startswith("http://") or pdf_path.startswith("https://"):
                paper_link = pdf_path
            else:
                paper_link = f"https://openreview.net{pdf_path}"
        record = AcceptedPaperRecord(
            venue=conf.venue,
            year=conf.year,
            conf_year=conf.conf_year,
            title=title,
            authors=authors_text,
            source_type=source.kind,
            source_url=source.url,
            paper_link=normalize_link(paper_link),
            keywords_raw="; ".join(content.get("keywords", [])) if isinstance(content.get("keywords", []), list) else normalize_whitespace(str(content.get("keywords", ""))),
            abstract_raw=normalize_whitespace(str(content.get("abstract", ""))),
            openreview_forum_id=normalize_whitespace(str(item.get("id", ""))),
            has_pdf_camera_ready="true" if paper_link else "",
            decision="Accept",
        )
        records.append(record)
    return records


def parse_simple_html_paper_list(text: str, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    pattern = re.compile(r"<a\s+href=\"([^\"]+)\"[^>]*>(.*?)</a>\s*[—-]\s*([^<]+)", re.I | re.S)
    records: list[AcceptedPaperRecord] = []
    for href, raw_title, raw_authors in pattern.findall(text):
        title = normalize_whitespace(html.unescape(re.sub(r"<[^>]+>", " ", raw_title)))
        authors = normalize_whitespace(html.unescape(raw_authors))
        records.append(
            AcceptedPaperRecord(
                venue=conf.venue,
                year=conf.year,
                conf_year=conf.conf_year,
                title=title,
                authors=authors,
                source_type=source.kind,
                source_url=source.url,
                paper_link=normalize_link(href),
                has_pdf_camera_ready="true" if href else "",
                decision="Accept",
            )
        )
    return records


def parse_curated_markdown_links(text: str, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    records: list[AcceptedPaperRecord] = []
    for title, link in pattern.findall(text):
        normalized = normalize_link(link)
        arxiv_id = extract_arxiv_id(normalized)
        records.append(
            AcceptedPaperRecord(
                venue=conf.venue,
                year=conf.year,
                conf_year=conf.conf_year,
                title=normalize_whitespace(title),
                authors="",
                source_type=source.kind,
                source_url=source.url,
                paper_link=normalized,
                arxiv_id=arxiv_id,
                arxiv_url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                openreview_forum_id=extract_openreview_forum_id(normalized),
                has_pdf_camera_ready="",
                decision="Accept",
            )
        )
    return records


def parse_cvpr_openaccess_html(text: str, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    """Parse CVF OpenAccess HTML (CVPR / ICCV).

    Structure has TWO <dd> blocks per paper:
      <dt class="ptitle"><br><a href="...html">Title</a></dt>
      <dd> ... author links ... </dd>
      <dd>
        [<a href="...pdf">pdf</a>]
        [<a href="http://arxiv.org/abs/...">arXiv</a>]
        ...
      </dd>
    """

    OPENACCESS_BASE = "https://openaccess.thecvf.com"

    venue_upper = conf.venue.upper()
    venue_pattern = f"(?:CVPR|ICCV)"

    block_re = re.compile(
        r"<dt class=\"ptitle\"><br><a href=\"([^\"]+)\">([^<]+)</a></dt>"
        r"\s*<dd>(.*?)</dd>"
        r"\s*<dd>(.*?)</dd>",
        re.I | re.S,
    )
    records: list[AcceptedPaperRecord] = []
    for href, raw_title, dd_authors, dd_links in block_re.findall(text):
        title = normalize_whitespace(html.unescape(raw_title))

        author_names: list[str] = []
        for name in re.findall(r">([^<]+)</a>", dd_authors):
            clean = normalize_whitespace(html.unescape(name))
            if not clean:
                continue
            lower = clean.lower()
            if lower in {"pdf", "supp", "bibtex", "arxiv"}:
                continue
            author_names.append(clean)
        authors = "; ".join(author_names)

        pdf_match = re.search(
            rf"href=\"(/content/{venue_pattern}\d{{4}}/papers/[^\"]+\.pdf)\"",
            dd_links,
        )
        pdf_url = ""
        if pdf_match:
            pdf_url = normalize_link(pdf_match.group(1), OPENACCESS_BASE)

        html_url = normalize_link(href, OPENACCESS_BASE)

        arxiv_match = re.search(r"href=\"(https?://arxiv\.org/abs/[^\"]+)\"", dd_links)
        arxiv_url = normalize_link(arxiv_match.group(1)) if arxiv_match else ""
        arxiv_id = extract_arxiv_id(arxiv_url)

        records.append(
            AcceptedPaperRecord(
                venue=conf.venue,
                year=conf.year,
                conf_year=conf.conf_year,
                title=title,
                authors=authors,
                source_type=source.kind,
                source_url=source.url,
                paper_link=pdf_url or html_url,
                arxiv_id=arxiv_id,
                arxiv_url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                keywords_raw="",
                abstract_raw="",
                openreview_forum_id="",
                has_pdf_camera_ready="true" if pdf_url else "",
                decision="Accept",
            )
        )
    return records


def parse_iclr_virtual_html(text: str, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    """Parse ICLR 2025 virtual conference papers list.

    The page contains blocks of:
      <li><a href="/virtual/2025/poster/27719">Point-SAM: ...</a></li>
    """

    pattern = re.compile(r"<li><a href=\"(/virtual/2025/poster/[^\"]+)\">([^<]+)</a></li>")
    records: list[AcceptedPaperRecord] = []
    for href, raw_title in pattern.findall(text):
        title = normalize_whitespace(html.unescape(raw_title))
        records.append(
            AcceptedPaperRecord(
                venue=conf.venue,
                year=conf.year,
                conf_year=conf.conf_year,
                title=title,
                authors="",
                source_type=source.kind,
                source_url=source.url,
                paper_link=normalize_link("https://iclr.cc" + href),
                keywords_raw="",
                abstract_raw="",
                openreview_forum_id="",
                has_pdf_camera_ready="",
                decision="Accept",
            )
        )
    return records


def _conference_base_url(venue: str) -> str:
    mapping = {
        "ICLR": "https://iclr.cc",
        "NEURIPS": "https://neurips.cc",
        "ICML": "https://icml.cc",
        "CVPR": "https://cvpr.thecvf.com",
        "ECCV": "https://eccv.ecva.net",
        "ICCV": "https://iccv.thecvf.com",
    }
    return mapping.get(venue.upper(), "")


def _normalize_decision(raw: str) -> str:
    """Normalize decision strings to canonical form.

    Keeps the semantic content but normalizes casing/whitespace.
    Examples:
      "Accept (poster)"          -> "Accept (Poster)"
      "Accept (oral)"            -> "Accept (Oral)"
      "Accept (spotlight)"       -> "Accept (Spotlight)"
      "Accept (Highlight)"       -> "Accept (Highlight)"
      "Accept (spotlight poster)" -> "Accept (Spotlight Poster)"
      "Accept: Oral"             -> "Accept (Oral)"
      "Accept: Poster (Highlight)" -> "Accept (Highlight)"
      "poster"                   -> "Accept (Poster)"
      "oral"                     -> "Accept (Oral)"
      "highlight"                -> "Accept (Highlight)"
    """
    if not raw:
        return ""
    stripped = raw.strip()
    if not stripped:
        return ""
    import re as _re

    m = _re.match(r"(?i)^accept\s*\((.+)\)$", stripped)
    if m:
        inner = m.group(1).strip().title()
        return f"Accept ({inner})"

    m = _re.match(r"(?i)^accept\s*:\s*(.+)$", stripped)
    if m:
        rest = m.group(1).strip()
        paren_m = _re.match(r"(?i)^poster\s*\((.+)\)$", rest)
        if paren_m:
            inner = paren_m.group(1).strip().title()
            return f"Accept ({inner})"
        return f"Accept ({rest.title()})"

    low = stripped.lower()
    if low in ("oral", "poster", "spotlight", "highlight"):
        return f"Accept ({stripped.title()})"

    if low.startswith("accept"):
        return "Accept"
    return stripped


def _infer_acceptance_type(decision: str, event_type: str, eventtype: str) -> str:
    """Infer standardized acceptance type from decision/event_type/eventtype.

    Returns one of: Oral, Spotlight, Highlight, Poster, Accept, or "".
    Priority: decision > event_type > eventtype.
    """
    low = (decision or "").lower()
    if "oral" in low:
        return "Oral"
    if "spotlight" in low:
        return "Spotlight"
    if "highlight" in low:
        return "Highlight"
    if "poster" in low:
        return "Poster"

    et_low = (event_type or "").lower()
    if "oral" in et_low and "poster" not in et_low:
        return "Oral"
    if "spotlight" in et_low:
        return "Spotlight"
    if "highlight" in et_low:
        return "Highlight"

    evt_low = (eventtype or "").lower()
    if "oral" in evt_low:
        return "Oral"
    if "poster" in evt_low:
        return "Poster"

    if low.startswith("accept"):
        return "Accept"
    return ""


def _acl_decision_from_id(paper_id: str, venue: str) -> str:
    """Infer ACL/EMNLP decision from Anthology paper ID.

    Examples:
      2024.acl-long.123  -> Main Long
      2024.findings-acl.456 -> Findings
      2024.emnlp-main.789 -> Main
      2024.acl-srw.12 -> SRW
      2024.acl-demo.34 -> Demo
      2024.acl-industry.56 -> Industry
    """
    low = paper_id.lower()
    if "findings" in low:
        return "Findings"
    if "-long." in low or "-main." in low:
        return "Main"
    if "-short." in low:
        return "Main Short"
    if "-industry." in low:
        return "Industry"
    if "-demo." in low:
        return "Demo"
    if "-srw." in low:
        return "SRW"
    return "Accept"


def parse_virtual_conference_json(payload: dict, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    """Parse the unified JSON from EventHosts virtual conference platforms.

    Works for ICLR, NeurIPS, ICML, CVPR, ECCV, etc.
    """
    import json as _json

    if isinstance(payload, str):
        payload = _json.loads(payload)

    results = payload.get("results", [])
    base_url = _conference_base_url(conf.venue)
    records: list[AcceptedPaperRecord] = []

    for item in results:
        title = normalize_whitespace(str(item.get("name", "")))
        if not title:
            continue

        authors_list = item.get("authors", [])
        if isinstance(authors_list, list):
            author_names = [
                normalize_whitespace(str(a.get("fullname", "")))
                for a in authors_list
                if isinstance(a, dict) and normalize_whitespace(str(a.get("fullname", "")))
            ]
            authors_text = "; ".join(author_names)
        else:
            authors_text = normalize_whitespace(str(authors_list))

        vsite_url = item.get("virtualsite_url", "") or ""
        paper_link = ""
        if vsite_url and base_url:
            paper_link = normalize_link(base_url + vsite_url)

        pdf_url = item.get("paper_pdf_url", "") or ""
        if pdf_url:
            pdf_url = normalize_link(pdf_url)

        source_url_field = item.get("sourceurl", "") or ""
        openreview_id = extract_openreview_forum_id(source_url_field)

        abstract_raw = normalize_whitespace(str(item.get("abstract", "")))

        keywords_list = item.get("keywords", [])
        if isinstance(keywords_list, list):
            keywords_raw = "; ".join(str(k) for k in keywords_list if k)
        else:
            keywords_raw = normalize_whitespace(str(keywords_list))

        raw_decision = (item.get("decision") or "")
        decision = _normalize_decision(raw_decision)
        raw_eventtype = normalize_whitespace(str(item.get("eventtype", "") or ""))
        raw_event_type = normalize_whitespace(str(item.get("event_type", "") or ""))
        acceptance_type = _infer_acceptance_type(raw_decision, raw_event_type, raw_eventtype)

        paper_url_field = (item.get("paper_url", "") or "").strip()
        if not openreview_id and paper_url_field:
            openreview_id = extract_openreview_forum_id(paper_url_field)

        records.append(
            AcceptedPaperRecord(
                venue=conf.venue,
                year=conf.year,
                conf_year=conf.conf_year,
                title=title,
                authors=authors_text,
                source_type=source.kind,
                source_url=source.url,
                paper_link=pdf_url or paper_link,
                arxiv_id="",
                arxiv_url="",
                keywords_raw=keywords_raw,
                abstract_raw=abstract_raw,
                openreview_forum_id=openreview_id,
                has_pdf_camera_ready="true" if pdf_url else "",
                decision=decision,
                acceptance_type=acceptance_type,
                topic=normalize_whitespace(str(item.get("topic", "") or "")),
                code_url=(item.get("url", "") or "").strip(),
                paper_url=paper_url_field,
                virtual_id=str(item.get("id", "") or ""),
                virtual_uid=str(item.get("uid", "") or ""),
                virtualsite_url=vsite_url,
                sourceid=str(item.get("sourceid", "") or ""),
                sourceurl=source_url_field,
                session=normalize_whitespace(str(item.get("session", "") or "")),
                eventtype=raw_eventtype,
                event_type=raw_event_type,
                room_name=normalize_whitespace(str(item.get("room_name", "") or "")),
                starttime=(item.get("starttime", "") or "").strip(),
                endtime=(item.get("endtime", "") or "").strip(),
                poster_position=normalize_whitespace(str(item.get("poster_position", "") or "")),
            )
        )

    return records


def parse_openreview_api_v2(payload: dict, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    """Parse notes fetched via OpenReview API v2 (search endpoint).

    Each note has content.title.value, content.authors.value, etc.
    """
    records: list[AcceptedPaperRecord] = []
    for item in payload.get("notes", []):
        content = item.get("content", {})
        title = normalize_whitespace(str(content.get("title", {}).get("value", "")))
        if not title:
            continue

        authors_raw = content.get("authors", {}).get("value", [])
        if isinstance(authors_raw, list):
            authors_text = "; ".join(normalize_whitespace(str(x)) for x in authors_raw if normalize_whitespace(str(x)))
        else:
            authors_text = normalize_whitespace(str(authors_raw))

        keywords_raw = content.get("keywords", {}).get("value", [])
        if isinstance(keywords_raw, list):
            keywords_text = "; ".join(keywords_raw)
        else:
            keywords_text = normalize_whitespace(str(keywords_raw))

        abstract_text = normalize_whitespace(str(content.get("abstract", {}).get("value", "")))

        forum_id = str(item.get("forum", item.get("id", "")))
        paper_link = f"https://openreview.net/forum?id={forum_id}" if forum_id else ""

        pdf_path = content.get("pdf", {}).get("value", "")
        pdf_url = ""
        if pdf_path:
            if str(pdf_path).startswith("http"):
                pdf_url = str(pdf_path)
            else:
                pdf_url = f"https://openreview.net{pdf_path}"

        records.append(
            AcceptedPaperRecord(
                venue=conf.venue,
                year=conf.year,
                conf_year=conf.conf_year,
                title=title,
                authors=authors_text,
                source_type=source.kind,
                source_url=source.url,
                paper_link=normalize_link(paper_link),
                arxiv_id="",
                arxiv_url="",
                keywords_raw=keywords_text,
                abstract_raw=abstract_text,
                openreview_forum_id=forum_id,
                has_pdf_camera_ready="true" if pdf_url else "",
                decision="Accept",
            )
        )
    return records


def parse_iclr_proceedings_html(text: str, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    """Parse ICLR proceedings page (proceedings.iclr.cc/paper_files/paper/YYYY).

    HTML structure:
      <div class="paper-content">
        <a title="paper title" href="/paper_files/paper/YYYY/hash/...html">Title</a>
        <span class="paper-authors">Author1, Author2, ...</span>
      ...
    """
    PROCEEDINGS_BASE = "https://proceedings.iclr.cc"

    block_re = re.compile(
        r'<a\s+title="paper title"\s+href="(/paper_files/paper/\d{4}/hash/[^"]+)"[^>]*>([^<]+)</a>\s*'
        r'<span class="paper-authors">([^<]+)</span>',
        re.I | re.S,
    )
    records: list[AcceptedPaperRecord] = []
    for href, raw_title, raw_authors in block_re.findall(text):
        title = normalize_whitespace(html.unescape(raw_title))
        authors = normalize_whitespace(html.unescape(raw_authors))
        paper_link = normalize_link(href, PROCEEDINGS_BASE)
        records.append(
            AcceptedPaperRecord(
                venue=conf.venue,
                year=conf.year,
                conf_year=conf.conf_year,
                title=title,
                authors=authors,
                source_type=source.kind,
                source_url=source.url,
                paper_link=paper_link,
                arxiv_id="",
                arxiv_url="",
                keywords_raw="",
                abstract_raw="",
                openreview_forum_id="",
                has_pdf_camera_ready="true",
                decision="Accept",
            )
        )
    return records


def parse_aaai_ojs_html(text: str, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    """Parse AAAI OJS issue page HTML.

    Each paper lives in a block like:
      <div class="obj_article_summary">
        <h3 class="title">
          <a id="article-NNNNN" href="https://ojs.aaai.org/.../article/view/NNNNN">Title</a>
        </h3>
        <div class="authors">Author1, Author2, ...</div>
        ...
        <a class="obj_galley_link pdf" href="...pdf_url...">PDF</a>
      </div>

    DOI derivation: article/view/NNNNN -> 10.1609/aaai.vXXiYY.NNNNN
    The volume (vXX) and issue (iYY) are embedded in the page URL or can be
    inferred from the year. We extract the article ID and store it; the full
    DOI is resolved during enrichment via the article page or S2 lookup.
    """
    block_re = re.compile(
        r'<div class="obj_article_summary">(.*?)</div>\s*(?:</li>|<div class="obj_article_summary">)',
        re.S,
    )
    title_re = re.compile(
        r'<a\s+id="article-(\d+)"\s+href="([^"]+)"[^>]*>\s*(.*?)\s*</a>',
        re.S,
    )
    authors_re = re.compile(r'<div class="authors">\s*(.*?)\s*</div>', re.S)
    pdf_re = re.compile(
        r'<a class="obj_galley_link pdf"\s+href="([^"]+)"',
        re.S,
    )

    records: list[AcceptedPaperRecord] = []
    for block in block_re.findall(text):
        tm = title_re.search(block)
        if not tm:
            continue
        article_id = tm.group(1).strip()
        article_url = tm.group(2).strip()
        title = normalize_whitespace(html.unescape(re.sub(r"<[^>]+>", " ", tm.group(3))))
        if not title:
            continue

        am = authors_re.search(block)
        authors = normalize_whitespace(html.unescape(am.group(1))) if am else ""

        pm = pdf_re.search(block)
        pdf_url = normalize_link(pm.group(1)) if pm else ""

        records.append(
            AcceptedPaperRecord(
                venue=conf.venue,
                year=conf.year,
                conf_year=conf.conf_year,
                title=title,
                authors=authors,
                source_type=source.kind,
                source_url=source.url,
                paper_link=pdf_url or normalize_link(article_url),
                has_pdf_camera_ready="true" if pdf_url else "",
                decision="Accept",
            )
        )
    return records


def parse_kdd_html(text: str, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    """Parse KDD official research-track paper list pages.

    Both KDD 2024 and 2025 use WordPress tables with alternating rows:
      Row 1 (title):  <tr><td><strong>Title</strong>[<br>DOI: ...]</td></tr>
      Row 2 (authors): <tr><td>Author1 (Affil); Author2 (Affil)</td></tr>
    """
    row_re = re.compile(r"<tr><td>(.*?)</td></tr>", re.S)
    rows = row_re.findall(text)

    records: list[AcceptedPaperRecord] = []
    i = 0
    while i < len(rows):
        cell = rows[i]
        strong_m = re.search(r"<strong>(.*?)</strong>", cell, re.S)
        if strong_m:
            raw_title = strong_m.group(1)
            raw_title = re.sub(r"<strong>|</strong>", "", raw_title)
            title = normalize_whitespace(html.unescape(re.sub(r"<[^>]+>", " ", raw_title)))

            doi_m = re.search(r"DOI:\s*(https://doi\.org/[^\s<]+)", cell)
            doi_url = doi_m.group(1).strip() if doi_m else ""

            authors = ""
            if i + 1 < len(rows):
                next_cell = rows[i + 1]
                if not re.search(r"<strong>", next_cell):
                    authors = normalize_whitespace(html.unescape(re.sub(r"<[^>]+>", " ", next_cell)))
                    i += 1

            records.append(
                AcceptedPaperRecord(
                    venue=conf.venue,
                    year=conf.year,
                    conf_year=conf.conf_year,
                    title=title,
                    authors=authors,
                    source_type=source.kind,
                    source_url=source.url,
                    paper_link=normalize_link(doi_url) if doi_url else "",
                    has_pdf_camera_ready="",
                    decision="Accept",
                )
            )
        i += 1
    return records


def parse_neurips_virtual_html(text: str, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    """Parse NeurIPS virtual conference papers list.

    The page contains blocks of:
      <li><a href="/virtual/YYYY/poster/NNNNN">Title</a></li>
    """
    year = conf.year
    pattern = re.compile(rf'<li><a href="(/virtual/{year}/poster/[^"]+)">([^<]+)</a></li>')
    records: list[AcceptedPaperRecord] = []
    for href, raw_title in pattern.findall(text):
        title = normalize_whitespace(html.unescape(raw_title))
        records.append(
            AcceptedPaperRecord(
                venue=conf.venue,
                year=conf.year,
                conf_year=conf.conf_year,
                title=title,
                authors="",
                source_type=source.kind,
                source_url=source.url,
                paper_link=normalize_link("https://neurips.cc" + href),
                keywords_raw="",
                abstract_raw="",
                openreview_forum_id="",
                has_pdf_camera_ready="",
                decision="Accept",
            )
        )
    return records


def parse_acl_anthology_html(text: str, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    """Parse ACL Anthology event page (e.g. https://aclanthology.org/events/acl-2024/).

    Only keeps main conference volumes (long, short, findings, main, industry,
    demo, srw) and skips workshop papers.
    """
    year = str(conf.year)
    venue_lower = conf.venue.lower()

    allowed_prefixes = [
        f"{year}.{venue_lower}-long.",
        f"{year}.{venue_lower}-short.",
        f"{year}.findings-{venue_lower}.",
        f"{year}.{venue_lower}-main.",
        f"{year}.{venue_lower}-industry.",
        f"{year}.{venue_lower}-demo.",
        f"{year}.{venue_lower}-srw.",
    ]

    def _is_main_volume(paper_id: str) -> bool:
        pid = paper_id.lower().strip("/")
        return any(pid.startswith(p) for p in allowed_prefixes)

    strip_tags_re = re.compile(r"<[^>]+>")

    entry_re = re.compile(
        r'<span\s+class=["\']?d-block["\']?[^>]*>'
        r'\s*<strong>\s*<a\s[^>]*href=["\']?/([^"\'>/]+/)["\']?[^>]*>(.*?)</a>\s*</strong>'
        r'(.*?)</span>',
        re.I | re.S,
    )

    author_re = re.compile(r'<a\s+href=["\']?/people/[^"\']*["\']?[^>]*>([^<]+)</a>', re.I)

    abstract_re = re.compile(
        r'<div\s+class=["\']card-body\s+p-3\s+small["\'][^>]*>(.*?)</div>',
        re.I | re.S,
    )
    abstracts_iter = abstract_re.finditer(text)
    abstract_map: dict[int, str] = {}
    for m in abstracts_iter:
        abstract_map[m.start()] = normalize_whitespace(html.unescape(strip_tags_re.sub(" ", m.group(1))))

    abstract_positions = sorted(abstract_map.keys())

    def _find_abstract_after(pos: int) -> str:
        for ap in abstract_positions:
            if ap > pos:
                return abstract_map[ap]
        return ""

    records: list[AcceptedPaperRecord] = []
    for m in entry_re.finditer(text):
        paper_id = m.group(1).strip("/")
        if not _is_main_volume(paper_id):
            continue

        raw_title = m.group(2)
        title = normalize_whitespace(html.unescape(strip_tags_re.sub(" ", raw_title)))
        if not title:
            continue

        tail = m.group(3)
        author_names = [normalize_whitespace(html.unescape(a)) for a in author_re.findall(tail)]
        authors_text = "; ".join(n for n in author_names if n)

        abstract = _find_abstract_after(m.end())

        paper_link = f"https://aclanthology.org/{paper_id}/"

        records.append(
            AcceptedPaperRecord(
                venue=conf.venue,
                year=conf.year,
                conf_year=conf.conf_year,
                title=title,
                authors=authors_text,
                source_type=source.kind,
                source_url=source.url,
                paper_link=normalize_link(paper_link),
                arxiv_id="",
                arxiv_url="",
                keywords_raw="",
                abstract_raw=abstract,
                openreview_forum_id="",
                has_pdf_camera_ready="true",
                decision=_acl_decision_from_id(paper_id, conf.venue),
            )
        )
    return records


def parse_kesen_siggraph_html(raw_html: str, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    """Parse Ke-Sen Huang's SIGGRAPH / SIGGRAPH Asia paper list HTML.

    Structure: <dt><B>Title</B> ... (SIG/TOG) ...</dt><dd>authors</dd>
    Sections separated by <h2>.
    """
    records: list[AcceptedPaperRecord] = []
    idx = 0

    dt_pattern = re.compile(r'<dt>\s*<B>(.*?)</B>(.*?)</dt>\s*<dd>(.*?)</dd>', re.S | re.I)

    for m in dt_pattern.finditer(raw_html):
        title = normalize_whitespace(html.unescape(re.sub(r'<[^>]+>', '', m.group(1))))
        if not title:
            continue

        dt_body = m.group(2)
        dd_body = m.group(3)

        paper_type = ""
        type_m = re.search(r'\(<B>(SIG(?:/TOG)?|TOG|SIG)</B>\)', dt_body)
        if type_m:
            paper_type = type_m.group(1)

        acm_doi = ""
        doi_m = re.search(r'<a\s+href="(https?://doi\.org/[^"]+)"[^>]*>\s*<img[^>]*alt="ACM DOI"', dt_body, re.I)
        if doi_m:
            acm_doi = doi_m.group(1).strip()

        preprint_url = ""
        preprint_m = re.search(r'<a\s+href="([^"]+)"[^>]*>\s*<img[^>]*alt="Author Preprint"', dt_body, re.I)
        if preprint_m:
            preprint_url = preprint_m.group(1).strip()

        abstract_page = ""
        abs_m = re.search(r'<a\s+href="([^"]+)"[^>]*>\s*<img[^>]*alt="Paper Abstract"', dt_body, re.I)
        if abs_m:
            abstract_page = abs_m.group(1).strip()

        paper_link = acm_doi or abstract_page or preprint_url

        arxiv_id = ""
        arxiv_url = ""
        for url_candidate in [preprint_url, abstract_page, paper_link]:
            aid = extract_arxiv_id(url_candidate)
            if aid:
                arxiv_id = aid
                arxiv_url = f"https://arxiv.org/abs/{aid}"
                break

        author_text = re.sub(r'<a[^>]*>([^<]*)</a>', r'\1', dd_body)
        author_text = re.sub(r'\([^)]*\)', '', author_text)
        author_text = re.sub(r'\*', '', author_text)
        author_text = re.sub(r'(?:Equal Contribution|Both authors contributed equally)[^,;]*', '', author_text, flags=re.I)
        author_names = [normalize_whitespace(a) for a in re.split(r'\s*,\s*', author_text) if normalize_whitespace(a)]
        authors_str = "; ".join(author_names)

        idx += 1
        short_id = f"{conf.venue}_{conf.year}_{idx:04d}"
        pid = f"{conf.conf_year}_{idx:04d}"

        records.append(AcceptedPaperRecord(
            paper_id=pid,
            short_id=short_id,
            venue=conf.venue,
            year=conf.year,
            conf_year=conf.conf_year,
            title=title,
            authors=authors_str,
            source_type=source.kind,
            source_url=source.url,
            paper_link=paper_link,
            arxiv_id=arxiv_id,
            arxiv_url=arxiv_url,
            keywords_raw=paper_type,
            abstract_raw="",
            openreview_forum_id="",
            has_pdf_camera_ready="yes" if acm_doi else "",
            decision="Accept",
        ))

    print(f"  [kesen-siggraph] parsed {len(records)} papers from {conf.conf_year}")
    return records


def _read_js_string_literal(s: str, quote_idx: int) -> tuple[str, int]:
    """Read a double-quoted JS string starting at quote_idx; return (value, index_after_closing_quote)."""
    if quote_idx >= len(s) or s[quote_idx] != '"':
        raise ValueError("expected opening double quote")
    i = quote_idx + 1
    parts: list[str] = []
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 1
            if i < len(s):
                parts.append(s[i])
                i += 1
            continue
        if c == '"':
            return "".join(parts), i + 1
        parts.append(c)
        i += 1
    raise ValueError("unterminated JS string")


def parse_acmmm_vue_accepted(js_text: str, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    """Parse ACM MM site: accepted paper titles/authors embedded in Vue chunk (contents: [...])."""
    marker_start = js_text.find("contents:[")
    if marker_start < 0:
        print(f"  [acmmm-vue] no contents:[ in chunk for {conf.conf_year}")
        return []
    arr_open = js_text.find("[", marker_start)
    depth = 0
    arr_close = -1
    for j in range(arr_open, len(js_text)):
        if js_text[j] == "[":
            depth += 1
        elif js_text[j] == "]":
            depth -= 1
            if depth == 0:
                arr_close = j
                break
    if arr_close < 0:
        print(f"  [acmmm-vue] unbalanced contents array for {conf.conf_year}")
        return []
    inner = js_text[arr_open + 1 : arr_close]

    title_marker = '{type:"paperTitle",text:"'
    author_marker = '{type:"paperAuthor",text:"'
    records: list[AcceptedPaperRecord] = []
    pos = 0
    while True:
        ti = inner.find(title_marker, pos)
        if ti < 0:
            break
        q_title = ti + len(title_marker) - 1
        title_raw, after_title = _read_js_string_literal(inner, q_title)
        ai = inner.find(author_marker, after_title)
        if ai < 0:
            break
        q_auth = ai + len(author_marker) - 1
        authors_raw, pos = _read_js_string_literal(inner, q_auth)

        title_plain = normalize_whitespace(html.unescape(re.sub(r"<[^>]+>", " ", title_raw)))
        id_m = re.match(r"^(\d{1,5})\s+(.+)$", title_plain, re.DOTALL)
        if not id_m:
            continue
        paper_num = id_m.group(1).strip()
        title = normalize_whitespace(id_m.group(2))
        if not title:
            continue
        authors_line = normalize_whitespace(html.unescape(re.sub(r"<[^>]+>", " ", authors_raw)))
        authors_text = "; ".join(a.strip() for a in authors_line.split(",") if a.strip())
        short_id = f"ACMMM_{conf.year}_{paper_num}"
        records.append(
            AcceptedPaperRecord(
                paper_id=short_id,
                short_id=short_id,
                title=title,
                authors=authors_text,
                venue=conf.venue,
                year=conf.year,
                conf_year=conf.conf_year,
                source_type=source.kind,
                source_url=source.url,
                decision="Accept",
            )
        )

    print(f"  [acmmm-vue] parsed {len(records)} papers from {conf.conf_year}")
    return records


JOURNAL_VENUES = {"TPAMI", "IJCV", "JMLR", "AIJ", "TNNLS"}

_NON_PAPER_RE = re.compile(
    r"(?:^|\b)(?:editorial\b|guest editorial|reviewers? list|editorial board$"
    r"|editor.s note|corrigendum\b|erratum\b|retraction\b"
    r"|table of contents|front cover|back cover)"
    r"|^correction to[:\s]",
    re.I,
)


def _is_non_paper(title: str) -> bool:
    """Return True if title matches known non-paper patterns (editorials, etc.)."""
    clean = re.sub(r"<[^>]+>", "", title).strip()
    return bool(_NON_PAPER_RE.search(clean))


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def parse_openalex_works(payload: list, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    records: list[AcceptedPaperRecord] = []
    for work in payload:
        title = normalize_whitespace(str(work.get("title", "") or ""))
        if not title:
            continue
        if _is_non_paper(title):
            continue

        authorships = work.get("authorships", []) or []
        author_names = []
        for a in authorships:
            author_info = a.get("author", {}) or {}
            name = (author_info.get("display_name", "") or "").strip()
            if name:
                author_names.append(name)
        authors_text = "; ".join(author_names)

        doi_raw = (work.get("doi") or "").strip()
        doi = doi_raw.replace("https://doi.org/", "") if doi_raw else ""

        abstract = normalize_whitespace(
            _reconstruct_abstract(work.get("abstract_inverted_index"))
        )

        biblio = work.get("biblio", {}) or {}

        records.append(
            AcceptedPaperRecord(
                venue=conf.venue,
                year=conf.year,
                conf_year=conf.conf_year,
                title=title,
                authors=authors_text,
                source_type=source.kind,
                source_url=f"openalex:{source.url}",
                paper_link=doi_raw or "",
                doi=doi,
                abstract_raw=abstract,
                decision="Accept",
                acceptance_type="Journal Article",
            )
        )
    print(f"  [openalex] parsed {len(records)} papers from {conf.conf_year}")
    return records


def parse_jmlr_html(text: str, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    """Parse JMLR volume page HTML.

    Each paper is a <dl> block:
      <dl>
      <dt>Title</dt>
      <dd><b><i>Author1, Author2</i></b>; (N):pages, year.
      <br>[<a href='abs_url'>abs</a>][<a href='pdf_url'>pdf</a>]...
      </dl>
    """
    block_re = re.compile(r"<dl>\s*<dt>(.*?)</dt>\s*<dd>(.*?)</dl>", re.S)
    author_re = re.compile(r"<b><i>(.*?)</i></b>", re.S)
    abs_re = re.compile(r"""href=["'](/papers/v\d+/[^"']+\.html)["'][^>]*>abs</a>""", re.I)
    pdf_re = re.compile(r"""href=["'](/papers/volume\d+/[^"']+\.pdf)["'][^>]*>pdf</a>""", re.I)
    code_re = re.compile(r"""href=["'](https?://[^"']+)["'][^>]*>code</a>""", re.I)

    records: list[AcceptedPaperRecord] = []
    for dt_raw, dd_raw in block_re.findall(text):
        title = normalize_whitespace(html.unescape(re.sub(r"<[^>]+>", " ", dt_raw)))
        if not title:
            continue
        if _is_non_paper(title):
            continue

        am = author_re.search(dd_raw)
        authors_raw = am.group(1) if am else ""
        authors_clean = normalize_whitespace(html.unescape(re.sub(r"<[^>]+>", "", authors_raw)))
        authors_text = "; ".join(a.strip() for a in authors_clean.split(",") if a.strip())

        abs_m = abs_re.search(dd_raw)
        abs_url = f"https://jmlr.org{abs_m.group(1)}" if abs_m else ""

        pdf_m = pdf_re.search(dd_raw)
        pdf_url = f"https://jmlr.org{pdf_m.group(1)}" if pdf_m else ""

        code_m = code_re.search(dd_raw)
        code_url = code_m.group(1) if code_m else ""

        records.append(
            AcceptedPaperRecord(
                venue=conf.venue,
                year=conf.year,
                conf_year=conf.conf_year,
                title=title,
                authors=authors_text,
                source_type=source.kind,
                source_url=source.url,
                paper_link=pdf_url or abs_url,
                code_url=code_url,
                decision="Accept",
                acceptance_type="Journal Article",
            )
        )
    print(f"  [jmlr] parsed {len(records)} papers from {conf.conf_year}")
    return records


def parse_acmmm_html(text: str, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    """Parse ACM MM accepted papers HTML: <p>ID\xa0<b>Title</b><br/>Authors</p>"""
    records: list[AcceptedPaperRecord] = []
    pattern = re.compile(
        r'<p[^>]*>\s*(\d{1,5})\s*[\xa0\s]*<b>(.*?)</b>\s*<br\s*/?\s*>(.*?)</p>',
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        paper_num = m.group(1).strip()
        title = normalize_whitespace(html.unescape(re.sub(r'<[^>]+>', '', m.group(2))))
        authors_raw = normalize_whitespace(html.unescape(re.sub(r'<[^>]+>', '', m.group(3))))
        authors_text = "; ".join(a.strip() for a in authors_raw.split(",") if a.strip())
        if not title:
            continue
        short_id = f"ACMMM_{conf.year}_{paper_num}"
        records.append(AcceptedPaperRecord(
            paper_id=short_id,
            short_id=short_id,
            title=title,
            authors=authors_text,
            venue=conf.venue,
            year=conf.year,
            conf_year=conf.conf_year,
            source_type=source.kind,
            source_url=source.url,
            decision="Accept",
        ))
    print(f"  [acmmm-html] parsed {len(records)} papers from {conf.conf_year}")
    return records


def parse_s2_bulk_papers(
    payload: list, conf: ConferenceYearConfig, source: SourceConfig,
) -> list[AcceptedPaperRecord]:
    """Parse Semantic Scholar bulk search results into AcceptedPaperRecords.

    Filters to arXiv-only papers (must have externalIds.ArXiv).
    """
    records: list[AcceptedPaperRecord] = []
    for paper in payload:
        ext_ids = paper.get("externalIds") or {}
        arxiv_id = (ext_ids.get("ArXiv") or "").strip()
        if not arxiv_id:
            continue

        title = normalize_whitespace(str(paper.get("title", "") or ""))
        if not title:
            continue

        authors_list = paper.get("authors") or []
        authors_text = "; ".join(
            (a.get("name") or "").strip()
            for a in authors_list
            if (a.get("name") or "").strip()
        )

        abstract = normalize_whitespace(str(paper.get("abstract", "") or ""))
        citation_count = paper.get("citationCount", 0) or 0
        doi = (ext_ids.get("DOI") or "").strip()
        oa_pdf = paper.get("openAccessPdf") or {}
        pdf_url = (oa_pdf.get("url") or "").strip()
        s2_url = (paper.get("url") or "").strip()

        records.append(AcceptedPaperRecord(
            title=title,
            authors=authors_text,
            venue=conf.venue,
            year=conf.year,
            conf_year=conf.conf_year,
            source_type=source.kind,
            source_url=source.url,
            arxiv_id=arxiv_id,
            arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
            abstract_raw=abstract,
            doi=doi,
            paper_link=pdf_url or s2_url,
            decision="Accept",
            acceptance_type="High-Impact Preprint",
            extras={"citation_count": str(citation_count)},
        ))
    print(f"  [s2-bulk] parsed {len(records)} arXiv papers from {conf.conf_year}")
    return records


def parse_hf_daily_papers(
    payload: list, conf: ConferenceYearConfig, source: SourceConfig,
) -> list[AcceptedPaperRecord]:
    """Parse HuggingFace Daily Papers API results."""
    records: list[AcceptedPaperRecord] = []
    for entry in payload:
        paper = entry.get("paper", {})
        arxiv_id = (paper.get("id") or "").strip()
        if not arxiv_id:
            continue

        title = normalize_whitespace(str(paper.get("title", "") or ""))
        if not title:
            continue

        authors_list = paper.get("authors") or []
        authors_text = "; ".join(
            (a.get("name") or "").strip()
            for a in authors_list
            if (a.get("name") or "").strip()
        )

        abstract = normalize_whitespace(str(paper.get("summary", "") or ""))
        upvotes = entry.get("_upvotes", entry.get("upvotes", 0))

        records.append(AcceptedPaperRecord(
            title=title,
            authors=authors_text,
            venue=conf.venue,
            year=conf.year,
            conf_year=conf.conf_year,
            source_type=source.kind,
            source_url=source.url,
            arxiv_id=arxiv_id,
            arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
            abstract_raw=abstract,
            paper_link=f"https://huggingface.co/papers/{arxiv_id}",
            decision="Accept",
            acceptance_type="Community Selected",
            extras={"hf_upvotes": str(upvotes)},
        ))
    print(f"  [hf-daily] parsed {len(records)} papers from {conf.conf_year}")
    return records


def parse_anthropic_research(
    payload: list, conf: ConferenceYearConfig, source: SourceConfig,
) -> list[AcceptedPaperRecord]:
    """Parse Anthropic research page scrape results."""
    from .normalize import extract_arxiv_id as _extract_arxiv_id
    records: list[AcceptedPaperRecord] = []
    for article in payload:
        title = normalize_whitespace(str(article.get("title", "") or ""))
        if not title:
            continue

        arxiv_url = (article.get("arxiv_url") or "").strip()
        arxiv_id = _extract_arxiv_id(arxiv_url) if arxiv_url else ""
        full_paper = (article.get("full_paper_url") or "").strip()
        page_url = (article.get("page_url") or "").strip()

        records.append(AcceptedPaperRecord(
            title=title,
            authors="Anthropic",
            venue=conf.venue,
            year=conf.year,
            conf_year=conf.conf_year,
            source_type=source.kind,
            source_url=source.url,
            arxiv_id=arxiv_id,
            arxiv_url=arxiv_url,
            abstract_raw=normalize_whitespace(str(article.get("description", "") or "")),
            paper_link=full_paper or arxiv_url or page_url,
            decision="Accept",
            acceptance_type="Technical Report",
            extras={
                "anthropic_page_url": page_url,
                "full_paper_url": full_paper,
                "published_date": article.get("date", ""),
            },
        ))
    print(f"  [anthropic] parsed {len(records)} articles from {conf.conf_year}")
    return records


PARSERS: dict[str, Callable[[object, ConferenceYearConfig, SourceConfig], list[AcceptedPaperRecord]]] = {
    "openreview_notes_json": parse_openreview_notes_json,
    "openreview_api_v2": parse_openreview_api_v2,
    "simple_html_paper_list": parse_simple_html_paper_list,
    "curated_markdown_links": parse_curated_markdown_links,
    "cvpr_openaccess_html": lambda payload, conf, source: parse_cvpr_openaccess_html(str(payload), conf, source),
    "iclr_virtual_html": lambda payload, conf, source: parse_iclr_virtual_html(str(payload), conf, source),
    "iclr_proceedings_html": lambda payload, conf, source: parse_iclr_proceedings_html(str(payload), conf, source),
    "virtual_conference_json": parse_virtual_conference_json,
    "acl_anthology_html": lambda payload, conf, source: parse_acl_anthology_html(str(payload), conf, source),
    "aaai_ojs_html": lambda payload, conf, source: parse_aaai_ojs_html(str(payload), conf, source),
    "kdd_html": lambda payload, conf, source: parse_kdd_html(str(payload), conf, source),
    "kesen_siggraph_html": lambda payload, conf, source: parse_kesen_siggraph_html(str(payload), conf, source),
    "acmmm_html": lambda payload, conf, source: parse_acmmm_html(str(payload), conf, source),
    "acmmm_vue_accepted": lambda payload, conf, source: parse_acmmm_vue_accepted(str(payload), conf, source),
    "openalex_works": parse_openalex_works,
    "jmlr_html": lambda payload, conf, source: parse_jmlr_html(str(payload), conf, source),
    "s2_bulk_papers": parse_s2_bulk_papers,
    "hf_daily_papers": parse_hf_daily_papers,
    "anthropic_research": parse_anthropic_research,
}


def parse_payload(payload: object, conf: ConferenceYearConfig, source: SourceConfig) -> list[AcceptedPaperRecord]:
    if source.parser not in PARSERS:
        raise ValueError(f"Unsupported parser: {source.parser}")
    return PARSERS[source.parser](payload, conf, source)
