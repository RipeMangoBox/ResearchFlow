from __future__ import annotations

import json
import os
import re
import signal
import time
import urllib.request
import urllib.parse
from pathlib import Path

from .models import SourceConfig


# ── Proxy support ─────────────────────────────────────────────
# Set VENUE_PROXY env var to route fetcher traffic through mihomo.
# mihomo's rules decide per-domain whether to proxy or go direct.
# Example: VENUE_PROXY=socks5h://127.0.0.1:7891

def _get_proxy_url() -> str:
    return os.environ.get("VENUE_PROXY", "").strip()


def _build_opener() -> urllib.request.OpenerDirector:
    """Build URL opener with SOCKS5 proxy if VENUE_PROXY is set."""
    proxy = _get_proxy_url()
    if proxy:
        proxy_handler = urllib.request.ProxyHandler({
            "http": proxy,
            "https": proxy,
        })
        return urllib.request.build_opener(proxy_handler)
    return urllib.request.build_opener()


def _curl_proxy_args() -> list[str]:
    """Return curl proxy args if VENUE_PROXY is set."""
    proxy = _get_proxy_url()
    if proxy:
        return ["--proxy", proxy]
    return []


class _FetchTimeout(Exception):
    pass


def _fetch_with_total_timeout(url: str, headers: dict, socket_timeout: int = 30, total_timeout: int = 120) -> bytes:
    """Fetch URL with both per-socket and total wall-clock timeout.

    Uses signal.SIGALRM in the main thread, falls back to threading.Timer
    in worker threads (where signals are unavailable).
    """
    import threading

    in_main_thread = threading.current_thread() is threading.main_thread()
    old_handler = None

    if in_main_thread:
        old_handler = signal.getsignal(signal.SIGALRM)
        def _alarm(signum, frame):
            raise _FetchTimeout(f"total timeout {total_timeout}s exceeded for {url}")
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(total_timeout)

    # For non-main threads, rely on socket_timeout (set generously)
    effective_socket_timeout = min(socket_timeout, total_timeout) if not in_main_thread else socket_timeout

    try:
        req = urllib.request.Request(url, headers=headers)
        opener = _build_opener()
        with opener.open(req, timeout=effective_socket_timeout) as resp:
            data = resp.read()
        return data
    except _FetchTimeout:
        raise
    except Exception:
        raise
    finally:
        if in_main_thread and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def fetch_text(source: SourceConfig, fixtures_dir: Path, timeout: int = 60) -> str:
    url = source.url
    if url.startswith("fixture://"):
        rel = url[len("fixture://"):]
        return (fixtures_dir / rel).read_text(encoding="utf-8")

    raw = _fetch_with_total_timeout(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/json,*/*"},
        socket_timeout=timeout,
        total_timeout=max(timeout * 3, 120),
    )
    return raw.decode("utf-8", errors="replace")


def fetch_json(source: SourceConfig, fixtures_dir: Path, timeout: int = 30) -> dict:
    return json.loads(fetch_text(source, fixtures_dir=fixtures_dir, timeout=timeout))


def fetch_openreview_api_v2(
    group: str,
    accepted_venue_prefixes: list[str],
    timeout: int = 60,
    page_size: int = 1000,
) -> dict:
    """Paginate through OpenReview API v2 search endpoint and return accepted notes.

    Uses /notes/search?query=*&group=<group> with offset pagination.
    Filters notes whose content.venue.value starts with any of accepted_venue_prefixes.
    Returns a dict compatible with openreview_notes_json parser: {"notes": [...]}.
    """
    base = "https://api2.openreview.net/notes/search"
    all_notes: list[dict] = []
    offset = 0
    total = None

    while True:
        params = urllib.parse.urlencode({
            "query": "*",
            "group": group,
            "limit": page_size,
            "offset": offset,
        })
        url = f"{base}?{params}"
        raw = _fetch_with_total_timeout(
            url,
            headers={
                "User-Agent": "resmax-accepted-index/1.0",
                "Accept": "application/json",
            },
            socket_timeout=timeout,
            total_timeout=max(timeout * 2, 90),
        )
        data = json.loads(raw.decode("utf-8", errors="replace"))

        notes = data.get("notes", [])
        if total is None:
            total = data.get("count", 0)
            print(f"  [OpenReview] group={group}, total submissions={total}")

        for note in notes:
            venue = note.get("content", {}).get("venue", {}).get("value", "")
            if any(venue.lower().startswith(p.lower()) for p in accepted_venue_prefixes):
                all_notes.append(note)

        offset += len(notes)
        if not notes or offset >= total:
            break
        time.sleep(0.5)

    print(f"  [OpenReview] accepted notes after filtering: {len(all_notes)}")
    return {"notes": all_notes}


def fetch_aaai_ojs_all_issues(
    archive_url: str,
    year_tag: str,
    timeout: int = 60,
) -> str:
    """Fetch all AAAI OJS Technical Tracks issue pages for a given year and concatenate.

    1. Fetches the archive page(s) to discover issue URLs matching year_tag (e.g. "AAAI-25").
    2. Fetches each matching issue page.
    3. Returns concatenated HTML of all issue pages.
    """
    import re as _re

    def _get(url: str) -> str:
        raw = _fetch_with_total_timeout(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "Accept": "text/html,*/*",
            },
            socket_timeout=timeout,
            total_timeout=max(timeout * 2, 90),
        )
        return raw.decode("utf-8", errors="replace")

    all_issue_urls: list[str] = []
    page = 1
    max_pages = 10
    found_year = False
    while page <= max_pages:
        page_url = archive_url if page == 1 else f"{archive_url}/{page}"
        html = _get(page_url)

        issue_pattern = _re.compile(
            r'<a\s+class="title"\s+href="(https://ojs\.aaai\.org/index\.php/AAAI/issue/view/\d+)">\s*'
            + _re.escape(year_tag) + r'\s+Technical\s+Tracks',
            _re.I | _re.S,
        )
        found = issue_pattern.findall(html)
        all_issue_urls.extend(found)

        if found:
            found_year = True
        elif found_year:
            break

        page += 1
        time.sleep(0.3)

    all_issue_urls = list(dict.fromkeys(all_issue_urls))
    print(f"  [AAAI OJS] found {len(all_issue_urls)} Technical Tracks issues for {year_tag}")

    combined_html = ""
    for i, url in enumerate(all_issue_urls):
        print(f"  [AAAI OJS] fetching issue {i+1}/{len(all_issue_urls)}: {url}")
        combined_html += _get(url) + "\n"
        time.sleep(0.3)

    return combined_html


def _http_get(url: str, timeout: int = 60) -> str:
    raw = _fetch_with_total_timeout(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"},
        socket_timeout=timeout,
        total_timeout=max(timeout * 3, 120),
    )
    return raw.decode("utf-8", errors="replace")


def fetch_acmmm_vue_accepted_chunk(base_url: str, chunk_name: str, timeout: int = 60) -> str:
    """Load ACM MM Vue SPA accepted-papers chunk: resolve app bundle hash, then chunk hash.

    base_url: e.g. https://2024.acmmm.org (no trailing path required).
    chunk_name: webpack chunk id for the Accepted Papers route, e.g. chunk-240a60f6.
    """
    base = base_url.rstrip("/")
    index_html = _http_get(base + "/", timeout=timeout)
    app_m = re.search(r'src="(/js/app\.[a-f0-9]+\.js)"', index_html)
    if not app_m:
        app_m = re.search(r"src='(/js/app\.[a-f0-9]+\.js)'", index_html)
    if not app_m:
        raise ValueError("Could not find /js/app.[hash].js in ACM MM index HTML")
    app_path = app_m.group(1)
    app_js = _http_get(base + app_path, timeout=timeout)
    chunk_re = re.compile(re.escape(f'"{chunk_name}"') + r':"([a-f0-9]+)"')
    hm = chunk_re.search(app_js)
    if not hm:
        raise ValueError(f'Chunk id "{chunk_name}" not found in app bundle')
    chunk_path = f"/js/{chunk_name}.{hm.group(1)}.js"
    print(f"  [acmmm-vue] fetching {base}{chunk_path}")
    return _http_get(base + chunk_path, timeout=timeout)


def fetch_openalex_works(
    source_id: str,
    year: int,
    api_key: str | None = None,
    per_page: int = 200,
    timeout: int = 30,
) -> list[dict]:
    """Fetch all works from an OpenAlex source for a given year via cursor paging."""
    import os

    if api_key is None:
        api_key = os.environ.get("OPENALEX_API_KEY", "")

    base = "https://api.openalex.org/works"
    select_fields = (
        "id,title,doi,publication_date,authorships,"
        "abstract_inverted_index,biblio,type,cited_by_count"
    )
    all_works: list[dict] = []
    cursor = "*"
    page_num = 0

    while cursor:
        params: dict[str, str] = {
            "filter": f"primary_location.source.id:{source_id},publication_year:{year}",
            "per_page": str(per_page),
            "cursor": cursor,
            "select": select_fields,
        }
        if api_key:
            params["api_key"] = api_key

        url = f"{base}?{urllib.parse.urlencode(params)}"
        raw = _fetch_with_total_timeout(
            url,
            headers={
                "User-Agent": "resmax-accepted-index/1.0",
                "Accept": "application/json",
            },
            socket_timeout=timeout,
            total_timeout=max(timeout * 3, 120),
        )
        data = json.loads(raw.decode("utf-8", errors="replace"))

        results = data.get("results", [])
        meta = data.get("meta", {})
        all_works.extend(results)
        page_num += 1

        total = meta.get("count", "?")
        if page_num == 1:
            print(f"  [OpenAlex] source={source_id}, year={year}, total={total}")

        next_cursor = meta.get("next_cursor")
        if not next_cursor or not results:
            break
        cursor = next_cursor
        time.sleep(0.2)

    print(f"  [OpenAlex] fetched {len(all_works)} works in {page_num} pages")
    return all_works


def fetch_s2_bulk_search(
    year: int,
    min_citation_count: int = 50,
    fields_of_study: str = "Computer Science",
    timeout: int = 30,
) -> list[dict]:
    """Fetch high-citation arXiv preprints via Semantic Scholar Bulk Search API.

    Uses cursor-based pagination. Returns papers that:
    - Are in the given fieldsOfStudy
    - Were published in the given year
    - Have >= min_citation_count citations
    - Have an arXiv external ID (i.e. are on arXiv)
    """
    import os

    api_key = os.environ.get("S2_API_KEY", "")
    base = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    fields = "paperId,externalIds,title,abstract,authors,year,citationCount,publicationTypes,openAccessPdf,url"
    all_papers: list[dict] = []
    token: str | None = None
    page_num = 0

    while True:
        params: dict[str, str] = {
            "query": "",
            "fieldsOfStudy": fields_of_study,
            "year": str(year),
            "minCitationCount": str(min_citation_count),
            "fields": fields,
        }
        if token:
            params["token"] = token

        url = f"{base}?{urllib.parse.urlencode(params)}"
        headers: dict[str, str] = {
            "User-Agent": "resmax-accepted-index/1.0",
            "Accept": "application/json",
        }
        if api_key:
            headers["x-api-key"] = api_key

        raw = _fetch_with_total_timeout(
            url, headers=headers,
            socket_timeout=timeout,
            total_timeout=max(timeout * 4, 180),
        )
        data = json.loads(raw.decode("utf-8", errors="replace"))

        papers = data.get("data", [])
        all_papers.extend(papers)
        page_num += 1

        if page_num == 1:
            total = data.get("total", "?")
            print(f"  [S2-bulk] year={year}, minCite={min_citation_count}, total={total}")

        token = data.get("token")
        if not token or not papers:
            break
        time.sleep(1.0)

    print(f"  [S2-bulk] fetched {len(all_papers)} papers in {page_num} pages")
    return all_papers


def _curl_json(url: str, timeout: int = 15) -> object:
    """Fetch JSON via curl subprocess (fallback for hosts where urllib fails)."""
    import subprocess
    result = subprocess.run(
        ["curl", "-s", "--max-time", str(timeout), *_curl_proxy_args(),
         "-H", "User-Agent: resmax-accepted-index/1.0", url],
        capture_output=True, text=True, timeout=timeout + 5,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed (rc={result.returncode}): {result.stderr[:200]}")
    return json.loads(result.stdout)


def fetch_hf_daily_papers(
    year: int,
    min_upvotes: int = 10,
    timeout: int = 15,
) -> list[dict]:
    """Fetch community-selected papers from HuggingFace Daily Papers API.

    Iterates day-by-day through the target year using the ``date`` query
    parameter (the only reliable pagination method — ``skip`` is broken
    on this endpoint).  Uses curl as HTTP client because Python urllib
    cannot connect to huggingface.co on some systems.
    """
    from datetime import date as _date, timedelta

    base = "https://huggingface.co/api/daily_papers"

    start = _date(year, 1, 1)
    today = _date.today()
    end = min(_date(year, 12, 31), today)
    all_papers: list[dict] = []
    seen_ids: set[str] = set()
    day = start
    fetch_days = 0

    while day <= end:
        url = f"{base}?date={day.isoformat()}&limit=100"
        try:
            data = _curl_json(url, timeout=timeout)
        except Exception as exc:
            print(f"  [HF-daily] {day}: fetch error: {exc}")
            day += timedelta(days=1)
            continue

        if not isinstance(data, list) or not data:
            day += timedelta(days=1)
            time.sleep(0.05)
            continue

        fetch_days += 1
        for entry in data:
            paper = entry.get("paper", {})
            arxiv_id = (paper.get("id") or "").strip()
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            upvotes = paper.get("upvotes", 0)
            if upvotes >= min_upvotes:
                entry["_upvotes"] = upvotes
                all_papers.append(entry)
                seen_ids.add(arxiv_id)

        day += timedelta(days=1)
        time.sleep(0.1)

    print(f"  [HF-daily] year={year}, minUpvotes={min_upvotes}, fetched {len(all_papers)} papers from {fetch_days} active days")
    return all_papers


def _curl_html(url: str, timeout: int = 10, retries: int = 2) -> str:
    """Fetch HTML via curl subprocess with retries."""
    import subprocess
    for attempt in range(retries + 1):
        result = subprocess.run(
            ["curl", "-s", "--max-time", str(timeout), *_curl_proxy_args(),
             "-H", "User-Agent: resmax-accepted-index/1.0", url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        if result.stdout and len(result.stdout) > 100:
            return result.stdout
        if attempt < retries:
            time.sleep(1.0)
    return result.stdout


def fetch_anthropic_research(year: int | None = None, timeout: int = 10) -> list[dict]:
    """Fetch Anthropic research articles from sitemap + per-page scraping.

    Returns a list of dicts with keys: slug, title, date, description,
    arxiv_url, full_paper_url, page_url.
    """
    import re
    import subprocess
    from datetime import date as _date

    # Step 1: get all research URLs from sitemap
    r = subprocess.run(
        ["curl", "-s", "--max-time", "10", *_curl_proxy_args(), "https://www.anthropic.com/sitemap.xml"],
        capture_output=True, text=True, timeout=15,
    )
    urls = re.findall(
        r"<loc>(https://www\.anthropic\.com/research/[^<]+)</loc>", r.stdout
    )
    urls = [u for u in urls if "/research/team/" not in u]
    print(f"  [anthropic] sitemap: {len(urls)} research URLs")

    # Step 2: scrape each page
    all_articles: list[dict] = []
    date_re = re.compile(
        r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})"
    )
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    for i, url in enumerate(urls):
        slug = url.split("/research/")[-1]
        try:
            html = _curl_html(url, timeout=timeout)
            if not html:
                continue

            # Title
            m = re.search(r"<title>([^<]+)</title>", html)
            if not m:
                continue
            title = m.group(1).split("\\")[0].split("—")[0].split("|")[0].strip()
            if title.lower() in ("anthropic", "research", ""):
                continue

            # Date — first match in page
            dm = date_re.search(html)
            pub_date = ""
            pub_year = 0
            if dm:
                raw_date = dm.group(1)
                parts = raw_date.replace(",", "").split()
                if len(parts) >= 3:
                    mon = month_map.get(parts[0][:3].lower(), 0)
                    day_num = int(parts[1])
                    yr = int(parts[2])
                    pub_date = f"{yr}-{mon:02d}-{day_num:02d}"
                    pub_year = yr

            # Filter by year if specified
            if year and pub_year and pub_year != year:
                continue

            # Description
            desc = ""
            m2 = re.search(r'og:description["\s]+content="([^"]+)"', html)
            if not m2:
                m2 = re.search(r'name="description"[^>]*content="([^"]+)"', html)
            if m2:
                import html as html_mod
                desc = html_mod.unescape(m2.group(1)).strip()

            # Links
            arxiv_urls = re.findall(r'href="(https?://arxiv\.org/abs/[^"]+)"', html)
            tc_urls = re.findall(r'href="(https?://transformer-circuits\.pub[^"]+)"', html)

            all_articles.append({
                "slug": slug,
                "title": title,
                "date": pub_date,
                "year": pub_year,
                "description": desc,
                "arxiv_url": arxiv_urls[0] if arxiv_urls else "",
                "full_paper_url": tc_urls[0] if tc_urls else "",
                "page_url": url,
            })
        except Exception as exc:
            print(f"  [anthropic] {slug}: error: {exc}")
        if (i + 1) % 20 == 0:
            print(f"  [anthropic] scraped {i + 1}/{len(urls)} pages ...")
        time.sleep(0.2)

    print(f"  [anthropic] fetched {len(all_articles)} articles" +
          (f" for year={year}" if year else ""))
    return all_articles
