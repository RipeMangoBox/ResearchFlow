"""Standalone metadata helper functions — each independently testable.

All functions are pure (no DB, no state) unless explicitly noted.
Import and call from enrich_service.py, parse_service.py, etc.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# 1. Title matching (Jaccard + substring, from resmax)
# ═══════════════════════════════════════════════════════════════════

def _normalize_title(title: str) -> str:
    """NFKD unicode normalize → strip accents → lowercase → alphanum only."""
    t = unicodedata.normalize("NFKD", title or "")
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def title_similarity(a: str, b: str) -> float:
    """Jaccard similarity of normalized title word sets."""
    wa = set(_normalize_title(a).split())
    wb = set(_normalize_title(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def is_title_match(query: str, candidate: str, threshold: float = 0.80) -> bool:
    """Robust title matching: Jaccard ≥ threshold, OR substring containment.

    This replaces the old _titles_similar() which only checked 30-char prefix.
    Handles: accent differences, punctuation, long/short title variants.
    """
    if title_similarity(query, candidate) >= threshold:
        return True
    nq = _normalize_title(query)
    nc = _normalize_title(candidate)
    if not nq or not nc:
        return False
    # Substring containment (handles truncated titles)
    if nq in nc or nc in nq:
        return True
    # No-space exact match (handles spacing differences)
    nq_ns = nq.replace(" ", "")
    nc_ns = nc.replace(" ", "")
    if nq_ns == nc_ns:
        return True
    # Relaxed substring on no-space versions (long titles only)
    if len(nq_ns) > 10 and len(nc_ns) > 10:
        if nq_ns in nc_ns or nc_ns in nq_ns:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════
# 2. PDF validation
# ═══════════════════════════════════════════════════════════════════

MAX_PDF_BYTES = 80 * 1024 * 1024  # 80 MB safety cap


def looks_like_pdf(data: bytes) -> bool:
    """Check %PDF- magic bytes. Rejects HTML error pages disguised as 200 OK."""
    return len(data) >= 5 and data[:5] == b"%PDF-"


def is_valid_abstract(text: str | None) -> bool:
    """Check abstract is real content, not a placeholder."""
    if not text or len(text.strip()) < 10:
        return False
    lower = text.strip().lower()
    placeholders = {"none", "null", "n/a", "na", "not available", "abstract", "tbd", ""}
    return lower not in placeholders


# ═══════════════════════════════════════════════════════════════════
# 3. URL normalization (from resmax data_contracts.py)
# ═══════════════════════════════════════════════════════════════════

_TRAILING_URL_PUNCT = " \t\r\n.,;!?)]}>'\""


def clean_url_token(raw: str) -> str:
    """Strip punctuation that commonly follows URLs in prose."""
    return (raw or "").strip().rstrip(_TRAILING_URL_PUNCT)


def normalize_http_url(raw: str) -> str:
    """Add https://, handle www., clean trailing punctuation."""
    url = clean_url_token(raw)
    if not url:
        return ""
    if url.startswith("www."):
        url = "https://" + url
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        if url.lower().startswith(("github.com/", "gitlab.com/")):
            url = "https://" + url
    return url


def normalize_repo_url(raw: str) -> str:
    """Canonical GitHub/GitLab repo URL: https://github.com/{owner}/{repo}."""
    import urllib.parse
    url = normalize_http_url(raw)
    if not url:
        return ""
    parsed = urllib.parse.urlsplit(url)
    host = parsed.netloc.lower().lstrip("www.")
    if host not in {"github.com", "gitlab.com"}:
        return url
    parts = [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return f"https://{host}/" + "/".join(parts) if parts else ""
    owner = parts[0].strip()
    repo = clean_url_token(parts[1]).strip()
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        return ""
    # Filter out GitHub feature paths
    _skip = {"features", "issues", "pulls", "blob", "tree", "wiki",
             "settings", "actions", "releases", "commit", "compare"}
    if repo.lower() in _skip:
        return ""
    return f"https://{host}/{owner}/{repo}"


def repo_cache_key(raw: str) -> str:
    """Lowercase owner/repo for deduplication."""
    import urllib.parse
    url = normalize_repo_url(raw)
    parsed = urllib.parse.urlsplit(url)
    if parsed.netloc.lower().lstrip("www.") not in {"github.com", "gitlab.com"}:
        return ""
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return ""
    return f"{parts[0]}/{parts[1]}".lower()


# ═══════════════════════════════════════════════════════════════════
# 4. Papers With Code dump loader
# ═══════════════════════════════════════════════════════════════════

_pwc_links: dict[str, list[str]] | None = None
_pwc_arxiv_to_url: dict[str, str] | None = None
_pwc_title_to_url: dict[str, str] | None = None


def load_pwc_dump(pwc_path: str) -> tuple[dict, dict, dict]:
    """Load Papers With Code links dump. Returns (links, arxiv_to_url, title_to_url).

    pwc_path: path to links-between-papers-and-code.json.gz or links.json
    Caches in module globals after first load.
    """
    global _pwc_links, _pwc_arxiv_to_url, _pwc_title_to_url
    if _pwc_links is not None:
        return _pwc_links, _pwc_arxiv_to_url, _pwc_title_to_url  # type: ignore

    import gzip
    import json
    from pathlib import Path

    path = Path(pwc_path)
    if not path.exists():
        logger.warning(f"PWC dump not found: {pwc_path}")
        _pwc_links, _pwc_arxiv_to_url, _pwc_title_to_url = {}, {}, {}
        return _pwc_links, _pwc_arxiv_to_url, _pwc_title_to_url

    logger.info(f"Loading PWC dump: {pwc_path}")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

    links: dict[str, list[str]] = {}
    arxiv_to_url: dict[str, str] = {}
    title_to_url: dict[str, str] = {}

    for entry in raw:
        paper_url = (entry.get("paper_url") or "").strip()
        repo_url = (entry.get("repo_url") or "").strip()
        if paper_url and repo_url:
            links.setdefault(paper_url, []).append(repo_url)

        # Build arxiv_id → paper_url index
        if paper_url:
            m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", paper_url)
            if m and m.group(1) not in arxiv_to_url:
                arxiv_to_url[m.group(1)] = paper_url
            # Build title → paper_url index (from PWC paper pages)
            paper_title = (entry.get("paper_title") or "").strip()
            if paper_title:
                norm = _normalize_title(paper_title)
                if norm and norm not in title_to_url:
                    title_to_url[norm] = paper_url

    logger.info(f"PWC loaded: {len(links)} paper→code, {len(arxiv_to_url)} arxiv→url, {len(title_to_url)} title→url")
    _pwc_links, _pwc_arxiv_to_url, _pwc_title_to_url = links, arxiv_to_url, title_to_url
    return links, arxiv_to_url, title_to_url


def lookup_pwc_code_url(title: str, arxiv_id: str = "", pwc_path: str = "") -> str | None:
    """Find code URL via Papers With Code dump. Returns normalized repo URL or None."""
    if not pwc_path:
        return None
    links, arxiv_to_url, title_to_url = load_pwc_dump(pwc_path)
    if not links:
        return None

    paper_url = None
    # Try arxiv_id first
    if arxiv_id:
        clean_id = re.sub(r"v\d+$", "", arxiv_id)
        paper_url = arxiv_to_url.get(clean_id)
    # Try normalized title
    if not paper_url and title:
        norm = _normalize_title(title)
        paper_url = title_to_url.get(norm)

    if paper_url and paper_url in links:
        repos = links[paper_url]
        # Prefer github.com repos
        best = repos[0]
        for r in repos:
            if "github.com" in r:
                best = r
                break
        return normalize_repo_url(best) or None
    return None


# ═══════════════════════════════════════════════════════════════════
# 5. Abstract scraping — venue-specific (CVF/AAAI/ACM)
# ═══════════════════════════════════════════════════════════════════

async def fetch_cvf_abstract(client, paper_link: str) -> str | None:
    """Scrape abstract from CVF OpenAccess page (CVPR/ICCV)."""
    html_url = paper_link or ""
    if "/papers/" in html_url and html_url.endswith(".pdf"):
        html_url = html_url.replace("/papers/", "/html/").replace("_paper.pdf", "_paper.html")
    elif not html_url.endswith(".html"):
        return None
    if "openaccess.thecvf.com" not in html_url:
        return None
    try:
        resp = await client.get(html_url, timeout=15)
        if resp.status_code != 200:
            return None
        m = re.search(r'<div id="abstract">\s*(.*?)\s*</div>', resp.text, re.S)
        if m:
            abstract = re.sub(r"<[^>]+>", "", m.group(1))
            abstract = re.sub(r"\s+", " ", abstract).strip()
            return abstract if len(abstract) > 20 else None
    except Exception:
        return None
    return None


async def fetch_aaai_abstract(client, paper_link: str) -> str | None:
    """Scrape abstract from AAAI OJS article page."""
    m = re.search(r"ojs\.aaai\.org/index\.php/AAAI/article/view/(\d+)", paper_link or "")
    if not m:
        return None
    url = f"https://ojs.aaai.org/index.php/AAAI/article/view/{m.group(1)}"
    try:
        resp = await client.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        m2 = re.search(r'<section class="item abstract">(.*?)</section>', resp.text, re.S)
        if m2:
            abstract = re.sub(r"<[^>]+>", "", m2.group(1))
            abstract = re.sub(r"\s+", " ", abstract).strip()
            if abstract.lower().startswith("abstract"):
                abstract = abstract[8:].strip()
            return abstract if len(abstract) > 20 else None
    except Exception:
        return None
    return None


async def fetch_acm_abstract(client, doi: str) -> str | None:
    """Scrape abstract from ACM Digital Library page (KDD/MM/SIGIR etc)."""
    if not doi:
        return None
    url = f"https://dl.acm.org/doi/{doi}"
    try:
        resp = await client.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        })
        if resp.status_code != 200:
            return None
        # Try <section class="abstract">
        m = re.search(r'<section[^>]*class="[^"]*abstract[^"]*"[^>]*>(.*?)</section>', resp.text, re.S)
        if not m:
            m = re.search(r'<div\s+class="abstractSection[^"]*"[^>]*>(.*?)</div>', resp.text, re.S)
        if m:
            abstract = re.sub(r"<[^>]+>", " ", m.group(1))
            abstract = re.sub(r"\s+", " ", abstract).strip()
            return abstract if len(abstract) > 20 else None
    except Exception:
        return None
    return None


async def search_openalex_abstract(client, title: str) -> dict | None:
    """Search OpenAlex by title, return {abstract, doi, title} if matched."""
    import urllib.parse
    encoded = urllib.parse.quote(title[:200])
    url = f"https://api.openalex.org/works?search={encoded}&per_page=3&select=id,title,abstract_inverted_index,doi"
    try:
        resp = await client.get(url, timeout=15, headers={"User-Agent": "ResearchFlow/0.2"})
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    for r in data.get("results", []):
        if not is_title_match(title, r.get("title", "")):
            continue
        result: dict = {"title": r.get("title", "")}
        raw_doi = r.get("doi", "")
        if raw_doi:
            result["doi"] = re.sub(r"^https?://doi\.org/", "", raw_doi)
        aii = r.get("abstract_inverted_index")
        if aii:
            word_positions = []
            for word, positions in aii.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort()
            abstract = " ".join(w for _, w in word_positions)
            if len(abstract) > 20:
                result["abstract"] = abstract
        return result if result.get("abstract") or result.get("doi") else None
    return None


async def search_crossref_abstract(client, title: str) -> dict | None:
    """Search CrossRef by title, return {abstract, doi, title} if matched."""
    import urllib.parse
    clean = re.sub(r"[^\w\s]", " ", title)
    clean = re.sub(r"\s+", " ", clean).strip()[:200]
    encoded = urllib.parse.quote(clean)
    url = f"https://api.crossref.org/works?query.title={encoded}&rows=3&select=DOI,title,abstract"
    try:
        resp = await client.get(url, timeout=20, headers={"User-Agent": "ResearchFlow/0.2"})
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    for item in data.get("message", {}).get("items", []):
        cand_title = (item.get("title") or [""])[0]
        cand_title = re.sub(r"<[^>]+>", "", cand_title)  # strip JATS XML
        if not is_title_match(title, cand_title):
            continue
        result: dict = {"title": cand_title}
        if item.get("DOI"):
            result["doi"] = item["DOI"]
        raw_abstract = item.get("abstract", "")
        if raw_abstract:
            abstract = re.sub(r"<[^>]+>", "", raw_abstract)
            abstract = re.sub(r"\s+", " ", abstract).strip()
            if len(abstract) > 20:
                result["abstract"] = abstract
        return result if result.get("abstract") or result.get("doi") else None
    return None


async def search_s2_abstract(client, title: str, s2_headers: dict | None = None) -> dict | None:
    """Search Semantic Scholar by title, return {abstract, arxiv_id, doi, title} if matched."""
    import urllib.parse
    clean = re.sub(r"[^\w\s]", " ", title)
    clean = re.sub(r"\s+", " ", clean).strip()[:200]
    encoded = urllib.parse.quote(clean)
    url = (f"https://api.semanticscholar.org/graph/v1/paper/search"
           f"?query={encoded}&limit=3&fields=title,abstract,externalIds,openAccessPdf")
    try:
        resp = await client.get(url, timeout=15, headers=s2_headers or {})
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    best, best_sim = None, 0.0
    for p in data.get("data", []):
        sim = title_similarity(title, p.get("title", ""))
        if sim > best_sim:
            best_sim = sim
            result: dict = {"title": p.get("title", "")}
            if p.get("abstract"):
                result["abstract"] = p["abstract"]
            ext = p.get("externalIds") or {}
            if ext.get("ArXiv"):
                result["arxiv_id"] = ext["ArXiv"]
            if ext.get("DOI"):
                result["doi"] = ext["DOI"]
            best = result

    return best if best_sim >= 0.80 and best and best.get("abstract") else None


async def search_arxiv_abstract(client, title: str) -> dict | None:
    """Search arXiv API by title keywords, return {abstract, arxiv_id, title} if matched."""
    import urllib.parse
    _STOPWORDS = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with",
                  "is", "are", "by", "from", "that", "this", "at", "or", "as", "its"}
    norm = _normalize_title(title)
    words = [w for w in norm.split() if len(w) > 2 and w not in _STOPWORDS][:6]
    if len(words) < 2:
        return None
    query = "+AND+".join(f"ti:{w}" for w in words)
    url = f"http://export.arxiv.org/api/query?search_query={query}&max_results=5&sortBy=relevance"
    try:
        resp = await client.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        body = resp.text
    except Exception:
        return None

    best, best_sim = None, 0.0
    for entry in re.findall(r"<entry>(.*?)</entry>", body, re.S):
        t_m = re.search(r"<title>(.*?)</title>", entry, re.S)
        a_m = re.search(r"<summary>(.*?)</summary>", entry, re.S)
        id_m = re.search(r"<id>http://arxiv\.org/abs/(\d{4}\.\d{4,5})", entry)
        if t_m and a_m:
            cand_title = re.sub(r"\s+", " ", t_m.group(1)).strip()
            sim = title_similarity(title, cand_title)
            if sim > best_sim:
                best_sim = sim
                abstract = re.sub(r"\s+", " ", a_m.group(1)).strip()
                best = {
                    "abstract": abstract,
                    "title": cand_title,
                    "arxiv_id": id_m.group(1) if id_m else "",
                }

    return best if best_sim >= 0.80 and best else None


# ═══════════════════════════════════════════════════════════════════
# 6. OpenReview review data extraction
# ═══════════════════════════════════════════════════════════════════

async def fetch_openreview_reviews(client, forum_id: str, venue_id: str = "") -> dict | None:
    """Fetch review data from OpenReview API v2.

    Returns {
        review_available, review_num_reviewers, review_scores,
        review_score_mean, review_confidence_scores, review_confidence_mean,
        reviews: [{reviewer_id, rating, confidence, summary, strengths, weaknesses}]
    } or None.
    """
    if not forum_id:
        return None
    url = f"https://api2.openreview.net/notes?forum={forum_id}&details=directReplies"
    try:
        resp = await client.get(url, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    reviews = []
    for note in data.get("notes", []):
        content = note.get("content", {})
        # Detect review notes (have rating or recommendation)
        rating_field = None
        for key in ("rating", "recommendation", "overall_assessment", "soundness_rating"):
            if key in content:
                rating_field = key
                break
        if not rating_field:
            continue

        rating_raw = content.get(rating_field, {})
        rating_val = rating_raw.get("value") if isinstance(rating_raw, dict) else rating_raw

        confidence_raw = content.get("confidence", {})
        confidence_val = confidence_raw.get("value") if isinstance(confidence_raw, dict) else confidence_raw

        # Parse numeric value from strings like "8: Strong Accept"
        def _parse_score(val) -> float | None:
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return float(val)
            s = str(val)
            m = re.match(r"(\d+(?:\.\d+)?)", s)
            return float(m.group(1)) if m else None

        rating_num = _parse_score(rating_val)
        confidence_num = _parse_score(confidence_val)

        def _get_text(field_name: str) -> str:
            v = content.get(field_name, {})
            if isinstance(v, dict):
                return str(v.get("value", ""))[:2000]
            return str(v)[:2000] if v else ""

        reviews.append({
            "reviewer_id": note.get("signatures", [""])[0].split("/")[-1] if note.get("signatures") else "",
            "rating": rating_num,
            "confidence": confidence_num,
            "summary": _get_text("summary"),
            "strengths": _get_text("strengths"),
            "weaknesses": _get_text("weaknesses"),
        })

    if not reviews:
        return None

    scores = [r["rating"] for r in reviews if r["rating"] is not None]
    confidences = [r["confidence"] for r in reviews if r["confidence"] is not None]

    return {
        "review_available": "yes",
        "review_num_reviewers": len(reviews),
        "review_scores": scores,
        "review_score_mean": round(sum(scores) / len(scores), 2) if scores else None,
        "review_confidence_scores": confidences,
        "review_confidence_mean": round(sum(confidences) / len(confidences), 2) if confidences else None,
        "reviews": reviews,
    }
