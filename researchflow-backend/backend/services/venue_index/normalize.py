from __future__ import annotations

import re
import urllib.parse


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def strip_latex(text: str) -> str:
    """Remove LaTeX math and commands from a title string.

    Handles: $O(n)$, \\mathcal{F}, \\textbf{x}, \\alpha, etc.
    """
    text = re.sub(r"\$([^$]*)\$", r"\1", text)  # $..$ → content
    text = re.sub(r"\\(?:mathcal|mathbb|mathrm|textbf|textit|text|operatorname)\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:alpha|beta|gamma|delta|epsilon|lambda|mu|sigma|theta|pi|omega|phi|psi|rho|tau|eta|zeta|kappa|nu|xi|chi)\b", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)  # remaining \commands
    text = re.sub(r"[{}^_~]", " ", text)  # braces and special TeX chars
    return text


def normalize_title(text: str) -> str:
    text = strip_latex(normalize_whitespace(text)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_authors(text: str) -> str:
    text = normalize_whitespace(text)
    if not text:
        return ""
    parts = [normalize_whitespace(p) for p in re.split(r"\s*(?:,|;| and )\s*", text) if normalize_whitespace(p)]
    return "; ".join(parts)


def normalize_link(url: str, base_url: str = "") -> str:
    url = (url or "").strip().strip("()[]{}<>\"'.,;")
    if not url:
        return ""
    if base_url and not url.startswith(("http://", "https://")):
        url = base_url.rstrip("/") + "/" + url.lstrip("/")
    u = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((u.scheme, u.netloc, u.path, u.query, ""))


def extract_arxiv_id(url: str) -> str:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", url or "", re.I)
    return m.group(1) if m else ""


def extract_openreview_forum_id(url: str) -> str:
    m = re.search(r"openreview\.net/forum\?(?:.*&)?id=([A-Za-z0-9_-]+)", url or "", re.I)
    return m.group(1) if m else ""


def extract_doi(url: str) -> str:
    """Extract DOI from a URL or DOI string.

    Handles:
      - https://doi.org/10.XXXX/...
      - https://ojs.aaai.org/.../article/view/NNNNN/NNNNN (AAAI OJS PDF link)
      - https://ojs.aaai.org/.../article/view/NNNNN (AAAI OJS article link)
    """
    url = (url or "").strip()
    m = re.match(r"https?://doi\.org/(10\.\d{4,}/[^\s,]+)", url)
    if m:
        return m.group(1)
    return ""


def extract_aaai_article_id(url: str) -> str:
    """Extract AAAI OJS article ID from paper_link.

    https://ojs.aaai.org/index.php/AAAI/article/view/28815/29555 -> 28815
    https://ojs.aaai.org/index.php/AAAI/article/view/28815 -> 28815
    """
    m = re.search(r"ojs\.aaai\.org/index\.php/AAAI/article/view/(\d+)", url or "")
    return m.group(1) if m else ""


def slugify_short_title(title: str, max_words: int = 3) -> str:
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", title or "") if p]
    return "".join(p.capitalize() for p in parts[:max_words]) or "Paper"
