r"""TeX source extraction — arXiv e-print → formulas, citations, figures.

When a paper has an arxiv_id, we can download the LaTeX source and extract:
  - Formulas: exact LaTeX from equation/align environments (zero OCR error)
  - Citations: \cite{key} bibkeys with context sentences
  - Figures: \includegraphics paths + \caption text
  - URLs: \url{} and \href{} (char-faithful, no I/l/1 confusion)

Uses arxiv-to-prompt to flatten multi-file LaTeX projects into a single text.
Falls back gracefully if arxiv-to-prompt is not installed or source unavailable.

Inspired by resmax's paper_source_fetch.py.
"""

import logging
import re

logger = logging.getLogger(__name__)

try:
    from arxiv_to_prompt import process_latex_source  # type: ignore
    HAS_ARXIV_TO_PROMPT = True
except ImportError:
    process_latex_source = None  # type: ignore[assignment]
    HAS_ARXIV_TO_PROMPT = False


def fetch_arxiv_tex(arxiv_id: str) -> str | None:
    """Download and flatten arXiv LaTeX source. Returns full TeX text or None."""
    if not HAS_ARXIV_TO_PROMPT or not arxiv_id:
        return None
    clean_id = re.sub(r"v\d+$", "", arxiv_id.strip())
    try:
        text = process_latex_source(clean_id, keep_comments=False, remove_appendix_section=False)
        return text if text and text.strip() else None
    except Exception as e:
        logger.debug(f"arxiv-to-prompt failed for {clean_id}: {e}")
        return None


# ── Formula extraction ──────────────────────────────────────────

# Match equation environments: equation, align, gather, multline, eqnarray, split
_EQUATION_ENV_RE = re.compile(
    r"\\begin\{(equation|align|gather|multline|eqnarray|split)\*?\}"
    r"(.*?)"
    r"\\end\{\1\*?\}",
    re.DOTALL,
)

# Match inline/display math: $...$, $$...$$, \[...\], \(...\)
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

# Match \label{} inside equations
_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")


def extract_formulas(tex: str) -> list[dict]:
    """Extract all formulas from TeX source with labels.

    Returns list of {latex, label, env_type, is_numbered}.
    """
    if not tex:
        return []

    formulas = []
    seen = set()

    # 1. Named equation environments (highest priority)
    for m in _EQUATION_ENV_RE.finditer(tex):
        env_type = m.group(1)
        body = m.group(2).strip()
        if len(body) < 3 or body in seen:
            continue
        seen.add(body)

        label_m = _LABEL_RE.search(body)
        label = label_m.group(1) if label_m else ""
        clean_body = _LABEL_RE.sub("", body).strip()

        formulas.append({
            "latex": clean_body,
            "label": label,
            "env_type": env_type,
            "is_numbered": "*" not in m.group(0).split("{")[1],
        })

    # 2. Display math ($$...$$ or \[...\])
    for m in _DISPLAY_MATH_RE.finditer(tex):
        body = (m.group(1) or m.group(2) or "").strip()
        if len(body) < 3 or body in seen:
            continue
        seen.add(body)

        label_m = _LABEL_RE.search(body)
        label = label_m.group(1) if label_m else ""
        clean_body = _LABEL_RE.sub("", body).strip()

        formulas.append({
            "latex": clean_body,
            "label": label,
            "env_type": "display",
            "is_numbered": False,
        })

    return formulas


# ── Citation extraction ─────────────────────────────────────────

# \cite{key1, key2}, \citep{}, \citet{}, \citeauthor{}, etc.
_CITE_RE = re.compile(r"\\cite[tp]?\*?\{([^}]+)\}")


def extract_citations(tex: str) -> list[dict]:
    """Extract citation keys with surrounding context.

    Returns list of {keys: [str], context: str}.
    """
    if not tex:
        return []

    citations = []
    lines = tex.split("\n")
    full_text = tex

    for m in _CITE_RE.finditer(full_text):
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        if not keys:
            continue

        # Get context: ~100 chars before and after
        start = max(0, m.start() - 100)
        end = min(len(full_text), m.end() + 100)
        context = full_text[start:end].replace("\n", " ").strip()

        citations.append({"keys": keys, "context": context})

    return citations


def extract_unique_bibkeys(tex: str) -> list[str]:
    """Return deduplicated list of all cited bibkeys in order of first appearance."""
    if not tex:
        return []
    seen = set()
    result = []
    for m in _CITE_RE.finditer(tex):
        for key in m.group(1).split(","):
            key = key.strip()
            if key and key not in seen:
                seen.add(key)
                result.append(key)
    return result


# ── Figure extraction ───────────────────────────────────────────

_FIGURE_ENV_RE = re.compile(
    r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}",
    re.DOTALL,
)
_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_CAPTION_RE = re.compile(r"\\caption(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_figures(tex: str) -> list[dict]:
    """Extract figure info: image path, caption, label.

    Returns list of {image_path, caption, label, is_subfigure}.
    """
    if not tex:
        return []

    figures = []

    for m in _FIGURE_ENV_RE.finditer(tex):
        body = m.group(1)

        # Get all image paths in this figure env
        images = _INCLUDEGRAPHICS_RE.findall(body)

        # Get caption
        cap_m = _CAPTION_RE.search(body)
        caption = cap_m.group(1).strip() if cap_m else ""

        # Get label
        label_m = _LABEL_RE.search(body)
        label = label_m.group(1) if label_m else ""

        if images:
            for img_path in images:
                figures.append({
                    "image_path": img_path.strip(),
                    "caption": caption,
                    "label": label,
                    "is_subfigure": len(images) > 1,
                })
        elif caption:
            # Figure env with caption but no \includegraphics (tikz, etc.)
            figures.append({
                "image_path": "",
                "caption": caption,
                "label": label,
                "is_subfigure": False,
            })

    return figures


# ── URL extraction ──────────────────────────────────────────────

_URL_RE = re.compile(r"\\(?:url|href)\{([^}]+)\}")
_GITHUB_RE = re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def extract_urls(tex: str) -> dict:
    """Extract URLs from TeX: github repos, project pages, general URLs.

    Returns {github_urls: [...], project_urls: [...], all_urls: [...]}.
    """
    if not tex:
        return {"github_urls": [], "project_urls": [], "all_urls": []}

    all_urls = []
    for m in _URL_RE.finditer(tex):
        url = m.group(1).strip()
        if url:
            all_urls.append(url)

    # Also find bare GitHub URLs in text
    for m in _GITHUB_RE.finditer(tex):
        url = m.group(0).rstrip(").,;:}")
        if url not in all_urls:
            all_urls.append(url)

    github_urls = list(dict.fromkeys(
        u for u in all_urls if "github.com/" in u and "github.com/topics" not in u
    ))
    project_urls = list(dict.fromkeys(
        u for u in all_urls if ".github.io/" in u
    ))

    return {
        "github_urls": github_urls[:5],
        "project_urls": project_urls[:3],
        "all_urls": list(dict.fromkeys(all_urls))[:20],
    }


# ── Full extraction orchestrator ────────────────────────────────

def extract_all_from_tex(arxiv_id: str) -> dict | None:
    """One-call extraction: download TeX → extract everything.

    Returns None if TeX source unavailable.
    Returns dict with formulas, citations, bibkeys, figures, urls, tex_chars.
    """
    tex = fetch_arxiv_tex(arxiv_id)
    if not tex:
        return None

    return {
        "formulas": extract_formulas(tex),
        "citations": extract_citations(tex),
        "bibkeys": extract_unique_bibkeys(tex),
        "figures": extract_figures(tex),
        "urls": extract_urls(tex),
        "tex_chars": len(tex),
        "source": "arxiv_tex",
    }
