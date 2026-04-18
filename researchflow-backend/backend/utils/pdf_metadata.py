"""Extract metadata directly from PDF text when no arxiv_id or external API available.

Parses first page text for:
  - Title (largest font or first prominent text)
  - Authors (names with affiliations)
  - Venue/acceptance status ("Published at ICLR 2026")
  - Abstract
  - Emails, URLs
"""

import re
import logging
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


def extract_metadata_from_pdf(pdf_path: str | Path) -> dict:
    """Extract metadata from PDF first 1-2 pages.

    Returns dict with: title, authors, affiliations, abstract,
    venue, acceptance_status, acceptance_type, emails, urls
    """
    doc = fitz.open(str(pdf_path))
    if len(doc) == 0:
        doc.close()
        return {}

    result = {}

    # ── Title: largest font on first page ────────────────────
    page0 = doc[0]
    blocks = page0.get_text("dict")["blocks"]

    # Find the largest font span (usually the title)
    max_font_size = 0
    title_spans = []

    # Skip patterns that are NOT titles
    skip_patterns = re.compile(
        r'arXiv:|^\d{4}\.\d+|^\[cs\.|^\[math\.|Published as|Under review|Anonymous',
        re.IGNORECASE
    )

    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size", 0)
                text = span.get("text", "").strip()
                if not text or len(text) < 5:
                    continue
                if skip_patterns.search(text):
                    continue
                if size > max_font_size:
                    max_font_size = size
                    title_spans = [(span, block["bbox"])]
                elif size == max_font_size and text:
                    title_spans.append((span, block["bbox"]))

    if title_spans:
        # Collect all spans at the largest font size, merge into title
        title_parts = []
        for span, _ in title_spans:
            text = span.get("text", "").strip()
            if text and len(text) > 2:
                title_parts.append(text)
        title = " ".join(title_parts).strip()
        # Clean up common artifacts
        title = re.sub(r'\s+', ' ', title)
        title = title.strip("*†‡§¶")
        if len(title) > 10:
            result["title"] = title

    # ── Full first page text for regex parsing ───────────────
    page0_text = page0.get_text()

    # ── Acceptance / venue from first page ───────────────────
    from backend.services.enrich_service import _parse_acceptance_from_comment
    acceptance = _parse_acceptance_from_comment(page0_text[:500])
    if acceptance:
        result["venue"] = acceptance.get("venue", "")
        result["acceptance_status"] = acceptance["acceptance_status"]
        result["acceptance_type"] = acceptance.get("acceptance_type", "")

    # ── Abstract ─────────────────────────────────────────────
    abstract_match = re.search(
        r'(?:Abstract|ABSTRACT)[.\s:—\-]*\n?([\s\S]{50,2000}?)(?=\n\s*\n\s*\d+[\s.]|\n\s*(?:Introduction|1[\s.]))',
        page0_text,
        re.IGNORECASE
    )
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        abstract = re.sub(r'\s+', ' ', abstract)
        result["abstract"] = abstract

    # ── Authors ──────────────────────────────────────────────
    # Authors are usually between title and abstract, in smaller font
    if title_spans:
        title_bottom = max(bbox[3] for _, bbox in title_spans)
        abstract_top = 9999

        # Find where abstract starts
        for block in blocks:
            if block["type"] != 0:
                continue
            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
            if re.search(r'(?:Abstract|ABSTRACT)', block_text):
                abstract_top = block["bbox"][1]
                break

        # Collect text blocks between title and abstract
        author_texts = []
        for block in blocks:
            if block["type"] != 0:
                continue
            y0 = block["bbox"][1]
            if title_bottom < y0 < abstract_top:
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "")
                block_text = block_text.strip()
                if block_text and len(block_text) > 3:
                    author_texts.append(block_text)

        if author_texts:
            author_block = " ".join(author_texts)
            result["authors_raw"] = author_block

            # Try to parse individual author names
            authors = _parse_author_names(author_block)
            if authors:
                result["authors"] = authors

    # ── Emails ───────────────────────────────────────────────
    emails = re.findall(r'[\w.+-]+@[\w.-]+\.\w+', page0_text)
    if emails:
        result["emails"] = list(set(emails))

    # ── URLs (project pages, code repos) ─────────────────────
    urls = re.findall(r'https?://[^\s\)\"\']+', page0_text)
    project_urls = [u for u in urls if any(k in u.lower() for k in
                    ["github.com", "gitlab.com", "huggingface.co", ".github.io"])]
    if project_urls:
        result["project_urls"] = project_urls

    doc.close()
    return result


def _parse_author_names(text: str) -> list[dict]:
    """Parse author names from the author block text."""
    # Remove footnote markers, affiliations in parentheses
    text = re.sub(r'[*†‡§¶∗]+', '', text)
    text = re.sub(r'\d+', '', text)  # Remove superscript numbers

    # Split by common separators
    # "John Smith, Jane Doe, Bob Wilson"
    # "John Smith  Jane Doe  Bob Wilson"
    parts = re.split(r'[,;]\s*|\s{3,}', text)

    authors = []
    for part in parts:
        name = part.strip()
        # Filter out non-name text
        if not name or len(name) < 3 or len(name) > 50:
            continue
        if any(kw in name.lower() for kw in
               ["university", "institute", "lab", "department", "school",
                "research", "corporation", "inc.", "ltd.", "@", "http"]):
            continue
        # Should contain at least 2 words (first + last name)
        words = name.split()
        if len(words) < 2 or len(words) > 5:
            continue
        # All words should be capitalized (names)
        if all(w[0].isupper() for w in words if w):
            authors.append({"name": name})

    return authors if authors else []
