"""Filename and string sanitization utilities."""

import re


def sanitize_filename(name: str) -> str:
    """Convert a string into a safe filename."""
    sanitized = re.sub(r"[^\w\s-]", "", name)
    sanitized = re.sub(r"\s+", "_", sanitized.strip())
    return sanitized


def parse_venue_year(venue_str: str) -> tuple[str, int | None]:
    """Parse combined venue string like 'CVPR 2025' into (venue, year).

    Returns (venue_name, year) or (venue_str, None) if no year found.
    """
    if not venue_str:
        return "", None

    # Try to extract year from end
    match = re.match(r"^(.+?)\s+(\d{4})$", venue_str.strip())
    if match:
        return match.group(1).strip(), int(match.group(2))

    # Try arXiv format: "arXiv 2025"
    match = re.match(r"^(arXiv)\s+(\d{4})$", venue_str.strip(), re.IGNORECASE)
    if match:
        return "arXiv", int(match.group(2))

    return venue_str.strip(), None


def extract_arxiv_id(url: str) -> str | None:
    """Extract arXiv ID from a URL like https://arxiv.org/abs/2410.05260."""
    if not url:
        return None
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", url)
    return match.group(1) if match else None
