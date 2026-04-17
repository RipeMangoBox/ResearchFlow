"""Parse YAML frontmatter from Markdown files."""

import re
from typing import Any

import yaml

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from a Markdown string.

    Returns (frontmatter_dict, body_text).
    If no frontmatter found, returns ({}, full_text).
    """
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    fm_raw = match.group(1)
    body = text[match.end():]
    try:
        fm = yaml.safe_load(fm_raw) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, body


def render_frontmatter(data: dict[str, Any]) -> str:
    """Render a dict as YAML frontmatter block."""
    # Use default_flow_style=False for readable output
    yaml_str = yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=200,
    )
    return f"---\n{yaml_str}---\n"


def extract_sections(body: str) -> dict[str, str]:
    """Extract Part I/II/III and core intuition from analysis body."""
    sections = {}
    # Match Part I, Part II, Part III sections (Chinese or English headers)
    part_patterns = [
        (r"(?:Part I[：:].*?)\n(.*?)(?=\nPart II|$)", "part_i"),
        (r"(?:Part II[：:].*?)\n(.*?)(?=\nPart III|$)", "part_ii"),
        (r"(?:Part III[：:].*?)\n(.*?)(?=\n#{1,3}\s|$)", "part_iii"),
    ]
    for pattern, key in part_patterns:
        match = re.search(pattern, body, re.DOTALL)
        if match:
            sections[key] = match.group(1).strip()

    # Extract core intuition / "Aha!" moment
    aha_match = re.search(
        r"(?:核心直觉|The \"Aha!\" Moment|核心直覺)\s*\n(.*?)(?=\n---|\n#{1,2}\s|$)",
        body, re.DOTALL,
    )
    if aha_match:
        sections["core_intuition"] = aha_match.group(1).strip()

    return sections


def sanitize_title(title: str) -> str:
    """Convert a paper title into a filesystem-safe string."""
    # Remove special chars, replace spaces with underscores
    sanitized = re.sub(r"[^\w\s-]", "", title)
    sanitized = re.sub(r"\s+", "_", sanitized.strip())
    return sanitized
