"""Centralized rate limiters and HTTP client factory for external APIs.

Importable anywhere:
    from backend.utils.api_clients import limiters, make_client

Rate limits are conservative defaults based on documented API policies.
Adjust via environment or override in tests.
"""

from __future__ import annotations

import httpx

from backend.config import settings
from backend.utils.rate_limiter import TokenBucketLimiter

# ── Rate limiter instances (singleton per process) ────────────────────
limiters: dict[str, TokenBucketLimiter] = {
    # arXiv API: official limit 3 req/s, stay under
    "arxiv": TokenBucketLimiter(rate=2.0, burst=3, name="arxiv"),
    # Semantic Scholar: with key ≈1 req/s, without key 100/5min ≈0.33/s
    "s2": TokenBucketLimiter(
        rate=0.9 if settings.s2_api_key else 0.3,
        burst=3,
        name="s2",
    ),
    # GitHub: unauthenticated 10 req/min, with token 5000 req/hr ≈1.4/s
    "github": TokenBucketLimiter(
        rate=1.0 if settings.github_token else 0.15,
        burst=3 if settings.github_token else 2,
        name="github",
    ),
    # Crossref: polite pool 50 req/s
    "crossref": TokenBucketLimiter(rate=5.0, burst=10, name="crossref"),
    # OpenAlex: 10 req/s
    "openalex": TokenBucketLimiter(rate=5.0, burst=10, name="openalex"),
    # HuggingFace: ~100 req/min
    "huggingface": TokenBucketLimiter(rate=1.5, burst=3, name="huggingface"),
    # Kimi VLM: conservative per-plan
    "kimi_vlm": TokenBucketLimiter(rate=0.5, burst=2, name="kimi_vlm"),
    # Unpaywall: polite use
    "unpaywall": TokenBucketLimiter(rate=2.0, burst=5, name="unpaywall"),
    # DBLP: no documented limit, be polite
    "dblp": TokenBucketLimiter(rate=2.0, burst=5, name="dblp"),
    # OpenReview: no documented limit
    "openreview": TokenBucketLimiter(rate=1.0, burst=3, name="openreview"),
}


def _github_headers() -> dict[str, str]:
    """Build GitHub API headers with optional token auth."""
    h: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
    if settings.github_token:
        h["Authorization"] = f"token {settings.github_token}"
    return h


def make_client(
    timeout: float = 30.0,
    follow_redirects: bool = True,
) -> httpx.AsyncClient:
    """Create a shared-config httpx.AsyncClient.

    Caller is responsible for `async with make_client() as c:` lifecycle.
    """
    return httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=follow_redirects,
        headers={"User-Agent": "ResearchFlow/0.2"},
    )
