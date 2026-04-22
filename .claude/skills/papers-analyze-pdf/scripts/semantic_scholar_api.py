#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


API_BASE = "https://api.semanticscholar.org/graph/v1"
DEFAULT_TIMEOUT = 20


class SemanticScholarError(RuntimeError):
    pass


@dataclass
class ApiResult:
    ok: bool
    status: str
    data: Optional[Dict[str, Any]] = None
    error: str = ""
    http_status: int = 0


def normalize_title(text: str) -> str:
    return "".join(ch.lower() for ch in (text or "") if ch.isalnum())


def build_headers(api_key: str = "") -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "ResearchFlow/semantic-scholar-api",
    }
    key = api_key.strip() or os.environ.get("RF_SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        headers["x-api-key"] = key
    return headers


def http_get_json(url: str, api_key: str = "", retries: int = 2, backoff_sec: float = 1.5) -> ApiResult:
    headers = build_headers(api_key)
    request = urllib.request.Request(url, headers=headers)
    last_error = ""
    last_status = 0
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return ApiResult(ok=True, status="ok", data=payload, http_status=response.status)
        except urllib.error.HTTPError as exc:
            last_status = exc.code
            body = exc.read().decode("utf-8", errors="replace").strip()
            try:
                parsed = json.loads(body) if body else {}
            except json.JSONDecodeError:
                parsed = {}
            message = parsed.get("message") or parsed.get("error") or body or str(exc)
            if exc.code == 429:
                last_error = message
                if attempt < retries:
                    time.sleep(backoff_sec * (attempt + 1))
                    continue
                return ApiResult(ok=False, status="rate_limited", error=message, http_status=exc.code)
            return ApiResult(ok=False, status="http_error", error=message, http_status=exc.code)
        except urllib.error.URLError as exc:
            last_error = str(exc.reason)
            if attempt < retries:
                time.sleep(backoff_sec * (attempt + 1))
                continue
            return ApiResult(ok=False, status="network_error", error=last_error, http_status=last_status)
    return ApiResult(ok=False, status="unknown_error", error=last_error, http_status=last_status)


def choose_best_candidate(candidates: List[Dict[str, Any]], title: str, year: str = "") -> Optional[Dict[str, Any]]:
    target = normalize_title(title)
    target_year = (year or "").strip()
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for candidate in candidates:
        cand_title = candidate.get("title", "")
        cand_norm = normalize_title(cand_title)
        score = 0
        if cand_norm == target:
            score += 100
        elif target and target in cand_norm:
            score += 60
        elif cand_norm and cand_norm in target:
            score += 40
        candidate_year = str(candidate.get("year") or "").strip()
        if target_year and candidate_year == target_year:
            score += 20
        if candidate.get("externalIds", {}).get("ArXiv"):
            score += 5
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        return None
    return scored[0][1]


def search_paper(title: str, year: str = "", api_key: str = "", limit: int = 5) -> ApiResult:
    query = urllib.parse.quote(title)
    fields = ",".join(
        [
            "paperId",
            "title",
            "year",
            "authors",
            "url",
            "externalIds",
            "citationCount",
            "influentialCitationCount",
            "referenceCount",
            "openAccessPdf",
        ]
    )
    url = f"{API_BASE}/paper/search?query={query}&limit={limit}&fields={fields}"
    result = http_get_json(url, api_key=api_key)
    if not result.ok:
        return result
    candidates = result.data.get("data", []) if result.data else []
    best = choose_best_candidate(candidates, title=title, year=year)
    if not best:
        return ApiResult(ok=False, status="not_found", error=f"No good Semantic Scholar match for: {title}")
    return ApiResult(ok=True, status="ok", data={"match": best, "candidates": candidates}, http_status=result.http_status)


def fetch_paper_details(paper_id: str, api_key: str = "") -> ApiResult:
    fields = ",".join(
        [
            "paperId",
            "title",
            "year",
            "authors",
            "url",
            "externalIds",
            "citationCount",
            "influentialCitationCount",
            "referenceCount",
            "openAccessPdf",
        ]
    )
    url = f"{API_BASE}/paper/{urllib.parse.quote(paper_id)}?fields={fields}"
    return http_get_json(url, api_key=api_key)


def _extract_edge_paper(item: Dict[str, Any], nested_key: str) -> Dict[str, Any]:
    node = item.get(nested_key) or item
    return {
        "paperId": node.get("paperId", ""),
        "title": node.get("title", ""),
        "year": node.get("year", ""),
    }


def fetch_references(paper_id: str, api_key: str = "", limit: int = 20) -> ApiResult:
    fields = "paperId,title,year"
    url = f"{API_BASE}/paper/{urllib.parse.quote(paper_id)}/references?fields={fields}&limit={limit}"
    result = http_get_json(url, api_key=api_key)
    if not result.ok:
        return result
    data = result.data.get("data", []) if result.data else []
    refs = [_extract_edge_paper(item, "citedPaper") for item in data]
    return ApiResult(ok=True, status="ok", data={"references": refs}, http_status=result.http_status)


def fetch_citations(paper_id: str, api_key: str = "", limit: int = 20) -> ApiResult:
    fields = "paperId,title,year"
    url = f"{API_BASE}/paper/{urllib.parse.quote(paper_id)}/citations?fields={fields}&limit={limit}"
    result = http_get_json(url, api_key=api_key)
    if not result.ok:
        return result
    data = result.data.get("data", []) if result.data else []
    citations = [_extract_edge_paper(item, "citingPaper") for item in data]
    return ApiResult(ok=True, status="ok", data={"citations": citations}, http_status=result.http_status)


def enrich_paper(title: str, year: str = "", api_key: str = "", limit: int = 20) -> ApiResult:
    search = search_paper(title=title, year=year, api_key=api_key)
    if not search.ok:
        return search
    match = search.data["match"]
    paper_id = match.get("paperId", "")
    details = fetch_paper_details(paper_id=paper_id, api_key=api_key)
    if not details.ok:
        return details
    refs = fetch_references(paper_id=paper_id, api_key=api_key, limit=limit)
    cits = fetch_citations(paper_id=paper_id, api_key=api_key, limit=limit)
    aggregate = {
        "search_match": match,
        "paper": details.data,
        "references": refs.data.get("references", []) if refs.ok and refs.data else [],
        "citations": cits.data.get("citations", []) if cits.ok and cits.data else [],
        "references_status": refs.status,
        "references_error": refs.error,
        "citations_status": cits.status,
        "citations_error": cits.error,
    }
    return ApiResult(ok=True, status="ok", data=aggregate, http_status=details.http_status)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query Semantic Scholar Graph API for one paper by title.")
    parser.add_argument("--title", required=True, help="Paper title")
    parser.add_argument("--year", default="", help="Publication year")
    parser.add_argument("--limit", type=int, default=20, help="Max references/citations to fetch")
    parser.add_argument("--api-key", default="", help="Semantic Scholar API key override")
    args = parser.parse_args()
    result = enrich_paper(title=args.title, year=args.year, api_key=args.api_key, limit=args.limit)
    output = {
        "ok": result.ok,
        "status": result.status,
        "http_status": result.http_status,
        "error": result.error,
        "data": result.data,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
