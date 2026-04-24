"""Venue index service — build, store, and query pre-crawled conference metadata.

Flow:
  1. build_venue_index(venues, years) — fetch + parse + store to DB
  2. lookup_paper(title, arxiv_id, doi) — find in local index (zero API cost)
  3. enrich_service calls lookup_paper before any API calls

Ported from resmax's accepted_index_builder, adapted for PostgreSQL storage.
"""

import json
import logging
import re
from pathlib import Path
from uuid import UUID

from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Paths
_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config"
_REGISTRY_PATH = _CONFIG_DIR / "source_registry.json"
_FIXTURES_DIR = _CONFIG_DIR / "fixtures"


# ── Registry loading ────────────────────────────────────────────

def load_registry(venues: list[str] | None = None, years: list[int] | None = None) -> list[dict]:
    """Load source registry, optionally filtered by venue/year."""
    data = json.loads(_REGISTRY_PATH.read_text())
    items = data.get("conference_years", [])
    if venues:
        venues_lower = {v.lower() for v in venues}
        items = [i for i in items if i["venue"].lower() in venues_lower]
    if years:
        items = [i for i in items if i["year"] in years]
    return [i for i in items if i.get("status") == "active"]


# ── Fetcher dispatch ────────────────────────────────────────────

def _fetch_source(source: dict, venue: str, year: int) -> str | list | dict:
    """Fetch raw data for a source config. Returns text/json."""
    from .fetchers import (
        fetch_text, fetch_openreview_api_v2, fetch_aaai_ojs_all_issues,
        fetch_acmmm_vue_accepted_chunk, fetch_openalex_works,
        fetch_s2_bulk_search, fetch_hf_daily_papers, fetch_anthropic_research,
    )
    from .models import SourceConfig

    kind = source["kind"]
    url = source["url"]
    parser_args = source.get("parser_args", "")

    sc = SourceConfig(kind=kind, url=url, parser=source.get("parser", ""),
                      parser_args=parser_args)

    if kind == "virtual_conference_json":
        return json.loads(fetch_text(sc, _FIXTURES_DIR))

    elif kind in ("cvpr_openaccess_html", "iclr_virtual_html", "iclr_proceedings_html",
                   "neurips_virtual_html", "kdd_html", "acmmm_html", "jmlr_html",
                   "kesen_siggraph_html", "acl_anthology_html"):
        return fetch_text(sc, _FIXTURES_DIR)

    elif kind == "openreview_api_v2":
        # parser_args contains group and venue_prefixes
        parts = (parser_args or "").split(",")
        group = parts[0] if parts else f"{venue}.cc/{year}/Conference"
        prefixes = parts[1:] if len(parts) > 1 else [venue]
        return fetch_openreview_api_v2(group, prefixes)

    elif kind == "aaai_ojs_multi_issue":
        year_tag = parser_args or f"AAAI-{year % 100:02d}"
        return fetch_aaai_ojs_all_issues(url, year_tag)

    elif kind == "acmmm_vue_accepted":
        chunk_name = parser_args or "chunk-240a60f6"
        return fetch_acmmm_vue_accepted_chunk(url, chunk_name)

    elif kind == "openalex_api":
        source_id = url  # e.g., "S199944782"
        oa_year = int(parser_args) if parser_args and parser_args.isdigit() else year
        return fetch_openalex_works(source_id, oa_year)

    elif kind == "s2_bulk_search":
        args_dict = {}
        for kv in (parser_args or "").split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                args_dict[k.strip()] = v.strip()
        s2_year = int(args_dict.get("year", year))
        min_cite = int(args_dict.get("minCitationCount", 50))
        if args_dict.get("minCitationCount") == "auto":
            min_cite = 50 if s2_year <= 2024 else (20 if s2_year == 2025 else 5)
        return fetch_s2_bulk_search(s2_year, min_cite)

    elif kind == "hf_daily_papers_api":
        args_dict = {}
        for kv in (parser_args or "").split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                args_dict[k.strip()] = v.strip()
        hf_year = int(args_dict.get("year", year))
        min_up = int(args_dict.get("minUpvotes", 10))
        return fetch_hf_daily_papers(hf_year, min_up)

    elif kind == "anthropic_sitemap_cached":
        return fetch_anthropic_research(year)

    else:
        logger.warning(f"Unknown source kind: {kind}")
        return ""


# ── Parser dispatch ─────────────────────────────────────────────

def _parse_source(parser_name: str, raw_data, venue: str, year: int, url: str) -> list[dict]:
    """Parse raw data into list of paper records."""
    from . import parsers

    conf_year = f"{venue}_{year}"
    parser_func = getattr(parsers, f"parse_{parser_name}", None)
    if not parser_func:
        logger.warning(f"Unknown parser: {parser_name}")
        return []

    try:
        from .models import AcceptedPaperRecord
        records = parser_func(raw_data, venue=venue, year=year, conf_year=conf_year)
        # Convert AcceptedPaperRecord to dicts
        result = []
        for r in records:
            if isinstance(r, AcceptedPaperRecord):
                d = {k: v for k, v in r.__dict__.items() if not k.startswith("_") and k != "extras"}
                d.update(r.extras or {})
                result.append(d)
            elif isinstance(r, dict):
                result.append(r)
        return result
    except Exception as e:
        logger.error(f"Parser {parser_name} failed for {venue} {year}: {e}")
        return []


# ── Build index ─────────────────────────────────────────────────

async def build_venue_index(
    session: AsyncSession,
    venues: list[str] | None = None,
    years: list[int] | None = None,
) -> dict:
    """Fetch + parse + store venue papers to DB.

    Returns summary: {venue_year: count, total: N, errors: [...]}
    """
    registry = load_registry(venues, years)
    if not registry:
        return {"error": "No matching entries in registry", "total": 0}

    summary = {"total": 0, "errors": [], "details": {}}

    for entry in registry:
        venue = entry["venue"]
        year = entry["year"]
        conf_year = entry.get("conf_year", f"{venue}_{year}")
        primary = entry.get("primary_source", {})

        logger.info(f"Building index for {conf_year} from {primary.get('kind')}")

        try:
            # Fetch
            raw_data = _fetch_source(primary, venue, year)

            # Parse
            parser_name = primary.get("parser", primary.get("kind", ""))
            records = _parse_source(parser_name, raw_data, venue, year, primary.get("url", ""))

            if not records:
                summary["errors"].append(f"{conf_year}: 0 records parsed")
                continue

            # Also parse auxiliary sources and merge
            for aux in entry.get("auxiliary_sources", []):
                try:
                    aux_data = _fetch_source(aux, venue, year)
                    aux_parser = aux.get("parser", aux.get("kind", ""))
                    aux_records = _parse_source(aux_parser, aux_data, venue, year, aux.get("url", ""))
                    if aux_records:
                        records = _merge_records(records, aux_records)
                except Exception as e:
                    logger.warning(f"Auxiliary source failed for {conf_year}: {e}")

            # Store to DB
            stored = await _store_records(session, records, conf_year)
            summary["details"][conf_year] = stored
            summary["total"] += stored
            logger.info(f"  {conf_year}: stored {stored} papers")

        except Exception as e:
            logger.error(f"Failed to build index for {conf_year}: {e}")
            summary["errors"].append(f"{conf_year}: {str(e)[:100]}")

    await session.flush()
    return summary


def _merge_records(primary: list[dict], auxiliary: list[dict]) -> list[dict]:
    """Merge auxiliary records into primary by title matching."""
    from .normalize import normalize_title

    # Build lookup from auxiliary
    aux_by_title = {}
    for r in auxiliary:
        nt = normalize_title(r.get("title", ""))
        if nt:
            aux_by_title[nt] = r

    # Merge: fill empty fields from auxiliary
    for r in primary:
        nt = normalize_title(r.get("title", ""))
        aux = aux_by_title.get(nt)
        if aux:
            for key, val in aux.items():
                if val and not r.get(key):
                    r[key] = val

    return primary


def _normalize_acceptance(raw_decision: str) -> str:
    """Normalize decision string → standard acceptance_type.

    Handles format differences across years:
      "Accept (Oral)" → "oral"
      "Accept: Poster (Highlight)" → "highlight"
      "Accept" → "poster"
    """
    if not raw_decision:
        return ""
    d = raw_decision.lower().strip()
    if "oral" in d:
        return "oral"
    if "highlight" in d or "spotlight" in d:
        return "highlight"
    if "poster" in d or "accept" in d:
        return "poster"
    if "workshop" in d:
        return "workshop"
    if "journal" in d:
        return "journal"
    return raw_decision[:50]


def _build_authors_json(raw_authors) -> list[dict]:
    """Convert various author formats to [{fullname, institution}].

    Handles:
      - VC JSON: [{"fullname": "X", "institution": "Y"}]
      - String: "A; B; C" or "A, B, C"
      - List of strings: ["A", "B"]
    """
    if not raw_authors:
        return []

    if isinstance(raw_authors, list):
        result = []
        for a in raw_authors:
            if isinstance(a, dict):
                result.append({
                    "fullname": a.get("fullname") or a.get("name", ""),
                    "institution": a.get("institution", ""),
                })
            elif isinstance(a, str):
                result.append({"fullname": a.strip(), "institution": ""})
        return result

    if isinstance(raw_authors, str):
        sep = ";" if ";" in raw_authors else ","
        return [{"fullname": n.strip(), "institution": ""}
                for n in raw_authors.split(sep) if n.strip()]

    return []


# Fields that go into the main columns (everything else → extra_data)
_MAIN_FIELDS = {
    "title", "authors", "venue", "year", "conf_year",
    "arxiv_id", "doi", "openreview_forum_id", "abstract_raw", "abstract",
    "paper_link", "pdf_url", "code_url", "decision", "acceptance_type",
    "keywords_raw", "keywords", "source_type", "source_url", "topic",
    "project_url", "name",
}


async def _store_records(session: AsyncSession, records: list[dict], conf_year: str) -> int:
    """Upsert records into venue_papers table with dedup.

    Dedup chain: arxiv_id > doi > openreview_forum_id > title_normalized+first_author
    Match → MERGE (fill empty fields). No match → INSERT.
    """
    from .normalize import normalize_title

    stored = 0
    merged = 0

    for r in records:
        title = (r.get("title") or r.get("name") or "").strip()
        if not title:
            continue

        arxiv_id = re.sub(r"v\d+$", "", (r.get("arxiv_id") or "").strip())
        doi = (r.get("doi") or "").strip()
        orf_id = (r.get("openreview_forum_id") or "").strip()
        title_norm = normalize_title(title)
        authors_json = _build_authors_json(r.get("authors"))
        raw_decision = (r.get("decision") or "")
        acc_type = (r.get("acceptance_type") or "") or _normalize_acceptance(raw_decision)
        abstract = (r.get("abstract_raw") or r.get("abstract") or "")[:5000]

        # Extra data: everything not in main columns
        extra = {k: v for k, v in r.items() if k not in _MAIN_FIELDS and v}

        # ── Dedup check ──
        existing_id = None
        if arxiv_id:
            row = (await session.execute(
                text("SELECT id FROM venue_papers WHERE arxiv_id = :v AND arxiv_id != '' LIMIT 1"), {"v": arxiv_id}
            )).first()
            if row: existing_id = row[0]

        if not existing_id and doi:
            row = (await session.execute(
                text("SELECT id FROM venue_papers WHERE doi = :v AND doi != '' LIMIT 1"), {"v": doi}
            )).first()
            if row: existing_id = row[0]

        if not existing_id and orf_id:
            row = (await session.execute(
                text("SELECT id FROM venue_papers WHERE openreview_forum_id = :v AND openreview_forum_id != '' LIMIT 1"), {"v": orf_id}
            )).first()
            if row: existing_id = row[0]

        if not existing_id and len(title_norm) > 10:
            row = (await session.execute(
                text("SELECT id, authors FROM venue_papers WHERE title_normalized = :v LIMIT 1"), {"v": title_norm}
            )).first()
            if row:
                # Cross-check first author
                first_author = (authors_json[0]["fullname"].split()[-1].lower() if authors_json else "")
                existing_str = json.dumps(row[1] or []).lower()
                if not first_author or first_author in existing_str:
                    existing_id = row[0]

        if existing_id:
            # MERGE: fill empty fields
            updates = []
            params: dict = {"eid": existing_id}
            for col, val in [
                ("abstract", abstract),
                ("arxiv_id", arxiv_id),
                ("doi", doi),
                ("openreview_forum_id", orf_id),
                ("pdf_url", (r.get("pdf_url") or "")[:500]),
                ("code_url", (r.get("code_url") or "")[:500]),
                ("acceptance_type", acc_type[:50]),
                ("decision", raw_decision[:100]),
                ("keywords", (r.get("keywords_raw") or r.get("keywords") or "")[:1000]),
                ("topic", (r.get("topic") or "")[:200]),
            ]:
                if val:
                    updates.append(f"{col} = CASE WHEN {col} IS NULL OR {col} = '' THEN :{col} ELSE {col} END")
                    params[col] = val
            # Always merge authors if current is empty
            if authors_json:
                updates.append("authors = CASE WHEN authors IS NULL OR authors = CAST('[]' AS jsonb) THEN CAST(:authors AS jsonb) ELSE authors END")
                params["authors"] = json.dumps(authors_json, ensure_ascii=False)
            if updates:
                updates.append("updated_at = now()")
                await session.execute(text(f"UPDATE venue_papers SET {', '.join(updates)} WHERE id = :eid"), params)
            merged += 1
        else:
            await session.execute(
                text("""
                    INSERT INTO venue_papers (
                        title, title_normalized, authors, venue, year, conf_year,
                        arxiv_id, doi, openreview_forum_id,
                        abstract, paper_link, pdf_url, code_url, project_url,
                        decision, acceptance_type, keywords, topic,
                        source_type, source_url, extra_data
                    ) VALUES (
                        :title, :title_norm, CAST(:authors AS jsonb), :venue, :year, :conf_year,
                        :arxiv_id, :doi, :orf_id,
                        :abstract, :paper_link, :pdf_url, :code_url, :project_url,
                        :decision, :acceptance_type, :keywords, :topic,
                        :source_type, :source_url, CAST(:extra_data AS jsonb)
                    )
                """),
                {
                    "title": title[:500],
                    "title_norm": title_norm,
                    "authors": json.dumps(authors_json, ensure_ascii=False),
                    "venue": (r.get("venue") or "")[:50],
                    "year": r.get("year", 0),
                    "conf_year": conf_year[:50],
                    "arxiv_id": arxiv_id[:30],
                    "doi": doi[:200],
                    "orf_id": orf_id[:50],
                    "abstract": abstract,
                    "paper_link": (r.get("paper_link") or "")[:500],
                    "pdf_url": (r.get("pdf_url") or "")[:500],
                    "code_url": (r.get("code_url") or "")[:500],
                    "project_url": (r.get("project_url") or "")[:500],
                    "decision": raw_decision[:100],
                    "acceptance_type": acc_type[:50],
                    "keywords": (r.get("keywords_raw") or r.get("keywords") or "")[:1000],
                    "topic": (r.get("topic") or "")[:200],
                    "source_type": (r.get("source_type") or "")[:50],
                    "source_url": (r.get("source_url") or "")[:500],
                    "extra_data": json.dumps(extra, ensure_ascii=False, default=str)[:5000],
                },
            )
            stored += 1

    logger.info(f"  {conf_year}: {stored} new + {merged} merged")
    return stored + merged


# ── Lookup ──────────────────────────────────────────────────────

async def lookup_paper(
    session: AsyncSession,
    *,
    title: str = "",
    arxiv_id: str = "",
    doi: str = "",
) -> dict | None:
    """Find a paper in the venue index. Returns enrichment dict or None.

    Priority: arxiv_id exact → doi exact → title normalized match.
    """
    # 1. arxiv_id exact match (fastest, most reliable)
    if arxiv_id:
        clean_id = re.sub(r"v\d+$", "", arxiv_id.strip())
        row = (await session.execute(
            text("SELECT * FROM venue_papers WHERE arxiv_id = :aid LIMIT 1"),
            {"aid": clean_id},
        )).mappings().first()
        if row:
            return _row_to_enrichment(row)

    # 2. doi exact match
    if doi:
        row = (await session.execute(
            text("SELECT * FROM venue_papers WHERE doi = :doi LIMIT 1"),
            {"doi": doi.strip()},
        )).mappings().first()
        if row:
            return _row_to_enrichment(row)

    # 3. title normalized match
    if title:
        from .normalize import normalize_title
        norm = normalize_title(title)
        if len(norm) > 10:  # skip too-short titles
            row = (await session.execute(
                text("SELECT * FROM venue_papers WHERE title_normalized = :tn LIMIT 1"),
                {"tn": norm},
            )).mappings().first()
            if row:
                return _row_to_enrichment(row)

    return None


def _row_to_enrichment(row) -> dict:
    """Convert a venue_papers row to an enrichment dict for enrich_service."""
    result = {
        "source": "venue_index",
        "venue": row["venue"],
        "year": row["year"],
        "conf_year": row["conf_year"],
        "decision": row["decision"],
        "acceptance_type": row["acceptance_type"],
    }
    if row.get("abstract"):
        result["abstract"] = row["abstract"]
    if row.get("authors"):
        # authors is JSONB: [{fullname, institution}]
        authors = row["authors"]
        if isinstance(authors, list):
            result["authors"] = authors
            result["authors_str"] = "; ".join(a.get("fullname", "") for a in authors if a.get("fullname"))
        else:
            result["authors"] = authors
    if row.get("arxiv_id"):
        result["arxiv_id"] = row["arxiv_id"]
    if row.get("doi"):
        result["doi"] = row["doi"]
    if row.get("paper_link"):
        result["paper_link"] = row["paper_link"]
    if row.get("pdf_url"):
        result["pdf_url"] = row["pdf_url"]
    if row.get("code_url"):
        result["code_url"] = row["code_url"]
    if row.get("keywords"):
        result["keywords"] = row["keywords"]
    if row.get("openreview_forum_id"):
        result["openreview_forum_id"] = row["openreview_forum_id"]
    return result


# ── Stats ───────────────────────────────────────────────────────

async def get_index_stats(session: AsyncSession) -> dict:
    """Get venue index coverage stats."""
    rows = (await session.execute(
        text("""
            SELECT conf_year, COUNT(*) as cnt,
                   SUM(CASE WHEN abstract != '' AND abstract IS NOT NULL THEN 1 ELSE 0 END) as with_abstract,
                   SUM(CASE WHEN arxiv_id != '' AND arxiv_id IS NOT NULL THEN 1 ELSE 0 END) as with_arxiv
            FROM venue_papers
            GROUP BY conf_year
            ORDER BY conf_year
        """)
    )).fetchall()

    stats = {}
    total = 0
    for r in rows:
        stats[r.conf_year] = {
            "papers": r.cnt,
            "with_abstract": r.with_abstract,
            "with_arxiv": r.with_arxiv,
        }
        total += r.cnt

    return {"total": total, "venues": stats}
