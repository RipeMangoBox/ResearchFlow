"""arXiv OAI-PMH client for bulk metadata harvesting.

Data source hierarchy (per arXiv official guidelines):
  A. OAI-PMH — bulk metadata sync (title, authors, abstract, categories, arxiv_id)
     Base URL: https://oaipmh.arxiv.org/oai (updated March 2025)
     Rate: polite, no hard limit (one request at a time, metadata updated daily)

  B. arXiv API — real-time keyword/title/category search (max 3s/req, max 2000/page)
     Base URL: https://export.arxiv.org/api/query

  C. Amazon S3 / Kaggle — full PDF corpus download (not per-paper scraping)

This module implements (A) for venue_papers arxiv_id matching.
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

OAI_BASE = "https://oaipmh.arxiv.org/oai"
OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "arx": "http://arxiv.org/OAI/arXivRaw/",
}

# arXiv categories relevant to ML/CV/NLP conferences
ML_CATEGORIES = {
    "cs.CV", "cs.LG", "cs.CL", "cs.AI", "cs.RO", "cs.MM", "cs.SD",
    "cs.NE", "cs.IR", "cs.GR", "cs.HC", "cs.MA", "cs.SE",
    "stat.ML", "eess.IV", "eess.AS", "eess.SP",
}


@dataclass
class ArxivRecord:
    arxiv_id: str = ""
    title: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    doi: str = ""
    license: str = ""


async def harvest_oaipmh(
    *,
    from_date: str = "",
    until_date: str = "",
    arxiv_set: str = "cs",
    max_records: int = 0,
    timeout: int = 60,
) -> list[ArxivRecord]:
    """Harvest metadata via OAI-PMH ListRecords.

    Args:
        from_date: Start date (YYYY-MM-DD). Records created/updated after this.
        until_date: End date (YYYY-MM-DD).
        arxiv_set: OAI-PMH set filter (e.g. "cs" for all CS, "cs.CV" for CV only).
        max_records: Stop after this many records (0 = unlimited).
        timeout: HTTP timeout per request.

    Returns:
        List of ArxivRecord with parsed metadata.
    """
    records: list[ArxivRecord] = []
    resumption_token: str | None = None
    page = 0

    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            if resumption_token:
                params = {"verb": "ListRecords", "resumptionToken": resumption_token}
            else:
                params = {
                    "verb": "ListRecords",
                    "metadataPrefix": "arXivRaw",
                    "set": arxiv_set,
                }
                if from_date:
                    params["from"] = from_date
                if until_date:
                    params["until"] = until_date

            try:
                resp = await client.get(OAI_BASE, params=params)
                if resp.status_code == 503:
                    # Retry-After header
                    wait = int(resp.headers.get("Retry-After", "30"))
                    logger.info(f"[oaipmh] 503, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
            except Exception as e:
                logger.error(f"[oaipmh] HTTP error page {page}: {e}")
                break

            root = ET.fromstring(resp.text)

            # Parse records
            for record_el in root.findall(".//oai:record", OAI_NS):
                header = record_el.find("oai:header", OAI_NS)
                if header is not None and header.get("status") == "deleted":
                    continue

                meta = record_el.find(".//arx:arXivRaw", OAI_NS)
                if meta is None:
                    continue

                rec = _parse_record(header, meta)
                if rec and rec.arxiv_id:
                    records.append(rec)

                if max_records and len(records) >= max_records:
                    logger.info(f"[oaipmh] Reached max_records={max_records}")
                    return records

            # Check for resumption token
            token_el = root.find(".//oai:resumptionToken", OAI_NS)
            if token_el is not None and token_el.text:
                resumption_token = token_el.text.strip()
                complete_size = token_el.get("completeListSize", "?")
                cursor = token_el.get("cursor", "?")
                page += 1
                if page % 10 == 0:
                    logger.info(f"[oaipmh] page {page}, {len(records)} records, cursor={cursor}/{complete_size}")
                await asyncio.sleep(1)  # polite delay
            else:
                break

    logger.info(f"[oaipmh] harvest complete: {len(records)} records in {page + 1} pages")
    return records


def _parse_record(header, meta) -> ArxivRecord | None:
    """Parse a single OAI-PMH arXivRaw record."""
    try:
        arxiv_id = _text(meta, "arx:id")
        if not arxiv_id:
            return None

        # Strip version suffix for consistency
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        title = _text(meta, "arx:title")
        # Clean up LaTeX and whitespace in title
        title = re.sub(r"\s+", " ", title).strip()

        abstract = _text(meta, "arx:abstract")
        abstract = re.sub(r"\s+", " ", abstract).strip()

        # Authors: "Author1, Author2, Author3" or newline-separated
        authors_raw = _text(meta, "arx:authors")
        authors = [a.strip() for a in re.split(r",\s*|\n", authors_raw) if a.strip()]

        # Categories
        categories_raw = _text(meta, "arx:categories")
        categories = categories_raw.split() if categories_raw else []

        created = _text(meta, "arx:created")
        updated = _text(meta, "arx:updated")
        doi = _text(meta, "arx:doi")
        license_url = _text(meta, "arx:license")

        return ArxivRecord(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            categories=categories,
            created=created,
            updated=updated,
            doi=doi,
            license=license_url,
        )
    except Exception as e:
        logger.warning(f"[oaipmh] parse error: {e}")
        return None


def _text(parent, tag: str) -> str:
    el = parent.find(tag, OAI_NS)
    return (el.text or "").strip() if el is not None else ""


async def match_venue_papers_with_oaipmh(
    session,
    from_date: str,
    until_date: str,
    conf_years: list[str] | None = None,
) -> dict:
    """Harvest arXiv OAI-PMH records and match against venue_papers by title.

    Fills: arxiv_id, doi (if empty). Does NOT overwrite existing values.

    Args:
        session: async SQLAlchemy session
        from_date: OAI-PMH from date (e.g. "2024-01-01")
        until_date: OAI-PMH until date (e.g. "2026-12-31")
        conf_years: optional filter for specific conf_years

    Returns:
        Stats dict with harvest count, match count, updates.
    """
    from sqlalchemy import text as sa_text
    from backend.services.venue_index.normalize import normalize_title

    # Step 1: Harvest from OAI-PMH
    logger.info(f"[oaipmh-match] Harvesting arXiv records from {from_date} to {until_date}")
    records = await harvest_oaipmh(
        from_date=from_date,
        until_date=until_date,
        arxiv_set="cs",  # All CS papers (includes CV, LG, CL, AI)
    )

    if not records:
        return {"harvested": 0, "matched": 0, "updated_arxiv": 0, "updated_doi": 0}

    # Step 2: Build title_norm → record index
    arxiv_index: dict[str, ArxivRecord] = {}
    for rec in records:
        tn = normalize_title(rec.title)
        if tn and len(tn) > 10:
            arxiv_index[tn] = rec

    logger.info(f"[oaipmh-match] Indexed {len(arxiv_index)} unique normalized titles from {len(records)} records")

    # Step 3: Load venue_papers needing arxiv_id
    where_clause = "(arxiv_id = '' OR arxiv_id IS NULL)"
    params: dict = {}
    if conf_years:
        where_clause += " AND conf_year = ANY(:conf_years)"
        params["conf_years"] = conf_years

    rows = (await session.execute(sa_text(f"""
        SELECT id, title_normalized, doi FROM venue_papers
        WHERE {where_clause}
    """), params)).fetchall()

    logger.info(f"[oaipmh-match] {len(rows)} venue_papers rows need arxiv_id")

    # Step 4: Match by normalized title
    updated_arxiv = 0
    updated_doi = 0
    for vp_id, title_norm, existing_doi in rows:
        rec = arxiv_index.get(title_norm)
        if not rec:
            continue

        updates = ["arxiv_id = :aid", "updated_at = now()"]
        update_params: dict = {"aid": rec.arxiv_id[:30], "id": vp_id}

        if not existing_doi and rec.doi:
            updates.append("doi = :doi")
            update_params["doi"] = rec.doi[:200]
            updated_doi += 1

        await session.execute(
            sa_text(f"UPDATE venue_papers SET {', '.join(updates)} WHERE id = :id"),
            update_params,
        )
        updated_arxiv += 1

    await session.flush()
    logger.info(f"[oaipmh-match] Done: matched {updated_arxiv}/{len(rows)}, doi+={updated_doi}")

    return {
        "harvested": len(records),
        "indexed_titles": len(arxiv_index),
        "venue_papers_needing": len(rows),
        "updated_arxiv": updated_arxiv,
        "updated_doi": updated_doi,
    }
