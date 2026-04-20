"""L2 parse service — extract structured content from PDF.

Parser Ensemble:
  1. GROBID → structured metadata (title, authors, affiliations, refs)
  2. PyMuPDF → text, sections, figure images (fast fallback)
  3. MinerU → formulas, tables, reading order (when available)

Results are merged with conflict marking.
"""

import logging
from dataclasses import asdict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.models.analysis import PaperAnalysis
from backend.models.enums import AnalysisLevel, PaperState
from backend.models.paper import Paper
from backend.services.object_storage import get_storage
from backend.utils.grobid_client import GrobidClient
from backend.utils.pdf_extract import parse_pdf

logger = logging.getLogger(__name__)


async def parse_paper_pdf(session: AsyncSession, paper_id: UUID) -> PaperAnalysis | None:
    """Run L2 parse on a paper's PDF using parser ensemble.

    1. Try GROBID for structured metadata + references
    2. Use PyMuPDF for text extraction + figure images
    3. Merge results, preferring GROBID for metadata
    """
    paper = await session.get(Paper, paper_id)
    if not paper:
        return None

    # Find the PDF path
    pdf_path = await _resolve_pdf_path(paper)
    if not pdf_path:
        logger.warning(f"No PDF found for paper {paper_id}")
        return None

    # ── Parser Ensemble ──────────────────────────────────────────

    # 1. PyMuPDF (always available, fast fallback)
    pymupdf_result = parse_pdf(pdf_path)

    # 2. GROBID (structured metadata + references)
    grobid_result = None
    grobid_refs = []
    grobid_authors = []
    grobid_client = GrobidClient(settings.grobid_url)

    if await grobid_client.is_alive():
        try:
            grobid_result = await grobid_client.parse_fulltext(pdf_path)
            if grobid_result.references:
                grobid_refs = [
                    {
                        "ref_id": r.ref_id,
                        "title": r.title,
                        "authors": r.authors,
                        "venue": r.venue,
                        "year": r.year,
                        "doi": r.doi,
                        "arxiv_id": r.arxiv_id,
                    }
                    for r in grobid_result.references
                    if r.title  # skip empty refs
                ]
            if grobid_result.authors:
                grobid_authors = [
                    {
                        "name": a.name,
                        "given_name": a.given_name,
                        "surname": a.surname,
                        "affiliation": a.affiliation,
                        "email": a.email,
                        "orcid": a.orcid,
                    }
                    for a in grobid_result.authors
                ]
            logger.info(
                f"GROBID parsed {paper_id}: "
                f"{len(grobid_refs)} refs, {len(grobid_authors)} authors"
            )
        except Exception as e:
            logger.warning(f"GROBID parse failed for {paper_id}: {e}")
    else:
        logger.info("GROBID not available, using PyMuPDF only")

    # ── S2 fallback for refs + authors when GROBID fails ─────────
    if not grobid_refs and paper.arxiv_id:
        try:
            import httpx
            from backend.services.enrich_service import _s2_headers
            S2_API = "https://api.semanticscholar.org/graph/v1"
            s2_id = f"ARXIV:{paper.arxiv_id}"
            async with httpx.AsyncClient(timeout=30) as client:
                # Fetch references
                resp = await client.get(
                    f"{S2_API}/paper/{s2_id}/references",
                    params={"fields": "title,year,venue,externalIds,authors", "limit": "100"},
                    headers=_s2_headers(),
                )
                if resp.status_code == 200:
                    for ref in resp.json().get("data", []):
                        cp = ref.get("citedPaper", {})
                        if cp.get("title"):
                            eids = cp.get("externalIds", {}) or {}
                            grobid_refs.append({
                                "ref_id": "", "title": cp["title"],
                                "authors": [a.get("name","") for a in (cp.get("authors") or [])[:5]],
                                "venue": cp.get("venue", ""), "year": str(cp.get("year", "")),
                                "doi": eids.get("DOI", ""), "arxiv_id": eids.get("ArXiv", ""),
                            })
                    logger.info(f"S2 fallback refs for {paper_id}: {len(grobid_refs)}")

                # Fetch detailed author info
                import asyncio as _asyncio
                await _asyncio.sleep(1.2)
                resp2 = await client.get(
                    f"{S2_API}/paper/{s2_id}/authors",
                    params={"fields": "name,affiliations,hIndex"},
                    headers=_s2_headers(),
                )
                if resp2.status_code == 200:
                    for a in resp2.json().get("data", []):
                        grobid_authors.append({
                            "name": a.get("name", ""), "given_name": "", "surname": "",
                            "affiliation": (a.get("affiliations") or [""])[0] if a.get("affiliations") else "",
                            "email": "", "orcid": "",
                        })
                    logger.info(f"S2 fallback authors for {paper_id}: {len(grobid_authors)}")
        except Exception as e:
            logger.debug(f"S2 fallback failed for {paper_id}: {e}")

    # ── Merge sections ───────────────────────────────────────────
    # Prefer GROBID sections if available (better structure), fall back to PyMuPDF
    def _clean_text(text: str) -> str:
        """Remove null bytes that PostgreSQL text columns cannot store."""
        return text.replace("\x00", "") if text else text

    merged_sections = {}
    if grobid_result and grobid_result.sections:
        merged_sections = {k: _clean_text(v[:5000]) for k, v in grobid_result.sections.items()}
        for k, v in pymupdf_result.sections.items():
            if k not in merged_sections:
                merged_sections[k] = _clean_text(v[:5000])
    else:
        merged_sections = {k: _clean_text(v[:5000]) for k, v in pymupdf_result.sections.items()}

    # ── Merge figure captions ────────────────────────────────────
    figure_captions = pymupdf_result.figure_captions
    if grobid_result and grobid_result.figure_captions:
        # GROBID captions may have better text; merge by checking overlap
        grobid_caps = grobid_result.figure_captions
        if len(grobid_caps) > len(figure_captions):
            figure_captions = grobid_caps

    # ── Merge table captions ─────────────────────────────────────
    table_captions = pymupdf_result.tables
    if grobid_result and grobid_result.table_captions:
        grobid_tabs = grobid_result.table_captions
        if len(grobid_tabs) > len(table_captions):
            table_captions = grobid_tabs

    # ── Supersede old L2 ─────────────────────────────────────────
    existing = await session.execute(
        select(PaperAnalysis).where(
            PaperAnalysis.paper_id == paper_id,
            PaperAnalysis.level == AnalysisLevel.L2_PARSE,
            PaperAnalysis.is_current.is_(True),
        )
    )
    old_analysis = existing.scalar_one_or_none()
    if old_analysis:
        old_analysis.is_current = False

    # ── Extract and upload figures ────────────────────────────────
    # Primary: VLM-guided precise extraction (1 API call)
    # Fallback: PyMuPDF heuristic extraction (free, CPU only)
    figure_image_records = []
    if (settings.anthropic_api_key or settings.openai_api_key) and pdf_path:
        try:
            from backend.services.figure_extraction_service import extract_figures_precise
            figure_image_records = await extract_figures_precise(
                pdf_path=pdf_path,
                paper_id=paper_id,
                paper_title=paper.title,
                session=session,
            )
            if figure_image_records:
                logger.info(f"VLM figure extraction: {len(figure_image_records)} figures for {paper_id}")
        except Exception as e:
            logger.warning(f"VLM figure extraction failed, falling back to heuristic: {e}")

    # Fallback: heuristic extraction if VLM didn't produce results
    if not figure_image_records:
        figure_image_records = await _upload_figure_images(
            paper_id, pymupdf_result.figure_images, figure_captions
        )

    # ── Formula extraction (VLM page scan, GROBID coords optional) ──
    extracted_formulas = pymupdf_result.formulas  # PyMuPDF regex as baseline
    grobid_formula_data = []

    if grobid_result and grobid_result.formulas:
        grobid_formula_data = [
            {"text": f.text, "label": f.label, "page": f.page, "bbox": f.bbox}
            for f in grobid_result.formulas
            if f.text
        ]

    if pdf_path and (settings.anthropic_api_key or settings.openai_api_key):
        try:
            from backend.services.formula_extraction_service import extract_formulas
            vlm_formulas = await extract_formulas(
                pdf_path=pdf_path,
                paper_id=paper_id,
                grobid_formulas=grobid_formula_data if grobid_formula_data else None,
                session=session,
            )
            if vlm_formulas:
                extracted_formulas = [f["latex"] for f in vlm_formulas if f.get("latex")]
                logger.info(f"Formula extraction: {len(vlm_formulas)} formulas via VLM for {paper_id}")
        except Exception as e:
            logger.warning(f"Formula extraction failed for {paper_id}: {e}")

    # ── Build parse metadata ─────────────────────────────────────
    parse_metadata = {
        "parsers_used": ["pymupdf"],
        "grobid_available": grobid_result is not None,
        "grobid_ref_count": len(grobid_refs),
        "grobid_author_count": len(grobid_authors),
        "pymupdf_section_count": len(pymupdf_result.sections),
        "pymupdf_formula_count": len(pymupdf_result.formulas),
        "grobid_formula_count": len(grobid_formula_data),
        "final_formula_count": len(extracted_formulas),
        "pymupdf_figure_count": len(pymupdf_result.figure_captions),
        "vlm_available": True,  # Claude API always available (no GPU needed)
    }
    if grobid_result:
        parse_metadata["parsers_used"].append("grobid")

    # ── Create L2 analysis ───────────────────────────────────────
    analysis = PaperAnalysis(
        paper_id=paper_id,
        level=AnalysisLevel.L2_PARSE,
        model_provider="local",
        model_name="ensemble_pymupdf_grobid",
        prompt_version="v2",
        schema_version="v2",
        confidence=1.0,
        extracted_sections=merged_sections,
        extracted_formulas=[_clean_text(f) for f in extracted_formulas] if extracted_formulas else extracted_formulas,
        extracted_tables=table_captions,
        figure_captions=figure_captions,
        extracted_figure_images=figure_image_records if figure_image_records else None,
        # Store GROBID-specific results in evidence_spans (repurposed for L2)
        evidence_spans={
            "grobid_references": grobid_refs,
            "grobid_authors": grobid_authors,
            "grobid_abstract": grobid_result.abstract if grobid_result else None,
            "grobid_keywords": grobid_result.keywords if grobid_result else [],
            "parse_metadata": parse_metadata,
        } if grobid_result else {"parse_metadata": parse_metadata},
        is_current=True,
    )
    session.add(analysis)

    # ── Update paper with GROBID metadata ────────────────────────
    if grobid_result:
        # Only fill missing fields (don't overwrite existing)
        if not paper.abstract and grobid_result.abstract:
            paper.abstract = grobid_result.abstract
        if not paper.authors and grobid_authors:
            paper.authors = grobid_authors
        if not paper.keywords and grobid_result.keywords:
            paper.keywords = grobid_result.keywords

        # Store references text for downstream analysis
        if not pymupdf_result.references_text and "references" in merged_sections:
            pymupdf_result.references_text = merged_sections.get("references", "")

    # Update paper state
    if paper.state in (PaperState.WAIT, PaperState.DOWNLOADED, PaperState.L1_METADATA,
                       PaperState.ENRICHED, PaperState.CANONICALIZED):
        paper.state = PaperState.L2_PARSED

    await session.flush()
    await session.refresh(analysis)

    return analysis


async def _resolve_pdf_path(paper: Paper) -> str | None:
    """Resolve the PDF file path from object storage or local path."""
    storage = get_storage()

    if paper.pdf_object_key:
        local_path = storage.get_local_path(paper.pdf_object_key)
        if local_path:
            return local_path

    if paper.pdf_path_local:
        import os
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        candidate = os.path.join(project_root, paper.pdf_path_local)
        if os.path.exists(candidate):
            return candidate

    return None


async def _upload_figure_images(
    paper_id: UUID,
    figure_images: list[dict],
    figure_captions: list[dict] | None = None,
) -> list[dict]:
    """Upload extracted figure images to object storage.

    Each figure gets:
      - Stored at papers/{paper_id}/figures/fig_001.png
      - public_url for CDN/Obsidian remote reference
      - Matched caption from GROBID/PyMuPDF
      - bbox and extraction_method metadata
    """
    if not figure_images:
        return []

    storage = get_storage()
    records = []
    captions = figure_captions or []

    for i, fig in enumerate(figure_images):
        ext = fig.get("ext", "png")
        object_key = f"papers/{paper_id}/figures/fig_{i+1:03d}.{ext}"
        try:
            await storage.put(object_key, fig["image_bytes"])
            record = {
                "figure_num": i + 1,
                "object_key": object_key,
                "page_num": fig.get("page_num", -1),
                "width": fig.get("width", 0),
                "height": fig.get("height", 0),
                "size_bytes": fig.get("size_bytes", 0),
                "extraction_method": fig.get("extraction_method", "unknown"),
                "bbox": fig.get("bbox"),
            }

            # Add public URL for report embedding
            public_url = storage.get_public_url(object_key)
            if public_url:
                record["public_url"] = public_url

            # Match caption by page proximity
            best_caption = _match_caption_to_figure(fig, captions, i)
            if best_caption:
                record["caption"] = best_caption.get("caption", "")
                record["caption_label"] = best_caption.get("label") or f"Figure {best_caption.get('figure_num', i+1)}"

            records.append(record)
        except Exception as e:
            logger.warning(f"Failed to store figure {i+1} for {paper_id}: {e}")

    return records


def _match_caption_to_figure(
    fig: dict,
    captions: list[dict],
    fig_index: int,
) -> dict | None:
    """Match a figure image to its caption by page number or index."""
    fig_page = fig.get("page_num", -1)

    # Try matching by page number (captions near the figure)
    page_captions = [c for c in captions if c.get("page_num") == fig_page]
    if page_captions:
        return page_captions[0]

    # Fallback: match by figure_num index
    for cap in captions:
        if cap.get("figure_num") == fig_index + 1:
            return cap

    # Fallback: match by order
    if fig_index < len(captions):
        return captions[fig_index]

    return None


async def parse_all_unprocessed(session: AsyncSession, limit: int = 10) -> list[dict]:
    """Parse PDFs for papers that have a PDF but no L2 analysis."""
    result = await session.execute(
        select(Paper)
        .where(
            Paper.pdf_path_local.isnot(None),
            Paper.state.in_([
                PaperState.WAIT, PaperState.DOWNLOADED,
                PaperState.L1_METADATA, PaperState.ENRICHED,
            ]),
        )
        .order_by(Paper.analysis_priority.desc().nullsfirst())
        .limit(limit)
    )
    papers = list(result.scalars().all())

    results = []
    for paper in papers:
        try:
            analysis = await parse_paper_pdf(session, paper.id)
            if analysis:
                section_names = list(analysis.extracted_sections.keys()) if analysis.extracted_sections else []
                grobid_data = analysis.evidence_spans or {}
                results.append({
                    "paper_id": str(paper.id),
                    "title": paper.title[:60],
                    "status": "parsed",
                    "sections": section_names,
                    "formulas": len(analysis.extracted_formulas or []),
                    "figures": len(analysis.figure_captions or []),
                    "tables": len(analysis.extracted_tables or []),
                    "grobid_refs": len(grobid_data.get("grobid_references", [])),
                    "grobid_authors": len(grobid_data.get("grobid_authors", [])),
                })
            else:
                results.append({
                    "paper_id": str(paper.id),
                    "title": paper.title[:60],
                    "status": "no_pdf",
                })
        except Exception as e:
            logger.error(f"Parse error for {paper.id}: {e}")
            results.append({
                "paper_id": str(paper.id),
                "title": paper.title[:60],
                "status": "error",
                "message": str(e)[:100],
            })

    await session.flush()
    return results
