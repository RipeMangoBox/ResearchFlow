"""L2 parse service — extract structured content from PDF.

Reads PDF from object storage, runs pymupdf extraction,
stores results in paper_analyses (level=l2_parse).
"""

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.analysis import PaperAnalysis
from backend.models.enums import AnalysisLevel, PaperState
from backend.models.paper import Paper, PaperAsset
from backend.services.object_storage import get_storage
from backend.utils.pdf_extract import parse_pdf

logger = logging.getLogger(__name__)


async def parse_paper_pdf(session: AsyncSession, paper_id: UUID) -> PaperAnalysis | None:
    """Run L2 parse on a paper's PDF.

    Extracts sections, formulas, tables, figure captions.
    Stores result in paper_analyses with level=l2_parse.
    """
    paper = await session.get(Paper, paper_id)
    if not paper:
        return None

    # Find the PDF — check object storage first, then local path
    storage = get_storage()
    pdf_data = None
    pdf_path = None

    if paper.pdf_object_key:
        local_path = storage.get_local_path(paper.pdf_object_key)
        if local_path:
            pdf_path = local_path
    if not pdf_path and paper.pdf_path_local:
        # Try the legacy local path (relative to project root)
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        candidate = os.path.join(project_root, paper.pdf_path_local)
        if os.path.exists(candidate):
            pdf_path = candidate

    if not pdf_path:
        logger.warning(f"No PDF found for paper {paper_id}")
        return None

    # Parse
    parsed = parse_pdf(pdf_path)

    # Check if L2 analysis already exists
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

    # Save figure images to object storage
    figure_image_records = []
    if parsed.figure_images:
        from backend.services.object_storage import get_storage
        storage = get_storage()
        for i, fig in enumerate(parsed.figure_images):
            ext = fig.get("ext", "png")
            object_key = f"figures/{paper_id}/fig_{i+1}.{ext}"
            try:
                await storage.put(object_key, fig["image_bytes"])
                figure_image_records.append({
                    "figure_num": i + 1,
                    "object_key": object_key,
                    "page_num": fig["page_num"],
                    "width": fig["width"],
                    "height": fig["height"],
                    "size_bytes": fig["size_bytes"],
                })
            except Exception as e:
                logger.warning(f"Failed to store figure {i+1} for {paper_id}: {e}")

    # Create L2 analysis
    analysis = PaperAnalysis(
        paper_id=paper_id,
        level=AnalysisLevel.L2_PARSE,
        model_provider="local",
        model_name="pymupdf",
        prompt_version="v1",
        schema_version="v1",
        confidence=1.0,  # deterministic extraction
        extracted_sections={k: v[:5000] for k, v in parsed.sections.items()},
        extracted_formulas=parsed.formulas,
        extracted_tables=parsed.tables,
        figure_captions=parsed.figure_captions,
        extracted_figure_images=figure_image_records if figure_image_records else None,
        is_current=True,
    )
    session.add(analysis)

    # Update paper state
    if paper.state in (PaperState.WAIT, PaperState.DOWNLOADED, PaperState.L1_METADATA,
                       PaperState.ENRICHED, PaperState.CANONICALIZED):
        paper.state = PaperState.L2_PARSED

    await session.flush()
    await session.refresh(analysis)
    return analysis


async def parse_all_unprocessed(session: AsyncSession, limit: int = 10) -> list[dict]:
    """Parse PDFs for papers that have a PDF but no L2 analysis."""
    # Find papers with PDF but no L2 analysis
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
                results.append({
                    "paper_id": str(paper.id),
                    "title": paper.title[:60],
                    "status": "parsed",
                    "sections": section_names,
                    "formulas": len(analysis.extracted_formulas or []),
                    "figures": len(analysis.figure_captions or []),
                    "tables": len(analysis.extracted_tables or []),
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
