"""Full pipeline service — end-to-end paper processing.

One call to run the complete pipeline for a paper:
  ingest → download_pdf → enrich → parse → skim → deep → delta_card → graph

Also supports: seed a domain from one paper by discovering related work.
"""

import logging
import re
from pathlib import Path
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.models.enums import AnalysisLevel, PaperState
from backend.models.paper import Paper, PaperAsset
from backend.models.enums import AssetType

logger = logging.getLogger(__name__)


# ── PDF Download ──────────────────────────────────────────────────

async def download_arxiv_pdf(session: AsyncSession, paper_id: UUID) -> bool:
    """Download PDF from arxiv for a paper. Returns True on success."""
    paper = await session.get(Paper, paper_id)
    if not paper or not paper.arxiv_id:
        return False

    # Already have PDF?
    if paper.pdf_path_local or paper.pdf_object_key:
        return True

    # Build arxiv PDF URL
    base_id = re.sub(r"v\d+$", "", paper.arxiv_id)
    pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"

    # Determine local path
    category = paper.category or "Uncategorized"
    venue_year = f"{paper.venue}_{paper.year}" if paper.venue and paper.year else "Unknown"
    filename = f"{paper.title_sanitized or base_id}.pdf"
    rel_path = f"{category}/{venue_year}/{filename}"

    pdf_dir = Path(settings.paper_pdfs_dir)
    local_path = pdf_dir / rel_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            resp = await client.get(pdf_url)
            resp.raise_for_status()
            if len(resp.content) < 1000:
                logger.warning(f"PDF too small for {paper.arxiv_id}: {len(resp.content)} bytes")
                return False
            local_path.write_bytes(resp.content)
    except Exception as e:
        logger.error(f"PDF download failed for {paper.arxiv_id}: {e}")
        return False

    # Update paper record
    paper.pdf_path_local = str(local_path)
    paper.state = PaperState.DOWNLOADED

    # Create asset record
    from backend.utils.sanitize import sanitize_filename
    asset = PaperAsset(
        paper_id=paper.id,
        asset_type=AssetType.RAW_PDF,
        object_key=f"papers/raw-pdf/{rel_path}",
        mime_type="application/pdf",
        size_bytes=len(resp.content),
    )
    session.add(asset)
    await session.flush()
    logger.info(f"Downloaded PDF for {paper.arxiv_id}: {local_path} ({len(resp.content)} bytes)")
    return True


# ── Full pipeline ─────────────────────────────────────────────────

async def run_full_pipeline(
    session: AsyncSession,
    paper_id: UUID,
    skip_existing: bool = True,
) -> dict:
    """Run the complete pipeline for a single paper.

    Steps: download_pdf → enrich → parse → skim → deep (→ delta_card_build is auto-triggered)

    Returns progress dict showing what was done at each step.
    """
    progress = {"paper_id": str(paper_id), "steps": {}}

    paper = await session.get(Paper, paper_id)
    if not paper:
        return {"error": "Paper not found"}

    # Step 1: Download PDF (if arxiv paper)
    if paper.arxiv_id and not paper.pdf_path_local:
        ok = await download_arxiv_pdf(session, paper_id)
        progress["steps"]["download_pdf"] = "success" if ok else "failed"
        if ok:
            await session.commit()
            await session.refresh(paper)
    else:
        progress["steps"]["download_pdf"] = "skipped (already have PDF or no arxiv_id)"

    # Step 2: Enrich metadata
    if not paper.abstract or not paper.authors:
        from backend.services import enrich_service
        enriched = await enrich_service.enrich_paper(session, paper_id)
        progress["steps"]["enrich"] = enriched if enriched else "no_data"
        await session.commit()
        await session.refresh(paper)
    else:
        progress["steps"]["enrich"] = "skipped (already enriched)"

    # Step 3: L2 Parse PDF
    from backend.models.analysis import PaperAnalysis
    has_l2 = (await session.execute(
        select(PaperAnalysis).where(
            PaperAnalysis.paper_id == paper_id,
            PaperAnalysis.level == AnalysisLevel.L2_PARSE,
            PaperAnalysis.is_current.is_(True),
        )
    )).scalar_one_or_none()

    if not has_l2 and paper.pdf_path_local:
        from backend.services import parse_service
        l2 = await parse_service.parse_paper_pdf(session, paper_id)
        progress["steps"]["parse_l2"] = f"sections={list(l2.extracted_sections.keys()) if l2 and l2.extracted_sections else []}" if l2 else "failed"
        await session.commit()
    elif has_l2:
        progress["steps"]["parse_l2"] = "skipped (already parsed)"
    else:
        progress["steps"]["parse_l2"] = "skipped (no PDF)"

    # Step 4: L3 Skim
    has_l3 = (await session.execute(
        select(PaperAnalysis).where(
            PaperAnalysis.paper_id == paper_id,
            PaperAnalysis.level == AnalysisLevel.L3_SKIM,
            PaperAnalysis.is_current.is_(True),
        )
    )).scalar_one_or_none()

    if not has_l3:
        from backend.services import analysis_service
        l3 = await analysis_service.skim_paper(session, paper_id)
        progress["steps"]["skim_l3"] = {
            "worth_deep_read": l3.worth_deep_read if l3 else None,
            "is_plugin_patch": l3.is_plugin_patch if l3 else None,
            "model": f"{l3.model_provider}/{l3.model_name}" if l3 else "failed",
        }
        await session.commit()
    else:
        progress["steps"]["skim_l3"] = "skipped (already skimmed)"

    # Step 5: L4 Deep (auto-triggers delta_card_build → graph)
    has_l4 = (await session.execute(
        select(PaperAnalysis).where(
            PaperAnalysis.paper_id == paper_id,
            PaperAnalysis.level == AnalysisLevel.L4_DEEP,
            PaperAnalysis.is_current.is_(True),
        )
    )).scalar_one_or_none()

    if not has_l4:
        from backend.services import analysis_service
        l4 = await analysis_service.deep_analyze_paper(session, paper_id)
        progress["steps"]["deep_l4"] = {
            "model": f"{l4.model_provider}/{l4.model_name}" if l4 else "failed",
        }
        await session.commit()
    else:
        progress["steps"]["deep_l4"] = "skipped (already analyzed)"

    # Check graph output
    from backend.models.delta_card import DeltaCard
    from backend.models.graph import IdeaDelta
    from sqlalchemy import func, text

    dc_count = (await session.execute(
        select(func.count()).select_from(DeltaCard).where(DeltaCard.paper_id == paper_id)
    )).scalar()
    idea_count = (await session.execute(
        select(func.count()).select_from(IdeaDelta).where(IdeaDelta.paper_id == paper_id)
    )).scalar()
    assertion_count = (await session.execute(text(
        "SELECT count(*) FROM graph_assertions ga "
        "JOIN graph_nodes gn ON ga.from_node_id = gn.id "
        "WHERE gn.ref_table = 'idea_deltas' AND gn.ref_id IN "
        "(SELECT id FROM idea_deltas WHERE paper_id = :pid)"
    ), {"pid": paper_id})).scalar()

    progress["graph_output"] = {
        "delta_cards": dc_count,
        "idea_deltas": idea_count,
        "assertions": assertion_count or 0,
    }

    # Final state
    await session.refresh(paper)
    progress["final_state"] = paper.state.value

    return progress


# ── Batch pipeline ────────────────────────────────────────────────

async def run_pipeline_batch(
    session: AsyncSession,
    limit: int = 5,
) -> list[dict]:
    """Run full pipeline on papers that need processing."""
    result = await session.execute(
        select(Paper).where(
            Paper.state.in_([PaperState.WAIT, PaperState.DOWNLOADED,
                            PaperState.CANONICALIZED, PaperState.ENRICHED,
                            PaperState.L2_PARSED]),
        ).order_by(Paper.created_at).limit(limit)
    )
    papers = list(result.scalars().all())

    results = []
    for paper in papers:
        try:
            progress = await run_full_pipeline(session, paper.id)
            results.append(progress)
        except Exception as e:
            logger.error(f"Pipeline error for {paper.id}: {e}")
            results.append({"paper_id": str(paper.id), "error": str(e)[:200]})

    return results
