"""Full pipeline service — end-to-end paper processing.

One call to run the complete pipeline for a paper:
  ingest → download_pdf → enrich → parse → skim → deep → delta_card → graph

Also supports: seed a domain from one paper by discovering related work.

L4 deep analysis uses the V6 ingest_workflow (agent pipeline).
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
    """Download PDF for a paper. Delegates to pdf_download_service.

    Handles arXiv, OpenReview, CVF, and generic OA URLs.
    """
    from backend.services.pdf_download_service import download_pdf_to_oss
    return await download_pdf_to_oss(session, paper_id)


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

    # Step 0: Triage scoring (always run first for prioritization)
    from backend.services import triage_service
    await triage_service.triage_paper(session, paper_id)
    await session.commit()
    await session.refresh(paper)
    progress["steps"]["triage"] = {
        "keep_score": paper.keep_score,
        "analysis_priority": paper.analysis_priority,
        "tier": paper.tier.value if paper.tier else None,
    }

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
        import httpx
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            enriched = await enrich_service.enrich_paper(session, paper, client)
        progress["steps"]["enrich"] = enriched if enriched else "no_data"
        await session.commit()
        await session.refresh(paper)
    else:
        progress["steps"]["enrich"] = "skipped (already enriched)"

    # Guard: if title is still a placeholder after enrich, arXiv was unreachable
    await session.refresh(paper)
    if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', paper.title or ''):
        progress["steps"]["warning"] = "Title still placeholder — arXiv API may be rate-limited. Retry later."
        # Still continue with PDF download/parse but skip LLM analysis
        logger.warning(f"Paper {paper_id} title is still placeholder after enrich")

    # Step 2.5: Venue resolution (OpenReview + DBLP + arXiv comments)
    if not paper.acceptance_type:
        try:
            from backend.services.venue_resolver_service import resolve_venue
            authors_list = None
            if paper.authors and isinstance(paper.authors, list):
                authors_list = [a.get("name", "") for a in paper.authors if isinstance(a, dict)]
            venue_result = await resolve_venue(
                session, paper.id,
                title=paper.title,
                authors=authors_list,
                arxiv_id=paper.arxiv_id or "",
                current_venue=paper.venue or "",
                current_year=paper.year or 0,
            )
            progress["steps"]["venue_resolve"] = {
                "venue": venue_result.get("venue"),
                "acceptance_status": venue_result.get("acceptance_status"),
                "sources_checked": venue_result.get("sources_checked"),
            }
            # Apply results to paper
            if venue_result.get("acceptance_status") and venue_result["acceptance_status"] != "unknown":
                paper.acceptance_type = venue_result["acceptance_status"]
                # Venue from accepted conference is more authoritative than arXiv default
                if venue_result.get("venue"):
                    paper.venue = venue_result["venue"][:100]
            elif venue_result.get("venue") and not paper.venue:
                paper.venue = venue_result["venue"][:100]
            await session.commit()
            await session.refresh(paper)
        except Exception as e:
            logger.warning(f"Venue resolution failed for {paper_id}: {e}")
            progress["steps"]["venue_resolve"] = f"error: {str(e)[:100]}"
    else:
        progress["steps"]["venue_resolve"] = "skipped (already resolved)"

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

    # Step 4: L3 Skim — REMOVED (merged into shallow_extractor agent in Step 5)
    # shallow_extractor produces paper_essence + method_delta which covers
    # all L3 skim output (problem_summary, method_summary, changed_slots).
    progress["steps"]["skim_l3"] = "merged_into_shallow_extractor"

    # Step 5: L4 Deep (6-agent pipeline → DeltaCard → graph)
    has_l4 = (await session.execute(
        select(PaperAnalysis).where(
            PaperAnalysis.paper_id == paper_id,
            PaperAnalysis.level == AnalysisLevel.L4_DEEP,
            PaperAnalysis.is_current.is_(True),
        )
    )).scalar_one_or_none()

    if not has_l4:
        from backend.services.ingest_workflow import IngestWorkflow
        workflow = IngestWorkflow(session)
        v6_result = await workflow.deep_ingest(paper_id)
        progress["steps"]["deep_l4"] = {
            "agents_run": v6_result.get("agents_run", []),
            "graph": v6_result.get("graph", {}),
        }
        await session.commit()
    else:
        progress["steps"]["deep_l4"] = "skipped (already analyzed)"
        # If L4 exists but DeltaCard is missing, re-run graph materialization
        if not paper.current_delta_card_id:
            try:
                from backend.services.delta_card_service import run_delta_card_pipeline
                from backend.services.analysis_service import _maybe_create_bottleneck, assign_paradigm
                l4_analysis = has_l4
                evidence_units = []
                if l4_analysis.evidence_summary:
                    for sentence in l4_analysis.evidence_summary.split("。")[:5]:
                        if sentence.strip():
                            evidence_units.append({
                                "atom_type": "evidence",
                                "claim": sentence.strip(),
                                "confidence": 0.7,
                                "basis": "text_stated",
                                "source_section": "evidence_summary",
                            })
                analysis_data = {
                    "problem_summary": l4_analysis.problem_summary,
                    "method_summary": l4_analysis.method_summary,
                    "evidence_summary": l4_analysis.evidence_summary,
                    "core_intuition": l4_analysis.core_intuition,
                    "changed_slots": l4_analysis.changed_slots or [],
                    "evidence_units": evidence_units,
                    "delta_card": {"paradigm": "unknown", "slots": {}, "is_structural": False},
                }
                bottleneck_id = await _maybe_create_bottleneck(session, paper, analysis_data)
                paradigm, slots = await assign_paradigm(
                    session, paper.category, paper.tags,
                    title=paper.title, abstract=paper.abstract,
                )
                await run_delta_card_pipeline(
                    session,
                    paper_id=paper.id,
                    analysis_id=l4_analysis.id,
                    analysis_data=analysis_data,
                    paradigm_id=paradigm.id if paradigm else None,
                    paradigm_name=paradigm.name if paradigm else None,
                    slots=[{"id": s["id"], "name": s["name"]} for s in slots] if slots else None,
                    changed_slots_graph=None,
                    bottleneck_id=bottleneck_id,
                )
                await session.refresh(paper)
                progress["steps"]["graph_repair"] = f"dc_ptr={'set' if paper.current_delta_card_id else 'failed'}"
                await session.commit()
            except Exception as e:
                logger.warning(f"Graph repair failed for {paper_id}: {e}")
                await session.rollback()
                await session.refresh(paper)
                progress["steps"]["graph_repair"] = f"error: {str(e)[:80]}"

    # Step 5.5: Post-L4 — assign ring and role_in_kb from DeltaCard
    # Taxonomy assignment is handled by IngestWorkflow._write_taxonomy_facets (agent-driven)
    # Citation discovery is handled by IngestWorkflow.discover_neighborhood (single entry)
    try:
        try:
            await session.commit()
        except Exception:
            await session.rollback()
        await session.refresh(paper)

        if paper.current_delta_card_id:
            if not paper.ring:
                from backend.models.delta_card import DeltaCard
                dc = await session.get(DeltaCard, paper.current_delta_card_id)
                if dc and dc.structurality_score is not None:
                    s = float(dc.structurality_score)
                    if s >= 0.7:
                        paper.ring = "baseline"
                    elif s >= 0.4:
                        paper.ring = "structural"
                    else:
                        paper.ring = "plugin"
            if not paper.role_in_kb:
                paper.role_in_kb = "extension"

        await session.commit()
        await session.refresh(paper)
        progress["steps"]["post_l4"] = "ring_and_role_assigned"
    except Exception as e:
        logger.warning(f"Post-L4 processing failed for {paper_id}: {e}")
        progress["steps"]["post_l4"] = f"error: {str(e)[:100]}"
        try:
            await session.rollback()
        except Exception:
            pass

    # Check graph output
    from backend.models.delta_card import DeltaCard
    from sqlalchemy import func, text

    dc_count = (await session.execute(
        select(func.count()).select_from(DeltaCard).where(DeltaCard.paper_id == paper_id)
    )).scalar()
    assertion_count = (await session.execute(text(
        "SELECT count(*) FROM graph_assertions ga "
        "JOIN graph_nodes gn ON ga.from_node_id = gn.id "
        "WHERE gn.ref_table = 'delta_cards' AND gn.ref_id IN "
        "(SELECT id FROM delta_cards WHERE paper_id = :pid)"
    ), {"pid": paper_id})).scalar()

    progress["graph_output"] = {
        "delta_cards": dc_count,
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
