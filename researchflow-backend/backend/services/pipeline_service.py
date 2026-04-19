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

    # Update paper record — store relative path for portability across environments
    paper.pdf_path_local = rel_path
    paper.state = PaperState.DOWNLOADED

    # Upload to object storage (if configured)
    object_key = f"papers/raw-pdf/{rel_path}"
    try:
        from backend.services.object_storage import get_storage
        storage = get_storage()
        await storage.put(object_key, resp.content)
        paper.pdf_object_key = object_key
        logger.info(f"Uploaded PDF to object storage: {object_key}")
    except Exception as e:
        logger.warning(f"Object storage upload failed for {paper.arxiv_id}, local only: {e}")

    # Create asset record
    from backend.services.object_storage import compute_checksum
    asset = PaperAsset(
        paper_id=paper.id,
        asset_type=AssetType.RAW_PDF,
        object_key=object_key,
        mime_type="application/pdf",
        size_bytes=len(resp.content),
        checksum=compute_checksum(resp.content),
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
        async with httpx.AsyncClient(follow_redirects=True) as client:
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
            if venue_result.get("venue") and not paper.venue:
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

    # Step 5.5: Post-L4 — fill paper fields from analysis + assign taxonomy
    try:
        try:
            await session.commit()
        except Exception:
            await session.rollback()
        await session.refresh(paper)
        # Get latest L4 analysis
        latest_l4 = (await session.execute(
            select(PaperAnalysis).where(
                PaperAnalysis.paper_id == paper_id,
                PaperAnalysis.level == AnalysisLevel.L4_DEEP,
                PaperAnalysis.is_current.is_(True),
            )
        )).scalar_one_or_none()

        if latest_l4:
            # Fill core_operator from core_intuition
            if not paper.core_operator and latest_l4.core_intuition:
                paper.core_operator = latest_l4.core_intuition[:200]
            # Fill primary_logic from method_summary (first 2 sentences)
            if not paper.primary_logic and latest_l4.method_summary:
                sentences = latest_l4.method_summary.split("。")[:2]
                paper.primary_logic = "。".join(sentences)[:300]
            # Fill claims from evidence_summary
            if not paper.claims and latest_l4.evidence_summary:
                paper.claims = [s.strip() for s in latest_l4.evidence_summary.split("。")[:3] if s.strip()]

            # Assign ring based on structurality_score
            if not paper.ring and paper.structurality_score is not None:
                s = float(paper.structurality_score)
                if s >= 0.7:
                    paper.ring = "baseline"
                elif s >= 0.4:
                    paper.ring = "structural"
                else:
                    paper.ring = "plugin"

            # Assign role_in_kb
            if not paper.role_in_kb:
                paper.role_in_kb = "extension"  # default

        await session.flush()

        # Taxonomy assignment — map paper fields to taxonomy facets
        from backend.models.taxonomy import TaxonomyNode, PaperFacet
        existing_facets = (await session.execute(
            select(PaperFacet).where(PaperFacet.paper_id == paper_id)
        )).scalars().all()

        if not existing_facets:
            # Auto-assign from paper.category, tags, mechanism_family, keywords
            assigned = 0
            for tag in (paper.tags or []):
                tag_clean = tag.replace("task/", "").replace("dataset/", "").replace("repr/", "").replace("opensource/", "")
                node = (await session.execute(
                    select(TaxonomyNode).where(
                        TaxonomyNode.name.ilike(f"%{tag_clean}%")
                    ).limit(1)
                )).scalar_one_or_none()
                if node:
                    facet = PaperFacet(
                        paper_id=paper_id, node_id=node.id,
                        facet_role=node.dimension, source="auto_tags", confidence=0.6,
                    )
                    session.add(facet)
                    assigned += 1

            # Map category to domain
            if paper.category:
                cat_node = (await session.execute(
                    select(TaxonomyNode).where(
                        TaxonomyNode.name.ilike(f"%{paper.category}%"),
                        TaxonomyNode.dimension == "domain",
                    ).limit(1)
                )).scalar_one_or_none()
                if cat_node:
                    session.add(PaperFacet(
                        paper_id=paper_id, node_id=cat_node.id,
                        facet_role="domain", source="auto_category", confidence=0.7,
                    ))
                    assigned += 1

            # Map mechanism_family
            if paper.mechanism_family:
                mech_node = (await session.execute(
                    select(TaxonomyNode).where(
                        TaxonomyNode.name.ilike(f"%{paper.mechanism_family.replace('_', '%')}%"),
                    ).limit(1)
                )).scalar_one_or_none()
                if mech_node:
                    session.add(PaperFacet(
                        paper_id=paper_id, node_id=mech_node.id,
                        facet_role="mechanism", source="auto_mech", confidence=0.6,
                    ))
                    assigned += 1

            # Map keywords (arXiv categories) to modality/paradigm
            keyword_map = {
                "cs.CV": ("modality", "Image"), "cs.CL": ("modality", "Text"),
                "cs.AI": ("domain", "Agent"), "cs.LG": ("learning_paradigm", "Reinforcement Learning"),
                "cs.RO": ("domain", "Embodied AI"),
            }
            for kw in (paper.keywords or []):
                if kw in keyword_map:
                    dim, name = keyword_map[kw]
                    kw_node = (await session.execute(
                        select(TaxonomyNode).where(
                            TaxonomyNode.name == name, TaxonomyNode.dimension == dim,
                        ).limit(1)
                    )).scalar_one_or_none()
                    if kw_node:
                        session.add(PaperFacet(
                            paper_id=paper_id, node_id=kw_node.id,
                            facet_role=dim, source="auto_arxiv_cat", confidence=0.5,
                        ))
                        assigned += 1

            progress["steps"]["taxonomy_assign"] = {"facets_assigned": assigned}
        else:
            progress["steps"]["taxonomy_assign"] = "skipped (already assigned)"

        await session.commit()
        await session.refresh(paper)
    except Exception as e:
        logger.warning(f"Post-L4 processing failed for {paper_id}: {e}")
        progress["steps"]["post_l4"] = f"error: {str(e)[:100]}"
        try:
            await session.rollback()
        except Exception:
            pass

    # Step 6: Discover related papers (references + citations via S2)
    try:
        from backend.services import discovery_service
        disc_result = await discovery_service.discover_related_papers(
            session, paper_id, max_references=20, max_citations=10
        )
        progress["steps"]["discover_citations"] = {
            "refs_found": disc_result.get("references_found", 0) if isinstance(disc_result, dict) else 0,
            "citations_found": disc_result.get("citations_found", 0) if isinstance(disc_result, dict) else 0,
            "new_papers_ingested": disc_result.get("new_papers", 0) if isinstance(disc_result, dict) else 0,
        }
        await session.commit()
    except Exception as e:
        logger.warning(f"Citation discovery failed for {paper_id}: {e}")
        progress["steps"]["discover_citations"] = f"error: {str(e)[:100]}"

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
