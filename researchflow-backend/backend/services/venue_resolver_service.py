"""Venue resolver — determines conference acceptance status.

Orchestrates: OpenReview → DBLP → metadata_observations → canonical resolver.
Follows authority ranking: official_conf > openreview > dblp > crossref > arxiv.
"""

import logging
import uuid
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.dblp_adapter import DBLPAdapter
from backend.services.metadata_resolver_service import record_observation, resolve_paper_metadata
from backend.services.openreview_adapter import OpenReviewAdapter

logger = logging.getLogger(__name__)


async def resolve_venue(
    session: AsyncSession,
    paper_id: UUID,
    title: str,
    authors: list[str] | None = None,
    arxiv_id: str = "",
    current_venue: str = "",
    current_year: int = 0,
) -> dict:
    """Resolve venue/acceptance status from multiple sources.

    Queries OpenReview and DBLP, records observations, then resolves canonical.
    Returns: {venue, acceptance_status, acceptance_type, review_scores, sources_checked}
    """
    sources_checked = []
    results = {}

    # 1. OpenReview check
    try:
        or_adapter = OpenReviewAdapter()
        or_result = await or_adapter.match_paper(
            title=title,
            authors=authors,
            venue=current_venue,
            year=current_year,
        )
        sources_checked.append("openreview")

        if or_result and or_result.acceptance_status != "unknown":
            # Record observations
            await record_observation(
                session,
                entity_type="paper",
                entity_id=paper_id,
                field_name="acceptance_status",
                value=or_result.acceptance_status,
                source="openreview",
                source_url=or_result.forum_url,
                confidence=or_result.confidence,
            )
            if or_result.acceptance_type:
                await record_observation(
                    session,
                    entity_type="paper",
                    entity_id=paper_id,
                    field_name="acceptance_type",
                    value=or_result.acceptance_type,
                    source="openreview",
                    source_url=or_result.forum_url,
                    confidence=or_result.confidence,
                )
            if or_result.review_scores:
                await record_observation(
                    session,
                    entity_type="paper",
                    entity_id=paper_id,
                    field_name="review_scores",
                    value={
                        "scores": or_result.review_scores,
                        "avg_score": or_result.avg_score,
                    },
                    source="openreview",
                    source_url=or_result.forum_url,
                    confidence=or_result.confidence,
                )
            if or_result.venue:
                await record_observation(
                    session,
                    entity_type="paper",
                    entity_id=paper_id,
                    field_name="venue",
                    value=or_result.venue,
                    source="openreview",
                    source_url=or_result.forum_url,
                    confidence=or_result.confidence,
                )

            results["openreview"] = {
                "venue": or_result.venue,
                "status": or_result.acceptance_status,
                "type": or_result.acceptance_type,
                "avg_score": or_result.avg_score,
            }
    except Exception as e:
        logger.warning(f"OpenReview check failed for {paper_id}: {e}")

    # 2. DBLP check
    try:
        dblp = DBLPAdapter()
        dblp_result = await dblp.verify_conference_acceptance(
            title=title,
            expected_venue=current_venue,
        )
        sources_checked.append("dblp")

        if dblp_result and dblp_result.dblp_key:
            if dblp_result.is_conference_accepted:
                await record_observation(
                    session,
                    entity_type="paper",
                    entity_id=paper_id,
                    field_name="acceptance_status",
                    value="accepted",
                    source="dblp",
                    source_url=dblp_result.url,
                    confidence=dblp_result.confidence,
                )
            if dblp_result.venue:
                await record_observation(
                    session,
                    entity_type="paper",
                    entity_id=paper_id,
                    field_name="venue",
                    value=dblp_result.venue,
                    source="dblp",
                    source_url=dblp_result.url,
                    confidence=dblp_result.confidence,
                )

            results["dblp"] = {
                "venue": dblp_result.venue,
                "accepted": dblp_result.is_conference_accepted,
                "key": dblp_result.dblp_key,
            }
    except Exception as e:
        logger.warning(f"DBLP check failed for {paper_id}: {e}")

    # 3. LLM judgment for ambiguous/conflicting cases
    if results and any(r for r in results.values() if isinstance(r, dict)):
        # Check if we have conflicting acceptance signals
        statuses = set()
        for src, r in results.items():
            if isinstance(r, dict) and r.get("status"):
                statuses.add(r["status"])
            if isinstance(r, dict) and r.get("accepted") is not None:
                statuses.add("accepted" if r["accepted"] else "rejected")

        if len(statuses) > 1 or (statuses and "unknown" in statuses):
            # Conflicting or uncertain — use LLM judgment
            try:
                from backend.services.vlm_extraction_service import judge_acceptance_status
                from backend.models.metadata import MetadataObservation
                from sqlalchemy import select as sa_select

                obs_result = await session.execute(
                    sa_select(MetadataObservation).where(
                        MetadataObservation.entity_id == paper_id,
                        MetadataObservation.field_name == "acceptance_status",
                    ).order_by(MetadataObservation.authority_rank)
                )
                obs_list = [
                    {"source": o.source, "value": o.value_json, "confidence": o.confidence}
                    for o in obs_result.scalars().all()
                ]
                if obs_list:
                    judgment = await judge_acceptance_status(
                        session, paper_id, title, obs_list
                    )
                    if judgment.get("accepted") is not None:
                        await record_observation(
                            session,
                            entity_type="paper",
                            entity_id=paper_id,
                            field_name="acceptance_status",
                            value="accepted" if judgment["accepted"] else "rejected",
                            source="llm_judgment",
                            confidence=judgment.get("confidence", 0.5),
                        )
                        results["llm_judgment"] = judgment
                        sources_checked.append("llm_judgment")
            except Exception as e:
                logger.warning(f"LLM acceptance judgment failed for {paper_id}: {e}")

    # 4. Resolve canonical from all observations
    canonical = await resolve_paper_metadata(session, paper_id)

    return {
        "venue": canonical.canonical_venue or current_venue,
        "acceptance_status": canonical.canonical_acceptance_status or "unknown",
        "year": canonical.canonical_year or current_year,
        "sources_checked": sources_checked,
        "results": results,
        "conflicts": canonical.unresolved_conflicts,
    }
