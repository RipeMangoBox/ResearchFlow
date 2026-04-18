"""Baseline comparator service — Step 3 of the 6-step analysis pipeline.

Defense line #2: "比较集不是论文自己说了算"
Auto-fills the comparison set from DB, not just the paper's self-reported baselines.

Queries:
1. Domain baselines — established baselines in the same paradigm frame
2. Same-concept papers — papers sharing mechanism families
3. Same-period strong papers — recent high-structurality papers in the same category
"""

import logging
from uuid import UUID

from sqlalchemy import desc, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.delta_card import DeltaCard
from backend.models.paper import Paper

logger = logging.getLogger(__name__)

# How many comparison papers to return per source
_MAX_PER_SOURCE = 5


async def build_compare_set(
    session: AsyncSession,
    paper_id: UUID,
    analysis_data: dict,
) -> dict:
    """Build an enriched comparison set for a paper.

    Combines the paper's self-reported baselines with DB-discovered comparisons.
    Returns a dict with comparison papers grouped by source.

    Defense line #2: comparison set is NOT just what the paper claims.
    """
    paper = await session.get(Paper, paper_id)
    if not paper:
        return {"error": "Paper not found", "comparisons": []}

    # Get current DeltaCard if exists
    dc = None
    if paper.current_delta_card_id:
        dc = await session.get(DeltaCard, paper.current_delta_card_id)

    frame_id = dc.frame_id if dc else None
    mechanism_ids = dc.mechanism_family_ids if dc else None

    # Collect comparison candidates from multiple sources
    seen_ids: set[UUID] = {paper_id}
    comparisons: list[dict] = []

    # ── Source 1: Domain baselines (established in same paradigm) ──
    if frame_id:
        baselines = await session.execute(
            select(
                DeltaCard.paper_id,
                Paper.title,
                Paper.title_sanitized,
                DeltaCard.structurality_score,
                DeltaCard.downstream_count,
            )
            .join(Paper, Paper.id == DeltaCard.paper_id)
            .where(
                DeltaCard.frame_id == frame_id,
                DeltaCard.is_established_baseline.is_(True),
                DeltaCard.status == "published",
                DeltaCard.paper_id != paper_id,
            )
            .order_by(desc(DeltaCard.downstream_count))
            .limit(_MAX_PER_SOURCE)
        )
        for row in baselines:
            if row.paper_id not in seen_ids:
                seen_ids.add(row.paper_id)
                comparisons.append({
                    "paper_id": str(row.paper_id),
                    "title": row.title,
                    "title_sanitized": row.title_sanitized,
                    "source": "domain_baseline",
                    "structurality_score": float(row.structurality_score) if row.structurality_score else None,
                    "downstream_count": row.downstream_count,
                })

    # ── Source 2: Same mechanism family ──
    if mechanism_ids:
        same_mech = await session.execute(
            select(
                DeltaCard.paper_id,
                Paper.title,
                Paper.title_sanitized,
                DeltaCard.structurality_score,
            )
            .join(Paper, Paper.id == DeltaCard.paper_id)
            .where(
                DeltaCard.mechanism_family_ids.overlap(mechanism_ids),
                DeltaCard.status != "deprecated",
                DeltaCard.paper_id != paper_id,
            )
            .order_by(desc(DeltaCard.structurality_score))
            .limit(_MAX_PER_SOURCE)
        )
        for row in same_mech:
            if row.paper_id not in seen_ids:
                seen_ids.add(row.paper_id)
                comparisons.append({
                    "paper_id": str(row.paper_id),
                    "title": row.title,
                    "title_sanitized": row.title_sanitized,
                    "source": "same_mechanism",
                    "structurality_score": float(row.structurality_score) if row.structurality_score else None,
                })

    # ── Source 3: Same-period strong papers in same category ──
    if paper.category:
        year_range = (
            (paper.year - 1, paper.year + 1)
            if paper.year
            else (2020, 2030)
        )
        strong_peers = await session.execute(
            select(
                Paper.id,
                Paper.title,
                Paper.title_sanitized,
                Paper.structurality_score,
                Paper.venue,
                Paper.year,
            )
            .where(
                Paper.category == paper.category,
                Paper.year.between(*year_range),
                Paper.id != paper_id,
                Paper.state == "l4_deep",
                Paper.structurality_score.isnot(None),
            )
            .order_by(desc(Paper.structurality_score))
            .limit(_MAX_PER_SOURCE)
        )
        for row in strong_peers:
            if row.id not in seen_ids:
                seen_ids.add(row.id)
                comparisons.append({
                    "paper_id": str(row.id),
                    "title": row.title,
                    "title_sanitized": row.title_sanitized,
                    "source": "strong_peer",
                    "structurality_score": float(row.structurality_score) if row.structurality_score else None,
                    "venue": row.venue,
                    "year": row.year,
                })

    # ── Source 4: Paper's self-reported baselines (from LLM output) ──
    reported_titles = analysis_data.get("baseline_paper_titles", [])
    if reported_titles:
        for title in reported_titles[:5]:
            if not isinstance(title, str):
                continue
            match = await session.execute(
                select(Paper.id, Paper.title, Paper.title_sanitized)
                .where(func.lower(Paper.title) == title.lower())
                .limit(1)
            )
            row = match.fetchone()
            if row and row.id not in seen_ids:
                seen_ids.add(row.id)
                comparisons.append({
                    "paper_id": str(row.id),
                    "title": row.title,
                    "title_sanitized": row.title_sanitized,
                    "source": "self_reported",
                })

    # ── Update DeltaCard with comparison baseline_paper_ids ──
    if dc and comparisons:
        baseline_ids = [
            UUID(c["paper_id"]) for c in comparisons
            if c["source"] in ("domain_baseline", "self_reported")
        ]
        if baseline_ids:
            dc.baseline_paper_ids = baseline_ids
            await session.flush()

    logger.info(
        f"Compare set for {paper_id}: {len(comparisons)} papers "
        f"({sum(1 for c in comparisons if c['source'] == 'domain_baseline')} baselines, "
        f"{sum(1 for c in comparisons if c['source'] == 'same_mechanism')} same-mech, "
        f"{sum(1 for c in comparisons if c['source'] == 'strong_peer')} peers, "
        f"{sum(1 for c in comparisons if c['source'] == 'self_reported')} self-reported)"
    )

    return {
        "paper_id": str(paper_id),
        "total_comparisons": len(comparisons),
        "comparisons": comparisons,
    }
