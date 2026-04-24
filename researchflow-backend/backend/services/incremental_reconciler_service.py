"""Incremental reconciler service — Step 6 of the 6-step analysis pipeline.

When a new paper is analyzed, reverse-update existing papers:
1. Update same_family_paper_ids on neighbor DeltaCards
2. Re-evaluate lineage edges if new paper is a better baseline
3. Update downstream_count and baseline promotion

Wraps and extends evolution_service.refresh_connections with
concept-aware reconciliation.
"""

import logging
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.delta_card import DeltaCard
from backend.models.lineage import DeltaCardLineage
from backend.models.paper import Paper

logger = logging.getLogger(__name__)


async def reconcile_neighbors(
    session: AsyncSession,
    paper_id: UUID,
    analysis_data: dict,
) -> dict:
    """Reverse-update neighbors after a new paper is analyzed.

    Steps:
    1. refresh_connections (same_family, downstream_count) — delegates to evolution_service
    2. Check if new paper should be added to neighbors' same_family lists
    3. If new paper is structural, check if it should become a baseline for others
    """
    stats = {"connections_refreshed": False, "neighbors_updated": 0, "baseline_candidates": 0}

    # ── Step 1: Core connection refresh ───────────────────────────
    try:
        from backend.services.evolution_service import refresh_connections
        conn_stats = await refresh_connections(session, paper_id)
        stats["connections_refreshed"] = True
        stats.update({f"conn_{k}": v for k, v in conn_stats.items()})
    except Exception as e:
        logger.warning(f"Connection refresh failed for {paper_id}: {e}")

    # ── Step 2: Update neighbors' same_family_paper_ids ───────────
    paper = await session.get(Paper, paper_id)
    if not paper:
        return stats

    # Get this paper's DeltaCard
    dc = None
    if paper.current_delta_card_id:
        dc = await session.get(DeltaCard, paper.current_delta_card_id)

    if dc and dc.method_node_ids:
        # Find all DeltaCards sharing mechanism families
        neighbors = await session.execute(
            select(DeltaCard).where(
                DeltaCard.method_node_ids.overlap(dc.method_node_ids),
                DeltaCard.paper_id != paper_id,
                DeltaCard.status != "deprecated",
            ).limit(30)
        )
        for neighbor_dc in neighbors.scalars():
            existing = list(neighbor_dc.same_family_paper_ids or [])
            if paper_id not in existing:
                existing.append(paper_id)
                neighbor_dc.same_family_paper_ids = existing[:20]  # cap at 20
                stats["neighbors_updated"] += 1

    # ── Step 3: Check baseline candidacy ──────────────────────────
    if dc and dc.structurality_score and float(dc.structurality_score) >= 0.6:
        # This is a structural paper — count children via delta_card_lineage table
        count_result = await session.execute(
            select(func.count()).select_from(DeltaCardLineage).where(
                DeltaCardLineage.parent_delta_card_id == dc.id,
                DeltaCardLineage.status.in_(["candidate", "published"]),
            )
        )
        downstream_count = count_result.scalar() or 0
        dc.downstream_count = downstream_count

        if downstream_count >= 3 and not dc.is_established_baseline:
            stats["baseline_candidates"] += 1
            logger.info(
                f"Paper {paper_id} is a baseline candidate: "
                f"struct={dc.structurality_score}, downstream={downstream_count}"
            )

    await session.flush()

    logger.info(f"Reconciliation for {paper_id}: {stats}")
    return stats
