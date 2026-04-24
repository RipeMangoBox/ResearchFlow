"""Quality scoring service — 4 quality metrics for the knowledge graph.

Metrics:
1. Idea correctness — structural completeness of an DeltaCard
2. Evidence grounding — evidence coverage and strong-edge grounding
3. DeltaCard completeness — field-level completeness check
4. KB quality report — aggregate quality across all published entities
"""

import logging
from uuid import UUID

from sqlalchemy import and_, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.delta_card import DeltaCard
from backend.models.evidence import EvidenceUnit


logger = logging.getLogger(__name__)

# Strong semantic edge types that should have linked evidence
STRONG_EDGE_TYPES = {"contradicts", "transferable_to"}


# ── 1. Idea Correctness ──────────────────────────────────────────

async def score_idea_correctness(
    session: AsyncSession,
    delta_card_id: UUID,
) -> dict:
    """Check structural completeness of an DeltaCard.

    Checks:
    - primary_bottleneck set
    - changed_slots non-empty
    - method_node_ids non-empty

    Returns dict with per-field booleans and overall score 0-1.
    """
    idea = await session.get(DeltaCard, delta_card_id)
    if not idea:
        return {"error": "DeltaCard not found", "overall_score": 0.0}

    has_bottleneck = idea.primary_bottleneck_id is not None
    has_changed_slots = bool(idea.changed_slots)
    has_mechanisms = bool(idea.method_node_ids)

    fields = [has_bottleneck, has_changed_slots, has_mechanisms]
    overall = sum(fields) / len(fields)

    return {
        "delta_card_id": str(delta_card_id),
        "has_primary_bottleneck": has_bottleneck,
        "has_changed_slots": has_changed_slots,
        "has_method_node_ids": has_mechanisms,
        "overall_score": round(overall, 3),
    }


# ── 2. Evidence Grounding ────────────────────────────────────────

async def score_evidence_grounding(
    session: AsyncSession,
    delta_card_id: UUID,
) -> dict:
    """Check evidence grounding for an DeltaCard.

    Checks:
    - Has >= 2 evidence units
    - Strong semantic edges (contradicts / transferable_to) have linked evidence

    Returns dict with evidence_count, has_min_evidence, strong_edges_grounded.
    """
    idea = await session.get(DeltaCard, delta_card_id)
    if not idea:
        return {"error": "DeltaCard not found"}

    # Count evidence units
    ev_result = await session.execute(
        select(func.count()).select_from(EvidenceUnit).where(
            EvidenceUnit.delta_card_id == delta_card_id
        )
    )
    evidence_count = ev_result.scalar() or 0
    has_min_evidence = evidence_count >= 2

    # Check strong edges: find edges where this idea is source or target
    # with edge_type in STRONG_EDGE_TYPES, and verify they have evidence_id set
    strong_edges_result = await session.execute(
        select(GraphAssertion).where(
            GraphAssertion.edge_type.in_(STRONG_EDGE_TYPES),
            (
                (GraphAssertion.source_type == "delta_card") & (GraphAssertion.source_id == delta_card_id)
            ) | (
                (GraphAssertion.target_type == "delta_card") & (GraphAssertion.target_id == delta_card_id)
            ),
        )
    )
    strong_edges = list(strong_edges_result.scalars().all())
    total_strong = len(strong_edges)
    grounded_strong = sum(1 for e in strong_edges if e.evidence_id is not None)
    strong_edges_grounded = (total_strong == 0) or (grounded_strong == total_strong)

    return {
        "delta_card_id": str(delta_card_id),
        "evidence_count": evidence_count,
        "has_min_evidence": has_min_evidence,
        "strong_edges_total": total_strong,
        "strong_edges_grounded": strong_edges_grounded,
        "strong_edges_grounded_count": grounded_strong,
    }


# ── 3. DeltaCard Completeness ────────────────────────────────────

async def score_delta_card_completeness(
    session: AsyncSession,
    delta_card_id: UUID,
) -> dict:
    """Check field-level completeness of a DeltaCard.

    Checks presence of:
    - frame_id
    - changed_slot_ids (non-empty)
    - evidence_refs >= 2
    - key_ideas_ranked (non-empty)
    - assumptions (non-empty)
    - failure_modes (non-empty)

    Returns completeness score 0-1.
    """
    card = await session.get(DeltaCard, delta_card_id)
    if not card:
        return {"error": "DeltaCard not found", "completeness_score": 0.0}

    checks = {
        "has_frame_id": card.frame_id is not None,
        "has_changed_slot_ids": bool(card.changed_slot_ids),
        "has_evidence_refs_gte_2": bool(card.evidence_refs) and len(card.evidence_refs) >= 2,
        "has_key_ideas_ranked": bool(card.key_ideas_ranked),
        "has_assumptions": bool(card.assumptions),
        "has_failure_modes": bool(card.failure_modes),
    }

    filled = sum(checks.values())
    total = len(checks)
    completeness = round(filled / total, 3)

    return {
        "delta_card_id": str(delta_card_id),
        **checks,
        "completeness_score": completeness,
    }


# ── 4. KB Quality Report ─────────────────────────────────────────

async def compute_kb_quality_report(session: AsyncSession) -> dict:
    """Aggregate quality across all published DeltaCards and DeltaCards.

    Returns summary stats for the entire knowledge base.
    """
    # Published DeltaCards
    published_ideas_result = await session.execute(
        select(DeltaCard).where(
            DeltaCard.publish_status.in_(["auto_published", "human_verified"])
        )
    )
    published_ideas = list(published_ideas_result.scalars().all())

    # All non-deprecated DeltaCards
    active_cards_result = await session.execute(
        select(DeltaCard).where(DeltaCard.status != "deprecated")
    )
    active_cards = list(active_cards_result.scalars().all())

    # Score each idea
    idea_scores = []
    for idea in published_ideas:
        correctness = await score_idea_correctness(session, idea.id)
        grounding = await score_evidence_grounding(session, idea.id)
        idea_scores.append({
            "delta_card_id": str(idea.id),
            "correctness_score": correctness.get("overall_score", 0.0),
            "has_min_evidence": grounding.get("has_min_evidence", False),
            "strong_edges_grounded": grounding.get("strong_edges_grounded", True),
        })

    # Score each delta card
    card_scores = []
    for card in active_cards:
        completeness = await score_delta_card_completeness(session, card.id)
        card_scores.append({
            "delta_card_id": str(card.id),
            "completeness_score": completeness.get("completeness_score", 0.0),
        })

    # Aggregate
    avg_idea_correctness = (
        sum(s["correctness_score"] for s in idea_scores) / len(idea_scores)
        if idea_scores else 0.0
    )
    ideas_with_min_evidence = sum(1 for s in idea_scores if s["has_min_evidence"])
    ideas_fully_grounded = sum(1 for s in idea_scores if s["strong_edges_grounded"])
    avg_card_completeness = (
        sum(s["completeness_score"] for s in card_scores) / len(card_scores)
        if card_scores else 0.0
    )

    return {
        "published_idea_count": len(published_ideas),
        "active_delta_card_count": len(active_cards),
        "avg_idea_correctness": round(avg_idea_correctness, 3),
        "ideas_with_min_evidence": ideas_with_min_evidence,
        "ideas_with_min_evidence_pct": round(
            ideas_with_min_evidence / len(idea_scores), 3
        ) if idea_scores else 0.0,
        "ideas_fully_grounded": ideas_fully_grounded,
        "ideas_fully_grounded_pct": round(
            ideas_fully_grounded / len(idea_scores), 3
        ) if idea_scores else 0.0,
        "avg_card_completeness": round(avg_card_completeness, 3),
        "idea_scores": idea_scores,
        "card_scores": card_scores,
    }
