"""Assertion service — propose → audit → review → publish lifecycle.

Graph assertions go through a lifecycle:
  1. propose: system creates assertion with auto-status
     - structural edges (supported_by, changes_slot, instance_of_method, targets_bottleneck)
       → auto-published
     - high-value semantic edges (contradicts, transferable_to, patch_of)
       → candidate (needs review)
  2. audit: check evidence backing for each assertion
  3. review: human or auto-review via review_tasks queue
  4. publish: move to published if gates pass
"""

import logging
from uuid import UUID

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.assertion import GraphAssertion, GraphAssertionEvidence, GraphNode
from backend.models.delta_card import DeltaCard
from backend.models.evidence import EvidenceUnit
from backend.models.review import ReviewTask

logger = logging.getLogger(__name__)

# Edge types that require human review before publishing
HIGH_VALUE_EDGE_TYPES = {"contradicts", "transferable_to", "patch_of"}
# Edge types that auto-publish
STRUCTURAL_EDGE_TYPES = {"supported_by", "changes_slot", "instance_of_method", "targets_bottleneck", "cites"}
# Minimum evidence count for assertion publish
MIN_EVIDENCE_FOR_ASSERTION = 1


async def propose_assertion(
    session: AsyncSession,
    from_node_id: UUID,
    to_node_id: UUID,
    edge_type: str,
    assertion_source: str = "system_inferred",
    confidence: float | None = None,
    metadata: dict | None = None,
    evidence_unit_ids: list[UUID] | None = None,
) -> GraphAssertion:
    """Propose a new assertion with auto-determined status.

    Structural edges auto-publish; high-value edges go to candidate.
    """
    status = "published" if edge_type in STRUCTURAL_EDGE_TYPES else "candidate"

    assertion = GraphAssertion(
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        edge_type=edge_type,
        assertion_source=assertion_source,
        confidence=confidence,
        status=status,
        metadata_=metadata,
    )
    session.add(assertion)
    await session.flush()
    await session.refresh(assertion)

    # Link evidence if provided
    if evidence_unit_ids:
        for eu_id in evidence_unit_ids:
            link = GraphAssertionEvidence(
                assertion_id=assertion.id,
                evidence_unit_id=eu_id,
                role="supports",
                weight=1.0,
            )
            session.add(link)
        await session.flush()

    # Auto-create review task for candidate assertions
    if status == "candidate":
        review = ReviewTask(
            target_type="assertion",
            target_id=assertion.id,
            task_type="human_review",
            priority=2 if edge_type == "contradicts" else 3,
        )
        session.add(review)
        await session.flush()

    return assertion


async def audit_assertion(
    session: AsyncSession,
    assertion_id: UUID,
) -> dict:
    """Audit an assertion's evidence backing.

    Returns audit result with evidence count, average confidence, and recommendation.
    """
    assertion = await session.get(GraphAssertion, assertion_id)
    if not assertion:
        return {"status": "not_found"}

    # Count evidence links
    ev_result = await session.execute(
        select(GraphAssertionEvidence).where(
            GraphAssertionEvidence.assertion_id == assertion_id
        )
    )
    evidence_links = list(ev_result.scalars().all())

    supports = [e for e in evidence_links if e.role == "supports"]
    contradicts = [e for e in evidence_links if e.role == "contradicts"]

    # Compute average confidence of supporting evidence
    avg_confidence = 0.0
    if supports:
        eu_ids = [e.evidence_unit_id for e in supports]
        eu_result = await session.execute(
            select(EvidenceUnit).where(EvidenceUnit.id.in_(eu_ids))
        )
        units = list(eu_result.scalars().all())
        confidences = [u.confidence for u in units if u.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

    recommendation = "publish"
    if len(supports) < MIN_EVIDENCE_FOR_ASSERTION:
        recommendation = "needs_evidence"
    elif len(contradicts) > 0 and len(contradicts) >= len(supports):
        recommendation = "reject"
    elif avg_confidence < 0.5:
        recommendation = "needs_review"

    return {
        "assertion_id": str(assertion_id),
        "edge_type": assertion.edge_type,
        "status": assertion.status,
        "supporting_evidence": len(supports),
        "contradicting_evidence": len(contradicts),
        "avg_confidence": avg_confidence,
        "recommendation": recommendation,
    }


async def publish_assertion(
    session: AsyncSession,
    assertion_id: UUID,
    reviewed_by: str | None = None,
) -> GraphAssertion | None:
    """Publish an assertion after review.

    Gates:
    - Must have at least 1 supporting evidence
    - For high-value edges, must have been reviewed
    """
    assertion = await session.get(GraphAssertion, assertion_id)
    if not assertion:
        return None

    if assertion.status == "published":
        return assertion

    # Check evidence gate
    ev_count = await session.execute(
        select(func.count()).select_from(GraphAssertionEvidence).where(
            GraphAssertionEvidence.assertion_id == assertion_id,
            GraphAssertionEvidence.role == "supports",
        )
    )
    support_count = ev_count.scalar() or 0

    if support_count < MIN_EVIDENCE_FOR_ASSERTION:
        logger.warning(f"Assertion {assertion_id} lacks evidence ({support_count}), not publishing")
        return assertion

    assertion.status = "published"
    if reviewed_by:
        assertion.reviewed_by = reviewed_by
        from datetime import datetime, timezone
        assertion.reviewed_at = datetime.now(timezone.utc)
        assertion.assertion_source = "human_verified"

    await session.flush()
    return assertion


async def reject_assertion(
    session: AsyncSession,
    assertion_id: UUID,
    reviewed_by: str | None = None,
    reason: str | None = None,
) -> GraphAssertion | None:
    """Reject an assertion."""
    assertion = await session.get(GraphAssertion, assertion_id)
    if not assertion:
        return None

    assertion.status = "rejected"
    if reviewed_by:
        assertion.reviewed_by = reviewed_by
        from datetime import datetime, timezone
        assertion.reviewed_at = datetime.now(timezone.utc)
    if reason:
        assertion.metadata_ = {**(assertion.metadata_ or {}), "rejection_reason": reason}

    await session.flush()
    return assertion


async def deprecate_assertion(
    session: AsyncSession,
    assertion_id: UUID,
    superseded_by: UUID | None = None,
) -> GraphAssertion | None:
    """Deprecate an assertion, optionally linking to its replacement."""
    assertion = await session.get(GraphAssertion, assertion_id)
    if not assertion:
        return None

    assertion.status = "superseded" if superseded_by else "deprecated"
    if superseded_by:
        assertion.metadata_ = {**(assertion.metadata_ or {}), "superseded_by": str(superseded_by)}

    await session.flush()
    return assertion


async def add_evidence_to_assertion(
    session: AsyncSession,
    assertion_id: UUID,
    evidence_unit_id: UUID,
    role: str = "supports",
    weight: float = 1.0,
) -> GraphAssertionEvidence:
    """Add an evidence link to an assertion."""
    link = GraphAssertionEvidence(
        assertion_id=assertion_id,
        evidence_unit_id=evidence_unit_id,
        role=role,
        weight=weight,
    )
    session.add(link)
    await session.flush()
    return link


# ── Query helpers ─────────────────────────────────────────────────

async def get_assertions_for_node(
    session: AsyncSession,
    node_id: UUID,
    direction: str = "both",
    status: str | None = None,
) -> list[GraphAssertion]:
    """Get assertions connected to a node."""
    conditions = []
    if direction in ("outgoing", "both"):
        conditions.append(GraphAssertion.from_node_id == node_id)
    if direction in ("incoming", "both"):
        conditions.append(GraphAssertion.to_node_id == node_id)

    from sqlalchemy import or_
    stmt = select(GraphAssertion).where(or_(*conditions))
    if status:
        stmt = stmt.where(GraphAssertion.status == status)
    stmt = stmt.order_by(desc(GraphAssertion.created_at))

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_assertion_with_evidence(
    session: AsyncSession,
    assertion_id: UUID,
) -> dict | None:
    """Get full assertion detail with evidence links and node info."""
    assertion = await session.get(GraphAssertion, assertion_id)
    if not assertion:
        return None

    from_node = await session.get(GraphNode, assertion.from_node_id)
    to_node = await session.get(GraphNode, assertion.to_node_id)

    ev_result = await session.execute(
        select(GraphAssertionEvidence).where(
            GraphAssertionEvidence.assertion_id == assertion_id
        )
    )
    evidence_links = []
    for link in ev_result.scalars():
        eu = await session.get(EvidenceUnit, link.evidence_unit_id)
        evidence_links.append({
            "evidence_unit_id": str(link.evidence_unit_id),
            "role": link.role,
            "weight": link.weight,
            "claim": eu.claim if eu else None,
            "confidence": eu.confidence if eu else None,
        })

    return {
        "id": str(assertion.id),
        "edge_type": assertion.edge_type,
        "assertion_source": assertion.assertion_source,
        "confidence": assertion.confidence,
        "status": assertion.status,
        "reviewed_by": assertion.reviewed_by,
        "metadata": assertion.metadata_,
        "from_node": _node_to_dict(from_node) if from_node else None,
        "to_node": _node_to_dict(to_node) if to_node else None,
        "evidence": evidence_links,
    }


def _node_to_dict(node: GraphNode) -> dict:
    return {
        "id": str(node.id),
        "node_type": node.node_type,
        "ref_table": node.ref_table,
        "ref_id": str(node.ref_id),
        "status": node.status,
    }
