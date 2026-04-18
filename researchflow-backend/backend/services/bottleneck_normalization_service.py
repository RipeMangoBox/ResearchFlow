"""Bottleneck normalization — cluster paper claims into canonical bottlenecks.

Workflow:
  1. L4 analysis creates PaperBottleneckClaim (may have bottleneck_id=NULL)
  2. This service clusters unlinked claims by embedding similarity
  3. High similarity (>0.9) → auto-link to existing ProjectBottleneck
  4. Medium (0.7-0.9) → create review task for human decision
  5. Low (<0.7) → create new ProjectBottleneck candidate
"""

import logging
from uuid import UUID

from sqlalchemy import and_, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.research import PaperBottleneckClaim, ProjectBottleneck, ProjectFocusBottleneck

logger = logging.getLogger(__name__)

AUTO_LINK_THRESHOLD = 0.88
CANDIDATE_THRESHOLD = 0.65


async def cluster_unlinked_claims(session: AsyncSession) -> dict:
    """Find unlinked PaperBottleneckClaims and cluster them against existing bottlenecks."""
    from backend.services.embedding_service import embed_text

    # Get unlinked claims
    unlinked = (await session.execute(
        select(PaperBottleneckClaim).where(
            PaperBottleneckClaim.bottleneck_id.is_(None)
        )
    )).scalars().all()

    if not unlinked:
        return {"unlinked": 0, "auto_linked": 0, "new_created": 0, "review_needed": 0}

    # Get existing bottlenecks with their titles for comparison
    existing_bns = (await session.execute(
        select(ProjectBottleneck).where(ProjectBottleneck.status == "active")
    )).scalars().all()

    # Embed existing bottleneck titles
    bn_embeddings = {}
    for bn in existing_bns:
        bn_embeddings[bn.id] = await embed_text(bn.title + " " + (bn.description or ""))

    stats = {"unlinked": len(unlinked), "auto_linked": 0, "new_created": 0, "review_needed": 0}

    for claim in unlinked:
        claim_emb = await embed_text(claim.claim_text)

        # Find best matching bottleneck
        best_bn_id = None
        best_sim = 0.0
        for bn_id, bn_emb in bn_embeddings.items():
            sim = _cosine_sim(claim_emb, bn_emb)
            if sim > best_sim:
                best_sim = sim
                best_bn_id = bn_id

        if best_sim >= AUTO_LINK_THRESHOLD and best_bn_id:
            # Auto-link
            claim.bottleneck_id = best_bn_id
            claim.confidence = round(best_sim, 3)
            stats["auto_linked"] += 1
        elif best_sim >= CANDIDATE_THRESHOLD and best_bn_id:
            # Create review task
            from backend.services.review_service import create_review_task
            await create_review_task(
                session,
                target_type="bottleneck_claim",
                target_id=claim.id,
                task_type="auto_review",
                priority=3,
                notes=f"Claim '{claim.claim_text[:60]}' may match bottleneck (sim={best_sim:.2f})",
            )
            stats["review_needed"] += 1
        else:
            # Create new bottleneck
            new_bn = ProjectBottleneck(
                title=claim.claim_text[:200],
                description=claim.claim_text,
                domain=None,
                status="active",
                related_paper_ids=[claim.paper_id],
            )
            session.add(new_bn)
            await session.flush()
            claim.bottleneck_id = new_bn.id
            claim.confidence = 1.0

            # Add to embedding cache for subsequent claims
            bn_embeddings[new_bn.id] = claim_emb
            stats["new_created"] += 1

    await session.flush()
    return stats


async def normalize_bottlenecks(session: AsyncSession) -> dict:
    """Merge near-duplicate ProjectBottlenecks using embedding similarity."""
    from backend.services.embedding_service import embed_text

    bns = (await session.execute(
        select(ProjectBottleneck).where(ProjectBottleneck.status == "active")
    )).scalars().all()

    if len(bns) < 2:
        return {"total": len(bns), "merged": 0}

    # Compute all embeddings
    embeddings = {}
    for bn in bns:
        embeddings[bn.id] = await embed_text(bn.title)

    merged = 0
    merged_ids = set()

    for i, bn_a in enumerate(bns):
        if bn_a.id in merged_ids:
            continue
        for bn_b in bns[i + 1:]:
            if bn_b.id in merged_ids:
                continue
            sim = _cosine_sim(embeddings[bn_a.id], embeddings[bn_b.id])
            if sim >= 0.92:
                # Merge bn_b into bn_a
                await session.execute(
                    text("UPDATE paper_bottleneck_claims SET bottleneck_id = :target WHERE bottleneck_id = :source"),
                    {"target": bn_a.id, "source": bn_b.id},
                )
                bn_b.status = "merged"
                merged_ids.add(bn_b.id)
                merged += 1

    await session.flush()
    return {"total": len(bns), "merged": merged}


async def get_unlinked_claims(session: AsyncSession, limit: int = 50) -> list[dict]:
    """Return claims not yet linked to a bottleneck."""
    from backend.models.paper import Paper
    result = await session.execute(
        select(PaperBottleneckClaim, Paper)
        .join(Paper, PaperBottleneckClaim.paper_id == Paper.id)
        .where(PaperBottleneckClaim.bottleneck_id.is_(None))
        .order_by(PaperBottleneckClaim.created_at)
        .limit(limit)
    )
    return [
        {
            "id": str(c.id),
            "paper_id": str(c.paper_id),
            "paper_title": p.title,
            "claim_text": c.claim_text,
            "is_fundamental": c.is_fundamental,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c, p in result
    ]


async def create_focus_bottleneck(
    session: AsyncSession,
    bottleneck_id: UUID,
    project_name: str | None = None,
    user_description: str | None = None,
    priority: int = 3,
    negative_constraints: list[str] | None = None,
) -> ProjectFocusBottleneck:
    """Create a project-level focus bottleneck (user decision)."""
    focus = ProjectFocusBottleneck(
        bottleneck_id=bottleneck_id,
        project_name=project_name,
        user_description=user_description,
        priority=priority,
        negative_constraints=negative_constraints,
        status="active",
    )
    session.add(focus)
    await session.flush()
    return focus


async def list_focus_bottlenecks(session: AsyncSession) -> list[dict]:
    """List all active project focus bottlenecks."""
    result = await session.execute(
        select(ProjectFocusBottleneck, ProjectBottleneck)
        .join(ProjectBottleneck, ProjectFocusBottleneck.bottleneck_id == ProjectBottleneck.id)
        .where(ProjectFocusBottleneck.status == "active")
        .order_by(ProjectFocusBottleneck.priority)
    )
    return [
        {
            "id": str(fb.id),
            "bottleneck_id": str(bn.id),
            "bottleneck_title": bn.title,
            "project_name": fb.project_name,
            "user_description": fb.user_description,
            "priority": fb.priority,
            "negative_constraints": fb.negative_constraints,
            "status": fb.status,
        }
        for fb, bn in result
    ]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
