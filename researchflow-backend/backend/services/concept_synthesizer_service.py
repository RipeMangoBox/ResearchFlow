"""Concept synthesizer service — Step 5 of the 6-step analysis pipeline.

When a new paper arrives, check if its mechanism/concept already exists
as a CanonicalIdea or MechanismFamily. If so, link; if not, create.

Produces cross-paper concept synthesis, not per-paper isolated concepts.
"""

import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.canonical_idea import CanonicalIdea, ContributionToCanonicalIdea
from backend.models.graph import MechanismFamily
from backend.models.delta_card import DeltaCard
from backend.models.paper import Paper

logger = logging.getLogger(__name__)


async def synthesize_concepts(
    session: AsyncSession,
    paper_id: UUID,
    analysis_data: dict,
) -> dict:
    """Link or create concepts for a newly analyzed paper.

    Steps:
    1. Resolve mechanism_family from same_family_method or existing DB match
    2. Resolve or create CanonicalIdea from the method's core contribution
    3. Create ContributionToCanonicalIdea linking paper → idea
    4. Update MechanismFamily if paper adds new understanding

    Returns summary of what was linked/created.
    """
    paper = await session.get(Paper, paper_id)
    if not paper:
        return {"error": "Paper not found"}

    stats = {"mechanism_linked": False, "idea_linked": False, "idea_created": False}

    # ── 1. Resolve MechanismFamily ────────────────────────────────
    family_name = analysis_data.get("same_family_method")
    mf = None

    if family_name:
        # Try exact match
        result = await session.execute(
            select(MechanismFamily).where(
                func.lower(MechanismFamily.name) == family_name.lower()
            ).limit(1)
        )
        mf = result.scalar_one_or_none()

        if not mf:
            # Try alias match
            result = await session.execute(
                select(MechanismFamily).where(
                    MechanismFamily.aliases.any(family_name)
                ).limit(1)
            )
            mf = result.scalar_one_or_none()

        if not mf:
            # Create new MechanismFamily
            mf = MechanismFamily(
                name=family_name,
                domain=paper.category,
                description=analysis_data.get("core_intuition", "")[:500],
            )
            session.add(mf)
            await session.flush()
            logger.info(f"Created MechanismFamily: {family_name}")

        # Link paper to mechanism
        paper.mechanism_family = mf.name
        stats["mechanism_linked"] = True

        # Update DeltaCard with mechanism family ID
        if paper.current_delta_card_id:
            dc = await session.get(DeltaCard, paper.current_delta_card_id)
            if dc:
                existing_ids = list(dc.mechanism_family_ids or [])
                if mf.id not in existing_ids:
                    existing_ids.append(mf.id)
                    dc.mechanism_family_ids = existing_ids

    # ── 2. Resolve or create CanonicalIdea ────────────────────────
    core_intuition = analysis_data.get("core_intuition", "")
    delta_card_data = analysis_data.get("delta_card", {})
    primary_gain = delta_card_data.get("primary_gain_source", "") if isinstance(delta_card_data, dict) else ""

    # Build a concept title from the method's core contribution
    concept_title = primary_gain or family_name or ""
    if not concept_title and core_intuition:
        # Use first sentence of core_intuition as title
        concept_title = core_intuition.split("。")[0].split(".")[0][:100]

    if not concept_title:
        logger.info(f"No concept title derivable for paper {paper_id}")
        await session.flush()
        return stats

    # Search for existing CanonicalIdea
    idea = None
    result = await session.execute(
        select(CanonicalIdea).where(
            func.lower(CanonicalIdea.title) == concept_title.lower(),
            CanonicalIdea.status.in_(["candidate", "established"]),
        ).limit(1)
    )
    idea = result.scalar_one_or_none()

    if not idea and mf:
        # Check if there's an idea linked to this mechanism family
        result = await session.execute(
            select(CanonicalIdea).where(
                CanonicalIdea.mechanism_family_id == mf.id,
                CanonicalIdea.status.in_(["candidate", "established"]),
            ).limit(1)
        )
        idea = result.scalar_one_or_none()

    if idea:
        # Link paper to existing idea
        idea.contribution_count = (idea.contribution_count or 0) + 1
        stats["idea_linked"] = True
    else:
        # Create new CanonicalIdea
        idea = CanonicalIdea(
            title=concept_title[:200],
            description=core_intuition[:1000] if core_intuition else "",
            domain=paper.category,
            mechanism_family_id=mf.id if mf else None,
            contribution_count=1,
            status="candidate",
        )
        session.add(idea)
        await session.flush()
        stats["idea_created"] = True
        logger.info(f"Created CanonicalIdea: {concept_title[:60]}")

    # ── 3. Create ContributionToCanonicalIdea ─────────────────────
    # Check if link already exists
    existing_link = await session.execute(
        select(ContributionToCanonicalIdea).where(
            ContributionToCanonicalIdea.canonical_idea_id == idea.id,
            ContributionToCanonicalIdea.paper_id == paper_id,
        ).limit(1)
    )
    if not existing_link.scalar_one_or_none():
        # Determine contribution type
        structurality = analysis_data.get("structurality_score", 0)
        if structurality and float(structurality) >= 0.7:
            contrib_type = "origin"
        elif structurality and float(structurality) >= 0.4:
            contrib_type = "extension"
        else:
            contrib_type = "instance"

        # Get idea_delta if exists
        from backend.models.graph import IdeaDelta
        idea_delta_result = await session.execute(
            select(IdeaDelta.id).where(
                IdeaDelta.paper_id == paper_id
            ).order_by(IdeaDelta.created_at.desc()).limit(1)
        )
        idea_delta_id = idea_delta_result.scalar_one_or_none()

        contribution = ContributionToCanonicalIdea(
            canonical_idea_id=idea.id,
            paper_id=paper_id,
            idea_delta_id=idea_delta_id,
            contribution_type=contrib_type,
            confidence=0.7,
            source="system_inferred",
        )
        session.add(contribution)

    await session.flush()

    logger.info(
        f"Concept synthesis for {paper_id}: "
        f"mechanism={mf.name if mf else 'none'}, "
        f"idea={idea.title[:40]}, "
        f"stats={stats}"
    )
    return stats
