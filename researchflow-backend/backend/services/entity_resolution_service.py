"""Entity resolution service — alias normalization and deduplication.

Resolves variant entity names to canonical forms:
  - "DDPM" → diffusion mechanism family
  - "flow matching" / "flow-matching" / "FM" → flow_matching
  - "vision-language model" / "VLM" → vlm_standard bottleneck

Uses the aliases table for fast lookup and supports auto-detection
of new aliases from LLM analysis output.
"""

import logging
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.graph import Slot
from backend.models.method import MethodNode
from backend.models.research import ProjectBottleneck
from backend.models.review import Alias

logger = logging.getLogger(__name__)


async def resolve_mechanism(
    session: AsyncSession,
    name: str,
) -> MethodNode | None:
    """Resolve a mechanism name to its canonical MethodNode.

    Checks: exact name match → aliases table → MethodNode.aliases array.
    """
    # 1. Exact match
    result = await session.execute(
        select(MethodNode).where(
            func.lower(MethodNode.name) == name.lower()
        )
    )
    mf = result.scalar_one_or_none()
    if mf:
        return mf

    # 2. Aliases table
    alias_result = await session.execute(
        select(Alias).where(
            Alias.entity_type == "method_family",
            func.lower(Alias.alias) == name.lower(),
        ).order_by(Alias.confidence.desc().nullslast()).limit(1)
    )
    alias = alias_result.scalar_one_or_none()
    if alias:
        return await session.get(MethodNode, alias.entity_id)

    # 3. MethodNode.aliases array (legacy)
    result = await session.execute(
        select(MethodNode).where(
            MethodNode.aliases.contains([name])
        ).limit(1)
    )
    mf = result.scalar_one_or_none()
    if mf:
        # Auto-register in aliases table for faster future lookups
        await register_alias(session, "method_family", mf.id, name, "auto_detected", 0.9)
        return mf

    return None


async def resolve_bottleneck(
    session: AsyncSession,
    name: str,
) -> ProjectBottleneck | None:
    """Resolve a bottleneck name to its canonical ProjectBottleneck."""
    # 1. Exact title match
    result = await session.execute(
        select(ProjectBottleneck).where(
            func.lower(ProjectBottleneck.title) == name.lower()
        )
    )
    bn = result.scalar_one_or_none()
    if bn:
        return bn

    # 2. Aliases table
    alias_result = await session.execute(
        select(Alias).where(
            Alias.entity_type == "bottleneck",
            func.lower(Alias.alias) == name.lower(),
        ).order_by(Alias.confidence.desc().nullslast()).limit(1)
    )
    alias = alias_result.scalar_one_or_none()
    if alias:
        return await session.get(ProjectBottleneck, alias.entity_id)

    # 3. Fuzzy title match (ilike)
    result = await session.execute(
        select(ProjectBottleneck).where(
            ProjectBottleneck.title.ilike(f"%{name}%")
        ).limit(1)
    )
    return result.scalar_one_or_none()


async def resolve_slot(
    session: AsyncSession,
    name: str,
    paradigm_id: UUID | None = None,
) -> Slot | None:
    """Resolve a slot name, optionally scoped to a paradigm."""
    stmt = select(Slot).where(func.lower(Slot.name) == name.lower())
    if paradigm_id:
        stmt = stmt.where(Slot.paradigm_id == paradigm_id)
    result = await session.execute(stmt.limit(1))
    slot = result.scalar_one_or_none()
    if slot:
        return slot

    # Aliases table
    alias_result = await session.execute(
        select(Alias).where(
            Alias.entity_type == "slot",
            func.lower(Alias.alias) == name.lower(),
        ).order_by(Alias.confidence.desc().nullslast()).limit(1)
    )
    alias = alias_result.scalar_one_or_none()
    if alias:
        return await session.get(Slot, alias.entity_id)

    return None


# ── Alias management ──────────────────────────────────────────────

async def register_alias(
    session: AsyncSession,
    entity_type: str,
    entity_id: UUID,
    alias_text: str,
    source: str = "auto_detected",
    confidence: float | None = None,
) -> Alias:
    """Register a new alias, skipping if duplicate."""
    result = await session.execute(
        select(Alias).where(
            Alias.entity_type == entity_type,
            Alias.entity_id == entity_id,
            func.lower(Alias.alias) == alias_text.lower(),
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        return existing

    alias = Alias(
        entity_type=entity_type,
        entity_id=entity_id,
        alias=alias_text,
        source=source,
        confidence=confidence,
    )
    session.add(alias)
    await session.flush()
    return alias


async def bulk_register_aliases(
    session: AsyncSession,
    aliases: list[dict],
) -> int:
    """Bulk register aliases from LLM output.

    Each dict: {entity_type, entity_id, alias, confidence?}
    Returns count of newly registered aliases.
    """
    count = 0
    for a in aliases:
        result = await session.execute(
            select(Alias).where(
                Alias.entity_type == a["entity_type"],
                Alias.entity_id == a["entity_id"],
                func.lower(Alias.alias) == a["alias"].lower(),
            )
        )
        if not result.scalar_one_or_none():
            alias = Alias(
                entity_type=a["entity_type"],
                entity_id=a["entity_id"],
                alias=a["alias"],
                source="auto_detected",
                confidence=a.get("confidence"),
            )
            session.add(alias)
            count += 1

    if count:
        await session.flush()
    return count


async def list_aliases(
    session: AsyncSession,
    entity_type: str | None = None,
    entity_id: UUID | None = None,
) -> list[dict]:
    """List aliases, optionally filtered."""
    stmt = select(Alias)
    if entity_type:
        stmt = stmt.where(Alias.entity_type == entity_type)
    if entity_id:
        stmt = stmt.where(Alias.entity_id == entity_id)
    stmt = stmt.order_by(Alias.entity_type, Alias.alias)

    result = await session.execute(stmt)
    return [
        {
            "id": str(a.id),
            "entity_type": a.entity_type,
            "entity_id": str(a.entity_id),
            "alias": a.alias,
            "source": a.source,
            "confidence": a.confidence,
        }
        for a in result.scalars()
    ]
