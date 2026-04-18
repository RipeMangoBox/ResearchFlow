"""Metadata resolver — picks canonical values from observation ledger.

For each field, selects the observation with the highest authority_rank
(lowest number = highest authority). Flags unresolved conflicts.
"""

import logging
import uuid
from datetime import datetime

from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.metadata import (
    CanonicalPaperMetadata,
    MetadataObservation,
    get_authority_rank,
)

logger = logging.getLogger(__name__)

RESOLVER_VERSION = "v1"


async def record_observation(
    session: AsyncSession,
    *,
    entity_type: str,
    entity_id: uuid.UUID,
    field_name: str,
    value: dict | str | int | list,
    source: str,
    source_url: str | None = None,
    confidence: float = 0.5,
) -> MetadataObservation:
    """Record a single metadata observation from an external source."""
    authority = get_authority_rank(field_name, source)

    # Wrap scalar values in a dict for JSONB storage
    if not isinstance(value, dict):
        value_json = {"value": value}
    else:
        value_json = value

    obs = MetadataObservation(
        entity_type=entity_type,
        entity_id=entity_id,
        field_name=field_name,
        value_json=value_json,
        source=source,
        source_url=source_url,
        confidence=confidence,
        authority_rank=authority,
    )
    session.add(obs)
    await session.flush()
    return obs


async def resolve_paper_metadata(
    session: AsyncSession,
    paper_id: uuid.UUID,
) -> CanonicalPaperMetadata:
    """Resolve canonical metadata for a paper from all observations.

    For each field, picks the observation with the lowest authority_rank
    (highest authority). If two sources have the same rank, picks the
    most recent. Flags conflicts where values differ significantly.
    """
    # Fetch all observations for this paper
    result = await session.execute(
        select(MetadataObservation)
        .where(
            MetadataObservation.entity_type == "paper",
            MetadataObservation.entity_id == paper_id,
        )
        .order_by(MetadataObservation.authority_rank, MetadataObservation.observed_at.desc())
    )
    observations = list(result.scalars().all())

    if not observations:
        # Return empty canonical (or existing)
        canonical = await session.get(CanonicalPaperMetadata, paper_id)
        if canonical:
            return canonical
        canonical = CanonicalPaperMetadata(paper_id=paper_id)
        session.add(canonical)
        await session.flush()
        return canonical

    # Group by field_name
    by_field: dict[str, list[MetadataObservation]] = {}
    for obs in observations:
        by_field.setdefault(obs.field_name, []).append(obs)

    # Resolve each field
    selected_ids: list[str] = []
    conflicts: list[dict] = []
    resolved: dict[str, any] = {}

    for field_name, obs_list in by_field.items():
        # Already sorted by authority_rank, then by recency
        best = obs_list[0]
        resolved[field_name] = best.value_json
        selected_ids.append(str(best.id))

        # Check for conflicts (different values from different sources)
        if len(obs_list) > 1:
            unique_values = set()
            for o in obs_list:
                # Simple string comparison of JSON values
                val_str = str(o.value_json.get("value", o.value_json))
                unique_values.add(val_str)

            if len(unique_values) > 1:
                conflict_group_id = uuid.uuid4()
                conflicts.append({
                    "field": field_name,
                    "sources": [
                        {"source": o.source, "value": o.value_json, "rank": o.authority_rank}
                        for o in obs_list[:5]  # top 5
                    ],
                    "reason": f"{len(unique_values)} different values from {len(obs_list)} sources",
                })
                # Mark conflict group
                for o in obs_list:
                    o.conflict_group_id = conflict_group_id

    # Upsert canonical
    canonical = await session.get(CanonicalPaperMetadata, paper_id)
    if not canonical:
        canonical = CanonicalPaperMetadata(paper_id=paper_id)
        session.add(canonical)

    # Map resolved fields to canonical columns
    _map_to_canonical(canonical, resolved)
    canonical.selected_observation_ids = selected_ids
    canonical.unresolved_conflicts = conflicts if conflicts else None
    canonical.resolved_at = datetime.utcnow()
    canonical.resolver_version = RESOLVER_VERSION

    await session.flush()
    return canonical


def _map_to_canonical(canonical: CanonicalPaperMetadata, resolved: dict) -> None:
    """Map resolved observation values to canonical columns."""
    field_mapping = {
        "title": "canonical_title",
        "authors": "canonical_authors",
        "affiliations": "canonical_affiliations",
        "venue": "canonical_venue",
        "acceptance_status": "canonical_acceptance_status",
        "year": "canonical_year",
        "citation_count": "canonical_citation_count",
        "code_url": "canonical_code_url",
    }

    for field_name, column_name in field_mapping.items():
        if field_name in resolved:
            value = resolved[field_name]
            # Unwrap {"value": x} format
            if isinstance(value, dict) and "value" in value and len(value) == 1:
                value = value["value"]
            setattr(canonical, column_name, value)


async def get_conflicts(
    session: AsyncSession,
    paper_id: uuid.UUID,
) -> list[dict]:
    """Get unresolved metadata conflicts for a paper."""
    canonical = await session.get(CanonicalPaperMetadata, paper_id)
    if not canonical or not canonical.unresolved_conflicts:
        return []
    return canonical.unresolved_conflicts
