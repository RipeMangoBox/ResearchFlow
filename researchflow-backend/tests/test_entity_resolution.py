"""Tests for entity_resolution_service — alias normalization."""

import uuid
import pytest
import pytest_asyncio

from backend.services import entity_resolution_service


@pytest.mark.asyncio
async def test_resolve_mechanism_exact(session, sample_mechanism):
    """Exact name match should resolve."""
    mf = await entity_resolution_service.resolve_mechanism(session, sample_mechanism.name)
    assert mf is not None
    assert mf.id == sample_mechanism.id


@pytest.mark.asyncio
async def test_resolve_mechanism_case_insensitive(session, sample_mechanism):
    """Case-insensitive match should work."""
    mf = await entity_resolution_service.resolve_mechanism(session, sample_mechanism.name.upper())
    assert mf is not None
    assert mf.id == sample_mechanism.id


@pytest.mark.asyncio
async def test_resolve_mechanism_via_legacy_alias(session, sample_mechanism):
    """Should resolve via MechanismFamily.aliases array."""
    mf = await entity_resolution_service.resolve_mechanism(session, "DDPM")
    assert mf is not None
    assert mf.id == sample_mechanism.id


@pytest.mark.asyncio
async def test_resolve_mechanism_not_found(session):
    """Unknown mechanism should return None."""
    mf = await entity_resolution_service.resolve_mechanism(session, "nonexistent_xyz")
    assert mf is None


@pytest.mark.asyncio
async def test_register_alias_dedup(session, sample_mechanism):
    """Registering same alias twice should not create duplicate."""
    alias1 = await entity_resolution_service.register_alias(
        session, "mechanism_family", sample_mechanism.id, "flow_matching",
    )
    alias2 = await entity_resolution_service.register_alias(
        session, "mechanism_family", sample_mechanism.id, "flow_matching",
    )
    assert alias1.id == alias2.id


@pytest.mark.asyncio
async def test_register_alias_case_insensitive_dedup(session, sample_mechanism):
    """Case-insensitive dedup should prevent 'DDPM' and 'ddpm' duplicates."""
    a1 = await entity_resolution_service.register_alias(
        session, "mechanism_family", sample_mechanism.id, "FlowMatch",
    )
    a2 = await entity_resolution_service.register_alias(
        session, "mechanism_family", sample_mechanism.id, "flowmatch",
    )
    assert a1.id == a2.id


@pytest.mark.asyncio
async def test_resolve_via_registered_alias(session, sample_mechanism):
    """After registering an alias, resolve should find it."""
    await entity_resolution_service.register_alias(
        session, "mechanism_family", sample_mechanism.id, "denoising_diffusion", confidence=0.95,
    )
    mf = await entity_resolution_service.resolve_mechanism(session, "denoising_diffusion")
    assert mf is not None
    assert mf.id == sample_mechanism.id


@pytest.mark.asyncio
async def test_bulk_register_aliases(session, sample_mechanism):
    """Bulk registration should skip duplicates and count new ones."""
    aliases = [
        {"entity_type": "mechanism_family", "entity_id": sample_mechanism.id, "alias": "new_alias_1"},
        {"entity_type": "mechanism_family", "entity_id": sample_mechanism.id, "alias": "new_alias_2"},
        {"entity_type": "mechanism_family", "entity_id": sample_mechanism.id, "alias": "new_alias_1"},  # dup
    ]
    count = await entity_resolution_service.bulk_register_aliases(session, aliases)
    assert count == 2  # 2 unique


@pytest.mark.asyncio
async def test_list_aliases(session, sample_mechanism):
    """Listing should return registered aliases."""
    await entity_resolution_service.register_alias(
        session, "mechanism_family", sample_mechanism.id, "test_list_alias",
    )
    results = await entity_resolution_service.list_aliases(
        session, entity_type="mechanism_family", entity_id=sample_mechanism.id,
    )
    assert any(a["alias"] == "test_list_alias" for a in results)
