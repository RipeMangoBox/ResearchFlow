"""Tests for delta_card_service — the core v3.1 pipeline."""

import uuid
import pytest
import pytest_asyncio

from backend.services import delta_card_service
from backend.models.delta_card import DeltaCard
from backend.models.graph import IdeaDelta
from backend.models.evidence import EvidenceUnit
from backend.models.assertion import GraphNode, GraphAssertion


@pytest.mark.asyncio
async def test_build_delta_card(session, sample_paper, sample_paradigm, sample_analysis_data):
    """DeltaCard should be created with correct fields from analysis data."""
    paradigm, slots = sample_paradigm

    card = await delta_card_service.build_delta_card(
        session,
        paper_id=sample_paper.id,
        analysis_id=None,
        analysis_data=sample_analysis_data,
        paradigm_id=paradigm.id,
        paradigm_name=paradigm.name,
        slot_ids=[s.id for s in slots],
        changed_slot_ids=[slots[0].id, slots[2].id],  # denoiser, sampling
    )

    assert card is not None
    assert card.paper_id == sample_paper.id
    assert card.frame_id == paradigm.id
    assert card.status == "draft"
    assert card.delta_statement  # non-empty
    assert card.changed_slot_ids is not None
    assert len(card.changed_slot_ids) == 2
    assert card.unchanged_slot_ids is not None
    assert len(card.unchanged_slot_ids) == 1  # conditioning


@pytest.mark.asyncio
async def test_persist_evidence_for_card(session, sample_paper, sample_analysis_data):
    """Evidence units should be linked to DeltaCard."""
    card = await delta_card_service.build_delta_card(
        session, paper_id=sample_paper.id, analysis_id=None,
        analysis_data=sample_analysis_data,
    )

    evidence = await delta_card_service.persist_evidence_for_card(
        session, sample_paper.id, None, card.id,
        sample_analysis_data["evidence_units"],
    )

    assert len(evidence) == 3
    assert all(eu.delta_card_id == card.id for eu in evidence)
    assert all(eu.paper_id == sample_paper.id for eu in evidence)

    # Card should have evidence_refs updated
    await session.refresh(card)
    assert card.evidence_refs is not None
    assert len(card.evidence_refs) == 3
    assert card.evidence_confidence is not None


@pytest.mark.asyncio
async def test_derive_idea_delta(session, sample_paper, sample_analysis_data):
    """IdeaDelta should be derived from DeltaCard with correct links."""
    card = await delta_card_service.build_delta_card(
        session, paper_id=sample_paper.id, analysis_id=None,
        analysis_data=sample_analysis_data,
    )
    evidence = await delta_card_service.persist_evidence_for_card(
        session, sample_paper.id, None, card.id,
        sample_analysis_data["evidence_units"],
    )

    idea = await delta_card_service.derive_idea_delta(
        session, card, evidence,
        changed_slots_graph=[{"slot_name": "denoiser", "change_type": "structural"}],
    )

    assert idea is not None
    assert idea.paper_id == sample_paper.id
    assert idea.delta_card_id == card.id
    assert idea.delta_statement == card.delta_statement
    assert idea.evidence_count == 3
    assert idea.publish_status == "draft"

    # Evidence should be linked to idea
    for eu in evidence:
        await session.refresh(eu)
        assert eu.idea_delta_id == idea.id


@pytest.mark.asyncio
async def test_propose_assertions(session, sample_paper, sample_paradigm, sample_analysis_data):
    """Assertions should be created for idea→evidence, idea→slot, etc."""
    paradigm, slots = sample_paradigm

    card = await delta_card_service.build_delta_card(
        session, paper_id=sample_paper.id, analysis_id=None,
        analysis_data=sample_analysis_data, paradigm_id=paradigm.id,
    )
    evidence = await delta_card_service.persist_evidence_for_card(
        session, sample_paper.id, None, card.id,
        sample_analysis_data["evidence_units"],
    )
    idea = await delta_card_service.derive_idea_delta(
        session, card, evidence,
        changed_slots_graph=[{"slot_name": "denoiser", "change_type": "structural"}],
    )

    assertions = await delta_card_service.propose_assertions(
        session, idea, evidence,
        paradigm_slots=[{"id": s.id, "name": s.name} for s in slots],
    )

    # Should have: 3 supported_by + 1 changes_slot (denoiser) = 4
    assert len(assertions) >= 4
    edge_types = [a.edge_type for a in assertions]
    assert "supported_by" in edge_types
    assert "changes_slot" in edge_types


@pytest.mark.asyncio
async def test_full_pipeline(session, sample_paper, sample_paradigm, sample_analysis_data):
    """Full pipeline should produce delta_card + idea + evidence + assertions."""
    paradigm, slots = sample_paradigm

    result = await delta_card_service.run_delta_card_pipeline(
        session,
        paper_id=sample_paper.id,
        analysis_id=None,
        analysis_data=sample_analysis_data,
        paradigm_id=paradigm.id,
        paradigm_name=paradigm.name,
        slots=[{"id": s.id, "name": s.name} for s in slots],
        changed_slots_graph=[
            {"slot_name": "denoiser", "change_type": "structural"},
            {"slot_name": "sampling", "change_type": "structural"},
        ],
    )

    assert "delta_card" in result
    assert "idea_delta" in result
    assert "evidence_units" in result
    assert "assertions" in result

    dc = result["delta_card"]
    idea = result["idea_delta"]
    assert dc.paper_id == sample_paper.id
    assert idea.delta_card_id == dc.id
    assert len(result["evidence_units"]) == 3
    assert len(result["assertions"]) >= 4


@pytest.mark.asyncio
async def test_check_and_publish(session, sample_paper, sample_analysis_data):
    """Publish gates should enforce evidence requirements."""
    card = await delta_card_service.build_delta_card(
        session, paper_id=sample_paper.id, analysis_id=None,
        analysis_data=sample_analysis_data,
    )
    evidence = await delta_card_service.persist_evidence_for_card(
        session, sample_paper.id, None, card.id,
        sample_analysis_data["evidence_units"],
    )
    idea = await delta_card_service.derive_idea_delta(session, card, evidence)

    dc_status, idea_status = await delta_card_service.check_and_publish(session, card, idea)

    # Card has no frame_id → should stay draft
    assert dc_status == "draft"
    # Idea has 3 evidence but confidence breakdown may not meet 0.85 threshold
    assert idea_status in ("draft", "auto_published")
