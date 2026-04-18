"""Tests for enhanced review_service — overrides, paradigm candidates, lineage."""

import uuid

import pytest
import pytest_asyncio

from backend.models.candidates import ParadigmCandidate
from backend.models.lineage import DeltaCardLineage
from backend.models.review import ReviewTask
from backend.services import review_service


@pytest.mark.asyncio
async def test_create_review_task(session):
    task = await review_service.create_review_task(
        session, target_type="delta_card", target_id=uuid.uuid4(),
        task_type="auto_review", priority=3, notes="test",
    )
    assert task.status == "pending"
    assert task.target_type == "delta_card"
    assert task.task_type == "auto_review"


@pytest.mark.asyncio
async def test_queue_stats_empty(session):
    stats = await review_service.queue_stats(session)
    assert stats["total_pending"] == 0


@pytest.mark.asyncio
async def test_approve_review_not_found(session):
    result = await review_service.approve_review(session, uuid.uuid4(), "tester")
    assert result["status"] == "not_found"


@pytest.mark.asyncio
async def test_reject_review_not_found(session):
    result = await review_service.reject_review(session, uuid.uuid4(), "tester", "bad")
    assert result["status"] == "not_found"


@pytest.mark.asyncio
async def test_apply_override_invalid_target(session):
    with pytest.raises(ValueError, match="Unknown target_type"):
        await review_service.apply_override(
            session, target_type="nonexistent", target_id=uuid.uuid4(),
            field_name="x", new_value="y",
        )


@pytest.mark.asyncio
async def test_apply_override_invalid_field(session):
    with pytest.raises(ValueError, match="not overridable"):
        await review_service.apply_override(
            session, target_type="paper", target_id=uuid.uuid4(),
            field_name="secret_field", new_value="hack",
        )


@pytest.mark.asyncio
async def test_apply_override_paper(session, sample_paper):
    result = await review_service.apply_override(
        session, target_type="paper", target_id=sample_paper.id,
        field_name="importance", new_value="S",
        reason="This paper is seminal", overridden_by="tester",
    )
    assert result["applied"] is True
    assert result["new_value"] == "S"

    # Check override was recorded
    overrides = await review_service.list_overrides(session, target_type="paper")
    assert len(overrides) == 1
    assert overrides[0]["field_name"] == "importance"


@pytest.mark.asyncio
async def test_paradigm_candidate_lifecycle(session):
    """Create → list → promote paradigm candidate."""
    cand = ParadigmCandidate(
        name="test_paradigm_xyz",
        domain="testing",
        description="A test paradigm",
        slots_json=[
            {"name": "encoder", "slot_type": "architecture", "description": "encodes input"},
            {"name": "decoder", "slot_type": "architecture", "description": "decodes output"},
        ],
        trigger_count=3,
        status="pending",
    )
    session.add(cand)
    await session.flush()

    # List candidates
    candidates = await review_service.list_paradigm_candidates(session, status="pending")
    assert len(candidates) == 1
    assert candidates[0]["name"] == "test_paradigm_xyz"

    # Promote
    result = await review_service.promote_paradigm_candidate(
        session, cand.id, reviewer="tester",
    )
    assert result is not None
    assert result["name"] == "test_paradigm_xyz"
    assert result["slots_created"] == 2

    # Verify candidate status updated
    await session.refresh(cand)
    assert cand.status == "approved"


@pytest.mark.asyncio
async def test_reject_paradigm_candidate(session):
    cand = ParadigmCandidate(
        name="bad_paradigm", domain="testing", trigger_count=1, status="pending",
    )
    session.add(cand)
    await session.flush()

    result = await review_service.reject_paradigm_candidate(
        session, cand.id, reviewer="tester", reason="Too niche",
    )
    assert result["status"] == "rejected"

    await session.refresh(cand)
    assert cand.status == "rejected"
    assert cand.review_notes == "Too niche"


@pytest.mark.asyncio
async def test_list_lineage_candidates_empty(session):
    items = await review_service.list_lineage_candidates(session)
    assert items == []
