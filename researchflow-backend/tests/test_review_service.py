"""Tests for review_service — audit queue and cascading decisions."""

import uuid
import pytest
import pytest_asyncio

from backend.services import review_service
from backend.models.assertion import GraphAssertion, GraphNode
from backend.models.delta_card import DeltaCard
from backend.models.graph import IdeaDelta


@pytest_asyncio.fixture
async def review_target_assertion(session):
    """Create a candidate assertion with review task."""
    node_a = GraphNode(node_type="idea_delta", ref_table="idea_deltas", ref_id=uuid.uuid4())
    node_b = GraphNode(node_type="evidence", ref_table="evidence_units", ref_id=uuid.uuid4())
    session.add_all([node_a, node_b])
    await session.flush()

    assertion = GraphAssertion(
        from_node_id=node_a.id, to_node_id=node_b.id,
        edge_type="contradicts", assertion_source="system_inferred",
        status="candidate",
    )
    session.add(assertion)
    await session.flush()
    return assertion


@pytest.mark.asyncio
async def test_create_review_task(session, review_target_assertion):
    """Review task should be created with correct defaults."""
    task = await review_service.create_review_task(
        session, "assertion", review_target_assertion.id,
        notes="Auto-created for contradicts edge",
    )
    assert task.status == "pending"
    assert task.target_type == "assertion"
    assert task.priority == 3


@pytest.mark.asyncio
async def test_list_reviews(session, review_target_assertion):
    """Listing should filter by status."""
    await review_service.create_review_task(session, "assertion", review_target_assertion.id)
    result = await review_service.list_reviews(session, status="pending")
    assert result["total"] >= 1
    assert len(result["tasks"]) >= 1


@pytest.mark.asyncio
async def test_approve_cascades_to_assertion(session, review_target_assertion):
    """Approving a review should publish the target assertion."""
    task = await review_service.create_review_task(
        session, "assertion", review_target_assertion.id,
    )

    result = await review_service.approve_review(session, task.id, "reviewer1")
    assert result["status"] == "approved"
    assert result["cascade"]["assertion_status"] == "published"

    # Verify assertion updated
    await session.refresh(review_target_assertion)
    assert review_target_assertion.status == "published"
    assert review_target_assertion.assertion_source == "human_verified"


@pytest.mark.asyncio
async def test_reject_cascades_to_assertion(session, review_target_assertion):
    """Rejecting should mark assertion as rejected with reason."""
    task = await review_service.create_review_task(
        session, "assertion", review_target_assertion.id,
    )

    result = await review_service.reject_review(
        session, task.id, "reviewer1", reason="Insufficient evidence",
    )
    assert result["status"] == "rejected"

    await session.refresh(review_target_assertion)
    assert review_target_assertion.status == "rejected"


@pytest.mark.asyncio
async def test_queue_stats(session, review_target_assertion):
    """Stats should count by status and type."""
    await review_service.create_review_task(session, "assertion", review_target_assertion.id)
    stats = await review_service.queue_stats(session)
    assert stats["total_pending"] >= 1
    assert "by_status" in stats


@pytest.mark.asyncio
async def test_create_override(session):
    """Human override should record old/new values."""
    target_id = uuid.uuid4()
    override = await review_service.create_override(
        session, "delta_card", target_id,
        field_name="structurality_score",
        old_value=0.3, new_value=0.8,
        reason="Manual re-assessment", overridden_by="hzh",
    )
    assert override.field_name == "structurality_score"
    assert override.new_value == {"value": 0.8}
    assert override.overridden_by == "hzh"


@pytest.mark.asyncio
async def test_list_overrides(session):
    """Listing overrides should return recent ones."""
    target_id = uuid.uuid4()
    await review_service.create_override(
        session, "delta_card", target_id, "score", 0.1, 0.9,
    )
    results = await review_service.list_overrides(session, target_type="delta_card")
    assert len(results) >= 1
