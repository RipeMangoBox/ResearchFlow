"""Tests for assertion_service — lifecycle management."""

import uuid
import pytest
import pytest_asyncio

from backend.services import assertion_service, delta_card_service
from backend.models.assertion import GraphNode, GraphAssertion, GraphAssertionEvidence
from backend.models.review import ReviewTask


@pytest_asyncio.fixture
async def two_nodes(session):
    """Create two graph nodes for testing assertions."""
    node_a = GraphNode(
        node_type="idea_delta", ref_table="idea_deltas", ref_id=uuid.uuid4(),
    )
    node_b = GraphNode(
        node_type="evidence", ref_table="evidence_units", ref_id=uuid.uuid4(),
    )
    session.add(node_a)
    session.add(node_b)
    await session.commit()
    return node_a, node_b


@pytest.mark.asyncio
async def test_propose_structural_assertion_auto_publishes(session, two_nodes):
    """Structural edges (supported_by) should auto-publish."""
    node_a, node_b = two_nodes
    assertion = await assertion_service.propose_assertion(
        session, node_a.id, node_b.id, "supported_by",
    )
    assert assertion.status == "published"
    assert assertion.edge_type == "supported_by"


@pytest.mark.asyncio
async def test_propose_high_value_assertion_stays_candidate(session, two_nodes):
    """High-value edges (contradicts) should stay candidate and create review task."""
    node_a, node_b = two_nodes
    assertion = await assertion_service.propose_assertion(
        session, node_a.id, node_b.id, "contradicts",
    )
    assert assertion.status == "candidate"

    # Should have created a review task
    from sqlalchemy import select
    result = await session.execute(
        select(ReviewTask).where(
            ReviewTask.target_type == "assertion",
            ReviewTask.target_id == assertion.id,
        )
    )
    task = result.scalar_one_or_none()
    assert task is not None
    assert task.task_type == "human_review"
    assert task.priority == 2  # contradicts gets priority 2


@pytest.mark.asyncio
async def test_audit_assertion_with_evidence(session, two_nodes):
    """Audit should count supporting evidence."""
    node_a, node_b = two_nodes
    assertion = await assertion_service.propose_assertion(
        session, node_a.id, node_b.id, "supported_by",
        evidence_unit_ids=[node_b.ref_id],  # FK won't resolve but link is created
    )

    audit = await assertion_service.audit_assertion(session, assertion.id)
    assert audit["assertion_id"] == str(assertion.id)
    assert audit["supporting_evidence"] >= 1


@pytest.mark.asyncio
async def test_publish_assertion(session, two_nodes):
    """Publishing should change status to published."""
    node_a, node_b = two_nodes
    assertion = await assertion_service.propose_assertion(
        session, node_a.id, node_b.id, "contradicts",
    )
    assert assertion.status == "candidate"

    # Add evidence so publish gate passes
    link = GraphAssertionEvidence(
        assertion_id=assertion.id, evidence_unit_id=node_b.ref_id, role="supports",
    )
    session.add(link)
    await session.flush()

    published = await assertion_service.publish_assertion(
        session, assertion.id, reviewed_by="test_user",
    )
    assert published.status == "published"
    assert published.reviewed_by == "test_user"
    assert published.assertion_source == "human_verified"


@pytest.mark.asyncio
async def test_reject_assertion(session, two_nodes):
    """Rejecting should set status and record reason."""
    node_a, node_b = two_nodes
    assertion = await assertion_service.propose_assertion(
        session, node_a.id, node_b.id, "transferable_to",
    )

    rejected = await assertion_service.reject_assertion(
        session, assertion.id, "reviewer1", "Not enough evidence",
    )
    assert rejected.status == "rejected"
    assert rejected.metadata_.get("rejection_reason") == "Not enough evidence"


@pytest.mark.asyncio
async def test_deprecate_assertion(session, two_nodes):
    """Deprecating with superseded_by should set status to superseded."""
    node_a, node_b = two_nodes
    old = await assertion_service.propose_assertion(
        session, node_a.id, node_b.id, "supported_by",
    )
    new = await assertion_service.propose_assertion(
        session, node_a.id, node_b.id, "supported_by",
    )

    deprecated = await assertion_service.deprecate_assertion(
        session, old.id, superseded_by=new.id,
    )
    assert deprecated.status == "superseded"


@pytest.mark.asyncio
async def test_get_assertion_with_evidence(session, two_nodes):
    """Full detail fetch should include node and evidence info."""
    node_a, node_b = two_nodes
    assertion = await assertion_service.propose_assertion(
        session, node_a.id, node_b.id, "supported_by",
    )

    detail = await assertion_service.get_assertion_with_evidence(session, assertion.id)
    assert detail is not None
    assert detail["from_node"]["node_type"] == "idea_delta"
    assert detail["to_node"]["node_type"] == "evidence"
