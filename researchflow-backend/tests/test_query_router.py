"""Tests for query_router_service — intent classification and routing."""

import pytest
import pytest_asyncio

from backend.services.query_router_service import _keyword_classify, route_query


class TestKeywordClassify:
    def test_bottleneck_keywords(self):
        intent, score = _keyword_classify("what bottleneck is blocking this direction?")
        assert intent == "bottleneck"
        assert score >= 1

    def test_mechanism_keywords(self):
        intent, score = _keyword_classify("what methods and techniques exist?")
        assert intent == "mechanism"
        assert score >= 1

    def test_lineage_keywords(self):
        intent, score = _keyword_classify("what does this method build on?")
        assert intent == "lineage"
        assert score >= 1

    def test_evidence_keywords(self):
        intent, score = _keyword_classify("show me the ablation and evidence")
        assert intent == "evidence"
        assert score >= 1

    def test_chinese_bottleneck(self):
        intent, score = _keyword_classify("这个方向的瓶颈是什么")
        assert intent == "bottleneck"

    def test_chinese_mechanism(self):
        intent, score = _keyword_classify("有哪些方法和策略")
        assert intent == "mechanism"

    def test_no_keywords_defaults_bottleneck(self):
        intent, score = _keyword_classify("transformer architecture for video")
        assert score == 0  # no keyword match


@pytest.mark.asyncio
async def test_route_query_with_explicit_intent(session):
    """Explicit intent should bypass classification."""
    result = await route_query(session, query="test", intent="evidence", limit=5)
    assert result["intent"] == "evidence"


@pytest.mark.asyncio
async def test_route_query_bottleneck(session):
    result = await route_query(session, query="bottleneck limitation", limit=5)
    assert result["intent"] == "bottleneck"
    assert "project_focus" in result
    assert "paper_claims" in result


@pytest.mark.asyncio
async def test_route_query_mechanism(session):
    result = await route_query(session, query="diffusion method approach", limit=5)
    assert result["intent"] == "mechanism"
    assert "canonical_ideas" in result
    assert "mechanism_families" in result


@pytest.mark.asyncio
async def test_route_query_lineage(session):
    result = await route_query(session, query="evolution history builds on", limit=5)
    assert result["intent"] == "lineage"
    assert "lineage_trees" in result


@pytest.mark.asyncio
async def test_route_query_evidence(session):
    result = await route_query(session, query="ablation experiment evidence code", limit=5)
    assert result["intent"] == "evidence"
    assert "evidence_units" in result
    assert "implementation_units" in result
