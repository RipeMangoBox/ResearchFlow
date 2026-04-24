"""Shared pytest fixtures for ResearchFlow backend tests.

Uses psycopg (sync-compatible async) instead of asyncpg to avoid
event loop issues with pytest-asyncio fixture/test task boundaries.
"""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.database import Base

# Use psycopg async driver (no event loop issues)
TEST_DB_URL = "postgresql+asyncpg://hzh@localhost:5432/researchflow_test"

CLEANUP_TABLES = [
    "taxonomy_versions",
    "contribution_to_canonical_idea", "canonical_ideas",
    "delta_card_lineage",
    "paradigm_candidates", "slot_candidates", "mechanism_candidates",
    "graph_assertion_evidence", "graph_assertions", "graph_nodes",
    "review_queue", "human_overrides", "aliases",
    "evidence_units", "delta_cards",
    "paper_analyses",
    "paper_assets", "paper_versions",
    "method_nodes", "method_edges", "method_applications",
    "paper_facets", "taxonomy_nodes", "taxonomy_edges",
    "direction_cards",
    "digests",
    "papers",
    "venue_papers",
]

_tables_created = False


@pytest_asyncio.fixture
async def session():
    """Per-test: create engine, session, cleanup after."""
    global _tables_created
    engine = create_async_engine(TEST_DB_URL, echo=False)

    if not _tables_created:
        import backend.models  # noqa: F401
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        _tables_created = True

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as sess:
        yield sess

    # Clean data
    async with engine.begin() as conn:
        for table in CLEANUP_TABLES:
            await conn.execute(text(f"DELETE FROM {table}"))

    await engine.dispose()


# ── Helpers ───────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def sample_paper(session):
    from backend.models.paper import Paper
    from backend.models.enums import PaperState
    paper = Paper(
        title="Test: Diffusion for Motion Gen",
        title_sanitized="test_diffusion_motion",
        venue="ICLR", year=2025, category="motion_generation",
        state=PaperState.L4_DEEP,
        tags=["diffusion", "motion"],
        keep_score=0.75,
    )
    session.add(paper)
    await session.commit()
    return paper


@pytest_asyncio.fixture
async def sample_analysis_data():
    return {
        "problem_summary": "Motion generation quality degrades",
        "method_summary": "Replace discrete tokenizer with continuous diffusion",
        "evidence_summary": "FID improved by 15%",
        "core_intuition": "连续扩散去噪保留更多运动细节",
        "is_plugin_patch": False, "worth_deep_read": True,
        "delta_card": {
            "paradigm": "motion_generation_diffusion",
            "is_structural": True, "primary_gain_source": "denoiser",
        },
        "evidence_units": [
            {"atom_type": "evidence", "claim": "FID improved 15%", "confidence": 0.9, "basis": "experiment_backed", "source_section": "Table 2", "conditions": "HumanML3D"},
            {"atom_type": "mechanism", "claim": "Diffusion preserves smoothness", "confidence": 0.85, "basis": "text_stated", "source_section": "Section 4.1"},
            {"atom_type": "boundary", "claim": "Degrades >10s", "confidence": 0.7, "basis": "inferred", "failure_modes": "Temporal drift"},
        ],
    }
