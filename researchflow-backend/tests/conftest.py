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
    "graph_assertion_evidence", "graph_assertions", "graph_nodes",
    "review_tasks", "human_overrides", "aliases",
    "implementation_units", "graph_edges",
    "evidence_units", "idea_deltas", "delta_cards",
    "method_deltas", "paper_analyses",
    "transfer_atoms", "paper_assets", "paper_versions",
    "slots", "paradigm_templates", "mechanism_families",
    "direction_cards", "reading_plans", "search_sessions",
    "user_bookmarks", "user_events", "user_feedback",
    "model_runs", "execution_memories", "jobs", "digests",
    "papers", "project_bottlenecks",
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
        core_operator="denoising score matching",
        keep_score=0.75, structurality_score=0.65,
    )
    session.add(paper)
    await session.commit()
    return paper


@pytest_asyncio.fixture
async def sample_paradigm(session):
    from backend.models.analysis import ParadigmTemplate
    from backend.models.graph import Slot
    paradigm = ParadigmTemplate(
        name=f"motion_gen_{uuid.uuid4().hex[:8]}", version="v1",
        domain="motion_generation",
        slots={"denoiser": {}, "conditioning": {}, "sampling": {}},
    )
    session.add(paradigm)
    await session.commit()
    slots = []
    for i, (name, stype) in enumerate([
        ("denoiser", "architecture"), ("conditioning", "architecture"), ("sampling", "inference"),
    ]):
        slot = Slot(paradigm_id=paradigm.id, name=name, slot_type=stype, sort_order=i)
        session.add(slot)
        slots.append(slot)
    await session.commit()
    for s in slots:
        await session.refresh(s)
    return paradigm, slots


@pytest_asyncio.fixture
async def sample_mechanism(session):
    from backend.models.graph import MechanismFamily
    mf = MechanismFamily(
        name=f"diffusion_{uuid.uuid4().hex[:8]}",
        domain="generative", description="DDPM family",
        aliases=["DDPM", "score-based"],
    )
    session.add(mf)
    await session.commit()
    return mf


@pytest_asyncio.fixture
async def sample_analysis_data():
    return {
        "problem_summary": "Motion generation quality degrades",
        "method_summary": "Replace discrete tokenizer with continuous diffusion",
        "evidence_summary": "FID improved by 15%",
        "core_intuition": "连续扩散去噪保留更多运动细节",
        "changed_slots": ["denoiser", "sampling"],
        "is_plugin_patch": False, "worth_deep_read": True,
        "structurality_score": 0.7, "transferability_score": 0.6,
        "delta_card": {
            "paradigm": "motion_generation_diffusion",
            "slots": {
                "denoiser": {"changed": True, "from": "VQ-VAE", "to": "DDPM", "change_type": "structural"},
                "sampling": {"changed": True, "from": "AR", "to": "DDIM", "change_type": "structural"},
            },
            "is_structural": True, "primary_gain_source": "denoiser",
        },
        "evidence_units": [
            {"atom_type": "evidence", "claim": "FID improved 15%", "confidence": 0.9, "basis": "experiment_backed", "source_section": "Table 2", "conditions": "HumanML3D"},
            {"atom_type": "mechanism", "claim": "Diffusion preserves smoothness", "confidence": 0.85, "basis": "text_stated", "source_section": "Section 4.1"},
            {"atom_type": "boundary", "claim": "Degrades >10s", "confidence": 0.7, "basis": "inferred", "failure_modes": "Temporal drift"},
        ],
        "confidence_notes": [
            {"claim": "FID improvement", "confidence": 0.9, "basis": "experiment_backed", "reasoning": "Clear results"},
        ],
    }
