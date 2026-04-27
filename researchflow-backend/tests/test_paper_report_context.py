"""Unit tests for the paper_report context builders in IngestWorkflow.

`_build_paper_metadata_block` is a pure function on the Paper-like object.
`_build_figures_block` formats a SQL row but is also straight-line; we can
test it by patching the session.execute method to return a fake row set.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.services.ingest_workflow import IngestWorkflow


def _paper(**kw):
    defaults = dict(
        title="HY-Motion: Scaling Flow Matching for Text-To-Motion",
        venue="arXiv",
        year=2025,
        acceptance_type="preprint",
        arxiv_id="2502.01234",
        paper_link="https://arxiv.org/abs/2502.01234",
        code_url="https://github.com/Tencent-Hunyuan/HY-Motion-1.0",
        doi=None,
        authors=["Alice", "Bob", "Carol"],
        method_family="HY-Motion",
        tags=["motion-generation", "flow-matching", "rlhf"],
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_metadata_block_lines_present():
    wf = IngestWorkflow.__new__(IngestWorkflow)  # bypass __init__
    p = _paper()
    out = wf._build_paper_metadata_block(p)
    assert "venue: arXiv (preprint)" in out
    assert "year: 2025" in out
    assert "arxiv: https://arxiv.org/abs/2502.01234" in out
    assert "code_url: https://github.com/Tencent-Hunyuan" in out
    assert "method_family: HY-Motion" in out
    # Tags are truncated to first 8 — check at least one is present
    assert "motion-generation" in out


def test_metadata_block_skips_missing_fields():
    wf = IngestWorkflow.__new__(IngestWorkflow)
    p = _paper(arxiv_id=None, paper_link=None, code_url=None,
               authors=None, doi=None)
    out = wf._build_paper_metadata_block(p)
    assert "arxiv:" not in out
    assert "paper_link:" not in out
    assert "code_url:" not in out
    assert "authors:" not in out


def test_metadata_block_empty_paper_returns_no_metadata():
    wf = IngestWorkflow.__new__(IngestWorkflow)
    empty = SimpleNamespace(
        title=None, venue=None, year=None, acceptance_type=None,
        arxiv_id=None, paper_link=None, code_url=None, doi=None,
        authors=None, method_family=None, tags=None,
    )
    out = wf._build_paper_metadata_block(empty)
    assert out == "(no metadata)"


# ── _build_figures_block ──────────────────────────────────────────────

def _row(figures):
    """Mimic SQLAlchemy Row with .extracted_figure_images on column 0."""
    r = MagicMock()
    r.extracted_figure_images = figures
    r.__getitem__ = lambda self, i: figures if i == 0 else None
    return r


@pytest.mark.asyncio
async def test_figures_block_formats_label_role_caption():
    wf = IngestWorkflow.__new__(IngestWorkflow)
    figures = [
        {"label": "Figure 1", "semantic_role": "pipeline",
         "caption": "Overall architecture diagram with three stages"},
        {"label": "Table 2", "semantic_role": "result",
         "caption": "Main benchmark comparison"},
    ]
    fake_session = MagicMock()
    fake_session.execute = AsyncMock(return_value=MagicMock(
        fetchall=lambda: [_row(figures)]
    ))
    wf.session = fake_session

    out = await wf._build_figures_block("11111111-1111-1111-1111-111111111111")
    assert "Figure 1 (role=pipeline)" in out
    assert "Overall architecture diagram with three stages" in out
    assert "Table 2 (role=result)" in out


@pytest.mark.asyncio
async def test_figures_block_handles_empty():
    wf = IngestWorkflow.__new__(IngestWorkflow)
    fake_session = MagicMock()
    fake_session.execute = AsyncMock(return_value=MagicMock(
        fetchall=lambda: []
    ))
    wf.session = fake_session
    out = await wf._build_figures_block("11111111-1111-1111-1111-111111111111")
    assert "no figures" in out
