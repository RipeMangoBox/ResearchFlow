"""Triage service — compute 4-dimension scores for papers.

Scores:
  keep_score           — should this enter the main KB or observation pool?
  analysis_priority    — how urgently should this be analyzed?
  structurality_score  — structural change vs plugin patch?
  extensionability_score — potential for cross-task/domain transfer?

Scoring rules based on project design doc:
  Tier A (open data) > B (open code) > C (accepted, no code) > D (preprint)
  Within tier: team credibility, task relevance, mechanism structure,
               reproduction asset completeness, freshness, evidence strength
"""

import re
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.enums import Importance, PaperState, Tier
from backend.models.paper import Paper


# ── Score computation ───────────────────────────────────────────

def compute_keep_score(paper: Paper) -> float:
    """Compute keep_score (0-1): should this paper enter the main KB?"""
    score = 0.0

    # Tier-based base score (highest weight)
    if paper.tier == Tier.A_OPEN_DATA:
        score += 0.40
    elif paper.tier == Tier.B_OPEN_CODE:
        score += 0.30
    elif paper.tier == Tier.C_ACCEPTED_NO_CODE:
        score += 0.20
    elif paper.tier == Tier.D_PREPRINT:
        score += 0.10
    else:
        # No tier assigned yet — infer from fields
        if paper.open_data:
            score += 0.40
        elif paper.open_code or paper.code_url:
            score += 0.30
        elif paper.venue and paper.venue not in ("arXiv", ""):
            score += 0.20
        else:
            score += 0.10

    # Importance bonus
    imp_bonus = {"S": 0.25, "A": 0.20, "B": 0.15, "C": 0.10, "D": 0.05}
    if paper.importance:
        score += imp_bonus.get(paper.importance.value, 0.10)

    # Venue prestige bonus
    top_venues = {"CVPR", "ICLR", "NeurIPS", "ICML", "ICCV", "ECCV", "SIGGRAPH", "ACL", "EMNLP"}
    if paper.venue and paper.venue.upper() in top_venues:
        score += 0.15

    # Has project link
    if paper.project_link:
        score += 0.05

    # Freshness (newer = better, max 0.10)
    if paper.year:
        current_year = datetime.now(timezone.utc).year
        age = current_year - paper.year
        if age <= 0:
            score += 0.10
        elif age <= 1:
            score += 0.07
        elif age <= 2:
            score += 0.04
        else:
            score += 0.01

    # Has abstract (metadata completeness)
    if paper.abstract:
        score += 0.05

    return min(score, 1.0)


def compute_analysis_priority(paper: Paper) -> float:
    """Compute analysis_priority (0-1): how urgently should this be analyzed?"""
    score = 0.0

    # Higher importance = higher priority
    imp_weight = {"S": 0.35, "A": 0.25, "B": 0.15, "C": 0.10, "D": 0.05}
    if paper.importance:
        score += imp_weight.get(paper.importance.value, 0.10)

    # Open assets → more analyzable
    if paper.open_code or paper.code_url:
        score += 0.15
    if paper.open_data:
        score += 0.10

    # Has PDF available
    if paper.pdf_path_local or paper.pdf_object_key:
        score += 0.15

    # Top venue
    top_venues = {"CVPR", "ICLR", "NeurIPS", "ICML", "ICCV"}
    if paper.venue and paper.venue.upper() in top_venues:
        score += 0.10

    # Freshness
    if paper.year:
        current_year = datetime.now(timezone.utc).year
        age = current_year - paper.year
        if age <= 0:
            score += 0.15
        elif age <= 1:
            score += 0.10
        elif age <= 2:
            score += 0.05

    return min(score, 1.0)


def compute_structurality_score(paper: Paper) -> float:
    """Compute structurality_score (0-1): structural change vs plugin patch?

    Without deep analysis, this is a rough estimate based on metadata.
    The score gets refined after L3/L4 analysis via method_deltas.
    """
    score = 0.5  # neutral default

    # Keyword signals in core_operator or title
    text = (paper.core_operator or "") + " " + (paper.title or "")
    text_lower = text.lower()

    structural_signals = [
        "framework", "architecture", "paradigm", "unified",
        "foundation", "scaling", "autoregressive", "end-to-end",
        "new representation", "novel framework",
    ]
    patch_signals = [
        "plug-in", "plugin", "adapter", "fine-tune", "finetune",
        "post-processing", "trick", "simple", "lightweight",
    ]

    structural_hits = sum(1 for s in structural_signals if s in text_lower)
    patch_hits = sum(1 for s in patch_signals if s in text_lower)

    score += structural_hits * 0.08
    score -= patch_hits * 0.08

    # Tag-based signals
    tags_str = " ".join(paper.tags or []).lower()
    if "framework" in tags_str or "architecture" in tags_str:
        score += 0.10
    if "plugin" in tags_str or "adapter" in tags_str:
        score -= 0.10

    return max(0.0, min(score, 1.0))


def compute_extensionability_score(paper: Paper) -> float:
    """Compute extensionability_score (0-1): cross-domain transfer potential?

    Without deep analysis, estimate from metadata signals.
    """
    score = 0.3  # conservative default

    # Open assets → easier to extend
    if paper.open_code or paper.code_url:
        score += 0.15
    if paper.open_data:
        score += 0.10

    # Cross-domain keywords
    text = (paper.core_operator or "") + " " + (paper.title or "") + " " + (paper.abstract or "")
    text_lower = text.lower()

    extension_signals = [
        "generaliz", "transfer", "universal", "multi-task",
        "multi-modal", "cross-domain", "scalab", "zero-shot",
        "any", "unified", "versatile",
    ]
    hits = sum(1 for s in extension_signals if s in text_lower)
    score += hits * 0.07

    # Multiple task tags
    task_tags = [t for t in (paper.tags or []) if t.startswith("task/")]
    if len(task_tags) >= 2:
        score += 0.10

    return max(0.0, min(score, 1.0))


def compute_all_scores(paper: Paper) -> dict[str, float]:
    """Compute all 4 scores for a paper."""
    return {
        "keep_score": round(compute_keep_score(paper), 3),
        "analysis_priority": round(compute_analysis_priority(paper), 3),
        "structurality_score": round(compute_structurality_score(paper), 3),
        "extensionability_score": round(compute_extensionability_score(paper), 3),
    }


# ── Batch triage ────────────────────────────────────────────────

async def triage_paper(session: AsyncSession, paper_id) -> Paper | None:
    """Score a single paper and update DB."""
    paper = await session.get(Paper, paper_id)
    if not paper:
        return None

    scores = compute_all_scores(paper)
    paper.keep_score = scores["keep_score"]
    paper.analysis_priority = scores["analysis_priority"]
    paper.structurality_score = scores["structurality_score"]
    paper.extensionability_score = scores["extensionability_score"]

    # Auto-assign tier if not set
    if paper.tier is None:
        if paper.open_data:
            paper.tier = Tier.A_OPEN_DATA
        elif paper.open_code or paper.code_url:
            paper.tier = Tier.B_OPEN_CODE
        elif paper.venue and paper.venue not in ("arXiv", ""):
            paper.tier = Tier.C_ACCEPTED_NO_CODE
        else:
            paper.tier = Tier.D_PREPRINT

    await session.flush()
    await session.refresh(paper)
    return paper


async def triage_all_unscored(session: AsyncSession) -> int:
    """Score all papers that don't have scores yet. Returns count."""
    result = await session.execute(
        select(Paper).where(Paper.keep_score.is_(None))
    )
    papers = result.scalars().all()
    for paper in papers:
        scores = compute_all_scores(paper)
        paper.keep_score = scores["keep_score"]
        paper.analysis_priority = scores["analysis_priority"]
        paper.structurality_score = scores["structurality_score"]
        paper.extensionability_score = scores["extensionability_score"]

        if paper.tier is None:
            if paper.open_data:
                paper.tier = Tier.A_OPEN_DATA
            elif paper.open_code or paper.code_url:
                paper.tier = Tier.B_OPEN_CODE
            elif paper.venue and paper.venue not in ("arXiv", ""):
                paper.tier = Tier.C_ACCEPTED_NO_CODE
            else:
                paper.tier = Tier.D_PREPRINT

    await session.flush()
    return len(papers)
