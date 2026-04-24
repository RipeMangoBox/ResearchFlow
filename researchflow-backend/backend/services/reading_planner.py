"""Reading planner — generate tiered reading recommendations.

Default order:
  1. Canonical baselines (foundational papers)
  2. Structural improvements (changed core architecture)
  3. Strong team follow-ups (recent, well-cited)
  4. Plugin patches (incremental improvements)
  5. Negative examples / boundary papers

Each paper is annotated with recommended reading depth:
  30s, 5min, or 20min deep read.
"""

import logging
from uuid import UUID

from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.delta_card import DeltaCard
from backend.models.enums import Importance, PaperState
from backend.models.paper import Paper

logger = logging.getLogger(__name__)


async def generate_reading_plan(
    session: AsyncSession,
    category: str | None = None,
    topic: str | None = None,
    max_papers: int = 15,
) -> dict:
    """Generate a tiered reading plan for a category or topic.

    Returns structured plan with papers sorted into tiers.
    """
    # Fetch candidate papers
    conditions = [
        Paper.state.in_([PaperState.CHECKED, PaperState.L4_DEEP,
                        PaperState.L3_SKIMMED, PaperState.L2_PARSED]),
        Paper.keep_score.isnot(None),
    ]
    if category:
        conditions.append(Paper.category == category)

    result = await session.execute(
        select(Paper)
        .where(and_(*conditions))
        .order_by(Paper.keep_score.desc().nullslast())
        .limit(max_papers * 3)  # Over-fetch for sorting into tiers
    )
    papers = list(result.scalars().all())

    if not papers:
        return {
            "category": category,
            "topic": topic,
            "tiers": {},
            "total_papers": 0,
            "reading_time_estimate": "0 min",
        }

    # Sort papers into tiers
    canonical = []
    structural = []
    follow_ups = []
    patches = []
    boundary = []

    # Pre-fetch DeltaCards for richer scoring
    paper_ids = [p.id for p in papers]
    dc_result = await session.execute(
        select(DeltaCard).where(
            DeltaCard.paper_id.in_(paper_ids),
            DeltaCard.status != "deprecated",
        )
    )
    dc_map = {dc.paper_id: dc for dc in dc_result.scalars()}

    for p in papers:
        dc = dc_map.get(p.id)
        entry = _make_entry(p, dc)

        # DeltaCard scores override paper-level scores when available
        struct_score = (dc.structurality_score if dc and dc.structurality_score else None) or p.structurality_score or 0.5
        keep_score = p.keep_score or 0.5
        is_old_important = (p.year and p.year <= 2023 and
                           p.importance in (Importance.S, Importance.A))

        if is_old_important and keep_score >= 0.6:
            # Foundational / canonical baseline
            entry["reading_depth"] = "20min"
            entry["tier_reason"] = "Canonical baseline — foundational work in this area"
            canonical.append(entry)
        elif struct_score >= 0.55 and keep_score >= 0.5:
            # Structural improvement
            entry["reading_depth"] = "20min" if struct_score >= 0.6 else "5min"
            entry["tier_reason"] = f"Structural change (score={struct_score:.2f})"
            structural.append(entry)
        elif keep_score >= 0.6:
            # Strong follow-up
            entry["reading_depth"] = "5min"
            entry["tier_reason"] = "Strong team / high-value follow-up"
            follow_ups.append(entry)
        elif struct_score < 0.45:
            # Patch / incremental
            entry["reading_depth"] = "30s"
            entry["tier_reason"] = f"Plugin patch (structurality={struct_score:.2f})"
            patches.append(entry)
        else:
            # Boundary / moderate
            entry["reading_depth"] = "5min"
            entry["tier_reason"] = "Moderate contribution — worth a skim"
            follow_ups.append(entry)

    # Trim to max_papers total
    plan_tiers = {}
    remaining = max_papers

    for tier_name, tier_papers in [
        ("canonical_baselines", canonical),
        ("structural_improvements", structural),
        ("strong_follow_ups", follow_ups),
        ("patches_and_boundary", patches + boundary),
    ]:
        take = min(len(tier_papers), remaining)
        if take > 0:
            plan_tiers[tier_name] = tier_papers[:take]
            remaining -= take

    # Estimate reading time
    total_time = 0
    for tier_papers in plan_tiers.values():
        for entry in tier_papers:
            depth = entry["reading_depth"]
            if depth == "30s":
                total_time += 1
            elif depth == "5min":
                total_time += 5
            else:
                total_time += 20

    return {
        "category": category,
        "topic": topic,
        "tiers": plan_tiers,
        "total_papers": sum(len(v) for v in plan_tiers.values()),
        "reading_time_estimate": f"{total_time} min",
        "reading_order": _explain_order(),
    }


def _make_entry(paper: Paper, dc: DeltaCard | None = None) -> dict:
    entry = {
        "paper_id": str(paper.id),
        "title": paper.title,
        "venue": paper.venue,
        "year": paper.year,
        "importance": paper.importance.value if paper.importance else None,
        "keep_score": paper.keep_score,
        "structurality_score": None,
        "core_operator": (paper.method_family or "")[:200],
        "open_code": bool(paper.code_url),
        "tags": list(paper.tags[:5]) if paper.tags else [],
        "reading_depth": "5min",
        "tier_reason": "",
    }
    if dc:
        entry["delta_card_id"] = str(dc.id)
        entry["delta_statement"] = (dc.delta_statement or "")[:200]
        entry["transferability_score"] = dc.transferability_score
        if dc.structurality_score:
            entry["structurality_score"] = dc.structurality_score
    return entry


def _explain_order() -> str:
    return (
        "推荐阅读顺序：\n"
        "1. Canonical baselines — 先读基线论文，建立领域认知框架\n"
        "2. Structural improvements — 再读结构性改进，理解方向演变\n"
        "3. Strong follow-ups — 看强团队的最新跟进\n"
        "4. Patches & boundary — 最后扫插件型工作和边界论文"
    )
