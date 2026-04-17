"""Direction proposal service — generate research direction cards.

Two-stage output:
  Stage 1: Propose 1-3 direction cards (bottleneck, rationale, cost, risk)
  Stage 2: Expand selected direction into detailed feasibility plan
"""

import json
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.direction import DirectionCard
from backend.models.paper import Paper
from backend.services.llm_service import call_llm

logger = logging.getLogger(__name__)

PROPOSE_SYSTEM = """You are a research strategy advisor. Given a research topic and the current knowledge base,
propose 1-3 concrete, actionable research directions.
For each direction, assess whether it's structural or incremental, estimate cost and risk.
Be specific — reference real techniques and papers, not vague generalities.
Output in Chinese."""

PROPOSE_PROMPT = """Based on the following topic and knowledge base context, propose 1-3 research directions.

Return a JSON array where each item has:
- title: direction name (Chinese)
- rationale: why this is worth pursuing (2-3 sentences, Chinese)
- is_structural: true if this is a structural change, false if incremental
- estimated_cost: rough experiment cost estimate
- max_risk: biggest risk
- confidence: 0-1 how confident you are this direction is viable
- required_assets: {{data: "...", code: "...", compute: "..."}}

Topic: {topic}
Category: {category}

Knowledge base context (top papers in this area):
{papers_context}

Return ONLY valid JSON array, no markdown fences."""

EXPAND_SYSTEM = """You are a research planning expert. Expand a research direction into a detailed feasibility plan.
Include: related paper chain, transferable mechanisms, baseline selection, minimal viable experiment,
data/code/compute requirements, failure points, success criteria, exit conditions.
Output in Chinese Markdown."""

EXPAND_PROMPT = """Expand this research direction into a detailed feasibility plan:

Direction: {title}
Rationale: {rationale}
Is structural: {is_structural}
Estimated cost: {estimated_cost}
Max risk: {max_risk}

Related papers in knowledge base:
{papers_context}

Generate a complete feasibility plan in Chinese Markdown with these sections:
## 相关论文链
## 可迁移的跨领域机制
## Baseline 选择
## 最小可行实验 (MVP)
## 数据/代码/算力条件
## 可能失败点
## 成功判据与退出条件"""


async def propose_directions(
    session: AsyncSession,
    topic: str,
    category: str | None = None,
    max_directions: int = 3,
) -> list[DirectionCard]:
    """Propose 1-3 research direction cards for a given topic."""

    # Gather related papers
    papers_context = await _gather_papers_context(session, category)

    prompt = PROPOSE_PROMPT.format(
        topic=topic,
        category=category or "General",
        papers_context=papers_context,
    )

    resp = await call_llm(
        prompt=prompt,
        system=PROPOSE_SYSTEM,
        max_tokens=2048,
        session=session,
        prompt_version="direction_propose_v1",
    )

    # Parse response
    directions_data = _parse_directions(resp.text)

    cards = []
    for d in directions_data[:max_directions]:
        card = DirectionCard(
            title=d.get("title", "Untitled"),
            rationale=d.get("rationale"),
            is_structural=d.get("is_structural"),
            required_assets=d.get("required_assets"),
            estimated_cost=d.get("estimated_cost"),
            max_risk=d.get("max_risk"),
            confidence=d.get("confidence"),
            source_topic=topic,
        )
        session.add(card)
        cards.append(card)

    await session.flush()
    for card in cards:
        await session.refresh(card)
    return cards


async def expand_direction(
    session: AsyncSession,
    direction_id: UUID,
) -> DirectionCard | None:
    """Expand a direction card into a detailed feasibility plan."""
    card = await session.get(DirectionCard, direction_id)
    if not card:
        return None

    papers_context = await _gather_papers_context(session, None)

    prompt = EXPAND_PROMPT.format(
        title=card.title,
        rationale=card.rationale or "",
        is_structural=card.is_structural,
        estimated_cost=card.estimated_cost or "Unknown",
        max_risk=card.max_risk or "Unknown",
        papers_context=papers_context,
    )

    resp = await call_llm(
        prompt=prompt,
        system=EXPAND_SYSTEM,
        max_tokens=3072,
        session=session,
        prompt_version="direction_expand_v1",
    )

    card.feasibility_plan_md = resp.text
    await session.flush()
    await session.refresh(card)
    return card


async def list_directions(session: AsyncSession, limit: int = 20) -> list[DirectionCard]:
    result = await session.execute(
        select(DirectionCard).order_by(DirectionCard.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


async def _gather_papers_context(session: AsyncSession, category: str | None) -> str:
    conditions = [Paper.keep_score.isnot(None)]
    if category:
        conditions.append(Paper.category == category)

    from sqlalchemy import and_, desc
    result = await session.execute(
        select(Paper).where(and_(*conditions)).order_by(desc(Paper.keep_score)).limit(10)
    )
    papers = result.scalars().all()

    lines = []
    for p in papers:
        line = f"- {p.title} ({p.venue} {p.year})"
        if p.core_operator:
            line += f"\n  Core: {p.core_operator[:150]}"
        line += f"\n  Struct: {p.structurality_score}, Keep: {p.keep_score}"
        lines.append(line)
    return "\n".join(lines) if lines else "No papers in knowledge base yet."


def _parse_directions(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return [{"title": "Parse error", "rationale": text[:200]}]
