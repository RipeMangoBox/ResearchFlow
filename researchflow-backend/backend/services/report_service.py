"""Report generation service — produce briefing reports from paper sets.

Three report tiers:
  quick (30s)     — one-paragraph summary per paper
  briefing (5min) — structured slot comparison + reading order
  deep_compare    — full cross-paper analysis with evidence

Reports follow a fixed structure per the design doc:
1. What problem does this set address
2. Canonical baseline in this direction
3. Which slots each paper changed (delta card table)
4. Plugin patch vs structural change classification
5. Open source status / reproduction value / evidence strength
6. Recommended reading order
7. Top 1-3 lines worth pursuing
"""

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.analysis import PaperAnalysis
from backend.models.delta_card import DeltaCard
from backend.models.enums import AnalysisLevel
from backend.models.paper import Paper
from backend.services.llm_service import call_llm

logger = logging.getLogger(__name__)


REPORT_SYSTEM = """You are a research report writer. Generate clear, structured research briefings in Chinese.
Your reports must follow the canonical delta card approach — compare papers against the domain's standard paradigm.
Never just retell each paper's story. Instead, align them to a shared framework and highlight what each one actually changed."""

QUICK_PROMPT = """Based on these papers, write a 30-second quick summary (约200字).
Include: this group is about what topic, key finding from each paper (1 sentence each), and which 1-2 papers are most worth reading.

Papers:
{paper_summaries}

Output in Chinese. Plain text, no JSON."""

BRIEFING_PROMPT = """Based on these papers, generate a 5-minute briefing report in Markdown (Chinese).

Required structure:
## 1. 这组论文在解决什么问题
(2-3 sentences)

## 2. 该方向的标准基线
(what is the canonical approach in this area)

## 3. 各论文改了哪些槽位
| 论文 | 改动槽位 | 改动类型 | 核心算子 |
|------|----------|----------|----------|
(one row per paper)

## 4. 结构性改进 vs 插件修补
(classify each paper)

## 5. 开源与复现情况
| 论文 | 开源状态 | 复现价值 | 证据强度 |
|------|----------|----------|----------|

## 6. 建议阅读顺序
(numbered list: start from baseline, then structural, then patches)

## 7. 最值得追的 1-3 条线
(which directions are most promising and why)

Papers:
{paper_summaries}

Output in Chinese Markdown."""

DEEP_PROMPT = """Based on these papers, generate a deep comparison report in Markdown (Chinese).

Include everything from the briefing format PLUS:
- Detailed method comparison across papers
- Evidence quality assessment for each paper's key claims
- Cross-paper transferable mechanisms
- Specific gaps and opportunities
- Recommended next experiments

Papers with analysis:
{paper_details}

Output in Chinese Markdown."""


async def generate_report(
    session: AsyncSession,
    paper_ids: list[UUID],
    report_type: str = "briefing",
    topic: str | None = None,
) -> dict:
    """Generate a briefing report for a set of papers.

    Args:
        paper_ids: List of paper UUIDs to include
        report_type: "quick", "briefing", or "deep_compare"
        topic: Optional topic context
    """
    # Fetch papers and their analyses
    papers_data = []
    for pid in paper_ids:
        paper = await session.get(Paper, pid)
        if not paper:
            continue

        # Get best analysis
        result = await session.execute(
            select(PaperAnalysis)
            .where(PaperAnalysis.paper_id == pid, PaperAnalysis.is_current.is_(True))
            .order_by(PaperAnalysis.level.desc())
            .limit(1)
        )
        analysis = result.scalar_one_or_none()

        # Get DeltaCard (v3) — falls back to None if not yet built
        dc_result = await session.execute(
            select(DeltaCard)
            .where(DeltaCard.paper_id == pid, DeltaCard.status != "deprecated")
            .order_by(DeltaCard.created_at.desc())
            .limit(1)
        )
        delta_card = dc_result.scalar_one_or_none()

        papers_data.append({
            "paper": paper,
            "analysis": analysis,
            "delta_card": delta_card,
        })

    if not papers_data:
        return {"error": "No valid papers found"}

    # Build paper summaries for prompt
    if report_type == "quick":
        text = _build_quick_summaries(papers_data)
        prompt = QUICK_PROMPT.format(paper_summaries=text)
        max_tokens = 1024
    elif report_type == "deep_compare":
        text = _build_deep_details(papers_data)
        prompt = DEEP_PROMPT.format(paper_details=text)
        max_tokens = 4096
    else:  # briefing
        text = _build_briefing_summaries(papers_data)
        prompt = BRIEFING_PROMPT.format(paper_summaries=text)
        max_tokens = 3072

    if topic:
        prompt = f"Topic context: {topic}\n\n{prompt}"

    # Call LLM
    resp = await call_llm(
        prompt=prompt,
        system=REPORT_SYSTEM,
        max_tokens=max_tokens,
        session=session,
        prompt_version=f"report_{report_type}_v1",
    )

    return {
        "report_type": report_type,
        "paper_count": len(papers_data),
        "paper_ids": [str(pid) for pid in paper_ids],
        "content": resp.text,
        "model": f"{resp.provider}/{resp.model}",
        "tokens": {"input": resp.input_tokens, "output": resp.output_tokens},
    }


def _build_quick_summaries(papers_data: list[dict]) -> str:
    """Build concise paper summaries for quick report."""
    lines = []
    for pd in papers_data:
        p = pd["paper"]
        a = pd["analysis"]
        line = f"- {p.title} ({p.venue} {p.year})"
        if a and a.core_intuition:
            line += f"\n  Core: {a.core_intuition[:150]}"
        elif p.core_operator:
            line += f"\n  Operator: {p.core_operator[:150]}"
        if p.open_code:
            line += " [open code]"
        lines.append(line)
    return "\n".join(lines)


def _build_briefing_summaries(papers_data: list[dict]) -> str:
    """Build structured summaries for briefing report using DeltaCard data."""
    lines = []
    for pd in papers_data:
        p = pd["paper"]
        a = pd["analysis"]
        dc = pd.get("delta_card")

        parts = [f"### {p.title}"]
        parts.append(f"Venue: {p.venue} {p.year} | Category: {p.category}")
        parts.append(f"Importance: {p.importance.value if p.importance else 'N/A'} | "
                     f"Structurality: {p.structurality_score or 'N/A'}")

        if p.core_operator:
            parts.append(f"Core operator: {p.core_operator[:300]}")

        # DeltaCard provides richer structured info than raw analysis
        if dc:
            parts.append(f"Delta: {dc.delta_statement[:400]}")
            if dc.baseline_paradigm:
                parts.append(f"Baseline paradigm: {dc.baseline_paradigm}")
            struct = f"{dc.structurality_score:.2f}" if dc.structurality_score else "N/A"
            transfer = f"{dc.transferability_score:.2f}" if dc.transferability_score else "N/A"
            parts.append(f"Structurality: {struct} | Transferability: {transfer}")
            if dc.key_ideas_ranked:
                ideas = [f"{ki['rank']}. {ki['statement'][:100]}" for ki in dc.key_ideas_ranked[:3]]
                parts.append(f"Key ideas:\n  " + "\n  ".join(ideas))
            if dc.assumptions:
                parts.append(f"Assumptions: {'; '.join(dc.assumptions[:3])}")
            if dc.failure_modes:
                parts.append(f"Failure modes: {'; '.join(dc.failure_modes[:3])}")
        elif a:
            # Fallback to analysis data
            if a.problem_summary:
                parts.append(f"Problem: {a.problem_summary[:200]}")
            if a.method_summary:
                parts.append(f"Method: {a.method_summary[:300]}")
            if a.changed_slots:
                parts.append(f"Changed slots: {', '.join(a.changed_slots)}")
            parts.append(f"Plugin patch: {a.is_plugin_patch} | Worth deep read: {a.worth_deep_read}")

        tags_str = ", ".join(p.tags[:8]) if p.tags else "N/A"
        parts.append(f"Tags: {tags_str}")
        parts.append(f"Open code: {p.open_code} | Open data: {p.open_data}")

        lines.append("\n".join(parts))

    return "\n\n---\n\n".join(lines)


def _build_deep_details(papers_data: list[dict]) -> str:
    """Build detailed content for deep comparison."""
    lines = []
    for pd in papers_data:
        p = pd["paper"]
        a = pd["analysis"]

        parts = [f"### {p.title} ({p.venue} {p.year})"]

        if p.core_operator:
            parts.append(f"Core operator: {p.core_operator}")
        if p.primary_logic:
            parts.append(f"Primary logic: {p.primary_logic}")

        if a:
            if a.problem_summary:
                parts.append(f"\nProblem:\n{a.problem_summary}")
            if a.method_summary:
                parts.append(f"\nMethod:\n{a.method_summary}")
            if a.evidence_summary:
                parts.append(f"\nEvidence:\n{a.evidence_summary}")
            if a.core_intuition:
                parts.append(f"\nCore intuition:\n{a.core_intuition}")
            if a.confidence_notes:
                parts.append(f"\nConfidence notes: {a.confidence_notes}")

        lines.append("\n".join(parts))

    return "\n\n===\n\n".join(lines)
