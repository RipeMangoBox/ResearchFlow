"""Analysis service — L3 skim using LLM.

L3 skim: lightweight card from title + abstract + key sections (~2K tokens).
L4 deep analysis has been moved to ingest_workflow.py (V6 agent pipeline).

Retained utilities:
  - skim_paper() / skim_batch() — L3 skim
  - _maybe_create_bottleneck() — auto-create bottleneck from analysis data
  - assign_paradigm() — re-exported for convenience
  - _gather_text() / _parse_json_safe() — helpers (moved from deleted analysis_steps.py)
"""

import json
import logging
import re
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.analysis import PaperAnalysis
from backend.models.enums import AnalysisLevel, PaperState
from backend.models.paper import Paper
from backend.services.llm_service import call_llm

logger = logging.getLogger(__name__)

# ── Prompt templates ────────────────────────────────────────────

L3_SYSTEM = """You are a critical research paper analyst. Produce a structured JSON analysis.

CRITICAL RULES:
1. Distinguish STORY from SUBSTANCE: what the paper CLAIMS vs what the EXPERIMENTS PROVE.
2. Distinguish structural changes from plugin patches. A true structural change rewires the pipeline; a plugin adds a module without changing the flow.
3. Be honest about confidence — mark speculative claims as such.
4. If this is a survey, benchmark, position, or theoretical paper (no new method proposed), set paper_type accordingly and set changed_slots to [].
5. Check if the paper's abstract claims match its experimental results. Flag discrepancies."""

L3_PROMPT_TEMPLATE = """Analyze this paper and return a JSON object with these fields:

- paper_type: ONE of "method" | "survey" | "benchmark" | "position" | "theoretical" | "dataset"
- problem_summary: What problem does this paper solve? Distinguish the PAPER'S CLAIM from what experiments actually prove. (2-3 sentences, Chinese)
- method_summary: What is the core method? What did they ACTUALLY change vs inherit from prior work? (3-5 sentences, Chinese)
- evidence_summary: Key experimental results. Are baselines strong and fair? Any cherry-picking? (2-3 sentences, Chinese)
- core_intuition: The "Aha!" moment — what changed and why it works. (2-3 sentences, Chinese)
- narrative_vs_substance: Does the paper's narrative (abstract/intro) match its experimental evidence? Note any discrepancies. (1-2 sentences, Chinese)
- changed_slots: Array of method slots this paper changes (e.g. ["denoiser", "conditioning", "sampling"]). Empty array [] for surveys/benchmarks.
- is_plugin_patch: Boolean — is this a small plugin fix or a structural change?
- worth_deep_read: Boolean — should this paper be analyzed in full depth?
- confidence_notes: Array of {{claim, confidence (0-1), basis ("text_stated"|"experiment_backed"|"inferred"|"speculative"), reasoning}}

Paper title: {title}
Venue: {venue} {year}
Category: {category}

{text_content}

Return ONLY valid JSON, no markdown fences."""


# ── L3 Skim ─────────────────────────────────────────────────────

async def skim_paper(session: AsyncSession, paper_id: UUID) -> PaperAnalysis | None:
    """Run L3 skim analysis on a paper.

    Uses title + abstract + intro/method/conclusion sections from L2 parse.
    """
    paper = await session.get(Paper, paper_id)
    if not paper:
        return None

    # Get L2 parse if available
    l2_result = await session.execute(
        select(PaperAnalysis).where(
            PaperAnalysis.paper_id == paper_id,
            PaperAnalysis.level == AnalysisLevel.L2_PARSE,
            PaperAnalysis.is_current.is_(True),
        )
    )
    l2 = l2_result.scalar_one_or_none()

    # Gather text: abstract + key L2 sections
    text_content = _gather_text(paper, l2) or f"Title only: {paper.title}"

    # Call LLM
    prompt = L3_PROMPT_TEMPLATE.format(
        title=paper.title,
        venue=paper.venue or "",
        year=paper.year or "",
        category=paper.category,
        text_content=text_content[:8000],  # Cap input
    )

    resp = await call_llm(
        prompt=prompt,
        system=L3_SYSTEM,
        max_tokens=2048,
        session=session,
        paper_id=paper_id,
        prompt_version="l3_v1",
    )

    # Parse JSON response
    analysis_data = _parse_json_response(
        resp.text,
        required_fields=["problem_summary", "method_summary", "changed_slots"],
    )

    # Mark old L3 as superseded
    old_l3 = await session.execute(
        select(PaperAnalysis).where(
            PaperAnalysis.paper_id == paper_id,
            PaperAnalysis.level == AnalysisLevel.L3_SKIM,
            PaperAnalysis.is_current.is_(True),
        )
    )
    old = old_l3.scalar_one_or_none()
    if old:
        old.is_current = False

    # Create L3 analysis
    analysis = PaperAnalysis(
        paper_id=paper_id,
        level=AnalysisLevel.L3_SKIM,
        model_provider=resp.provider,
        model_name=resp.model,
        prompt_version="l3_v1",
        schema_version="v1",
        confidence=0.7,
        problem_summary=analysis_data.get("problem_summary"),
        method_summary=analysis_data.get("method_summary"),
        evidence_summary=analysis_data.get("evidence_summary"),
        core_intuition=analysis_data.get("core_intuition"),
        changed_slots=analysis_data.get("changed_slots"),
        is_plugin_patch=analysis_data.get("is_plugin_patch"),
        worth_deep_read=analysis_data.get("worth_deep_read"),
        confidence_notes=analysis_data.get("confidence_notes"),
        is_current=True,
    )
    session.add(analysis)

    # Update state
    if paper.state in (PaperState.WAIT, PaperState.DOWNLOADED, PaperState.L1_METADATA,
                       PaperState.L2_PARSED, PaperState.ENRICHED):
        paper.state = PaperState.L3_SKIMMED

    await session.flush()
    await session.refresh(analysis)
    return analysis


# ── Bottleneck creation ─────────────────────────────────────────

async def _maybe_create_bottleneck(
    session: AsyncSession,
    paper: Paper,
    analysis_data: dict,
) -> UUID | None:
    """Auto-create a ProjectBottleneck + PaperBottleneckClaim from analysis output."""
    bn_data = analysis_data.get("bottleneck_addressed")
    if not bn_data or not isinstance(bn_data, dict) or not bn_data.get("title"):
        return None

    from backend.models.research import ProjectBottleneck, PaperBottleneckClaim

    # Dedup: check if similar bottleneck already exists
    existing = await session.execute(
        select(ProjectBottleneck).where(
            func.lower(ProjectBottleneck.title) == bn_data["title"].lower()
        ).limit(1)
    )
    bn = existing.scalar_one_or_none()
    if bn:
        if bn.related_paper_ids and paper.id not in bn.related_paper_ids:
            bn.related_paper_ids = list(bn.related_paper_ids) + [paper.id]
        elif not bn.related_paper_ids:
            bn.related_paper_ids = [paper.id]
    else:
        bn = ProjectBottleneck(
            title=bn_data["title"],
            description=bn_data.get("description", ""),
            domain=paper.category,
            related_paper_ids=[paper.id],
            status="active",
        )
        session.add(bn)
        await session.flush()
        logger.info(f"Auto-created bottleneck: {bn.title} (paper={paper.id})")

    claim = PaperBottleneckClaim(
        paper_id=paper.id,
        bottleneck_id=bn.id,
        claim_text=bn_data.get("description", bn_data["title"]),
        is_primary=True,
        is_fundamental=bn_data.get("is_fundamental"),
        confidence=0.7,
        source="system_inferred",
    )
    session.add(claim)
    await session.flush()
    return bn.id


# ── Paradigm assignment (re-export) ─────────────────────────────

async def assign_paradigm(session, category, tags, *, title=None, abstract=None):
    """Re-export from frame_assign_service for convenience."""
    from backend.services.frame_assign_service import assign_paradigm as _assign
    return await _assign(session, category, tags, title=title, abstract=abstract)


# ── Batch operations ────────────────────────────────────────────

async def skim_batch(session: AsyncSession, limit: int = 5) -> list[dict]:
    """Run L3 skim on papers that are ready (have L2 or abstract but no L3)."""
    result = await session.execute(
        select(Paper)
        .where(
            Paper.state.in_([PaperState.L2_PARSED, PaperState.ENRICHED,
                            PaperState.DOWNLOADED, PaperState.L1_METADATA]),
        )
        .order_by(Paper.analysis_priority.desc().nullsfirst())
        .limit(limit)
    )
    papers = list(result.scalars().all())

    results = []
    for paper in papers:
        try:
            analysis = await skim_paper(session, paper.id)
            results.append({
                "paper_id": str(paper.id),
                "title": paper.title[:60],
                "status": "skimmed",
                "worth_deep_read": analysis.worth_deep_read if analysis else None,
                "is_plugin_patch": analysis.is_plugin_patch if analysis else None,
            })
        except Exception as e:
            logger.error(f"Skim error for {paper.id}: {e}")
            results.append({
                "paper_id": str(paper.id),
                "title": paper.title[:60],
                "status": "error",
                "message": str(e)[:100],
            })
    await session.flush()
    return results


# ── Helpers (moved from deleted analysis_steps.py) ──────────────

def _gather_text(paper: Paper, l2: PaperAnalysis | None) -> str:
    """Gather paper text from abstract + L2 extracted sections."""
    parts = []
    if paper.abstract:
        parts.append(f"Abstract:\n{paper.abstract}")
    if l2 and l2.extracted_sections:
        for key, val in l2.extracted_sections.items():
            if key != "references":
                parts.append(f"\n## {key.title()}\n{val}")
    return "\n\n".join(parts)


def _parse_json_safe(text: str) -> dict:
    """Parse JSON from LLM response, tolerating markdown fences and truncation."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract JSON object between first { and last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, dict) and len(result) >= 3:
                return result
        except json.JSONDecodeError:
            pass

    # 3. Brute-force repair for truncated JSON
    if start >= 0:
        fragment = text[start:]
        result = _repair_truncated_json(fragment)
        if result is not None:
            logger.info(f"JSON repaired from truncated response (len={len(text)})")
            return result

    logger.error(f"JSON parse failed (len={len(text)}): {text[:300]}")
    return {}


def _repair_truncated_json(text: str) -> dict | None:
    """Brute-force repair of truncated JSON by trying bracket closures."""
    candidates = set()

    for m in re.finditer(r',\s*\n\s*"', text):
        candidates.add(m.start())
    for m in re.finditer(r',\s*"', text):
        candidates.add(m.start())
    for m in re.finditer(r'[}\]]\s*,', text):
        candidates.add(m.end())
    for m in re.finditer(r'"\s*,', text):
        candidates.add(m.end() - 1)
    for m in re.finditer(r'(?:true|false|null|\d)\s*,', text):
        candidates.add(m.end() - 1)
    candidates.add(len(text))

    for pos in sorted(candidates, reverse=True):
        prefix = text[:pos].rstrip().rstrip(",")
        opens = prefix.count("{") - prefix.count("}")
        open_sq = prefix.count("[") - prefix.count("]")
        if opens < 0 or open_sq < 0:
            continue
        for suffix in [
            "]" * open_sq + "}" * opens,
            '"' + "]" * open_sq + "}" * opens,
            '"}' + "]" * max(0, open_sq) + "}" * max(0, opens - 1),
        ]:
            attempt = prefix + suffix
            try:
                result = json.loads(attempt)
                if isinstance(result, dict) and len(result) > 0:
                    return result
            except json.JSONDecodeError:
                continue
    return None


def _parse_json_response(text: str, required_fields: list[str] | None = None) -> dict:
    """Parse JSON from LLM response with validation."""
    parsed = _parse_json_safe(text)
    if not parsed:
        logger.error(f"FAILED to parse LLM JSON: {text[:300]}")
        return {}
    if required_fields:
        missing = [f for f in required_fields if parsed.get(f) is None]
        if missing:
            logger.warning(f"LLM JSON missing required fields: {missing}")
    return parsed


def _build_report_md(paper: Paper, data: dict) -> str:
    """Build a Markdown report from analysis data."""
    parts = [f"# {paper.title}\n"]
    if data.get("problem_summary"):
        parts.append(f"## Part I：问题与挑战\n\n{data['problem_summary']}\n")
    if data.get("method_summary"):
        parts.append(f"## Part II：方法与洞察\n\n{data['method_summary']}\n")
    if data.get("core_intuition"):
        parts.append(f"### 核心直觉\n\n{data['core_intuition']}\n")
    if data.get("evidence_summary"):
        parts.append(f"## Part III：证据与局限\n\n{data['evidence_summary']}\n")
    return "\n".join(parts)
