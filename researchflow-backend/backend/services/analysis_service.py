"""Analysis service — L3 skim and L4 deep analysis using LLM.

L3 skim: lightweight card from title + abstract + key sections (~2K tokens)
L4 deep: full analysis from complete text (~10-20K tokens)

Both produce structured output: problem/method/evidence summaries,
changed_slots, delta card, confidence_notes.
"""

import json
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.analysis import MethodDelta, PaperAnalysis
from backend.models.enums import AnalysisLevel, PaperState
from backend.models.paper import Paper
from backend.services.llm_service import call_llm

logger = logging.getLogger(__name__)

# ── Prompt templates ────────────────────────────────────────────

L3_SYSTEM = """You are a research paper analyst. Produce a structured JSON analysis.
Focus on what the paper actually changes relative to the standard approach in its field.
Distinguish structural changes from plugin patches.
Be honest about confidence — mark speculative claims as such."""

L3_PROMPT_TEMPLATE = """Analyze this paper and return a JSON object with these fields:

- problem_summary: What problem does this paper solve? (2-3 sentences, Chinese)
- method_summary: What is the core method? (3-5 sentences, Chinese)
- evidence_summary: Key experimental results and ablations. (2-3 sentences, Chinese)
- core_intuition: The "Aha!" moment — what changed and why it works. (2-3 sentences, Chinese)
- changed_slots: Array of method slots this paper changes (e.g. ["denoiser", "conditioning", "sampling"])
- is_plugin_patch: Boolean — is this a small plugin fix or a structural change?
- worth_deep_read: Boolean — should this paper be analyzed in full depth?
- confidence_notes: Array of {{claim, confidence (0-1), basis ("text_stated"|"experiment_backed"|"inferred"|"speculative"), reasoning}}

Paper title: {title}
Venue: {venue} {year}
Category: {category}

{text_content}

Return ONLY valid JSON, no markdown fences."""

L4_SYSTEM = """You are a senior research analyst producing deep paper analysis.
Your analysis must:
1. Align findings to the domain's canonical paradigm — specify which slots changed
2. Distinguish code-verified facts from speculative interpretations
3. Extract atomic evidence units with source anchors
4. Assess cross-domain transferability
Output in Chinese. Be rigorous about evidence quality."""

L4_PROMPT_TEMPLATE = """Produce a comprehensive analysis of this paper in JSON format:

{{
  "problem_summary": "Part I: 问题与挑战 (detailed, 200-400 words)",
  "method_summary": "Part II: 方法与洞察 (detailed, 400-600 words)",
  "evidence_summary": "Part III: 证据与局限 (detailed, 200-400 words)",
  "core_intuition": "核心直觉 — 什么变了，为什么有效 (100-200 words)",
  "changed_slots": ["slot1", "slot2"],
  "is_plugin_patch": false,
  "worth_deep_read": true,
  "method_category": "ONE of: structural_architecture | plugin_module | reward_design | training_recipe | data_augmentation | inference_optimization | loss_function | representation_change | evaluation_method | theoretical_analysis",
  "improvement_type": "ONE of: fundamental_rethink | component_replacement | additive_plugin | hyperparameter_trick | data_scaling | combination_of_existing",
  "bottleneck_addressed": {{
    "title": "The core bottleneck/limitation this paper addresses (1 sentence)",
    "description": "Why this bottleneck matters and what makes it hard (2-3 sentences)",
    "is_fundamental": true/false
  }},
  "structurality_score": 0.0-1.0,
  "extensionability_score": 0.0-1.0,
  "transferability_score": 0.0-1.0,
  "delta_card": {{
    "paradigm": "inferred standard paradigm name",
    "slots": {{
      "slot_name": {{"changed": true/false, "from": "old", "to": "new", "change_type": "structural|plugin"}}
    }},
    "is_structural": true/false,
    "primary_gain_source": "which slot contributes most to improvement"
  }},
  "evidence_units": [
    {{
      "atom_type": "mechanism|evidence|boundary|transfer",
      "claim": "specific claim",
      "confidence": 0.0-1.0,
      "basis": "code_verified|experiment_backed|text_stated|inferred|speculative",
      "source_section": "e.g. Table 3, Section 4.2",
      "conditions": "under what conditions this holds",
      "failure_modes": "known failure cases"
    }}
  ],
  "confidence_notes": [
    {{"claim": "...", "confidence": 0.0-1.0, "basis": "...", "reasoning": "..."}}
  ]
}}

Paper title: {title}
Venue: {venue} {year}
Category: {category}
Tags: {tags}
Core operator: {core_operator}
Primary logic: {primary_logic}

Full text (extracted sections):
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

    # Gather text: abstract + key L2 sections
    text_parts = []
    if paper.abstract:
        text_parts.append(f"Abstract:\n{paper.abstract}")

    # Get L2 parse if available
    l2_result = await session.execute(
        select(PaperAnalysis).where(
            PaperAnalysis.paper_id == paper_id,
            PaperAnalysis.level == AnalysisLevel.L2_PARSE,
            PaperAnalysis.is_current.is_(True),
        )
    )
    l2 = l2_result.scalar_one_or_none()
    if l2 and l2.extracted_sections:
        for key in ("introduction", "method", "conclusion", "abstract"):
            if key in l2.extracted_sections:
                text_parts.append(f"\n{key.title()}:\n{l2.extracted_sections[key][:2000]}")

    if not text_parts and not paper.abstract:
        text_parts.append(f"Title only: {paper.title}")

    text_content = "\n\n".join(text_parts)

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
    analysis_data = _parse_json_response(resp.text)

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


# ── L4 Deep ─────────────────────────────────────────────────────

async def deep_analyze_paper(session: AsyncSession, paper_id: UUID) -> PaperAnalysis | None:
    """Run L4 deep analysis on a paper.

    Uses full extracted text from L2 parse + all metadata.
    Produces full report + delta card + evidence units.
    """
    paper = await session.get(Paper, paper_id)
    if not paper:
        return None

    # Gather full text from L2
    text_parts = []
    if paper.abstract:
        text_parts.append(f"Abstract:\n{paper.abstract}")

    l2_result = await session.execute(
        select(PaperAnalysis).where(
            PaperAnalysis.paper_id == paper_id,
            PaperAnalysis.level == AnalysisLevel.L2_PARSE,
            PaperAnalysis.is_current.is_(True),
        )
    )
    l2 = l2_result.scalar_one_or_none()
    if l2 and l2.extracted_sections:
        for key, val in l2.extracted_sections.items():
            if key != "references":
                text_parts.append(f"\n## {key.title()}\n{val}")

    text_content = "\n\n".join(text_parts)

    prompt = L4_PROMPT_TEMPLATE.format(
        title=paper.title,
        venue=paper.venue or "",
        year=paper.year or "",
        category=paper.category,
        tags=", ".join(paper.tags or []),
        core_operator=paper.core_operator or "N/A",
        primary_logic=paper.primary_logic or "N/A",
        text_content=text_content[:30000],  # Larger cap for L4
    )

    resp = await call_llm(
        prompt=prompt,
        system=L4_SYSTEM,
        max_tokens=4096,
        session=session,
        paper_id=paper_id,
        prompt_version="l4_v1",
    )

    analysis_data = _parse_json_response(resp.text)

    # Mark old L4 as superseded
    old_l4 = await session.execute(
        select(PaperAnalysis).where(
            PaperAnalysis.paper_id == paper_id,
            PaperAnalysis.level == AnalysisLevel.L4_DEEP,
            PaperAnalysis.is_current.is_(True),
        )
    )
    old = old_l4.scalar_one_or_none()
    if old:
        old.is_current = False

    # Build full report markdown
    full_report = _build_report_md(paper, analysis_data)

    # Create L4 analysis
    analysis = PaperAnalysis(
        paper_id=paper_id,
        level=AnalysisLevel.L4_DEEP,
        model_provider=resp.provider,
        model_name=resp.model,
        prompt_version="l4_v1",
        schema_version="v1",
        confidence=0.8,
        problem_summary=analysis_data.get("problem_summary"),
        method_summary=analysis_data.get("method_summary"),
        evidence_summary=analysis_data.get("evidence_summary"),
        core_intuition=analysis_data.get("core_intuition"),
        changed_slots=analysis_data.get("changed_slots"),
        is_plugin_patch=analysis_data.get("is_plugin_patch"),
        worth_deep_read=analysis_data.get("worth_deep_read"),
        full_report_md=full_report,
        confidence_notes=analysis_data.get("confidence_notes"),
        evidence_spans=analysis_data.get("evidence_units"),
        is_current=True,
    )
    session.add(analysis)

    # Legacy: Create method delta if provided (backward compat)
    delta_data = analysis_data.get("delta_card")
    if delta_data and isinstance(delta_data, dict):
        delta = MethodDelta(
            paper_id=paper_id,
            analysis_id=None,
            paradigm_name=delta_data.get("paradigm", "unknown"),
            slots=delta_data.get("slots", {}),
            is_structural=delta_data.get("is_structural"),
            primary_gain_source=delta_data.get("primary_gain_source"),
        )
        session.add(delta)

    # Update state
    if paper.state != PaperState.CHECKED:
        paper.state = PaperState.L4_DEEP

    # Save method_category and improvement_type as tags
    method_cat = analysis_data.get("method_category")
    improvement_type = analysis_data.get("improvement_type")
    new_tags = list(paper.tags or [])
    if method_cat and f"method/{method_cat}" not in new_tags:
        new_tags.append(f"method/{method_cat}")
    if improvement_type and f"improvement/{improvement_type}" not in new_tags:
        new_tags.append(f"improvement/{improvement_type}")
    if new_tags != list(paper.tags or []):
        paper.tags = new_tags

    await session.flush()
    await session.refresh(analysis)

    # ── Auto-create bottleneck from LLM output ────────────────
    bottleneck_id = await _maybe_create_bottleneck(session, paper, analysis_data)

    # ── Graph pipeline (DeltaCard → IdeaDelta → Assertions) ───
    await _build_idea_graph(session, paper, analysis, analysis_data, bottleneck_id=bottleneck_id)

    return analysis


async def _maybe_create_bottleneck(
    session: AsyncSession,
    paper: Paper,
    analysis_data: dict,
) -> UUID | None:
    """Auto-create a ProjectBottleneck + PaperBottleneckClaim from L4 output."""
    bn_data = analysis_data.get("bottleneck_addressed")
    if not bn_data or not isinstance(bn_data, dict) or not bn_data.get("title"):
        return None

    from backend.models.research import ProjectBottleneck, PaperBottleneckClaim
    from sqlalchemy import func

    # Dedup: check if similar bottleneck already exists
    existing = await session.execute(
        select(ProjectBottleneck).where(
            func.lower(ProjectBottleneck.title) == bn_data["title"].lower()
        ).limit(1)
    )
    bn = existing.scalar_one_or_none()
    if bn:
        # Link this paper to existing bottleneck
        if bn.related_paper_ids and paper.id not in bn.related_paper_ids:
            bn.related_paper_ids = list(bn.related_paper_ids) + [paper.id]
        elif not bn.related_paper_ids:
            bn.related_paper_ids = [paper.id]
    else:
        # Create new bottleneck in global ontology
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

    # Always create a paper-level claim (paper says it solves this)
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


async def _build_idea_graph(
    session: AsyncSession,
    paper: Paper,
    analysis: PaperAnalysis,
    analysis_data: dict,
    bottleneck_id: UUID | None = None,
) -> None:
    """Post-L4 hook: build DeltaCard → IdeaDelta → Assertions.

    Pipeline: frame_assign → delta_card_build → evidence → idea_derive → assert → publish
    DeltaCard is the intermediate truth layer; IdeaDelta is derived from it.
    """
    from backend.services.frame_assign_service import assign_paradigm
    from backend.services.delta_card_service import run_delta_card_pipeline

    try:
        # 1. Frame assign (with LLM fallback for unknown domains)
        paradigm, slots = await assign_paradigm(
            session, paper.category, paper.tags,
            title=paper.title, abstract=paper.abstract,
        )

        # 2. Extract changed_slots in graph format
        delta_card_data = analysis_data.get("delta_card", {})
        changed_slots_graph = []
        raw_slots = analysis_data.get("changed_slots", [])
        # Build a lookup from delta_card.slots for change_type enrichment
        dc_slot_info = {}
        if isinstance(delta_card_data, dict) and "slots" in delta_card_data:
            for name, info in delta_card_data["slots"].items():
                if isinstance(info, dict):
                    dc_slot_info[name] = info

        if isinstance(raw_slots, list) and raw_slots:
            for s in raw_slots:
                if isinstance(s, str):
                    info = dc_slot_info.get(s, {})
                    changed_slots_graph.append({
                        "slot_name": s,
                        "from": info.get("from"),
                        "to": info.get("to"),
                        "change_type": info.get("change_type", "structural" if analysis_data.get("structurality_score", 0) >= 0.5 else "plugin"),
                    })
                elif isinstance(s, dict):
                    changed_slots_graph.append(s)
        elif isinstance(delta_card_data, dict) and "slots" in delta_card_data:
            for name, info in delta_card_data["slots"].items():
                if isinstance(info, dict) and info.get("changed"):
                    changed_slots_graph.append({
                        "slot_name": name,
                        "from": info.get("from"),
                        "to": info.get("to"),
                        "change_type": info.get("change_type", "unknown"),
                    })

        # 3. Run full DeltaCard pipeline
        result = await run_delta_card_pipeline(
            session,
            paper_id=paper.id,
            analysis_id=analysis.id,
            analysis_data=analysis_data,
            paradigm_id=paradigm.id if paradigm else None,
            paradigm_name=paradigm.name if paradigm else None,
            slots=[{"id": s["id"], "name": s["name"]} for s in slots] if slots else None,
            changed_slots_graph=changed_slots_graph if changed_slots_graph else None,
            bottleneck_id=bottleneck_id,
            model_provider=analysis.model_provider,
            model_name=analysis.model_name,
        )

        dc = result["delta_card"]
        idea = result["idea_delta"]

        # Propagate L4-derived scores back to paper (more accurate than triage heuristics)
        if dc.structurality_score is not None:
            paper.structurality_score = dc.structurality_score
        if dc.extensionability_score is not None:
            paper.extensionability_score = dc.extensionability_score

        logger.info(
            f"Graph built for paper {paper.id}: "
            f"DeltaCard={dc.id}({dc.status}), "
            f"IdeaDelta={idea.id}({idea.publish_status}), "
            f"evidence={len(result['evidence_units'])}, "
            f"assertions={len(result['assertions'])}"
        )

    except Exception as e:
        logger.error(f"Graph pipeline error for paper {paper.id}: {e}")
        # Don't fail the analysis — graph is additive


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


# ── Helpers ─────────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling common formatting issues."""
    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    logger.warning(f"Failed to parse LLM JSON response: {text[:200]}")
    return {}


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
