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

L4_SYSTEM = """You are a senior research analyst producing deep, critical paper analysis.

Your analysis must:
1. Align findings to the domain's canonical paradigm — specify which slots changed
2. DISTINGUISH STORY FROM SUBSTANCE: what the paper claims in abstract/intro vs what experiments actually prove
3. VERIFY BASELINES: are they the strongest available? Are comparisons fair? Are any strong baselines suspiciously absent?
4. Extract atomic evidence units with source anchors — distinguish code-verified from text-stated from inferred
5. Assess cross-domain transferability
6. For surveys/benchmarks/theoretical papers: adapt analysis — don't force method/slot analysis on non-method papers

CRITICAL: Do not take the paper's self-assessment at face value. Cross-check claims against tables/figures. Flag if:
- Abstract claims improvement X% but tables show it's only on subset
- Baselines are from 2+ years ago when stronger recent ones exist
- Ablations don't isolate the claimed contribution
- The 'structural change' is actually just adding a module

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
  "key_equations": [
    {{"latex": "LaTeX formula string (use standard LaTeX notation)", "slot_affected": "which component this equation defines", "explanation": "why this equation is important (1 sentence, Chinese)"}}
  ],
  "key_figures": [
    {{"fig_ref": "Table 2 / Figure 3 / Algorithm 1", "caption": "what this figure shows", "evidence_for": "which claim this figure supports (1 sentence, Chinese)"}}
  ],
  "paper_type": "ONE of: method | survey | benchmark | position | theoretical | dataset",
  "narrative_vs_substance": "Does abstract/intro match experimental results? Note discrepancies (1-2 sentences, Chinese)",
  "baseline_paper_titles": ["list of paper titles this work directly builds on or compares against"],
  "baseline_fairness": "Are baselines the strongest available? Any strong methods suspiciously absent? (1-2 sentences, Chinese)",
  "same_family_method": "name of the broader method family this belongs to (e.g. direct_preference_optimization, group_relative_reward, constitutional_ai)",
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


# ── L4 Deep (6-step pipeline) ──────────────────────────────────

async def deep_analyze_paper(session: AsyncSession, paper_id: UUID) -> PaperAnalysis | None:
    """Run L4 deep analysis — 6-step pipeline with independent retry.

    Pipeline:
      Step 1: extract_evidence  (equations/figures/evidence_spans)
      Step 2: build_delta_card  (baseline/slots/mechanisms/bottleneck)
      Step 3: build_compare_set (auto-fill comparison set from DB)
      Step 4: propose_lineage   (builds_on/extends/replaces)
      Step 5: synthesize_concept(multi-paper → update Concept)
      Step 6: reconcile_neighbors(reverse-update old papers)

    Defense lines:
      #1 Step 1 reads method/experiments FIRST, then cross-checks abstract.
      #2 Step 3 auto-fills comparison set from DB, not just paper self-report.
      #3 High-value claims require ≥2 evidence refs (enforced by publish gate).
    """
    from backend.services.analysis_steps import (
        _gather_text,
        merge_step_outputs,
        run_step1_extract_evidence,
        run_step2_build_delta,
    )

    paper = await session.get(Paper, paper_id)
    if not paper:
        return None

    # ── Gather text from L2 parse ─────────────────────────────
    l2_result = await session.execute(
        select(PaperAnalysis).where(
            PaperAnalysis.paper_id == paper_id,
            PaperAnalysis.level == AnalysisLevel.L2_PARSE,
            PaperAnalysis.is_current.is_(True),
        )
    )
    l2 = l2_result.scalar_one_or_none()
    text_content = _gather_text(paper, l2)

    # ── Step 1: Extract evidence ──────────────────────────────
    logger.info(f"[L4 Step 1/6] extract_evidence for {paper_id}")
    step1_data = await run_step1_extract_evidence(session, paper, text_content)

    # ── Step 2: Build delta card ──────────────────────────────
    logger.info(f"[L4 Step 2/6] build_delta for {paper_id}")
    step2_data = await run_step2_build_delta(session, paper, text_content, step1_data)

    # Merge into unified analysis_data (backward-compatible shape)
    analysis_data = merge_step_outputs(step1_data, step2_data)

    # ── Confidence based on input quality ─────────────────────
    input_length = len(text_content)
    if input_length < 500:
        analysis_data["_input_quality"] = "title_only"
        analysis_confidence = 0.3
    elif input_length < 2000:
        analysis_data["_input_quality"] = "abstract_only"
        analysis_confidence = 0.5
    else:
        analysis_data["_input_quality"] = "full_text"
        analysis_confidence = 0.8

    # ── Supersede old L4 ──────────────────────────────────────
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

    # ── Build report and persist PaperAnalysis ────────────────
    full_report = _build_report_md(paper, analysis_data)

    model_provider = analysis_data.get("_model_provider", "unknown")
    model_name = analysis_data.get("_model_name", "unknown")

    analysis = PaperAnalysis(
        paper_id=paper_id,
        level=AnalysisLevel.L4_DEEP,
        model_provider=model_provider,
        model_name=model_name,
        prompt_version="l4_step_v1",
        schema_version="v2",
        confidence=analysis_confidence,
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

    # Legacy MethodDelta (backward compat)
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

    # Update state + tags
    if paper.state != PaperState.CHECKED:
        paper.state = PaperState.L4_DEEP

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

    # ── Steps 3-6: each wrapped with session recovery ─────────
    # If any step throws a DB error the session enters failed state.
    # We must rollback to recover before the next step can proceed.

    # ── Step 3: Build comparison set (defense line #2) ────────
    logger.info(f"[L4 Step 3/6] build_compare_set for {paper_id}")
    try:
        from backend.services.baseline_comparator_service import build_compare_set
        await build_compare_set(session, paper.id, analysis_data)
    except Exception as e:
        logger.warning(f"Step 3 (compare_set) failed for {paper.id}: {e}")
        await session.rollback()

    # ── Step 4: Propose lineage ───────────────────────────────
    logger.info(f"[L4 Step 4/6] propose_lineage for {paper_id}")
    try:
        from backend.services.evolution_service import link_to_parent_baselines
        if paper.current_delta_card_id:
            await link_to_parent_baselines(session, paper.current_delta_card_id, analysis_data)
    except Exception as e:
        logger.warning(f"Step 4 (lineage) failed for {paper.id}: {e}")
        await session.rollback()

    # ── Step 5: Synthesize concepts ───────────────────────────
    logger.info(f"[L4 Step 5/6] synthesize_concept for {paper_id}")
    try:
        from backend.services.concept_synthesizer_service import synthesize_concepts
        await synthesize_concepts(session, paper.id, analysis_data)
    except Exception as e:
        logger.warning(f"Step 5 (concept) failed for {paper.id}: {e}")
        await session.rollback()

    # ── Step 6: Reconcile neighbors ───────────────────────────
    logger.info(f"[L4 Step 6/6] reconcile_neighbors for {paper_id}")
    try:
        from backend.services.incremental_reconciler_service import reconcile_neighbors
        await reconcile_neighbors(session, paper.id, analysis_data)
    except Exception as e:
        logger.warning(f"Step 6 (reconcile) failed for {paper.id}: {e}")
        await session.rollback()

    # ── Auto-export to paperAnalysis/ Markdown ────────────────
    try:
        from backend.services.export_service import export_paper_analysis
        await export_paper_analysis(session, paper.id)
    except Exception as e:
        logger.warning(f"Auto-export to paperAnalysis/ failed for {paper.id}: {e}")
        await session.rollback()

    logger.info(f"[L4 complete] 6-step pipeline done for {paper_id}")
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

def _parse_json_response(text: str, required_fields: list[str] | None = None) -> dict:
    """Parse JSON from LLM response with validation.

    Handles: markdown fences, leading/trailing text, truncated JSON.
    Validates: required fields present and non-None.
    """
    # Reuse the robust parser from analysis_steps
    from backend.services.analysis_steps import _parse_json_safe
    parsed = _parse_json_safe(text)

    if not parsed:
        logger.error(f"FAILED to parse LLM JSON (total failure): {text[:300]}")
        return {}

    # Validate required fields (use `is None`, not `not` — empty list is valid)
    if required_fields:
        missing = [f for f in required_fields if parsed.get(f) is None]
        if missing:
            logger.warning(f"LLM JSON missing required fields: {missing}. Available keys: {list(parsed.keys())}")

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
