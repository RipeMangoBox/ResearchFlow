"""Analysis pipeline steps — split L4 into focused, independently retryable steps.

Step 1: extract_evidence  — equations, figures, evidence spans (read method first, not abstract)
Step 2: build_delta_card  — baseline, changed/unchanged slots, mechanisms, bottleneck claim

Each step has its own prompt, JSON schema, and retry logic.
Defense line #1: Step 1 prompt reads method/equations/ablations BEFORE abstract.
Defense line #3: High-value claims require ≥2 evidence references.
"""

import json
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.analysis import PaperAnalysis
from backend.models.enums import AnalysisLevel
from backend.models.paper import Paper
from backend.services.llm_service import call_llm

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

# ── Step 1: Extract Evidence ──────────────────────────────────────
#
# Defense line #1: "先看改动不先听故事"
# Prompt instructs LLM to read method/experiments/ablations FIRST,
# then cross-check against abstract claims.

STEP1_SYSTEM = """You are a meticulous research paper evidence extractor.

CRITICAL ORDER OF READING:
1. FIRST read the Method / Approach / Algorithm section — identify what actually changed.
2. THEN read Experiments / Results / Ablations — extract concrete evidence.
3. ONLY THEN read Abstract / Introduction — check if claims match evidence.

Do NOT take the abstract at face value. Extract what the paper DOES, not what it SAYS.
Output in Chinese where prose is needed. Output ONLY valid JSON."""

STEP1_PROMPT = """Extract evidence from this paper. Follow the reading order strictly.

Paper: {title}
Venue: {venue} {year}

{text_content}

Return JSON:
{{
  "key_equations": [
    {{"latex": "LaTeX string", "slot_affected": "which pipeline component", "explanation": "why important (Chinese, 1 sentence)"}}
  ],
  "key_figures": [
    {{"fig_ref": "Table 2 / Figure 3", "caption": "what it shows", "evidence_for": "which claim it supports (Chinese)"}}
  ],
  "evidence_units": [
    {{
      "atom_type": "mechanism | evidence | boundary | transfer",
      "claim": "specific factual claim (Chinese)",
      "confidence": 0.0-1.0,
      "basis": "code_verified | experiment_backed | text_stated | inferred | speculative",
      "source_section": "e.g. Table 3, Section 4.2",
      "source_page": null,
      "source_quote": "verbatim quote if available",
      "conditions": "under what conditions this holds",
      "failure_modes": "known failure cases"
    }}
  ],
  "narrative_vs_substance": "Does abstract match experimental evidence? Note discrepancies (Chinese, 1-2 sentences)",
  "baseline_fairness": "Are baselines the strongest available? Any strong methods suspiciously absent? (Chinese, 1-2 sentences)",
  "paper_type": "method | survey | benchmark | position | theoretical | dataset"
}}

Rules:
- Extract 3-8 evidence_units, each anchored to a specific section/table/figure.
- For key_equations: only equations that define the CORE change, max 4.
- For key_figures: only figures/tables that provide KEY evidence, max 4.
- Every evidence_unit with confidence >= 0.8 MUST have a source_section.
- If a claim appears only in the abstract but has no experimental backing, set basis="text_stated" and confidence <= 0.5."""

# ── Step 2: Build Delta Card ─────────────────────────────────────
#
# Uses Step 1 output as grounding. Focuses on structural analysis:
# what baseline, what slots changed, what mechanism, what bottleneck.

STEP2_SYSTEM = """You are a senior research analyst building a structured modification card.

You will receive the paper text AND evidence extracted in a prior step.
Your job: determine WHAT changed relative to the baseline paradigm.

CRITICAL:
1. Distinguish structural changes (rewire pipeline) from plugin patches (add module).
2. A paper that only adds a loss term is NOT structural — it's a plugin.
3. Identify the ACTUAL baseline, not just what the paper claims to improve upon.
4. Be honest about structurality — most papers are plugins.

Output in Chinese where prose is needed. Output ONLY valid JSON."""

STEP2_PROMPT = """Build a structured delta card for this paper.

Paper: {title}
Venue: {venue} {year}
Category: {category}
Tags: {tags}

Prior evidence extraction:
{step1_json}

Full text:
{text_content}

Return JSON:
{{
  "problem_summary": "问题与挑战 (200-400 words, Chinese)",
  "method_summary": "方法与洞察 — 重点说改了什么、怎么改的 (400-600 words, Chinese)",
  "evidence_summary": "证据与局限 (200-400 words, Chinese)",
  "core_intuition": "核心直觉 — 什么变了，为什么有效 (100-200 words, Chinese)",
  "changed_slots": ["slot1", "slot2"],
  "unchanged_slots": ["slot3", "slot4"],
  "is_plugin_patch": true/false,
  "method_category": "structural_architecture | plugin_module | reward_design | training_recipe | data_augmentation | inference_optimization | loss_function | representation_change | evaluation_method | theoretical_analysis",
  "improvement_type": "fundamental_rethink | component_replacement | additive_plugin | hyperparameter_trick | data_scaling | combination_of_existing",
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
  "bottleneck_addressed": {{
    "title": "核心瓶颈 (1 sentence, Chinese)",
    "description": "为什么这个瓶颈难解决 (2-3 sentences, Chinese)",
    "is_fundamental": true/false
  }},
  "baseline_paper_titles": ["list of paper titles this work builds on"],
  "same_family_method": "method family name (e.g. direct_preference_optimization)",
  "confidence_notes": [
    {{"claim": "...", "confidence": 0.0-1.0, "basis": "...", "reasoning": "..."}}
  ]
}}

Rules:
- changed_slots: ONLY slots where the paper makes a real modification.
- unchanged_slots: slots inherited from baseline without change.
- structurality_score: 0.0 (pure plugin/trick) to 1.0 (fundamental rethink).
  Most papers should be 0.2-0.5. Reserve 0.7+ for truly structural work.
- Cross-check your delta_card.slots against the evidence from Step 1."""


# ── Step execution ────────────────────────────────────────────────

def _parse_json_safe(text: str) -> dict:
    """Parse JSON from LLM response, tolerating markdown fences and truncation."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Handle truncated JSON — proxy APIs may cut responses mid-stream
    if start >= 0:
        fragment = text[start:]
        # Try to repair: close unclosed strings and brackets
        repaired = _repair_truncated_json(fragment)
        if repaired:
            try:
                result = json.loads(repaired)
                logger.warning(f"JSON repaired from truncated response (len={len(text)})")
                return result
            except json.JSONDecodeError:
                pass

    logger.error(f"JSON parse failed (len={len(text)}): {text[:300]}...TAIL: {text[-200:]}")
    return {}


def _repair_truncated_json(text: str) -> str | None:
    """Attempt to repair truncated JSON by closing unclosed structures.

    Strategy: track bracket/brace stack, find last complete top-level
    key-value pair, truncate there, close all open structures.
    """
    # Track state through the JSON
    last_top_comma = -1
    in_string = False
    escape_next = False
    stack = []  # Track open brackets: '{', '['

    for i, c in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if c in ('{', '['):
            stack.append(c)
        elif c == '}':
            if stack and stack[-1] == '{':
                stack.pop()
            if not stack:
                return text[:i + 1]  # Complete JSON found
        elif c == ']':
            if stack and stack[-1] == '[':
                stack.pop()
        elif c == ',':
            # Track last comma at top-level of the root object (stack depth = 1)
            if len(stack) == 1 and stack[0] == '{':
                last_top_comma = i

    if last_top_comma <= 0:
        return None

    # Truncate at last complete top-level value, close all open structures
    repaired = text[:last_top_comma]
    # Close any remaining open structures in reverse order
    for bracket in reversed(stack):
        if bracket == '{':
            repaired += "\n}"
        elif bracket == '[':
            repaired += "\n]"

    return repaired


async def _call_with_retry(
    prompt: str,
    system: str,
    max_tokens: int,
    session: AsyncSession,
    paper_id: UUID,
    prompt_version: str,
    required_fields: list[str],
) -> dict:
    """Call LLM with retry on missing required fields."""
    for attempt in range(1 + MAX_RETRIES):
        resp = await call_llm(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            session=session,
            paper_id=paper_id,
            prompt_version=prompt_version,
        )
        data = _parse_json_safe(resp.text)
        data["_model_provider"] = resp.provider
        data["_model_name"] = resp.model

        missing = [f for f in required_fields if not data.get(f)]
        if not missing:
            return data

        if attempt < MAX_RETRIES:
            # Log partial response for debugging
            for f in missing:
                val = data.get(f)
                logger.warning(f"  Field '{f}' value: {repr(val)[:100]}")
            logger.warning(
                f"Step {prompt_version} attempt {attempt+1}: "
                f"missing {missing}, retrying... "
                f"(parsed {len(data)} keys: {[k for k in data if not k.startswith('_')]})"
            )
        else:
            logger.warning(
                f"Step {prompt_version}: still missing {missing} "
                f"after {MAX_RETRIES} retries. Proceeding with partial data."
            )
    return data


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


async def run_step1_extract_evidence(
    session: AsyncSession,
    paper: Paper,
    text_content: str,
) -> dict:
    """Step 1: Extract evidence — equations, figures, evidence spans.

    Defense line #1: reads method/experiments first, then cross-checks abstract.
    """
    prompt = STEP1_PROMPT.format(
        title=paper.title,
        venue=paper.venue or "",
        year=paper.year or "",
        text_content=text_content[:25000],
    )

    return await _call_with_retry(
        prompt=prompt,
        system=STEP1_SYSTEM,
        max_tokens=4096,
        session=session,
        paper_id=paper.id,
        prompt_version="l4_step1_v1",
        required_fields=["evidence_units", "paper_type"],
    )


async def run_step2_build_delta(
    session: AsyncSession,
    paper: Paper,
    text_content: str,
    step1_data: dict,
) -> dict:
    """Step 2: Build delta card — baseline, slots, mechanism, bottleneck.

    Uses Step 1 evidence as grounding to prevent hallucination.
    """
    # Pass Step 1 output (sans internal fields) as context
    step1_clean = {k: v for k, v in step1_data.items() if not k.startswith("_")}

    prompt = STEP2_PROMPT.format(
        title=paper.title,
        venue=paper.venue or "",
        year=paper.year or "",
        category=paper.category,
        tags=", ".join(paper.tags or []),
        step1_json=json.dumps(step1_clean, ensure_ascii=False, indent=2)[:8000],
        text_content=text_content[:20000],
    )

    return await _call_with_retry(
        prompt=prompt,
        system=STEP2_SYSTEM,
        max_tokens=8000,
        session=session,
        paper_id=paper.id,
        prompt_version="l4_step2_v1",
        required_fields=["problem_summary", "method_summary", "changed_slots", "delta_card"],
    )


def merge_step_outputs(step1: dict, step2: dict) -> dict:
    """Merge Step 1 + Step 2 outputs into a unified analysis_data dict.

    This produces the same shape as the old monolithic L4 output,
    so downstream code (delta_card_service, etc.) works unchanged.
    """
    merged = {}

    # From Step 1: evidence, equations, figures
    merged["key_equations"] = step1.get("key_equations", [])
    merged["key_figures"] = step1.get("key_figures", [])
    merged["evidence_units"] = step1.get("evidence_units", [])
    merged["narrative_vs_substance"] = step1.get("narrative_vs_substance")
    merged["baseline_fairness"] = step1.get("baseline_fairness")
    merged["paper_type"] = step1.get("paper_type") or step2.get("paper_type")

    # From Step 2: summaries, delta card, scores
    merged["problem_summary"] = step2.get("problem_summary")
    merged["method_summary"] = step2.get("method_summary")
    merged["evidence_summary"] = step2.get("evidence_summary")
    merged["core_intuition"] = step2.get("core_intuition")
    merged["changed_slots"] = step2.get("changed_slots", [])
    merged["unchanged_slots"] = step2.get("unchanged_slots", [])
    merged["is_plugin_patch"] = step2.get("is_plugin_patch")
    merged["worth_deep_read"] = True  # If we got to L4, it's worth reading
    merged["method_category"] = step2.get("method_category")
    merged["improvement_type"] = step2.get("improvement_type")
    merged["structurality_score"] = step2.get("structurality_score")
    merged["extensionability_score"] = step2.get("extensionability_score")
    merged["transferability_score"] = step2.get("transferability_score")
    merged["delta_card"] = step2.get("delta_card", {})
    merged["bottleneck_addressed"] = step2.get("bottleneck_addressed")
    merged["baseline_paper_titles"] = step2.get("baseline_paper_titles", [])
    merged["same_family_method"] = step2.get("same_family_method")
    merged["confidence_notes"] = step2.get("confidence_notes", [])

    # Provenance
    merged["_model_provider"] = step2.get("_model_provider")
    merged["_model_name"] = step2.get("_model_name")

    return merged
