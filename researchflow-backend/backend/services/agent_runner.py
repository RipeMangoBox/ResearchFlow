"""Agent Runner — executes individual agents and manages runs.

Each agent is a prompt template + LLM call + output parser.
AgentRun records are created for tracking, and results go to the blackboard.
"""

import json
import logging
import time
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.agent import AgentBlackboardItem, AgentRun, PaperExtraction
from backend.services.llm_service import call_llm

logger = logging.getLogger(__name__)


class AgentRunner:
    """Executes agents: build prompt, call LLM, parse output, save to blackboard."""

    # ── Agent Prompt Templates ───────────────────────────────────────────

    AGENT_PROMPTS = {
        # ── Shallow Phase Agents (merged: shallow_paper + method_delta_lite) ──

        "shallow_extractor": {
            "system": """You are a research paper analyst. Given paper context (abstract, introduction excerpt, method excerpt, experiment excerpt, figure/table captions), extract BOTH a PaperEssence and a MethodDelta in a single pass.

Output a single JSON object with TWO top-level keys: "paper_essence" and "method_delta".

{
  "paper_essence": {
    "problem_statement": str — the core problem (1-2 sentences),
    "core_claim": str — main claim/thesis (1 sentence),
    "method_summary": str — concise method description (2-3 sentences),
    "main_contributions": list[str] — 2-5 key contributions,
    "target_tasks": list[str] — tasks addressed (e.g., "video question answering"),
    "target_modalities": list[str] — modalities (e.g., "text", "image", "video"),
    "training_paradigm": str — e.g., "supervised", "RLHF", "DPO",
    "limitations": list[str] — 1-3 limitations,
    "evidence_refs": list[dict] — {"claim": str, "confidence": float 0-1, "basis": "code_verified"|"experiment_backed"|"text_stated"|"inferred"|"speculative", "reasoning": str}
  },
  "method_delta": {
    "proposed_method_name": str — name of proposed method/model,
    "baseline_methods": list[dict] — {"name": str, "role": "primary_baseline"|"secondary_baseline"|"component"},
    "changed_slots": list[dict] — {
      "slot_name": str — component changed (e.g., "reward_function", "attention_mechanism"),
      "baseline_value": str — what the baseline uses,
      "proposed_value": str — what this paper proposes,
      "change_type": "replace"|"modify"|"add"|"remove",
      "is_novel": bool
    },
    "is_plugin_patch": bool — true if plug-in module addable to any baseline,
    "is_structural_change": bool — true if changes overall architecture/pipeline,
    "should_create_method_node": bool — true if novel enough for KB method node,
    "creation_reason": str | null,
    "key_equations": list[str] — 1-3 semantic descriptions of key equations
  }
}

Rules:
1. Extract ONLY what is stated or directly supported. Mark inferences explicitly.
2. Confidence: 0.9+ stated facts, 0.7-0.9 well-supported, 0.5-0.7 inferences, <0.5 speculation.
3. Focus on WHAT changed in method_delta, not how well it works.
4. changed_slots: capture actual architectural/algorithmic delta, not surface differences.
5. If a field cannot be determined, use null or empty list.

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "ShallowExtract",
            "item_type": "shallow_extract",
            "phase": "shallow",
        },

        "reference_role": {
            "system": """You classify each reference citation in a research paper by its functional role.

Given the reference list and citation contexts from a paper, classify each reference.

Output a JSON object:
{
  "classifications": [
    {
      "ref_index": str — reference number/label (e.g., "[1]", "[23]"),
      "ref_title": str — title of the referenced paper (if available),
      "role": str — one of: "direct_baseline", "method_source", "formula_source", "dataset_source", "benchmark_source", "comparison_baseline", "same_task_prior_work", "survey_or_taxonomy", "background_citation", "implementation_reference", "unimportant_related_work",
      "confidence": float — 0-1 confidence in the role assignment,
      "where_mentioned": list[str] — sections where this reference appears (e.g., ["introduction", "method", "experiments"]),
      "recommended_ingest_level": str — one of: "deep" (should fully analyze), "abstract" (read abstract only), "metadata" (just track existence), "skip" (not relevant),
      "reason": str — brief justification for the role and ingest level
    }
  ],
  "anchor_baselines": list[str] — ref_indexes that are direct baselines the paper builds upon,
  "method_sources": list[str] — ref_indexes providing core algorithmic ideas
}

Rules:
1. A reference mentioned in the method section AND experiment tables is likely a direct_baseline or comparison_baseline.
2. References cited only in related work with no method/experiment connection are usually same_task_prior_work or background_citation.
3. Prioritize "deep" ingest for direct_baseline and method_source references.
4. A single reference can only have ONE primary role — pick the most specific one.

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "ReferenceRoleMap",
            "item_type": "reference_role_map",
            "phase": "shallow",
        },

        # method_delta_lite — merged into shallow_extractor (Phase 2A)
        # score — removed, replaced by deterministic scoring_engine (Phase 2B)

        # ── Deep Phase Agents ────────────────────────────────────────────

        "deep_analyzer": {
            "system": """You are a comprehensive deep analysis agent. You perform THREE tasks in a single pass:
1. Exhaustive method slot decomposition (MethodDeltaFull)
2. Experiment result extraction (ExperimentMatrix)
3. Formula and figure analysis (FormulaFigureAnalysis)

CRITICAL READING ORDER: Read Method/Algorithm sections FIRST, then Experiments/Ablations, then Abstract/Introduction. Do NOT rely on abstract claims — verify against actual content.

Output a single JSON object with THREE top-level keys: "method", "experiment", "formulas".

{
  "method": {
    "proposed_method_name": str,
    "baseline_methods": [{"name": str, "role": "primary_baseline"|"secondary_baseline"|"component_source", "paper_title": str|null, "evidence_refs": [{"claim": str, "section": str, "confidence": float}]}],
    "changed_slots": [{"slot_name": str, "baseline_value": str, "proposed_value": str, "change_type": "modified"|"added"|"removed"|"replaced", "is_novel": bool, "evidence_refs": [{"claim": str, "section": str, "confidence": float}]}],
    "new_components": [{"name": str, "description": str, "role_in_pipeline": str}],
    "pipeline_modules": [{"name": str, "input": str, "output": str, "is_new": bool, "replaces": str|null}],
    "should_create_method_node": bool,
    "should_create_lineage_edge": bool,
    "lineage_parent": str|null
  },
  "experiment": {
    "main_results": [{"benchmark": str, "metric": str, "proposed_score": float|str, "baseline_scores": [{"name": str, "score": float|str}], "improvement": str, "is_sota": bool, "evidence_refs": [{"table_or_figure": str, "section": str, "confidence": float}]}],
    "ablations": [{"component_removed": str, "effect": str, "delta_value": float|str|null, "delta_metric": str|null, "supports_core_claim": bool}],
    "costs": {"training_compute": str|null, "inference_latency": str|null, "model_parameters": str|null, "gpu_type": str|null, "training_time": str|null},
    "fairness_assessment": {"are_comparisons_fair": bool, "are_baselines_strongest": bool, "missing_baselines": [str], "potential_issues": [str], "overall_evidence_strength": float}
  },
  "formulas": {
    "key_formulas": [{"latex": str, "name": str, "explanation_zh": str, "slot_affected": str|null, "differs_from_baseline": bool, "baseline_formula_latex": str|null}],
    "pipeline_figure": {"description": str, "modules": [{"name": str, "role": str}], "flow_description": str}|null,
    "figure_roles": [{"fig_ref": str, "semantic_role": "motivation"|"pipeline"|"architecture"|"result"|"failure_case"|"ablation_visual"|"comparison", "description_zh": str}],
    "formula_derivation_steps": [{"step": int, "from_formula": str, "to_formula": str, "explanation": str, "technique_used": str|null}]
  }
}

Rules:
1. method.changed_slots: be specific — "replaces 2-layer MLP with GLU" not "uses modified MLP".
2. experiment: extract EXACT numbers from tables. Do not paraphrase or approximate.
3. formulas: include ALL key formulas, especially the main objective/loss and novel equations.
4. Every evidence_refs should point to specific sections/equations/tables.
5. Distinguish "claims novelty" from "actually novel based on method description".
6. For ablations, check if they actually support core claims.
7. explanation_zh in formulas should be in Chinese, accessible to graduate students.

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "DeepAnalysis",
            "item_type": "deep_analysis",
            "phase": "deep",
        },

        # ── Graph Phase Agents ──────────────────────────────────────────

        "graph_candidate": {
            "system": """You are a knowledge graph candidate extraction agent. You generate node and edge candidates from paper analysis results for inclusion in a structured knowledge graph.

CRITICAL: Distinguish carefully between different relation types:
- "cites" = the paper merely references another work (bibliographic)
- "builds_on" = the paper's method directly extends or modifies another method (methodological lineage)
- "mentions" = a dataset/benchmark is named but not used
- "evaluates_on" = the paper actually runs experiments on this dataset/benchmark
Do NOT conflate these. A paper citing BERT in related work is NOT the same as a paper that builds on BERT's architecture.

Given the paper's PaperEssence, MethodDelta, task/mechanism/dataset facets, ExperimentMatrix, and ReferenceRoleMap, extract a GraphCandidates JSON object.

Output a single JSON object with these fields:

{
  "node_candidates": [
    {
      "node_type": "task" | "method" | "mechanism" | "dataset" | "benchmark" | "lineage" | "lab",
      "name": str,
      "name_zh": str | null,
      "one_liner": str,
      "evidence_refs": [{"claim": str, "section": str, "confidence": float}],
      "confidence": float
    }
  ],
  "edge_candidates": [
    {
      "source_type": str,
      "source_ref": str,
      "target_type": str,
      "target_ref": str,
      "relation_type": str,
      "slot_name": str | null,
      "one_liner": str,
      "evidence_refs": [{"claim": str, "section": str, "confidence": float}],
      "confidence": float
    }
  ],
  "lineage_candidates": [
    {
      "child_method": str,
      "parent_method": str,
      "relation": "builds_on" | "extends" | "replaces",
      "changed_slots": [str],
      "evidence": str
    }
  ]
}

Field definitions:
- node_type: The category of the entity. "task" = a research task/problem (e.g., "video QA"). "method" = a named method or model (e.g., "GRPO", "ViT"). "mechanism" = a reusable technique/component (e.g., "cross-attention", "contrastive loss"). "dataset" = a training/evaluation dataset. "benchmark" = an evaluation benchmark. "lineage" = a method evolution chain. "lab" = a research team/organization.
- name: Canonical English name. Use the most widely recognized name (e.g., "BERT" not "Bidirectional Encoder Representations from Transformers").
- name_zh: Chinese name if applicable (e.g., "视频问答" for "Video QA"). null for proper nouns that don't have Chinese equivalents.
- one_liner: A single sentence describing this entity. For methods, focus on what it does differently.
- source_ref / target_ref: The name of the source/target node. Must match a name in node_candidates or an existing KB node.
- relation_type: One of: "proposes_method", "evaluates_on", "uses_dataset", "compares_against", "modifies_slot", "extends_method", "cites_as_baseline", "belongs_to_task", "part_of_lineage", "produced_by_lab".
- slot_name: For "modifies_slot" edges, which slot is modified. null for other relation types.
- confidence: 0-1. Use 0.9+ only for explicitly stated facts. 0.7-0.9 for well-supported inferences. 0.5-0.7 for reasonable guesses. Below 0.5 = don't include.
- lineage_candidates: Method evolution relationships. "builds_on" = extends with modifications. "extends" = adds new capabilities without changing core. "replaces" = fundamentally different approach to the same problem.
- changed_slots: For lineage edges, list the slot names that changed between parent and child.

Rules:
1. Only propose nodes that are substantively discussed in the paper — not every citation deserves a node.
2. For method nodes, only create candidates for methods that have enough detail to profile (name + what it does + how it differs).
3. Edge confidence should reflect how clearly the paper establishes the relationship, not how important the relationship is.
4. A paper typically proposes 1 method, evaluates on 2-5 benchmarks, compares against 3-10 baselines, and belongs to 1-3 tasks. Deviate from these ranges only with strong evidence.
5. Do NOT create duplicate nodes for the same entity with different names — pick the canonical name.
6. lineage_candidates should only include relationships where the paper's method clearly descends from the parent method (not just cites it).

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "GraphCandidates",
            "item_type": "graph_candidates",
            "phase": "graph",
        },
        "kb_profiler": {
            "system": """You are a knowledge graph profiling agent. You generate profiles for BOTH nodes and edges in a single batch.

Given a list of node candidates (with metadata and connected papers) and edge candidates (with source/target nodes and evidence), generate profiles for all of them.

Output a single JSON object with TWO top-level keys: "node_profiles" and "edge_profiles".

{
  "node_profiles": [
    {
      "node_name": str,
      "one_liner": str — single sentence (40-80 chars) capturing the entity's essence,
      "short_intro_md": str — 2-3 paragraphs in Chinese Markdown,
      "detailed_md": str — wiki page in Chinese with sections: 概述, 核心特点, 技术细节, 相关工作, 应用场景,
      "structured_json": dict — type-dependent fields (see below),
      "evidence_refs": [{"claim": str, "source_paper": str, "confidence": float}]
    }
  ],
  "edge_profiles": [
    {
      "source_ref": str,
      "target_ref": str,
      "relation_type": str,
      "one_liner": str — contextual sentence in Chinese (e.g., "本文以 GRPO 为 baseline，修改了 reward design"),
      "relation_summary": str — 2-3 sentences (Chinese) explaining WHY connected,
      "source_context": str — from source perspective (Chinese),
      "target_context": str — from target perspective (Chinese),
      "evidence_refs": [{"claim": str, "source_paper": str, "confidence": float}]
    }
  ]
}

structured_json by node_type:
  task: {"input_modalities": [str], "output_modalities": [str], "evaluation_metrics": [str], "representative_benchmarks": [str]}
  method: {"architecture_type": str, "training_paradigm": str, "key_components": [str], "parent_method": str|null, "year_introduced": int|null}
  mechanism: {"mechanism_category": str, "applicable_to": [str], "key_formula": str|null, "advantages": [str], "limitations": [str]}
  dataset: {"domain": str, "size": str, "modalities": [str], "annotation_type": str}
  benchmark: {"task": str, "metrics": [str], "dataset_used": str|null}

Rules:
1. Base ALL content on provided evidence. Do NOT hallucinate facts or paper titles.
2. Use null for insufficient information rather than guessing.
3. short_intro_md and detailed_md in Chinese; structured_json values in English.
4. Edge one_liners must be contextual, not generic (no "这两个有关系").
5. detailed_md: 500-1500 chars. Don't pad with filler.
6. Match node_name / source_ref / target_ref exactly to the input candidate names.

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "KBProfiles",
            "item_type": "kb_profiles",
            "phase": "profile",
        },

        # ── Report Phase Agents ──────────────────────────────────────────

        "paper_report": {
            "system": """You are a structured paper report generation agent. You produce a comprehensive 10-section report for a single paper based on all verified analysis artifacts.

CRITICAL: Base everything on the verified extractions provided in the context. Do NOT invent facts, numbers, or paper titles. If a section cannot be filled due to missing data, write a brief note explaining what is missing rather than fabricating content.

Given all verified blackboard items (PaperEssence, MethodDelta, ExperimentMatrix, FormulaFigureAnalysis, ReferenceRoleMap, GraphCandidates) and selected evidence, generate a PaperReport JSON object.

Output a single JSON object with these fields:

{
  "title_zh": str,
  "title_en": str,
  "sections": [
    {
      "section_type": str,
      "title": str,
      "body_md": str
    }
  ]
}

The sections array MUST contain exactly 10 sections in this order:

1. section_type: "metadata"
   title: "基本信息"
   body_md: Paper title, authors, venue, year, links (code/data). Formatted as a metadata block.

2. section_type: "core_claim"
   title: "核心主张"
   body_md: The paper's central thesis in 1-2 sentences, followed by the key evidence supporting it. Include confidence assessment.

3. section_type: "motivation"
   title: "研究动机"
   body_md: What problem does this paper address? Why is it important? What gap in prior work does it fill? Reference specific prior work limitations.

4. section_type: "pipeline"
   title: "方法流程"
   body_md: Step-by-step description of the proposed method pipeline. Use the pipeline_modules from MethodDelta. Include a text-based flow diagram if possible.

5. section_type: "formula"
   title: "关键公式"
   body_md: List key formulas with LaTeX notation and Chinese explanations. Show derivation steps if available. Highlight which formulas are novel vs. borrowed.

6. section_type: "experiment"
   title: "实验结果"
   body_md: Summarize main results with exact numbers. Include benchmark, metric, score, and comparison to baselines. Note ablation findings. Assess evidence strength.

7. section_type: "related_work"
   title: "相关工作"
   body_md: Categorize references by role (baseline, method source, etc.). Highlight the most important 3-5 references and their relationship to this paper.

8. section_type: "lineage"
   title: "方法谱系"
   body_md: Where does this method sit in its evolution chain? What did it inherit from its parent method? What did it change? Use slot-level language.

9. section_type: "limitations"
   title: "局限与展望"
   body_md: Stated limitations from the paper + inferred limitations from the analysis. Potential future directions. Be honest about weaknesses.

10. section_type: "knowledge_position"
    title: "知识图谱定位"
    body_md: How this paper connects to the broader knowledge graph. Which task nodes, method nodes, and dataset nodes does it touch? What is its contribution to the field's structure?

Rules:
1. Each section body_md should be 200-600 characters in Chinese.
2. Use Markdown formatting (headers, bold, lists, code blocks for formulas).
3. Include specific numbers, paper titles, and method names — not vague references.
4. If a verified extraction is missing (e.g., no ExperimentMatrix), write a brief note in the relevant section: "（实验数据尚未提取，待补充）".
5. title_zh should be a Chinese title for the report (not the paper title — a descriptive report title).
6. title_en should be the original paper title in English.

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "PaperReport",
            "item_type": "paper_report",
            "phase": "report",
        },
        # quality_audit — removed (Phase 2E), replaced by deterministic validation
    }

    def __init__(self, session: AsyncSession, llm_service=None):
        """Initialize with DB session and optional LLM service override.

        Args:
            session: Async SQLAlchemy session for DB operations.
            llm_service: Optional override; if None, uses the module-level call_llm.
        """
        self.session = session
        self._call_llm = llm_service or call_llm

    async def run_agent(
        self,
        agent_name: str,
        context: dict,
        *,
        paper_id: UUID | None = None,
        candidate_id: UUID | None = None,
        domain_id: UUID | None = None,
    ) -> dict:
        """Execute an agent: create run record, call LLM, parse, save to blackboard.

        Args:
            agent_name: Key in AGENT_PROMPTS.
            context: Dict from ContextPackBuilder.build() with keys:
                     system_prompt, user_content, token_budget, metadata.
            paper_id: Optional paper ID for tracking.
            candidate_id: Optional candidate ID for tracking.
            domain_id: Optional domain ID for tracking.

        Returns:
            Parsed JSON result dict from the LLM.

        Raises:
            ValueError: If agent_name is unknown.
            RuntimeError: If LLM call or JSON parsing fails.
        """
        if agent_name not in self.AGENT_PROMPTS:
            raise ValueError(
                f"Unknown agent: {agent_name!r}. "
                f"Available: {list(self.AGENT_PROMPTS.keys())}"
            )

        agent_config = self.AGENT_PROMPTS[agent_name]
        phase = agent_config.get("phase", "unknown")

        # 1. Create AgentRun record
        run = AgentRun(
            paper_id=paper_id,
            candidate_id=candidate_id,
            domain_id=domain_id,
            agent_name=agent_name,
            phase=phase,
            status="running",
        )
        self.session.add(run)
        await self.session.flush()  # get run.id
        run_id = run.id

        start_time = time.monotonic()

        try:
            # 2. Build the prompt
            system_prompt = agent_config["system"]
            user_content = context.get("user_content", "")

            if not user_content.strip():
                raise ValueError(
                    f"Empty user_content for agent {agent_name}. "
                    "Context pack may have failed to load any data."
                )

            # 3. Call LLM
            token_budget = context.get("token_budget", 4096)
            max_output_tokens = min(token_budget, 8192)

            llm_response = await self._call_llm(
                prompt=user_content,
                system=system_prompt,
                max_tokens=max_output_tokens,
                temperature=0.2,
                session=self.session,
                paper_id=paper_id,
                prompt_version=f"agent_{agent_name}_v1",
            )

            # 4. Parse JSON response
            result = self._parse_json_response(llm_response.text, agent_name)

            # 5. Save to blackboard
            item_type = agent_config["item_type"]
            blackboard_item = AgentBlackboardItem(
                run_id=run_id,
                paper_id=paper_id,
                candidate_id=candidate_id,
                item_type=item_type,
                value_json=result,
                producer_agent=agent_name,
                is_verified=False,
            )
            self.session.add(blackboard_item)

            # 6. Update AgentRun — success
            duration_ms = int((time.monotonic() - start_time) * 1000)
            run.status = "success"
            run.model_name = llm_response.model
            run.input_token_count = llm_response.input_tokens
            run.output_token_count = llm_response.output_tokens
            run.duration_ms = duration_ms
            run.cost_usd = self._estimate_cost_from_response(llm_response)
            run.completed_at = datetime.now(timezone.utc)

            await self.session.flush()

            logger.info(
                "Agent %s completed in %dms (in=%d, out=%d tokens)",
                agent_name, duration_ms,
                llm_response.input_tokens, llm_response.output_tokens,
            )
            return result

        except Exception as e:
            # Update AgentRun — failure
            duration_ms = int((time.monotonic() - start_time) * 1000)
            run.status = "failed"
            run.duration_ms = duration_ms
            run.error_message = str(e)[:500]
            run.completed_at = datetime.now(timezone.utc)

            await self.session.flush()

            logger.error("Agent %s failed after %dms: %s", agent_name, duration_ms, e)
            raise

    async def save_extraction(
        self,
        paper_id: UUID,
        extraction_type: str,
        value: dict,
        run_id: UUID,
    ) -> PaperExtraction:
        """Save a structured extraction to paper_extractions table.

        Creates a new versioned extraction, marking previous versions as non-current.
        """
        # Find current version number for this paper+type
        from sqlalchemy import func as sa_func, select

        max_version = (
            await self.session.execute(
                select(sa_func.coalesce(
                    sa_func.max(PaperExtraction.extraction_version), 0
                )).where(
                    PaperExtraction.paper_id == paper_id,
                    PaperExtraction.extraction_type == extraction_type,
                )
            )
        ).scalar()

        new_version = (max_version or 0) + 1

        extraction = PaperExtraction(
            paper_id=paper_id,
            extraction_type=extraction_type,
            value_json=value,
            producer_run_id=run_id,
            extraction_version=new_version,
            review_status="auto",
        )
        self.session.add(extraction)
        await self.session.flush()

        logger.info(
            "Saved extraction %s v%d for paper %s",
            extraction_type, new_version, paper_id,
        )
        return extraction

    # ── Private Helpers ──────────────────────────────────────────────────

    def _parse_json_response(self, text: str, agent_name: str) -> dict:
        """Parse LLM response text as JSON, handling common formatting issues."""
        cleaned = text.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            # Remove opening fence (with optional language tag)
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]
            else:
                cleaned = cleaned[3:]  # strip just the ```
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(
                "Agent %s returned invalid JSON: %s\nResponse preview: %.500s",
                agent_name, e, text,
            )
            raise RuntimeError(
                f"Agent {agent_name} returned invalid JSON: {e}"
            ) from e

        if not isinstance(result, dict):
            raise RuntimeError(
                f"Agent {agent_name} returned {type(result).__name__}, expected dict"
            )

        return result

    @staticmethod
    def _estimate_cost_from_response(resp) -> float:
        """Estimate cost from LLMResponse (mirrors llm_service._estimate_cost)."""
        costs = {
            "kimi-k2.6": (1.0 / 1_000_000, 4.0 / 1_000_000),
            "so-4.6": (3.0 / 1_000_000, 15.0 / 1_000_000),
            "claude-haiku-4.5": (0.80 / 1_000_000, 4.0 / 1_000_000),
            "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
            "gpt-4o": (2.50 / 1_000_000, 10.0 / 1_000_000),
            "mock": (0, 0),
        }
        rate = costs.get(resp.model, (1.0 / 1_000_000, 3.0 / 1_000_000))
        return resp.input_tokens * rate[0] + resp.output_tokens * rate[1]
