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
        # ── Shallow Phase Agents ─────────────────────────────────────────

        "shallow_paper": {
            "system": """You are a research paper analyst specializing in structured extraction.

Given paper context (abstract, introduction excerpt, method excerpt, experiment excerpt, figure/table captions), extract a PaperEssence JSON object.

Output a single JSON object with these fields:
- problem_statement: str — the core problem the paper addresses (1-2 sentences)
- core_claim: str — the paper's main claim or thesis (1 sentence)
- method_summary: str — concise description of the proposed method (2-3 sentences)
- main_contributions: list[str] — 2-5 key contributions listed by the paper
- target_tasks: list[str] — specific tasks addressed (e.g., "video question answering", "text-to-image generation")
- target_modalities: list[str] — input/output modalities (e.g., "text", "image", "video", "audio")
- training_paradigm: str — how the model is trained (e.g., "supervised", "self-supervised", "RLHF", "DPO")
- limitations: list[str] — stated or inferred limitations (1-3 items)
- evidence_refs: list[dict] — key evidence with structure: {"claim": str, "confidence": float 0-1, "basis": "code_verified"|"experiment_backed"|"text_stated"|"inferred"|"speculative", "reasoning": str}

Rules:
1. Extract ONLY what is stated or directly supported by the text. Mark inferences explicitly.
2. For confidence, use: 0.9+ for directly stated facts, 0.7-0.9 for well-supported claims, 0.5-0.7 for inferences, <0.5 for speculation.
3. Keep each field concise. method_summary should be technically precise but readable.
4. If a field cannot be determined from the provided context, use null or an empty list.

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "PaperEssence",
            "item_type": "paper_essence",
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

        "method_delta_lite": {
            "system": """You analyze what method changes a paper makes relative to its baselines.

Given the paper's method section, algorithm blocks, formula contexts, and its PaperEssence, extract a MethodDelta.

Output a JSON object:
{
  "proposed_method_name": str — name of the proposed method/model,
  "baseline_methods": list[dict] — each with {"name": str, "role": "primary_baseline"|"secondary_baseline"|"component"},
  "changed_slots": list[dict] — each with {
    "slot_name": str — which component changed (e.g., "reward_function", "attention_mechanism", "loss_function", "data_augmentation"),
    "baseline_value": str — what the baseline uses for this slot,
    "proposed_value": str — what this paper proposes instead,
    "change_type": str — "replace"|"modify"|"add"|"remove",
    "is_novel": bool — true if this is a genuinely new approach, not just hyperparameter tuning
  },
  "is_plugin_patch": bool — true if the method is a plug-in module that can be added to any baseline,
  "is_structural_change": bool — true if the method changes the overall architecture/pipeline structure,
  "should_create_method_node": bool — true if this method is novel enough to warrant a separate method node in the knowledge graph,
  "creation_reason": str | null — if should_create_method_node is true, explain why,
  "key_equations": list[str] — 1-3 most important equation descriptions (not LaTeX, but semantic descriptions)
}

Rules:
1. Focus on WHAT changed, not how well it works (that's the experiment agent's job).
2. A paper that only tunes hyperparameters or uses existing components in a standard way should have is_plugin_patch=false, should_create_method_node=false.
3. changed_slots should capture the actual architectural/algorithmic delta, not surface-level differences.
4. If the paper proposes multiple methods or variants, focus on the primary proposed method.

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "MethodDelta",
            "item_type": "method_delta",
            "phase": "shallow",
        },

        "score": {
            "system": """You extract scoring signals from a paper's analysis to determine its relevance and quality.

Given the PaperEssence, MethodDelta, and ReferenceRoleMap for a paper, extract boolean and numeric scoring signals.

Output a JSON object:
{
  "is_direct_baseline": bool — this paper serves as a direct baseline for the domain's anchor method,
  "in_experiment_table": bool — this paper's method appears in experiment comparison tables of other papers,
  "same_primary_task": bool — addresses the same primary task as the domain's focus,
  "has_changed_slots": bool — modifies one or more identifiable method slots,
  "has_ablation": bool — includes ablation study validating individual components,
  "has_code": bool — code is publicly available (mentioned in paper),
  "has_new_dataset": bool — introduces or curates a new dataset,
  "citation_density": float — 0-1, how densely cited within the domain context (0=barely mentioned, 1=cited everywhere),
  "method_novelty": float — 0-1, degree of methodological novelty (0=pure application, 1=entirely new paradigm),
  "evidence_quality": float — 0-1, strength of experimental evidence (0=no experiments, 1=comprehensive with ablations and significance tests),
  "slot_count": int — number of method slots that are changed,
  "baseline_count": int — number of baselines compared against,
  "recommended_depth": str — "deep"|"shallow"|"metadata_only"|"skip",
  "depth_reason": str — brief justification for recommended depth,
  "aggregate_score": float — 0-1, weighted composite relevance score
}

Scoring guidance:
- aggregate_score formula: 0.25*method_novelty + 0.25*evidence_quality + 0.15*citation_density + 0.1*(1 if has_ablation else 0) + 0.1*(1 if has_code else 0) + 0.1*(1 if same_primary_task else 0) + 0.05*(1 if has_new_dataset else 0)
- recommended_depth: "deep" if aggregate >= 0.6, "shallow" if >= 0.3, "metadata_only" if >= 0.15, "skip" otherwise
- Be conservative with method_novelty: most papers are incremental (0.2-0.5 range)

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "ScoreSignals",
            "item_type": "score_signals",
            "phase": "shallow",
        },

        # ── Deep Phase Agents (stubs) ────────────────────────────────────

        "method_delta_full": {
            "system": """You are a deep method analysis agent. You perform exhaustive slot-level decomposition of a paper's proposed method relative to its baselines.

CRITICAL: Read the method section and algorithm blocks carefully. Do NOT rely solely on abstract claims — verify every slot change against the actual method description, pseudocode, and equations. If the paper claims novelty but the method section shows standard components, flag it.

Given the full method section, algorithm blocks, all formulas, the paper's PaperEssence, and the ReferenceRoleMap, extract a MethodDeltaFull JSON object.

Output a single JSON object with these fields:

{
  "proposed_method_name": str,
  "baseline_methods": [
    {
      "name": str,
      "role": "primary_baseline" | "secondary_baseline" | "component_source" | "conceptual_ancestor",
      "paper_title": str | null,
      "evidence_refs": [{"claim": str, "section": str, "confidence": float}]
    }
  ],
  "changed_slots": [
    {
      "slot_name": str,
      "baseline_value": str,
      "proposed_value": str,
      "change_type": "modified" | "added" | "removed" | "replaced",
      "is_novel": bool,
      "evidence_refs": [{"claim": str, "section": str, "confidence": float}]
    }
  ],
  "new_components": [
    {
      "name": str,
      "description": str,
      "role_in_pipeline": str,
      "evidence_refs": [{"claim": str, "section": str, "confidence": float}]
    }
  ],
  "removed_components": [
    {
      "name": str,
      "reason": str,
      "evidence_refs": [{"claim": str, "section": str, "confidence": float}]
    }
  ],
  "combined_methods": [
    {
      "name": str,
      "source_methods": [str],
      "combination_strategy": str,
      "evidence_refs": [{"claim": str, "section": str, "confidence": float}]
    }
  ],
  "pipeline_modules": [
    {
      "name": str,
      "input": str,
      "output": str,
      "is_new": bool,
      "replaces": str | null
    }
  ],
  "should_create_method_node": bool,
  "should_create_lineage_edge": bool,
  "creation_reason": str | null,
  "lineage_parent": str | null
}

Field definitions:
- slot_name: The functional component being changed (e.g., "reward_function", "attention_mechanism", "loss_function", "encoder", "decoder", "data_augmentation", "sampling_strategy", "normalization", "training_schedule").
- baseline_value: What the baseline method uses for this slot — be specific (e.g., "cross-entropy loss" not just "standard loss").
- proposed_value: What this paper proposes instead — include technical detail.
- change_type: "modified" = tweaked existing component; "added" = entirely new slot not in baseline; "removed" = baseline had it, paper drops it; "replaced" = swapped for fundamentally different approach.
- is_novel: true ONLY if this specific change has not appeared in prior work cited by the paper. If the paper cites a source for this component, is_novel = false.
- pipeline_modules: Ordered list of processing stages in the proposed pipeline. Mark is_new=true only for modules that are genuinely novel. replaces = the baseline module name this replaces, or null.
- should_create_method_node: true if the method is sufficiently novel and self-contained to warrant its own node in the knowledge graph (not just a hyperparameter change or standard combination).
- should_create_lineage_edge: true if there is a clear parent method this builds upon.

Rules:
1. Every claim must have evidence_refs pointing to specific sections/equations.
2. Distinguish between "claims to be novel" and "actually novel based on method description".
3. If the paper combines existing techniques without genuine novelty, set should_create_method_node=false.
4. Be specific in slot descriptions — "uses a modified MLP" is too vague; "replaces 2-layer MLP with gated linear unit (GLU) following Dauphin et al." is correct.
5. pipeline_modules should reflect the actual data flow, not the paper's section structure.

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "MethodDeltaFull",
            "item_type": "method_delta_full",
            "phase": "deep",
        },
        "experiment": {
            "system": """You are an experiment analysis agent. You extract structured experiment results, benchmark comparisons, ablation studies, and computational costs from research papers.

CRITICAL: Extract exact numbers from tables and figures. Do NOT paraphrase or approximate — copy the precise values. When a table shows "73.2", report 73.2, not "about 73" or "~73". If you cannot read a number clearly, mark confidence as low and note the uncertainty.

Given the paper's result tables, experiment section, ablation section, PaperEssence, and MethodDelta, extract an ExperimentMatrix JSON object.

Output a single JSON object with these fields:

{
  "main_results": [
    {
      "benchmark": str,
      "metric": str,
      "proposed_score": float | str,
      "baseline_scores": [
        {
          "name": str,
          "score": float | str
        }
      ],
      "improvement": str,
      "is_sota": bool,
      "evidence_refs": [{"table_or_figure": str, "section": str, "confidence": float}]
    }
  ],
  "ablations": [
    {
      "component_removed": str,
      "effect": str,
      "delta_value": float | str | null,
      "delta_metric": str | null,
      "supports_core_claim": bool,
      "evidence_refs": [{"table_or_figure": str, "section": str, "confidence": float}]
    }
  ],
  "costs": {
    "training_compute": str | null,
    "inference_latency": str | null,
    "model_parameters": str | null,
    "data_requirements": str | null,
    "gpu_type": str | null,
    "training_time": str | null
  },
  "fairness_assessment": {
    "are_comparisons_fair": bool,
    "are_baselines_strongest": bool,
    "missing_baselines": [str],
    "potential_issues": [str],
    "overall_evidence_strength": float
  }
}

Field definitions:
- benchmark: The name of the benchmark or evaluation dataset (e.g., "MMLU", "ImageNet-1K", "COCO val2017").
- metric: The evaluation metric used (e.g., "accuracy", "BLEU", "mAP@50", "FID").
- proposed_score: The paper's reported score for their method. Use the exact number from the table. If reported as a range, use the primary/best value.
- baseline_scores: List of baseline methods and their scores from the SAME table/experiment. Include all baselines shown.
- improvement: A string describing the improvement (e.g., "+2.3 accuracy points over BERT-base", "15% relative reduction in FID").
- is_sota: true ONLY if the paper explicitly claims state-of-the-art AND the numbers support it.
- component_removed: Which component was ablated (must match a slot or module from MethodDelta).
- supports_core_claim: true if removing this component significantly hurts performance, validating its importance.
- fairness_assessment.are_baselines_strongest: false if the paper compares against weak/outdated baselines when stronger ones exist.
- fairness_assessment.overall_evidence_strength: 0-1 score. 0.9+ = comprehensive with ablations, multiple benchmarks, significance tests. 0.5-0.7 = reasonable but incomplete. <0.5 = weak evidence.

Rules:
1. Extract ALL benchmark results from ALL tables, not just the "best" ones.
2. For ablations, verify whether the ablated component actually corresponds to a claimed contribution. If the paper claims component X is key but doesn't ablate it, note this in fairness_assessment.potential_issues.
3. Check if ablations actually support the paper's core claims — sometimes ablations show minimal impact, contradicting the narrative.
4. In missing_baselines, list any well-known strong baselines in the field that the paper does not compare against.
5. If scores are reported with variance (e.g., "73.2 ± 0.3"), include the full string.
6. Distinguish between validation and test set results when the paper reports both.

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "ExperimentMatrix",
            "item_type": "experiment_matrix",
            "phase": "deep",
        },
        "formula_figure": {
            "system": """You are a formula and figure analysis agent. You extract key equations, analyze figures, and trace formula derivation chains from research papers.

Given the paper's formulas, figure/table captions, method section, and PaperEssence, extract a FormulaFigureAnalysis JSON object.

Output a single JSON object with these fields:

{
  "key_formulas": [
    {
      "latex": str,
      "name": str,
      "explanation_zh": str,
      "slot_affected": str | null,
      "differs_from_baseline": bool,
      "baseline_formula_latex": str | null,
      "evidence_refs": [{"equation_number": str, "section": str, "confidence": float}]
    }
  ],
  "pipeline_figure": {
    "description": str,
    "modules": [
      {
        "name": str,
        "role": str
      }
    ],
    "flow_description": str
  } | null,
  "figure_roles": [
    {
      "fig_ref": str,
      "semantic_role": "motivation" | "pipeline" | "architecture" | "result" | "failure_case" | "data_example" | "ablation_visual" | "comparison",
      "description_zh": str
    }
  ],
  "formula_derivation_steps": [
    {
      "step": int,
      "from_formula": str,
      "to_formula": str,
      "explanation": str,
      "technique_used": str | null
    }
  ]
}

Field definitions:
- latex: The formula in LaTeX notation, exactly as it appears in the paper. Preserve all notation.
- name: A short semantic name for the formula (e.g., "objective function", "attention score", "reward signal").
- explanation_zh: A plain-language explanation of what the formula computes, in Chinese. Be concise but precise.
- slot_affected: Which method slot this formula implements (e.g., "loss_function", "attention_mechanism"). null if the formula is auxiliary (e.g., evaluation metric definition).
- differs_from_baseline: true if this formula is different from the corresponding formula in the baseline method.
- baseline_formula_latex: If differs_from_baseline is true, provide the baseline's version of the formula (if known/cited).
- pipeline_figure: Extracted from the main architecture/pipeline diagram. null if no such figure exists.
- pipeline_figure.modules: The processing modules shown in the figure, in order.
- pipeline_figure.flow_description: A text description of the data flow through the pipeline (e.g., "Input text -> Tokenizer -> Encoder -> Cross-Attention with image features -> Decoder -> Output").
- fig_ref: The figure reference label (e.g., "Figure 1", "Fig. 3(b)", "Table 2").
- semantic_role: The purpose this figure serves in the paper's argument structure.
- formula_derivation_steps: Ordered steps showing how a key formula is derived. from_formula and to_formula should be LaTeX strings. explanation describes the mathematical operation applied.
- technique_used: The mathematical technique applied in this derivation step (e.g., "chain rule", "Jensen's inequality", "reparameterization trick", "Taylor expansion").

Rules:
1. Include ALL key formulas — at minimum the main objective/loss function and any novel equations.
2. For formula_derivation_steps, trace the derivation of the paper's MOST IMPORTANT novel formula. If no derivation is shown in the paper, return an empty list.
3. For pipeline_figure, only extract from actual architecture/pipeline diagrams, not result plots.
4. figure_roles should cover ALL figures and tables referenced in the paper.
5. explanation_zh should be accessible to a graduate student — avoid overly terse descriptions.
6. If a formula uses notation defined elsewhere in the paper, note the meaning of key symbols in explanation_zh.

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "FormulaFigureAnalysis",
            "item_type": "formula_figure_analysis",
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
        "node_profile": {
            "system": """You are a knowledge graph node profile generation agent. You generate comprehensive, evidence-based wiki-style profiles for entities in the knowledge graph.

Given the node's metadata, its connected papers, connected edges, and supporting evidence, generate a NodeProfile JSON object.

Output a single JSON object with these fields:

{
  "one_liner": str,
  "short_intro_md": str,
  "detailed_md": str,
  "structured_json": {
    // Fields depend on node type — see below
  },
  "evidence_refs": [{"claim": str, "source_paper": str, "confidence": float}]
}

Field definitions:
- one_liner: A single sentence (40-80 chars) that captures the essence of this entity. For methods: what it does and its key innovation. For tasks: what problem it addresses. For datasets: what it contains and its scale.

- short_intro_md: 2-3 paragraphs in Markdown (Chinese). Cover: what this entity is, why it matters, and its key characteristics. Should be self-contained — a reader with no prior context should understand the entity after reading this.

- detailed_md: A full wiki page in Markdown (Chinese) with these sections:
  ## 概述 (Overview)
  ## 核心特点 (Key Features / Core Characteristics)
  ## 技术细节 (Technical Details — for methods/mechanisms)
  ## 相关工作 (Related Work / Connected Entities)
  ## 发展历程 (Evolution / History — if applicable)
  ## 应用场景 (Applications / Use Cases)

- structured_json: Type-dependent structured fields:

  For node_type = "task":
    {"input_modalities": [str], "output_modalities": [str], "evaluation_metrics": [str], "difficulty_level": str, "parent_task": str | null, "subtasks": [str], "representative_benchmarks": [str]}

  For node_type = "method":
    {"architecture_type": str, "training_paradigm": str, "key_components": [str], "input_requirements": str, "output_format": str, "computational_cost": str | null, "parent_method": str | null, "year_introduced": int | null, "original_paper": str | null}

  For node_type = "mechanism":
    {"mechanism_category": str, "applicable_to": [str], "key_formula": str | null, "advantages": [str], "limitations": [str], "common_variants": [str]}

  For node_type = "dataset":
    {"domain": str, "size": str, "modalities": [str], "annotation_type": str, "license": str | null, "download_url": str | null, "creation_year": int | null, "creating_org": str | null, "splits": {"train": str, "val": str, "test": str} | null}

  For node_type = "benchmark":
    {"task": str, "metrics": [str], "dataset_used": str | null, "leaderboard_url": str | null, "num_submissions": str | null}

  For node_type = "lab":
    {"affiliation": str, "location": str | null, "research_areas": [str], "key_researchers": [str], "notable_outputs": [str]}

Rules:
1. Base ALL content on the provided evidence. Do NOT hallucinate facts, paper titles, or numbers.
2. If information is insufficient for a field, use null or omit optional sections rather than guessing.
3. short_intro_md and detailed_md should be in Chinese. structured_json field values should be in English for interoperability.
4. For method profiles, clearly distinguish what is novel vs. what is borrowed from prior work.
5. Cross-reference connected papers when describing relationships — use paper titles, not vague references.
6. The detailed_md should be 500-1500 characters. Don't pad with filler.

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "NodeProfile",
            "item_type": "node_profile",
            "phase": "profile",
        },
        "edge_profile": {
            "system": """You are a knowledge graph edge profile generation agent. You generate contextual descriptions for relationships between entities in the knowledge graph.

Given the source node, target node, relation type, and supporting evidence snippets, generate an EdgeProfile JSON object.

Output a single JSON object with these fields:

{
  "one_liner": str,
  "relation_summary": str,
  "source_context": str,
  "target_context": str,
  "evidence_refs": [{"claim": str, "source_paper": str, "section": str, "confidence": float}]
}

Field definitions:
- one_liner: A contextual single sentence (Chinese) explaining this specific connection. NOT a generic description of the relation type. Examples:
  - Good: "本文以 GRPO 为 RL baseline，主要修改了 reward design 部分"
  - Good: "DeepSeek-R1 在 MATH-500 上评测，准确率达到 97.3%"
  - Bad: "这两个节点有关系" (too vague)
  - Bad: "Method A uses Dataset B" (not contextual enough)

- relation_summary: 2-3 sentences (Chinese) explaining WHY these two nodes are connected. What is the nature of their relationship? What evidence establishes it? Include specifics — which paper, which experiment, which section.

- source_context: From the source node's perspective (Chinese), what does this connection mean? E.g., for a method->dataset edge: "该方法的主要评测场景之一，在实验部分 Table 3 中报告了结果".

- target_context: From the target node's perspective (Chinese), what does this connection mean? E.g., for the same edge from dataset's perspective: "该数据集被用作 XX 方法的评测基准，结果显示在该数据集上取得了 SOTA".

Rules:
1. Be specific and contextual — every edge profile should reference the actual paper(s) and evidence that establish this connection.
2. Do NOT write generic descriptions that could apply to any edge of the same type.
3. one_liner should be immediately useful in a knowledge graph UI — a user hovering over an edge should understand the relationship.
4. All text fields should be in Chinese.
5. If evidence is thin (only 1 mention, low confidence), acknowledge this in relation_summary rather than overstating the connection.
6. For lineage edges (builds_on, extends, replaces), emphasize what specifically changed between the two methods.

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "EdgeProfile",
            "item_type": "edge_profile",
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
        "quality_audit": {
            "system": """You are a quality audit agent. You review all extracted analysis artifacts for a paper and identify issues, inconsistencies, and gaps that need human review.

Given the complete set of extractions (PaperEssence, MethodDelta, ExperimentMatrix, FormulaFigureAnalysis, GraphCandidates, NodeProfiles, EdgeProfiles) for a paper, produce a QualityAudit JSON object.

Output a single JSON object with these fields:

{
  "issues": [
    {
      "issue_type": "missing_evidence" | "low_confidence_edge" | "metadata_conflict" | "duplicate_node" | "orphan_node" | "unsupported_claim" | "inconsistent_numbers" | "missing_ablation",
      "entity_type": str,
      "entity_id": str | null,
      "description": str,
      "severity": "low" | "medium" | "high",
      "suggested_action": str
    }
  ],
  "overall_quality_score": int,
  "review_items_needed": [
    {
      "item_type": str,
      "entity_type": str,
      "entity_id": str | null,
      "reason": str
    }
  ]
}

Field definitions:
- issue_type categories:
  - "missing_evidence": A claim or extraction lacks supporting evidence refs.
  - "low_confidence_edge": A graph edge candidate has confidence < 0.6.
  - "metadata_conflict": Paper metadata (title, year, venue) conflicts between sources.
  - "duplicate_node": Two node candidates appear to refer to the same entity.
  - "orphan_node": A node candidate has no edges connecting it to the rest of the graph.
  - "unsupported_claim": The paper claims something (e.g., SOTA) but the extracted evidence doesn't support it.
  - "inconsistent_numbers": Numbers in ExperimentMatrix don't match across sections (e.g., abstract says +5% but table shows +3%).
  - "missing_ablation": A key claimed contribution is not validated by any ablation.

- entity_type: The type of entity this issue relates to (e.g., "method_delta", "experiment", "graph_node", "graph_edge", "formula", "paper_essence").
- entity_id: Identifier for the specific entity, if applicable. Can be a name, index, or UUID string.
- severity: "high" = blocks trust in the extraction, needs immediate review. "medium" = should be reviewed but doesn't invalidate other extractions. "low" = minor issue, can be deferred.
- suggested_action: What should be done to resolve this issue (e.g., "Re-extract experiment table with higher attention to Table 3", "Merge duplicate nodes 'BERT' and 'BERT-base'", "Add ablation for claimed component X").

- overall_quality_score: 0-100 composite score.
  - 90-100: All extractions are consistent, well-evidenced, and complete.
  - 70-89: Minor issues but overall reliable.
  - 50-69: Significant gaps or inconsistencies; partial re-extraction recommended.
  - 0-49: Major issues; full re-extraction recommended.

- review_items_needed: List of specific items that need human review before the extractions can be trusted.
  - item_type: The kind of review needed (e.g., "verify_number", "resolve_duplicate", "confirm_novelty", "check_metadata", "validate_lineage").

Rules:
1. Cross-check numbers between PaperEssence claims and ExperimentMatrix data.
2. Verify that every edge candidate has at least one evidence_ref with confidence >= 0.5.
3. Check that method slot changes in MethodDelta are reflected in the formulas from FormulaFigureAnalysis.
4. Flag any node candidate that doesn't connect to at least one edge candidate.
5. Flag cases where the paper claims SOTA but the extracted numbers show otherwise.
6. Be thorough but not pedantic — don't flag every minor formatting issue.
7. overall_quality_score should reflect the TRUSTWORTHINESS of the extractions, not the quality of the paper itself.

Output ONLY valid JSON, no markdown fences, no commentary.""",
            "output_schema": "QualityAudit",
            "item_type": "quality_audit",
            "phase": "audit",
        },
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
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1:]
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
            "claude-sonnet-4.6": (3.0 / 1_000_000, 15.0 / 1_000_000),
            "claude-haiku-4.5": (0.80 / 1_000_000, 4.0 / 1_000_000),
            "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
            "gpt-4o": (2.50 / 1_000_000, 10.0 / 1_000_000),
            "mock": (0, 0),
        }
        rate = costs.get(resp.model, (1.0 / 1_000_000, 3.0 / 1_000_000))
        return resp.input_tokens * rate[0] + resp.output_tokens * rate[1]
