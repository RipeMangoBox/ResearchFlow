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
    "target_tasks": list[str] — research TASKS this paper addresses, NOT
        method names. A task is the *problem* being solved. Pick 1-3 from
        this canonical list (use exact spelling):
          ["Image Classification", "Object Detection", "Semantic Segmentation",
           "Instance Segmentation", "Depth Estimation",
           "Image Generation", "Video Generation", "Image Editing",
           "Image Captioning", "Visual Question Answering", "Visual Reasoning",
           "Cross-Modal Retrieval", "Cross-Modal Matching",
           "Video Understanding", "Action Recognition", "Action Segmentation",
           "Object Tracking", "3D Reconstruction", "Pose Estimation",
           "Speech Recognition", "Speech Synthesis", "Audio Generation",
           "Text Classification", "Text Generation", "Machine Translation",
           "Reasoning", "Code Generation", "Math Reasoning",
           "Reinforcement Learning", "Imitation Learning", "Federated Learning",
           "Continual Learning", "Domain Adaptation", "Few-Shot Learning",
           "Self-Supervised Learning", "Contrastive Learning",
           "Adversarial Robustness", "OOD Detection", "Anomaly Detection",
           "Neural Architecture Search", "Model Compression", "Pruning",
           "Quantization", "Knowledge Distillation",
           "Agent", "Embodied AI", "Robotics", "Autonomous Driving",
           "Medical Imaging", "Time Series Forecasting",
           "Recommender System", "Graph Learning", "Knowledge Graph",
           "Benchmark / Evaluation", "Fairness", "Privacy", "Interpretability"]
        STRICT RULES (avoid hub bloat):
          - Pick MAXIMUM 2 tasks. One is preferred.
          - Use "Benchmark / Evaluation" ONLY if the paper's PRIMARY contribution
            is introducing a new benchmark/dataset. Papers that merely evaluate
            on existing benchmarks should use the underlying task instead.
          - Use "Agent" ONLY if agent behavior is the core research subject.
            Papers that use an LLM in an agentic loop but contribute methods to
            another task (e.g., Reasoning, Code Generation) should use that task.
          - Use "Reasoning" ONLY for general logical/multi-step reasoning. Use
            "Math Reasoning" / "Visual Reasoning" / "Code Generation" when the
            paper targets one of those specifically.
          - Prefer the most SPECIFIC matching task over a generic one.
        If absolutely none fits, write a SHORT canonical name (≤4 words),
        NEVER the paper's method/system name (e.g., NEVER "COS-PLAY" or
        "WebGen-R1" — those are methods, not tasks),
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
            "system": """You are a deep paper analysis report writer. Produce a single linear narrative — a reader should be able to start at section 1 and finish section 7 with a complete understanding of the paper's background, core innovation, framework, formulas, experiments, and position in the field.

CRITICAL RULES
- Base every fact on the provided analysis artifacts (PaperEssence, MethodDelta, DeepAnalysis, ReferenceRoleMap, GraphCandidates) and the figures_available list. Do NOT invent.
- Write in Chinese with English terms (model/dataset/metric names) where natural.
- Use EXACT numbers, EXACT figure labels (e.g. "Figure 3", "Table 2"), EXACT method names. Never write "显著提升" without a number.
- If a piece of evidence is missing, just OMIT that sentence/bullet/cell entirely. NEVER write "（数据待补充）", "（待补充）", "TBD", "n/a" or similar placeholders — they pollute the rendered notes.
- Total length 4000-7000 Chinese characters across all 7 sections.

OUTPUT JSON SCHEMA

{
  "title_zh": str — short Chinese descriptor (≤24 chars), e.g. "运动生成的Flow Matching大模型范式",
  "title_en": str — original paper title,
  "sections": [ {"section_type": str, "title": str, "body_md": str} ],   // exactly 7
  "figure_placements": [
    {"marker": str, "preferred_labels": [str], "semantic_role": str, "section_hint": str}
  ]
}

SECTION SPEC (7 sections, in order)

1. section_type: "metadata_overview"   title: "概览"
   body_md (300-500 chars): A markdown table with rows:
     | 中文题名 | {title_zh} |
     | 英文题名 | {title_en} |
     | 会议/期刊 | {venue} ({acceptance_type if known}) |
     | 链接 | [arXiv]({arxiv_url}) · [Code]({code_url} ⭐{stars if known}) · [Project]({project_url}) |
     | 主要任务 | {tasks} |
     | 主要 baseline | {baselines} |
   Then ONE-LINE TL;DR formatted as a `> [!abstract]` callout: "因为「{problem}」，作者在「{baseline}」基础上改了「{change}」，在「{benchmark}」上取得「{result}」"
   Then 2-3 bullets of 关键性能 (specific numbers).

2. section_type: "background_motivation"   title: "背景与动机"
   body_md (800-1200 chars): Open with the PROBLEM in plain language + a concrete example. Then describe HOW 2-3 named existing methods handle it (1-2 sentences each). Then explain WHY they fall short — the SPECIFIC limitation that motivates this paper. End with 1-sentence preview of what this paper does. If a motivation/teaser figure exists in figures_available, insert {{FIG:motivation}}.

3. section_type: "core_innovation"   title: "核心创新"
   body_md (400-700 chars): The ONE key insight in essence. Format: "核心洞察：X，因为 Y，从而使 Z 成为可能。" Then a small "与 baseline 的差异" table (3 cols: 维度 | Baseline | 本文). Do NOT insert figure here.

4. section_type: "framework_overview"   title: "整体框架"
   body_md (700-1100 chars): MUST start with {{FIG:pipeline}} or {{FIG:architecture}} marker (the overall framework diagram). Then describe data flow: input → module A → module B → ... → output. List each major module in 1 sentence (input/output/role). End with an ASCII or mermaid flow if helpful. Reader should know ALL components after this section.

5. section_type: "module_formulas"   title: "核心模块与公式推导"
   body_md (1200-2200 chars): Pick the 2-3 most important modules. For EACH module use this template:
     ### 模块 N: {名称}（对应框架图 {位置}）
     **直觉**: 一句话为什么这样设计。
     **Baseline 公式** ({baseline_name}): $$L_{{base}} = ...$$
     符号: $\\theta$ = ..., ...（只解释关键符号）
     **变化点**: 为什么 baseline 不够 → 改了什么假设/项/权重。
     **本文公式（推导）**:
     $$\\text{{Step 1}}: ... \\quad \\text{{加入了 X 项以解决 Y}}$$
     $$\\text{{Step 2}}: ... \\quad \\text{{重归一化以保证 Z}}$$
     $$\\text{{最终}}: L_{{final}} = ...$$
     **对应消融**: Table N 显示移除该项 ΔX%。
   Progress from simplest/most fundamental module to most complex. If the paper doesn't actually derive from a baseline, state the formula then explain each symbol — but always show the baseline form first when one exists.

6. section_type: "experiment_analysis"   title: "实验与分析"
   body_md (800-1400 chars):
   - **CRITICAL: Do NOT reconstruct the paper's results table as a markdown `|...|` table** — Markdown table rendering breaks easily on complex headers, multi-row cells, or long numbers. Instead, refer to the original table image with a `{{TBL:result}}` (or `{{TBL:ablation}}` / `{{TBL:comparison}}`) marker, then summarize the 2-3 most consequential numbers in prose.
   - Open with one or two **prose paragraphs** stating: which benchmark(s) the paper evaluates on, the headline number (e.g. "本文方法在 MS-COCO 上 mAP 达到 54.3，相比 baseline DINO 提升 +2.1"), and which gap this number actually closes. Insert `{{TBL:result}}` BEFORE these paragraphs so the reader sees the source table first.
   - For result figures (qualitative grids, scatter plots, etc.) use `{{FIG:result}}` separately.
   - Cover ablation: which removed component costs the most. Use `{{TBL:ablation}}` if there is a dedicated ablation table; otherwise `{{FIG:ablation}}` for an ablation plot. Quote the SPECIFIC delta (e.g. "去掉 X 后 FID 0.045 → 0.228, +0.183").
   - End with a fairness check: are the named baselines actually the strongest available for this benchmark? compute/data budget? failure modes the authors disclose?

7. section_type: "lineage_positioning"   title: "方法谱系与知识库定位"
   body_md (400-700 chars): Method family + parent method (named). Which slots changed (architecture / objective / training_recipe / data_curation / inference). Direct baselines (named, with how this paper differs in 1 line each). 2-3 follow-up directions. Tag this paper with: modality / paradigm / scenario / mechanism / constraint facets.

FIGURE / TABLE PLACEMENT RULES (CRITICAL)
- Two marker families:
    - `{{FIG:xxx}}` — for FIGURE images (diagrams, plots, photos). Match against figures_available entries with `type=figure`.
    - `{{TBL:xxx}}` — for TABLE images (numeric result tables, ablation tables). Match against figures_available entries with `type=table`.
- Use markers ONLY for items that ACTUALLY exist in figures_available (you'll be given a list with label + type + semantic_role + caption).
- For each marker, output a figure_placements entry with:
    "marker": the exact marker string used in body_md (incl. `{{TBL:...}}` form)
    "preferred_labels": the ACTUAL labels from figures_available (e.g. ["Table 1", "Figure 3"]) — match by best fit
    "semantic_role": one of motivation/pipeline/architecture/result/ablation/comparison/qualitative/example
    "section_hint": which section_type the marker is in
- Sections that MUST contain a marker if a matching item exists:
    - framework_overview → `{{FIG:pipeline}}` or `{{FIG:architecture}}`
    - experiment_analysis → `{{TBL:result}}` (table) AND/OR `{{FIG:result}}` (figure) as appropriate
- Do NOT cluster all markers in one place — distribute across the narrative where they actually help understanding.
- NEVER write markdown `|...|` tables for paper results — always use `{{TBL:xxx}}` to embed the table image instead.

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
        """Parse LLM response text as JSON, handling common formatting issues.

        Handles: markdown fences, truncated JSON, leading text before JSON.
        """
        cleaned = text.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]
            else:
                cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()

        # If response starts with non-JSON text, find the first {
        if cleaned and cleaned[0] != "{":
            brace_pos = cleaned.find("{")
            if brace_pos != -1:
                cleaned = cleaned[brace_pos:]

        # Try direct parse first
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            # Layer 1: minimal escape fix — only doubles `\X` where X is NOT a
            # valid JSON escape character. Preserves intentional escapes like
            # `\n` exactly. Recovers most lightly-broken outputs.
            try:
                escaped = self._fix_invalid_escapes(cleaned)
                result = json.loads(escaped)
            except json.JSONDecodeError:
                # Layer 2: aggressive — double EVERY lone `\` (preserves `\\`).
                # Cost: intentional `\n` in source becomes literal `\\n`. Win:
                # recovers Kimi LaTeX-laden outputs that emit single `\theta`,
                # `\frac`, etc. Worth the tradeoff for Chinese reports where
                # newlines are real `\n` chars (already passed) not escapes.
                try:
                    import re as _re
                    aggressive = _re.sub(r"(?<!\\)\\(?!\\)", r"\\\\", cleaned)
                    result = json.loads(aggressive)
                    logger.warning(
                        "Agent %s: recovered via aggressive backslash doubling",
                        agent_name,
                    )
                except json.JSONDecodeError:
                    # Layer 3: truncation repair (close open braces/brackets)
                    result = self._repair_truncated_json(cleaned, agent_name)

        if not isinstance(result, dict):
            raise RuntimeError(
                f"Agent {agent_name} returned {type(result).__name__}, expected dict"
            )

        return result

    @staticmethod
    def _fix_invalid_escapes(text: str) -> str:
        r"""Inside JSON string literals, replace ``\X`` with ``\\X`` whenever
        X is NOT a valid JSON escape character (``"``, ``\\``, ``/``, ``b``,
        ``f``, ``n``, ``r``, ``t``, ``u``).

        This salvages LaTeX-laden Kimi outputs like ``"\text{...}"`` →
        ``"\\text{...}"``. Without this, `json.loads` raises
        ``Invalid \\escape`` and the agent run fails.
        """
        valid_escapes = set('"\\/bfnrtu')
        out = []
        in_string = False
        escape_pending = False
        for ch in text:
            if not in_string:
                if ch == '"':
                    in_string = True
                out.append(ch)
                continue
            # in_string == True
            if escape_pending:
                # The previous char was `\`. If `ch` is a valid escape, keep
                # both as-is; otherwise we already prepended an extra `\`
                # below — so just emit `ch`.
                out.append(ch)
                escape_pending = False
                continue
            if ch == '\\':
                escape_pending = True
                out.append('\\')
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                continue
            out.append(ch)
        # Second pass: turn `\X` (where X not in valid_escapes) into `\\X`.
        # Doing this with a regex is simpler and covers all positions.
        import re as _re
        return _re.sub(
            r'\\([^"\\/bfnrtu])',
            lambda m: '\\\\' + m.group(1),
            ''.join(out),
        )

    @staticmethod
    def _repair_truncated_json(text: str, agent_name: str) -> dict:
        """Attempt to repair truncated JSON by closing open braces/brackets."""
        # Count unclosed delimiters
        open_braces = 0
        open_brackets = 0
        in_string = False
        escape_next = False

        for ch in text:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                open_braces += 1
            elif ch == "}":
                open_braces -= 1
            elif ch == "[":
                open_brackets += 1
            elif ch == "]":
                open_brackets -= 1

        # If we're inside a string, close it
        repaired = text.rstrip()
        if in_string:
            repaired += '"'

        # Close brackets then braces
        repaired += "]" * max(0, open_brackets)
        repaired += "}" * max(0, open_braces)

        try:
            result = json.loads(repaired)
            logger.warning(
                "Agent %s: repaired truncated JSON (added %d } and %d ])",
                agent_name, max(0, open_braces), max(0, open_brackets),
            )
            return result
        except json.JSONDecodeError as e:
            logger.error(
                "Agent %s returned unrepairable JSON: %s\nPreview: %.500s",
                agent_name, e, text,
            )
            raise RuntimeError(
                f"Agent {agent_name} returned invalid JSON: {e}"
            ) from e

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
