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
            "system": "You perform deep method delta analysis with full slot decomposition. TODO: implement full prompt.",
            "output_schema": "MethodDeltaFull",
            "item_type": "method_delta_full",
            "phase": "deep",
        },
        "experiment": {
            "system": "You extract structured experiment results, benchmark comparisons, and ablation studies. TODO: implement full prompt.",
            "output_schema": "ExperimentMatrix",
            "item_type": "experiment_matrix",
            "phase": "deep",
        },
        "formula_figure": {
            "system": "You extract and analyze formulas, figures, and their relationships to method components. TODO: implement full prompt.",
            "output_schema": "FormulaFigureAnalysis",
            "item_type": "formula_figure_analysis",
            "phase": "deep",
        },

        # ── Graph Phase Agents (stubs) ───────────────────────────────────

        "graph_candidate": {
            "system": "You propose knowledge graph nodes and edges from paper analysis results. TODO: implement full prompt.",
            "output_schema": "GraphCandidates",
            "item_type": "graph_candidates",
            "phase": "graph",
        },
        "node_profile": {
            "system": "You generate comprehensive profiles for knowledge graph nodes. TODO: implement full prompt.",
            "output_schema": "NodeProfile",
            "item_type": "node_profile",
            "phase": "profile",
        },
        "edge_profile": {
            "system": "You generate profiles for knowledge graph edges with evidence. TODO: implement full prompt.",
            "output_schema": "EdgeProfile",
            "item_type": "edge_profile",
            "phase": "profile",
        },

        # ── Report Phase Agents (stubs) ──────────────────────────────────

        "paper_report": {
            "system": "You generate comprehensive structured reports from all verified analysis results. TODO: implement full prompt.",
            "output_schema": "PaperReport",
            "item_type": "paper_report",
            "phase": "report",
        },
        "quality_audit": {
            "system": "You audit the quality and consistency of extracted analysis artifacts. TODO: implement full prompt.",
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
            "claude-sonnet-4-20250514": (3.0 / 1_000_000, 15.0 / 1_000_000),
            "claude-haiku-4-5-20251001": (0.80 / 1_000_000, 4.0 / 1_000_000),
            "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
            "gpt-4o": (2.50 / 1_000_000, 10.0 / 1_000_000),
            "mock": (0, 0),
        }
        rate = costs.get(resp.model, (1.0 / 1_000_000, 3.0 / 1_000_000))
        return resp.input_tokens * rate[0] + resp.output_tokens * rate[1]
