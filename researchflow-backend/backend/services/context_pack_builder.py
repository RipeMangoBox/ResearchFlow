"""Context Pack Builder — assembles tailored context for each agent.

4 context layers: Global → Domain → Paper → Run
Each agent gets a different subset with a token budget.
"""

import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.agent import AgentBlackboardItem
from backend.models.analysis import PaperAnalysis
from backend.models.domain import DomainSpec
from backend.models.evidence import EvidenceUnit
from backend.models.method import MethodNode
from backend.models.taxonomy import TaxonomyNode

logger = logging.getLogger(__name__)


class ContextPackBuilder:
    """Builds context packs for agents from 4 layers: global, domain, paper, run."""

    # ── Global Schema Constants ──────────────────────────────────────────

    GLOBAL_NODE_TYPES = (
        "Node types: T__Task, M__Method, C__Mechanism, P__Paper, "
        "D__Dataset, L__Lineage, Lab__Team"
    )
    GLOBAL_RELATION_TYPES = (
        "Relation types: proposes_method, evaluates_on, uses_dataset, "
        "compares_against, modifies_slot, extends_method, cites_as_baseline, "
        "belongs_to_task, part_of_lineage, produced_by_lab"
    )
    GLOBAL_REFERENCE_ROLE_DEFS = """Reference roles:
- direct_baseline: paper method section explicitly builds on this
- method_source: core algorithmic idea comes from this
- formula_source: key equations derived from this
- dataset_source: introduces dataset used in experiments
- benchmark_source: introduces benchmark used for evaluation
- comparison_baseline: appears in main experiment table as comparison
- same_task_prior_work: prior work on same task
- survey_or_taxonomy: survey/overview paper
- background_citation: general background reference
- implementation_reference: used for implementation details
- unimportant_related_work: not directly relevant"""

    GLOBAL_SLOT_TYPES = (
        "Slot types: architecture, objective, data_pipeline, "
        "inference_strategy, training_recipe, reward_design, "
        "credit_assignment, exploration_strategy"
    )
    GLOBAL_EDGE_RULES = """Edge creation rules:
- proposes_method: paper introduces a new method node
- evaluates_on: paper runs experiments on a dataset/benchmark
- uses_dataset: paper uses dataset for training/fine-tuning
- compares_against: paper compares with a baseline in experiment tables
- modifies_slot: paper changes a specific slot in an existing method
- extends_method: paper extends an existing method without replacing it
- cites_as_baseline: paper cites another as its baseline method
- belongs_to_task: paper addresses a specific task
- part_of_lineage: method belongs to a lineage chain
- produced_by_lab: paper/method produced by a research team"""

    GLOBAL_EXPERIMENT_SCHEMA = """Experiment schema:
- benchmark_name: str — name of the benchmark
- dataset_name: str — dataset used
- metric_name: str — evaluation metric
- proposed_value: float — result of proposed method
- baseline_values: dict[str, float] — baseline name → value
- delta_abs: float — absolute improvement
- delta_pct: float — percentage improvement
- is_sota: bool — claims state-of-the-art
- ablation_rows: list[dict] — ablation study rows"""

    GLOBAL_SCORE_SIGNAL_DEFS = """Score signal definitions:
- is_direct_baseline: bool — this paper is a direct baseline for the anchor
- in_experiment_table: bool — appears in experiment comparison tables
- same_primary_task: bool — addresses the same primary task
- has_changed_slots: bool — modifies one or more method slots
- has_ablation: bool — includes ablation study
- has_code: bool — code is publicly available
- has_new_dataset: bool — introduces a new dataset
- citation_density: float — how often cited in the paper body (0-1)
- method_novelty: float — degree of method novelty (0-1)
- evidence_quality: float — strength of experimental evidence (0-1)"""

    GLOBAL_PROFILE_SCHEMA = """Node profile schema:
- name: str — canonical name
- type: str — node type (Task/Method/Mechanism/...)
- aliases: list[str] — alternative names
- description: str — concise description
- key_papers: list[str] — representative paper titles
- connected_tasks: list[str] — related tasks
- connected_methods: list[str] — related methods
- evolution_stage: str — seed/emerging/established
- summary_stats: dict — paper count, citation count, etc."""

    GLOBAL_EDGE_PROFILE_SCHEMA = """Edge profile schema:
- source_node: str — source node name
- target_node: str — target node name
- relation_type: str — edge type
- evidence_count: int — number of supporting evidence units
- confidence: float — aggregate confidence
- representative_papers: list[str] — papers supporting this edge
- description: str — natural language description of the relation"""

    GLOBAL_REPORT_SECTION_SCHEMA = """Report section schema:
- paper_essence: PaperEssence — core summary
- method_analysis: MethodDelta — slot-level changes
- experiment_matrix: ExperimentMatrix — benchmark results
- reference_map: ReferenceRoleMap — classified references
- contribution_assessment: ContributionAssessment — novelty/significance
- limitations_and_future: str — limitations and future work
- quality_audit: QualityAudit — evidence quality assessment"""

    _GLOBAL_ITEMS = {
        "node_types": GLOBAL_NODE_TYPES,
        "relation_types": GLOBAL_RELATION_TYPES,
        "reference_role_definitions": GLOBAL_REFERENCE_ROLE_DEFS,
        "slot_types": GLOBAL_SLOT_TYPES,
        "edge_rules": GLOBAL_EDGE_RULES,
        "experiment_schema": GLOBAL_EXPERIMENT_SCHEMA,
        "score_signal_definitions": GLOBAL_SCORE_SIGNAL_DEFS,
        "profile_schema": GLOBAL_PROFILE_SCHEMA,
        "edge_profile_schema": GLOBAL_EDGE_PROFILE_SCHEMA,
        "report_section_schema": GLOBAL_REPORT_SECTION_SCHEMA,
    }

    # ── Pack Configurations ──────────────────────────────────────────────

    PACK_CONFIGS = {
        # ── Shallow Phase (merged) ──
        "shallow_extractor": {
            "global": ["node_types", "relation_types", "slot_types"],
            "domain": ["scope", "existing_tasks_summary", "existing_methods_summary", "baselines"],
            "paper": [
                "abstract", "introduction_excerpt", "method_excerpt",
                "experiment_excerpt", "figure_table_captions",
                "algorithm_blocks", "formula_contexts",
            ],
            "run": [],
            "token_budget": 18_000,
        },
        "reference_role": {
            "global": ["reference_role_definitions"],
            "domain": ["anchor_paper_titles"],
            "paper": ["reference_list", "citation_contexts"],
            "run": [],
            "token_budget": 30_000,
        },

        # ── Deep Phase (merged) ──
        "deep_analyzer": {
            "global": ["slot_types", "relation_types", "experiment_schema"],
            "domain": ["method_profiles", "baseline_profiles", "known_benchmarks"],
            "paper": [
                "method_section_full", "algorithm_blocks", "all_formula_contexts",
                "result_tables", "experiment_section", "ablation_section",
            ],
            "run": ["shallow_extract", "reference_role_map"],
            "token_budget": 40_000,
        },
        "graph_candidate": {
            "global": ["node_types", "relation_types", "edge_rules"],
            "domain": ["graph_summary", "task_hierarchy", "method_hierarchy"],
            "paper": [],
            "run": [
                "shallow_extract", "deep_analysis",
                "reference_role_map",
            ],
            "token_budget": 20_000,
        },

        # ── Profile Phase (merged) ──
        "kb_profiler": {
            "global": ["profile_schema", "edge_profile_schema"],
            "domain": [],
            "paper": [],
            "run": ["graph_candidates"],
            "token_budget": 20_000,
        },

        # ── Report Phase ──
        "paper_report": {
            "global": ["report_section_schema"],
            "domain": [],
            "paper": ["selected_evidence"],
            "run": ["ALL_VERIFIED"],
            "token_budget": 80_000,
        },

        # ── Legacy aliases (for backward compat during transition) ──
        "shallow_paper": {
            "global": ["node_types", "relation_types"],
            "domain": ["scope", "existing_tasks_summary", "existing_methods_summary"],
            "paper": ["abstract", "introduction_excerpt", "method_excerpt", "experiment_excerpt", "figure_table_captions"],
            "run": [],
            "token_budget": 12_000,
        },
    }

    def __init__(self, session: AsyncSession):
        self.session = session

    async def build(
        self,
        pack_name: str,
        *,
        paper_id: UUID | None = None,
        candidate_id: UUID | None = None,
        domain_id: UUID | None = None,
        run_items: dict | None = None,
    ) -> dict:
        """Build a context pack for the given agent.

        Returns dict with keys: system_prompt, user_content, token_budget, metadata.
        """
        if pack_name not in self.PACK_CONFIGS:
            raise ValueError(f"Unknown pack: {pack_name!r}. "
                             f"Available: {list(self.PACK_CONFIGS.keys())}")

        config = self.PACK_CONFIGS[pack_name]
        budget = config["token_budget"]

        # Assemble layers
        parts: list[str] = []

        # Layer 1: Global
        if config["global"]:
            global_text = await self._load_global_context(config["global"])
            if global_text:
                parts.append(f"=== GLOBAL SCHEMA ===\n{global_text}")

        # Layer 2: Domain
        if config["domain"] and domain_id:
            domain_text = await self._load_domain_context(domain_id, config["domain"])
            if domain_text:
                parts.append(f"=== DOMAIN CONTEXT ===\n{domain_text}")

        # Layer 3: Paper
        if config["paper"] and paper_id:
            paper_text = await self._load_paper_context(paper_id, config["paper"])
            if paper_text:
                parts.append(f"=== PAPER CONTEXT ===\n{paper_text}")

        # Layer 4: Run (blackboard items)
        if config["run"]:
            run_text = await self._load_run_context(
                paper_id, candidate_id, config["run"], run_items,
            )
            if run_text:
                parts.append(f"=== RUN CONTEXT ===\n{run_text}")

        user_content = "\n\n".join(parts)
        user_content = self._truncate_to_budget(user_content, budget)

        return {
            "system_prompt": f"You are the {pack_name} agent.",
            "user_content": user_content,
            "token_budget": budget,
            "metadata": {
                "pack_name": pack_name,
                "paper_id": str(paper_id) if paper_id else None,
                "candidate_id": str(candidate_id) if candidate_id else None,
                "domain_id": str(domain_id) if domain_id else None,
                "layers_included": [
                    layer for layer in ("global", "domain", "paper", "run")
                    if config[layer]
                ],
            },
        }

    # ── Layer Loaders ────────────────────────────────────────────────────

    async def _load_global_context(self, items: list[str]) -> str:
        """Return static schema definitions for the requested global items."""
        sections: list[str] = []
        for item in items:
            text = self._GLOBAL_ITEMS.get(item)
            if text:
                sections.append(text)
            else:
                logger.warning("Unknown global context item: %s", item)
        return "\n\n".join(sections)

    async def _load_domain_context(self, domain_id: UUID, items: list[str]) -> str:
        """Query DB for domain-level context."""
        sections: list[str] = []

        # Load DomainSpec for scope-related items
        domain = (
            await self.session.execute(
                select(DomainSpec).where(DomainSpec.id == domain_id)
            )
        ).scalar_one_or_none()

        if not domain:
            logger.warning("Domain %s not found", domain_id)
            return ""

        for item in items:
            if item == "scope":
                scope_parts = []
                if domain.scope_tasks:
                    scope_parts.append(f"Tasks: {', '.join(domain.scope_tasks)}")
                if domain.scope_modalities:
                    scope_parts.append(f"Modalities: {', '.join(domain.scope_modalities)}")
                if domain.scope_paradigms:
                    scope_parts.append(f"Paradigms: {', '.join(domain.scope_paradigms)}")
                if domain.negative_scope:
                    scope_parts.append(f"Excluded: {', '.join(domain.negative_scope)}")
                if scope_parts:
                    sections.append(f"[Domain Scope]\n" + "\n".join(scope_parts))

            elif item == "anchor_paper_titles":
                if domain.seed_paper_ids:
                    # Return seed paper IDs as anchors (titles resolved at query time)
                    sections.append(
                        f"[Anchor Papers]\n"
                        f"Seed paper IDs: {', '.join(str(pid) for pid in domain.seed_paper_ids)}"
                    )

            elif item in ("existing_tasks_summary", "task_hierarchy"):
                rows = (
                    await self.session.execute(
                        select(TaxonomyNode.name, TaxonomyNode.description)
                        .where(TaxonomyNode.dimension == "task")
                        .order_by(TaxonomyNode.sort_order)
                        .limit(50)
                    )
                ).all()
                if rows:
                    lines = [f"- {r.name}: {r.description or '(no desc)'}" for r in rows]
                    sections.append(f"[Existing Tasks]\n" + "\n".join(lines))

            elif item in ("existing_methods_summary", "existing_methods_with_slots",
                          "method_profiles", "method_hierarchy"):
                rows = (
                    await self.session.execute(
                        select(MethodNode.name, MethodNode.type, MethodNode.maturity,
                               MethodNode.description)
                        .order_by(MethodNode.downstream_count.desc())
                        .limit(50)
                    )
                ).all()
                if rows:
                    lines = []
                    for r in rows:
                        line = f"- {r.name} ({r.type}, {r.maturity})"
                        if r.description:
                            line += f": {r.description[:120]}"
                        lines.append(line)
                    sections.append(f"[Existing Methods]\n" + "\n".join(lines))

                # For items requesting slots, load from paradigm_templates.slots JSONB
                if item in ("existing_methods_with_slots", "method_profiles"):
                    from backend.models.analysis import ParadigmTemplate
                    pt_rows = (
                        await self.session.execute(
                            select(ParadigmTemplate.name, ParadigmTemplate.slots).limit(10)
                        )
                    ).all()
                    slot_lines = []
                    for pt in pt_rows:
                        if pt.slots and isinstance(pt.slots, dict):
                            for sname, sinfo in pt.slots.items():
                                desc = sinfo.get("description", "") if isinstance(sinfo, dict) else ""
                                slot_lines.append(f"  - {pt.name}/{sname}: {desc}")
                    if slot_lines:
                        sections.append(f"[Method Slots]\n" + "\n".join(slot_lines))

            elif item in ("baselines", "baseline_profiles", "known_baselines"):
                rows = (
                    await self.session.execute(
                        select(MethodNode.name, MethodNode.description)
                        .where(MethodNode.maturity == "established_baseline")
                        .limit(30)
                    )
                ).all()
                if rows:
                    lines = [f"- {r.name}: {r.description or ''}" for r in rows]
                    sections.append(f"[Known Baselines]\n" + "\n".join(lines))

            elif item == "known_benchmarks":
                rows = (
                    await self.session.execute(
                        select(TaxonomyNode.name, TaxonomyNode.description)
                        .where(TaxonomyNode.dimension.in_(["benchmark", "dataset"]))
                        .limit(30)
                    )
                ).all()
                if rows:
                    lines = [f"- {r.name}: {r.description or ''}" for r in rows]
                    sections.append(f"[Known Benchmarks]\n" + "\n".join(lines))

            elif item == "graph_summary":
                # Count nodes by dimension
                rows = (
                    await self.session.execute(
                        select(TaxonomyNode.dimension,
                               func.count(TaxonomyNode.id).label("cnt"))
                        .group_by(TaxonomyNode.dimension)
                    )
                ).all()
                if rows:
                    lines = [f"- {r.dimension}: {r.cnt} nodes" for r in rows]
                    sections.append(f"[Graph Summary]\n" + "\n".join(lines))

            else:
                logger.debug("Unhandled domain context item: %s", item)

        return "\n\n".join(sections)

    async def _load_paper_context(self, paper_id: UUID, items: list[str]) -> str:
        """Query DB for paper-level context (sections, evidence)."""
        sections: list[str] = []

        # Load the current analysis
        analysis = (
            await self.session.execute(
                select(PaperAnalysis)
                .where(
                    PaperAnalysis.paper_id == paper_id,
                    PaperAnalysis.is_current.is_(True),
                )
                .order_by(PaperAnalysis.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        extracted = analysis.extracted_sections if analysis else {}
        if not isinstance(extracted, dict):
            extracted = {}

        for item in items:
            if item == "abstract":
                text = extracted.get("abstract", "")
                if text:
                    sections.append(f"[Abstract]\n{text}")

            elif item in ("introduction_excerpt",):
                text = extracted.get("introduction", "")
                if text:
                    # Excerpt: first 2000 chars
                    sections.append(f"[Introduction Excerpt]\n{text[:2000]}")

            elif item in ("method_excerpt", "method_section"):
                text = extracted.get("method", "") or extracted.get("methodology", "")
                if text:
                    sections.append(f"[Method Section]\n{text[:4000]}")

            elif item == "method_section_full":
                text = extracted.get("method", "") or extracted.get("methodology", "")
                if text:
                    sections.append(f"[Method Section (Full)]\n{text}")

            elif item in ("experiment_excerpt", "experiment_section"):
                text = extracted.get("experiments", "") or extracted.get("results", "")
                if text:
                    sections.append(f"[Experiment Section]\n{text[:4000]}")

            elif item == "ablation_section":
                text = extracted.get("ablation", "") or extracted.get("ablation_study", "")
                if text:
                    sections.append(f"[Ablation Section]\n{text}")

            elif item == "figure_table_captions":
                captions = analysis.figure_captions if analysis else None
                if captions:
                    import json
                    sections.append(
                        f"[Figure/Table Captions]\n{json.dumps(captions, ensure_ascii=False, indent=1)}"
                    )

            elif item == "result_tables":
                tables = analysis.extracted_tables if analysis else None
                if tables:
                    import json
                    sections.append(
                        f"[Result Tables]\n{json.dumps(tables, ensure_ascii=False, indent=1)}"
                    )

            elif item in ("algorithm_blocks",):
                text = extracted.get("algorithm", "") or extracted.get("algorithms", "")
                if text:
                    sections.append(f"[Algorithm Blocks]\n{text}")

            elif item in ("formula_contexts", "all_formula_contexts"):
                formulas = analysis.extracted_formulas if analysis else None
                if formulas:
                    sections.append(f"[Formulas]\n" + "\n".join(formulas))

            elif item == "reference_list":
                text = extracted.get("references", "")
                if text:
                    sections.append(f"[Reference List]\n{text}")

            elif item == "citation_contexts":
                text = extracted.get("citation_contexts", "")
                if text:
                    sections.append(f"[Citation Contexts]\n{text}")

            elif item == "selected_evidence":
                evidence_rows = (
                    await self.session.execute(
                        select(EvidenceUnit.atom_type, EvidenceUnit.claim,
                               EvidenceUnit.confidence, EvidenceUnit.source_section)
                        .where(EvidenceUnit.paper_id == paper_id)
                        .limit(50)
                    )
                ).all()
                if evidence_rows:
                    lines = [
                        f"- [{e.atom_type}] {e.claim} "
                        f"(conf={e.confidence or 'N/A'}, sec={e.source_section or '?'})"
                        for e in evidence_rows
                    ]
                    sections.append(f"[Selected Evidence]\n" + "\n".join(lines))

            else:
                logger.debug("Unhandled paper context item: %s", item)

        return "\n\n".join(sections)

    async def _load_run_context(
        self,
        paper_id: UUID | None,
        candidate_id: UUID | None,
        items: list[str],
        run_items: dict | None,
    ) -> str:
        """Load blackboard items or use provided run_items dict."""
        import json

        sections: list[str] = []

        # If run_items dict is provided directly, use it
        if run_items:
            for item in items:
                if item == "ALL_VERIFIED":
                    # Include all provided items
                    for key, value in run_items.items():
                        val_str = json.dumps(value, ensure_ascii=False, indent=1) if isinstance(value, (dict, list)) else str(value)
                        sections.append(f"[{key}]\n{val_str}")
                    break
                value = run_items.get(item)
                if value is not None:
                    val_str = json.dumps(value, ensure_ascii=False, indent=1) if isinstance(value, (dict, list)) else str(value)
                    sections.append(f"[{item}]\n{val_str}")
            return "\n\n".join(sections)

        # Otherwise query the blackboard
        if not paper_id and not candidate_id:
            return ""

        # Build filter conditions
        conditions = []
        if paper_id:
            conditions.append(AgentBlackboardItem.paper_id == paper_id)
        if candidate_id:
            conditions.append(AgentBlackboardItem.candidate_id == candidate_id)

        for item in items:
            if item == "ALL_VERIFIED":
                # Load all verified blackboard items
                rows = (
                    await self.session.execute(
                        select(AgentBlackboardItem.item_type,
                               AgentBlackboardItem.value_json)
                        .where(
                            *conditions,
                            AgentBlackboardItem.is_verified.is_(True),
                        )
                    )
                ).all()
                for r in rows:
                    val_str = json.dumps(r.value_json, ensure_ascii=False, indent=1)
                    sections.append(f"[{r.item_type}]\n{val_str}")
                break

            rows = (
                await self.session.execute(
                    select(AgentBlackboardItem.value_json)
                    .where(
                        *conditions,
                        AgentBlackboardItem.item_type == item,
                    )
                    .order_by(AgentBlackboardItem.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if rows is not None:
                val_str = json.dumps(rows, ensure_ascii=False, indent=1)
                sections.append(f"[{item}]\n{val_str}")

        return "\n\n".join(sections)

    def _truncate_to_budget(self, text: str, budget: int) -> str:
        """Truncate text to fit within token budget (4 chars ~ 1 token)."""
        char_limit = budget * 4
        if len(text) <= char_limit:
            return text
        logger.info(
            "Truncating context from %d to %d chars (budget=%d tokens)",
            len(text), char_limit, budget,
        )
        return text[:char_limit] + "\n\n... [truncated to fit token budget]"
