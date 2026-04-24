"""V6 Ingest Workflow — orchestrates candidate → shallow → deep → profile → report.

This is the main entry point for the V6 pipeline. Papers flow through:
  Import → DiscoveryScore → [Shallow Ingest] → DeepIngestScore → [Deep Ingest] → [Profile] → [Report]

Each phase is independently retryable. Score gates prevent low-value papers from consuming resources.
"""

import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.services import candidate_service
from backend.services.candidate_service import _normalize_title
from backend.services.context_pack_builder import ContextPackBuilder
from backend.services.agent_runner import AgentRunner
from backend.services.scoring_engine import ScoringEngine

logger = logging.getLogger(__name__)


class IngestWorkflow:
    """V6 multi-phase, score-gated paper ingestion workflow."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.engine = ScoringEngine()
        self.runner = AgentRunner(session)
        self.ctx_builder = ContextPackBuilder(self.session)

    # ── Phase 1: Import & Score ────────────────────────────────────────

    async def import_and_score(
        self,
        source: str | dict,
        *,
        discovery_source: str = "manual_import",
        discovery_reason: str | None = None,
        relation_hint: str | None = None,
        discovered_from_paper_id: UUID | None = None,
        domain_id: UUID | None = None,
    ) -> dict:
        """Create a candidate and compute its DiscoveryScore.

        Routes based on score:
            >= 75 → enqueue shallow_ingest
            60-74 → candidate_pool
            40-59 → metadata_only
            < 40  → archive

        Returns:
            {candidate_id, discovery_score, decision, status}
        """
        # Normalize source into create_candidate kwargs
        if isinstance(source, dict):
            title = source.get("title", "Untitled")
            create_kwargs = {
                "arxiv_id": source.get("arxiv_id"),
                "doi": source.get("doi"),
                "paper_link": source.get("url") or source.get("paper_link"),
                "abstract": source.get("abstract"),
                "authors_json": source.get("authors_json"),
                "venue": source.get("venue"),
                "year": source.get("year"),
                "citation_count": source.get("citation_count"),
                "code_url": source.get("code_url"),
                "metadata_json": source.get("metadata_json"),
            }
        else:
            title = source
            create_kwargs = {}

        candidate = await candidate_service.create_candidate(
            self.session,
            title=title,
            discovery_source=discovery_source,
            discovery_reason=discovery_reason or "v6_import",
            discovered_from_paper_id=discovered_from_paper_id,
            discovered_from_domain_id=domain_id,
            relation_hint=relation_hint,
            **create_kwargs,
        )

        # Load domain for scoring context
        domain = None
        if domain_id:
            from backend.models.domain import DomainSpec
            domain = await self.session.get(DomainSpec, domain_id)

        # Extract signals and compute discovery score
        signals = candidate_service._extract_discovery_signals(candidate, domain)
        score_result = self.engine.compute_discovery_score(signals, domain=domain)

        # Persist score via candidate_service
        try:
            await candidate_service.score_candidate(
                self.session, candidate.id, domain=domain,
            )
        except Exception as e:
            logger.warning(
                "score_candidate failed for %s, using engine result directly: %s",
                candidate.id, e,
            )

        # Route based on decision
        decision = score_result.decision
        if decision == "shallow_ingest":
            candidate.status = "accepted"
        elif decision == "candidate_pool":
            candidate.status = "scoring"
        elif decision == "metadata_only":
            candidate.status = "metadata_resolved"
        else:  # archive
            candidate.status = "rejected"
            candidate.reject_reason = "discovery_score below 40"

        await self.session.flush()

        return {
            "candidate_id": str(candidate.id),
            "discovery_score": score_result.total,
            "decision": decision,
            "status": candidate.status,
        }

    # ── Phase 2: Shallow Ingest ────────────────────────────────────────

    async def shallow_ingest(self, candidate_id: UUID) -> dict:
        """Promote candidate to Paper (L1) and run shallow-phase agents.

        Agents run sequentially:
            1. ShallowPaperAgent → paper_essence
            2. ReferenceRoleAgent → reference_role_map
            3. MethodDeltaAgent-lite → method_delta
            4. ScoreAgent → score_signals

        Then computes DeepIngestScore and routes:
            >= 88 → enqueue deep_ingest
            80-87 → enqueue deep_ingest (review_needed)
            68-79 → visible_graph (L2)
            < 68  → stays L1

        Returns:
            {paper_id, deep_ingest_score, decision, agents_run, references_discovered}
        """
        # Load and promote candidate
        candidate = await candidate_service.get_candidate(self.session, candidate_id)
        if not candidate:
            return {"error": f"Candidate {candidate_id} not found"}

        paper = await candidate_service.promote_candidate(
            self.session, candidate_id, absorption_level=1,
        )
        paper_id = paper.id
        await self.session.flush()

        agents_run = []
        agent_results = {}

        # Run shallow agents sequentially (merged: 2 agents instead of 4)
        shallow_agents = [
            ("shallow_extractor", "shallow_extract"),
            ("reference_role", "reference_role_map"),
        ]

        for agent_name, result_key in shallow_agents:
            try:
                context = await self.ctx_builder.build(
                    agent_name,
                    paper_id=paper_id,
                )
            except Exception as e:
                logger.warning(
                    "Context pack build failed for %s/%s: %s",
                    agent_name, paper_id, e,
                )
                context = {
                    "user_content": (
                        f"Paper: {paper.title}\n"
                        f"Abstract: {paper.abstract or 'N/A'}"
                    ),
                    "token_budget": 4096,
                }

            try:
                result = await self.runner.run_agent(
                    agent_name, context, paper_id=paper_id,
                )
                agents_run.append(agent_name)
                agent_results[result_key] = result
            except Exception as e:
                logger.error(
                    "%s agent failed for %s: %s", agent_name, paper_id, e,
                )
                agent_results[result_key] = {}

        # Build deep-ingest signals deterministically (Phase 2B: no score agent)
        shallow_extract = agent_results.get("shallow_extract", {})
        paper_essence = shallow_extract.get("paper_essence", {})
        method_delta = shallow_extract.get("method_delta", {})
        ref_role_map = agent_results.get("reference_role_map", {})

        # Derive scoring signals from shallow_extract + reference_role_map
        changed_slots = method_delta.get("changed_slots", [])
        evidence_refs = paper_essence.get("evidence_refs", [])
        classifications = ref_role_map.get("classifications", [])
        anchor_baselines = ref_role_map.get("anchor_baselines", [])

        has_code = any(
            e.get("basis") == "code_verified" for e in evidence_refs
        ) or "code" in (paper_essence.get("method_summary") or "").lower()
        has_ablation = len(changed_slots) > 1  # multiple slots implies ablation likely
        is_direct_baseline = len(anchor_baselines) > 0
        same_primary_task = bool(paper_essence.get("target_tasks"))
        has_changed_slots = len(changed_slots) > 0
        baseline_count = len(method_delta.get("baseline_methods", []))
        novel_count = sum(1 for s in changed_slots if s.get("is_novel"))
        method_novelty = min(1.0, novel_count * 0.3 + 0.2 * int(method_delta.get("is_structural_change", False)))
        evidence_quality = min(1.0, sum(
            e.get("confidence", 0.5) for e in evidence_refs
        ) / max(len(evidence_refs), 1))

        deep_signals = {
            "domain_fit": evidence_quality * 60 + method_novelty * 40,
            "relation_type": "same_task_prior" if same_primary_task else "",
            "reusable_knowledge": (
                50 * int(bool(method_delta.get("should_create_method_node")))
                + 30 * int(has_changed_slots)
                + 20 * int(has_ablation)
            ),
            "evidence_quality": evidence_quality * 100,
            "experiment_value": (
                40 * int(has_ablation)
                + 30 * int(baseline_count >= 3)
                + 30 * int(bool(paper_essence.get("limitations")))
            ),
            "has_code": has_code,
            "has_data": False,
            "has_model": False,
            "has_benchmark": False,
            "novelty_freshness": method_novelty * 100,
            "source_type": candidate.discovery_source or "",
            "is_direct_baseline": is_direct_baseline,
            "in_experiment_table": is_direct_baseline,  # approximation
            "same_primary_task": same_primary_task,
            "has_changed_slots": has_changed_slots,
            "has_ablation": has_ablation,
            "is_baseline": is_direct_baseline,
            "has_strong_ablation": has_ablation and len(changed_slots) >= 2,
            "fills_graph_gap": False,
        }

        deep_score_result = self.engine.compute_deep_ingest_score(deep_signals)
        decision = deep_score_result.decision

        # Update paper state based on routing
        from backend.models.enums import PaperState
        if decision in ("auto_full_paper", "full_review_needed"):
            paper.state = PaperState.L2_PARSED
        elif decision == "shallow_card":
            paper.state = PaperState.L2_PARSED
        # else stays at L1

        await self.session.flush()

        # ── Step A: 处理已知论文引用 (对比流程，不重新分析) ──
        ref_map = agent_results.get("reference_role_map", {})
        classifications = ref_map.get("classifications", [])
        known_result = await self._handle_known_references(
            paper_id, classifications,
            domain_id=candidate.discovered_from_domain_id,
        )
        known_titles = known_result.get("known_titles_set", set())

        # ── Step B: 只对真正未知的引用做递归发现 ──
        references_discovered = 0
        for ref_cls in classifications:
            ref_title = ref_cls.get("ref_title")
            if not ref_title:
                continue
            # 跳过已在 KB 中的论文（已在 Step A 中建了关系边）
            if _normalize_title(ref_title) in known_titles:
                continue
            ingest_level = ref_cls.get("recommended_ingest_level", "skip")
            if ingest_level in ("deep", "abstract"):
                try:
                    await self.import_and_score(
                        {"title": ref_title},
                        discovery_source="s2_reference",
                        discovery_reason=f"ref_role={ref_cls.get('role', 'unknown')}",
                        relation_hint=ref_cls.get("role"),
                        discovered_from_paper_id=paper_id,
                    )
                    references_discovered += 1
                except Exception as e:
                    logger.debug("Ref import failed for %s: %s", ref_title, e)

        return {
            "paper_id": str(paper_id),
            "deep_ingest_score": deep_score_result.total,
            "decision": decision,
            "agents_run": agents_run,
            "references_discovered": references_discovered,
            "known_refs_linked": known_result.get("known_refs_linked", 0),
            "edges_created": known_result.get("edges_created", 0),
        }

    # ── Phase 3: Deep Ingest ──────────────────────────────────────────

    async def deep_ingest(
        self,
        paper_id: UUID,
        *,
        review_needed: bool = False,
    ) -> dict:
        """Run deep-phase agents, profile qualifying nodes/edges, generate report.

        Deep agents (sequential, merged):
            1. DeepAnalyzer → deep_analysis (method + experiment + formulas)
            2. GraphCandidateAgent → graph_candidates

        Profile agents (batched):
            3. KBProfiler → node + edge profiles (single call)

        Report agents:
            4. PaperReportAgent → 10-section structured report

        Returns:
            {paper_id, agents_run, nodes_created, edges_created, report_sections}
        """
        from backend.models.paper import Paper
        from backend.models.enums import PaperState

        paper = await self.session.get(Paper, paper_id)
        if not paper:
            return {"error": f"Paper {paper_id} not found"}

        agents_run = []
        agent_results = {}
        nodes_created = 0
        edges_created = 0
        report_sections = []

        # ── Deep agents (sequential, each with own context pack) ──

        deep_agents = [
            ("deep_analyzer", "deep_analysis"),
            ("graph_candidate", "graph_candidates"),
        ]

        for agent_name, result_key in deep_agents:
            try:
                context = await self.ctx_builder.build(
                    agent_name,
                    paper_id=paper_id,
                )
            except Exception as e:
                logger.warning(
                    "Context pack build failed for %s/%s: %s",
                    agent_name, paper_id, e,
                )
                context = {
                    "user_content": (
                        f"Paper: {paper.title}\n"
                        f"Abstract: {paper.abstract or 'N/A'}"
                    ),
                    "token_budget": 8192,
                }

            try:
                result = await self.runner.run_agent(
                    agent_name, context, paper_id=paper_id,
                )
                agents_run.append(agent_name)
                agent_results[result_key] = result
            except Exception as e:
                logger.error(
                    "%s agent failed for %s: %s", agent_name, paper_id, e,
                )
                agent_results[result_key] = {}

        # ── Scoring & Profile agents (batched into single LLM call) ──

        graph_candidates = agent_results.get("graph_candidates", {})

        # Collect qualifying node and edge candidates for batch profiling
        qualifying_nodes = []
        node_candidates = graph_candidates.get("node_candidates", graph_candidates.get("nodes", []))
        for node_cand in node_candidates:
            node_signals = {
                "evidence_count": node_cand.get("evidence_count", 30),
                "connected_paper_quality": node_cand.get("connected_paper_quality", 50),
                "source_diversity": node_cand.get("source_diversity", 40),
                "structural_importance": node_cand.get("structural_importance", 40),
                "name_stability": node_cand.get("name_stability", 60),
                "profile_completeness": node_cand.get("profile_completeness", 30),
            }
            node_score = self.engine.compute_node_promotion_score(node_signals)
            if node_score.total >= 75:
                qualifying_nodes.append(node_cand)

        qualifying_edges = []
        edge_candidates = graph_candidates.get("edge_candidates", graph_candidates.get("edges", []))
        for edge_cand in edge_candidates:
            edge_signals = {
                "evidence_directness": edge_cand.get("evidence_directness", 40),
                "relation_specificity": edge_cand.get("relation_specificity", 50),
                "extractor_agreement": edge_cand.get("extractor_agreement", 50),
                "source_reliability": edge_cand.get("source_reliability", 50),
                "graph_consistency": edge_cand.get("graph_consistency", 40),
                "description_quality": edge_cand.get("description_quality", 40),
            }
            edge_score = self.engine.compute_edge_confidence_score(edge_signals)
            if edge_score.total >= 70:
                qualifying_edges.append(edge_cand)

        # Run batched KBProfiler for all qualifying candidates (single LLM call)
        if qualifying_nodes or qualifying_edges:
            try:
                import json
                profile_context = {
                    "user_content": (
                        f"Paper: {paper.title}\n\n"
                        f"Node candidates to profile ({len(qualifying_nodes)}):\n"
                        f"{json.dumps(qualifying_nodes, ensure_ascii=False, default=str)}\n\n"
                        f"Edge candidates to profile ({len(qualifying_edges)}):\n"
                        f"{json.dumps(qualifying_edges, ensure_ascii=False, default=str)}"
                    ),
                    "token_budget": 8192,
                }
                profile_result = await self.runner.run_agent(
                    "kb_profiler", profile_context, paper_id=paper_id,
                )
                agents_run.append("kb_profiler")
                nodes_created = len(profile_result.get("node_profiles", []))
                edges_created = len(profile_result.get("edge_profiles", []))
            except Exception as e:
                logger.error("kb_profiler agent failed: %s", e)

        # ── Report agent ──

        try:
            report_context = {
                "user_content": (
                    f"Paper: {paper.title}\n"
                    f"Agent results: {agent_results}\n"
                    f"Nodes created: {nodes_created}, Edges created: {edges_created}"
                ),
                "token_budget": 8192,
            }
            report_result = await self.runner.run_agent(
                "paper_report", report_context, paper_id=paper_id,
            )
            agents_run.append("paper_report")
            report_sections = list(report_result.get("sections", {}).keys())
        except Exception as e:
            logger.error("paper_report agent failed for %s: %s", paper_id, e)

        # quality_audit removed (Phase 2E) — replaced by deterministic validation

        # ── Phase 3A: Materialize agent outputs to DeltaCard/GraphAssertion ──
        delta_card_result = {}
        try:
            delta_card_result = await self._materialize_to_graph(
                paper, agent_results,
            )
            logger.info(
                "Materialized graph for %s: dc=%s, idea=%s, evidence=%d, assertions=%d",
                paper_id,
                delta_card_result.get("delta_card_id"),
                delta_card_result.get("delta_card_id"),
                delta_card_result.get("evidence_count", 0),
                delta_card_result.get("assertion_count", 0),
            )
        except Exception as e:
            logger.error("Graph materialization failed for %s: %s", paper_id, e)

        # Update paper absorption level
        paper.absorption_level = 3
        paper.state = PaperState.L4_DEEP
        await self.session.flush()

        return {
            "paper_id": str(paper_id),
            "agents_run": agents_run,
            "nodes_created": nodes_created,
            "edges_created": edges_created,
            "report_sections": report_sections,
            "graph": delta_card_result,
        }

    async def _materialize_to_graph(self, paper, agent_results: dict) -> dict:
        """Convert agent outputs to DeltaCard → IdeaDelta → GraphAssertions.

        Bridges agent blackboard outputs to the delta_card_service pipeline,
        then runs evolution/concept/reconciliation steps.
        """
        from backend.services.delta_card_service import run_delta_card_pipeline
        from backend.services.evolution_service import link_to_parent_baselines
        from backend.services.concept_synthesizer_service import synthesize_concepts
        from backend.services.incremental_reconciler_service import reconcile_neighbors

        # Extract data from merged agent outputs
        shallow = agent_results.get("shallow_extract", {})
        deep = agent_results.get("deep_analysis", {})
        paper_essence = shallow.get("paper_essence", {})
        method_delta_lite = shallow.get("method_delta", {})
        method_full = deep.get("method", {})
        experiment = deep.get("experiment", {})
        formulas = deep.get("formulas", {})

        # Build analysis_data dict matching delta_card_service input format
        analysis_data = {
            "problem_summary": paper_essence.get("problem_statement", ""),
            "method_summary": paper_essence.get("method_summary", ""),
            "evidence_summary": "",  # built from experiment fairness
            "core_intuition": paper_essence.get("core_claim", ""),
            "changed_slots": [
                s.get("slot_name", "") for s in method_full.get("changed_slots", method_delta_lite.get("changed_slots", []))
            ],
            "is_plugin_patch": method_delta_lite.get("is_plugin_patch", False),
            "structurality_score": 0.7 if method_delta_lite.get("is_structural_change") else 0.3,
            "extensionability_score": None,
            "transferability_score": None,
            "delta_card": {
                "paradigm": paper_essence.get("training_paradigm", "unknown"),
                "slots": {
                    s.get("slot_name", ""): {
                        "changed": True,
                        "from": s.get("baseline_value", ""),
                        "to": s.get("proposed_value", ""),
                        "change_type": s.get("change_type", "modified"),
                    }
                    for s in method_full.get("changed_slots", method_delta_lite.get("changed_slots", []))
                },
                "is_structural": method_delta_lite.get("is_structural_change", False),
                "primary_gain_source": method_full.get("proposed_method_name", method_delta_lite.get("proposed_method_name", "")),
            },
            "bottleneck_addressed": None,
            "same_family_method": method_full.get("proposed_method_name", method_delta_lite.get("proposed_method_name", "")),
            "confidence_notes": paper_essence.get("evidence_refs", []),
            "key_equations": formulas.get("key_formulas", []),
            "key_figures": formulas.get("figure_roles", []),
            "evidence_units": [
                {
                    "atom_type": "evidence",
                    "claim": ref.get("claim", ""),
                    "confidence": ref.get("confidence", 0.5),
                    "basis": ref.get("basis", "text_stated"),
                    "source_section": ref.get("reasoning", ""),
                }
                for ref in paper_essence.get("evidence_refs", [])
            ],
        }

        # Build evidence_summary from experiment fairness
        fairness = experiment.get("fairness_assessment", {})
        if fairness:
            parts = []
            if fairness.get("are_comparisons_fair") is False:
                parts.append("Baseline comparisons may not be fair.")
            if fairness.get("missing_baselines"):
                parts.append(f"Missing baselines: {', '.join(fairness['missing_baselines'])}")
            if fairness.get("potential_issues"):
                parts.append(f"Issues: {'; '.join(fairness['potential_issues'])}")
            analysis_data["evidence_summary"] = " ".join(parts) if parts else "Evidence assessment complete."

        # Assign paradigm
        from backend.services.analysis_service import assign_paradigm
        paradigm, slots = await assign_paradigm(
            self.session, paper.category, paper.tags,
            title=paper.title, abstract=paper.abstract,
        )

        # Build changed_slots_graph for delta_card_service
        changed_slots_graph = [
            {
                "slot_name": s.get("slot_name", ""),
                "from": s.get("baseline_value", ""),
                "to": s.get("proposed_value", ""),
                "change_type": "structural" if method_delta_lite.get("is_structural_change") else "plugin",
            }
            for s in method_full.get("changed_slots", method_delta_lite.get("changed_slots", []))
        ]

        # Run delta_card_service pipeline
        result = await run_delta_card_pipeline(
            self.session,
            paper_id=paper.id,
            analysis_id=None,
            analysis_data=analysis_data,
            paradigm_id=paradigm.id if paradigm else None,
            paradigm_name=paradigm.name if paradigm else None,
            slots=[{"id": s["id"], "name": s["name"]} for s in slots] if slots else None,
            changed_slots_graph=changed_slots_graph if changed_slots_graph else None,
            bottleneck_id=None,
        )

        dc = result.get("delta_card")

        # Post-delta steps (each wrapped to not block on failure)
        if dc and dc.id:
            try:
                await link_to_parent_baselines(self.session, dc.id, analysis_data)
            except Exception as e:
                logger.warning("link_to_parent_baselines failed: %s", e)

        try:
            await synthesize_concepts(self.session, paper.id, analysis_data)
        except Exception as e:
            logger.warning("synthesize_concepts failed: %s", e)

        try:
            await reconcile_neighbors(self.session, paper.id, analysis_data)
        except Exception as e:
            logger.warning("reconcile_neighbors failed: %s", e)

        # ── Write taxonomy facets (Layer A activation) ──
        try:
            await self._write_taxonomy_facets(paper, paper_essence, method_delta_lite, experiment)
        except Exception as e:
            logger.warning("taxonomy facet write failed: %s", e)

        return {
            "delta_card_id": str(dc.id) if dc else None,
            "evidence_count": len(result.get("evidence_units", [])),
            "assertion_count": len(result.get("assertions", [])),
        }

    async def _write_taxonomy_facets(
        self, paper, paper_essence: dict, method_delta: dict, experiment: dict,
    ) -> None:
        """Auto-create TaxonomyNodes and PaperFacets from agent outputs.

        Activates Layer A of the 4-layer architecture:
          1. paper.category → TaxonomyNode(dimension='domain')
          2. target_tasks → TaxonomyNode(dimension='task') + PaperFacet(primary_task)
          3. proposed_method_name → PaperFacet(primary_method)
          4. experiment benchmarks → TaxonomyNode(dimension='dataset') + PaperFacet(dataset)
        """
        from sqlalchemy import func, select
        from backend.models.taxonomy import TaxonomyNode, PaperFacet

        async def get_or_create_taxnode(name: str, dimension: str) -> TaxonomyNode:
            """Find or create a taxonomy node by name + dimension."""
            result = await self.session.execute(
                select(TaxonomyNode).where(
                    func.lower(TaxonomyNode.name) == name.lower(),
                    TaxonomyNode.dimension == dimension,
                ).limit(1)
            )
            node = result.scalar_one_or_none()
            if node:
                return node
            node = TaxonomyNode(
                name=name,
                dimension=dimension,
                status="candidate",
            )
            self.session.add(node)
            await self.session.flush()
            return node

        async def add_facet(paper_id, node_id, role: str, source: str = "auto_agent"):
            """Add a PaperFacet if not exists."""
            existing = await self.session.execute(
                select(PaperFacet.id).where(
                    PaperFacet.paper_id == paper_id,
                    PaperFacet.node_id == node_id,
                    PaperFacet.facet_role == role,
                ).limit(1)
            )
            if existing.scalar_one_or_none():
                return
            self.session.add(PaperFacet(
                paper_id=paper_id,
                node_id=node_id,
                facet_role=role,
                source=source,
            ))

        # 1. Domain from paper.category
        if paper.category:
            domain_node = await get_or_create_taxnode(paper.category, "domain")
            await add_facet(paper.id, domain_node.id, "domain")

        # 2. Tasks from paper_essence.target_tasks
        target_tasks = paper_essence.get("target_tasks", [])
        for i, task_name in enumerate(target_tasks[:3]):
            if task_name:
                task_node = await get_or_create_taxnode(task_name, "task")
                role = "primary_task" if i == 0 else "secondary_task"
                await add_facet(paper.id, task_node.id, role)

        # 3. Modalities from paper_essence.target_modalities
        for mod in paper_essence.get("target_modalities", [])[:3]:
            if mod:
                mod_node = await get_or_create_taxnode(mod, "modality")
                await add_facet(paper.id, mod_node.id, "modality")

        # 4. Training paradigm
        paradigm = paper_essence.get("training_paradigm")
        if paradigm and paradigm != "unknown":
            para_node = await get_or_create_taxnode(paradigm, "learning_paradigm")
            await add_facet(paper.id, para_node.id, "paradigm")

        # 5. Datasets from experiment benchmarks
        main_results = experiment.get("main_results", [])
        seen_datasets = set()
        for result in main_results[:5]:
            benchmark = result.get("benchmark", "")
            if benchmark and benchmark not in seen_datasets:
                seen_datasets.add(benchmark)
                ds_node = await get_or_create_taxnode(benchmark, "dataset")
                await add_facet(paper.id, ds_node.id, "dataset")

        await self.session.flush()

    # ── Phase 4: Neighborhood Discovery ───────────────────────────────

    async def discover_neighborhood(
        self,
        paper_id: UUID,
        *,
        max_references: int = 30,
        max_citations: int = 50,
        max_related: int = 10,
        domain_id: UUID | None = None,
    ) -> dict:
        """Discover neighborhood via Semantic Scholar and create candidates.

        Uses the existing discovery_service to fetch refs, citations, related
        papers, then routes each through import_and_score.

        Returns:
            {total_discovered, candidates_created, candidates_existing,
             by_source: {refs, citations, related}}
        """
        from backend.services.discovery_service import (
            _s2_to_dict,
            _s2_get,
            find_paper_on_s2,
            S2_API,
            S2_FIELDS,
        )
        import asyncio
        import httpx
        from backend.models.paper import Paper

        paper = await self.session.get(Paper, paper_id)
        if not paper:
            return {"error": f"Paper {paper_id} not found"}

        # Find paper on S2
        s2_paper = await find_paper_on_s2(self.session, paper_id)
        if not s2_paper:
            return {
                "total_discovered": 0,
                "candidates_created": 0,
                "candidates_existing": 0,
                "by_source": {"refs": 0, "citations": 0, "related": 0},
                "error": "Paper not found on Semantic Scholar",
            }

        s2_id = s2_paper.get("paperId")
        if not s2_id:
            return {
                "total_discovered": 0,
                "candidates_created": 0,
                "candidates_existing": 0,
                "by_source": {"refs": 0, "citations": 0, "related": 0},
                "error": "No S2 paperId",
            }

        discovered_papers: list[tuple[dict, str, str]] = []  # (paper_dict, source, reason)

        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            # References
            refs_data = await _s2_get(
                client, f"{S2_API}/paper/{s2_id}/references",
                {"fields": S2_FIELDS, "limit": str(max_references)},
            )
            if refs_data and refs_data.get("data"):
                for ref in refs_data["data"]:
                    cited = ref.get("citedPaper", {})
                    if cited and cited.get("title"):
                        discovered_papers.append((
                            _s2_to_dict(cited),
                            "s2_reference",
                            f"reference of {paper.title[:80]}",
                        ))

            await asyncio.sleep(0.5)

            # Citations
            cit_data = await _s2_get(
                client, f"{S2_API}/paper/{s2_id}/citations",
                {"fields": S2_FIELDS, "limit": str(max_citations)},
            )
            if cit_data and cit_data.get("data"):
                for cit in cit_data["data"]:
                    citing = cit.get("citingPaper", {})
                    if citing and citing.get("title"):
                        discovered_papers.append((
                            _s2_to_dict(citing),
                            "s2_citation",
                            f"cites {paper.title[:80]}",
                        ))

            await asyncio.sleep(0.5)

            # Related (recommendations)
            rec_data = await _s2_get(
                client, f"{S2_API}/paper/{s2_id}/recommendations",
                {"fields": S2_FIELDS, "limit": str(max_related)},
            )
            if rec_data and rec_data.get("recommendedPapers"):
                for rec in rec_data["recommendedPapers"]:
                    if rec.get("title"):
                        discovered_papers.append((
                            _s2_to_dict(rec),
                            "s2_recommendation",
                            f"related to {paper.title[:80]}",
                        ))

        # Import each discovered paper through the scoring gate
        created = 0
        existing = 0
        by_source = {"refs": 0, "citations": 0, "related": 0}

        for disc_paper, source, reason in discovered_papers:
            source_key = {
                "s2_reference": "refs",
                "s2_citation": "citations",
                "s2_recommendation": "related",
            }.get(source, "related")

            try:
                result = await self.import_and_score(
                    disc_paper,
                    discovery_source=source,
                    discovery_reason=reason,
                    discovered_from_paper_id=paper_id,
                    domain_id=domain_id,
                )
                by_source[source_key] += 1
                # Heuristic: if status is not "rejected" and score > 0, count as created
                # The create_candidate dedup logic handles existing detection internally
                if result.get("status") != "rejected":
                    created += 1
                else:
                    existing += 1
            except Exception as e:
                logger.debug(
                    "Import failed for %s: %s",
                    disc_paper.get("title", "?"), e,
                )

        await self.session.flush()

        return {
            "total_discovered": len(discovered_papers),
            "candidates_created": created,
            "candidates_existing": existing,
            "by_source": by_source,
        }

    # ── Phase 5: Process Reference Roles ──────────────────────────────

    async def process_reference_roles(
        self,
        paper_id: UUID,
        domain_id: UUID | None = None,
    ) -> dict:
        """Process reference role classifications and import recommended refs.

        Loads reference_role_map from the blackboard for this paper.
        Based on recommended_ingest_level:
            - "deep" → import_and_score with high priority
            - "abstract" → import_and_score
            - "metadata" → create candidate without scoring
            - "skip" → ignore

        Returns:
            {full_imported, shallow_imported, metadata_imported, ignored}
        """
        from sqlalchemy import select
        from backend.models.agent import AgentBlackboardItem

        # Load latest reference_role_map from blackboard
        stmt = (
            select(AgentBlackboardItem)
            .where(
                AgentBlackboardItem.paper_id == paper_id,
                AgentBlackboardItem.item_type == "reference_role_map",
            )
            .order_by(AgentBlackboardItem.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        bb_item = result.scalar_one_or_none()

        if not bb_item or not bb_item.value_json:
            return {
                "full_imported": 0,
                "shallow_imported": 0,
                "metadata_imported": 0,
                "ignored": 0,
            }

        ref_map = bb_item.value_json
        classifications = ref_map.get("classifications", [])

        full_imported = 0
        shallow_imported = 0
        metadata_imported = 0
        ignored = 0

        for ref_cls in classifications:
            ingest_level = ref_cls.get("recommended_ingest_level", "skip")
            ref_title = ref_cls.get("ref_title")
            if not ref_title:
                ignored += 1
                continue

            role = ref_cls.get("role", "unknown")

            if ingest_level == "deep":
                try:
                    await self.import_and_score(
                        {"title": ref_title},
                        discovery_source="method_section_citation",
                        discovery_reason=f"ref_role={role}, recommended=deep",
                        relation_hint=role,
                        discovered_from_paper_id=paper_id,
                        domain_id=domain_id,
                    )
                    full_imported += 1
                except Exception as e:
                    logger.debug("Deep ref import failed for %s: %s", ref_title, e)

            elif ingest_level == "abstract":
                try:
                    await self.import_and_score(
                        {"title": ref_title},
                        discovery_source="s2_reference",
                        discovery_reason=f"ref_role={role}, recommended=abstract",
                        relation_hint=role,
                        discovered_from_paper_id=paper_id,
                        domain_id=domain_id,
                    )
                    shallow_imported += 1
                except Exception as e:
                    logger.debug("Shallow ref import failed for %s: %s", ref_title, e)

            elif ingest_level == "metadata":
                try:
                    await candidate_service.create_candidate(
                        self.session,
                        title=ref_title,
                        discovery_source="s2_reference",
                        discovery_reason=f"ref_role={role}, metadata_only",
                        relation_hint=role,
                        discovered_from_paper_id=paper_id,
                        discovered_from_domain_id=domain_id,
                    )
                    metadata_imported += 1
                except Exception as e:
                    logger.debug("Metadata ref import failed for %s: %s", ref_title, e)

            else:
                ignored += 1

        await self.session.flush()

        return {
            "full_imported": full_imported,
            "shallow_imported": shallow_imported,
            "metadata_imported": metadata_imported,
            "ignored": ignored,
        }

    # ── Known Reference Handling (对比流程) ────────────────────────

    _ROLE_TO_RELATION = {
        "direct_baseline": "builds_on",
        "method_source": "extends",
        "formula_source": "extends",
        "comparison_baseline": "compares_against",
        "dataset_source": "evaluates_on",
        "benchmark_source": "evaluates_on",
        "same_task_prior": "cites",
        "survey_or_taxonomy": "cites",
    }

    async def _handle_known_references(
        self,
        paper_id: UUID,
        classifications: list[dict],
        *,
        domain_id: UUID | None = None,
    ) -> dict:
        """处理引用了 KB 中已有论文的情况。

        对比流程：不重新分析已知论文，而是：
        1. 创建 当前论文→已知论文 的关系边 (GraphEdgeCandidate)
        2. 更新已知论文的 downstream 计数
        3. 触发已知论文 profile 的 staleness 递增
        """
        from backend.services import candidate_service
        from backend.models.kb import GraphEdgeCandidate

        known_refs_linked = 0
        edges_created = 0
        known_titles_set: set[str] = set()

        important_roles = {
            "direct_baseline", "method_source", "formula_source",
            "comparison_baseline", "dataset_source", "benchmark_source",
            "same_task_prior",
        }

        for ref_cls in classifications:
            role = ref_cls.get("role", "")
            if role not in important_roles:
                continue

            ref_title = ref_cls.get("ref_title")
            if not ref_title:
                continue

            norm = _normalize_title(ref_title)

            try:
                # 查找 papers 表中的已有论文
                kb_paper = await candidate_service.find_existing_paper(
                    self.session,
                    normalized_title=norm,
                )
                if not kb_paper:
                    continue

                # ── 找到已知论文，执行对比流程 ──
                known_titles_set.add(norm)
                known_refs_linked += 1

                # 1. 创建关系边候选
                relation = self._ROLE_TO_RELATION.get(role, "cites")
                edge = GraphEdgeCandidate(
                    paper_id=paper_id,
                    source_entity_type="paper",
                    source_entity_id=paper_id,
                    target_entity_type="paper",
                    target_entity_id=kb_paper.id,
                    relation_type=relation,
                    one_liner=ref_cls.get("reason", ""),
                    confidence_score=ref_cls.get("confidence", 0.7),
                    evidence_refs={
                        "source_paper_id": str(paper_id),
                        "role": role,
                        "where_mentioned": ref_cls.get("where_mentioned", []),
                    },
                    status="candidate",
                )
                self.session.add(edge)
                edges_created += 1

                # 2. 更新下游计数
                if kb_paper.cited_by_count is not None:
                    kb_paper.cited_by_count = (kb_paper.cited_by_count or 0) + 1
                else:
                    kb_paper.cited_by_count = 1

                # 3. 触发 profile staleness 递增
                try:
                    from backend.services import node_profile_service
                    await node_profile_service.increment_staleness(
                        self.session, "paper", kb_paper.id
                    )
                except Exception as e:
                    logger.debug("profile staleness update skipped for %s: %s", kb_paper.id, e)

                logger.info(
                    "已知论文重遇: %s → %s (%s), 创建 %s 边",
                    ref_title[:50], kb_paper.title[:50], role, relation,
                )

            except Exception as e:
                logger.debug("Known ref check failed for %s: %s", ref_title, e)

        if edges_created:
            await self.session.flush()

        return {
            "known_refs_linked": known_refs_linked,
            "edges_created": edges_created,
            "known_titles_set": known_titles_set,
        }

    # ── Full V6 Pipeline ─────────────────────────────────────────────

    async def run_full_v6_pipeline(
        self,
        source: str,
        *,
        domain_id: UUID | None = None,
    ) -> dict:
        """Run the complete V6 pipeline end-to-end.

        import_and_score → shallow_ingest → deep_ingest
        + discover_neighborhood + process_reference_roles

        Returns complete pipeline result dict.
        """
        result = {
            "source": source if isinstance(source, str) else str(source),
            "phases": {},
        }

        # Phase 1: Import & Score
        try:
            import_result = await self.import_and_score(
                source, domain_id=domain_id,
            )
            result["phases"]["import_and_score"] = import_result
        except Exception as e:
            logger.error("import_and_score failed: %s", e)
            result["phases"]["import_and_score"] = {"error": str(e)}
            return result

        candidate_id = import_result.get("candidate_id")
        decision = import_result.get("decision", "archive")

        if decision not in ("shallow_ingest",):
            result["final_decision"] = decision
            return result

        # Phase 2: Shallow Ingest
        if candidate_id:
            from uuid import UUID as _UUID
            cid = _UUID(candidate_id) if isinstance(candidate_id, str) else candidate_id
            try:
                shallow_result = await self.shallow_ingest(cid)
                result["phases"]["shallow_ingest"] = shallow_result
            except Exception as e:
                logger.error("shallow_ingest failed: %s", e)
                result["phases"]["shallow_ingest"] = {"error": str(e)}
                result["final_decision"] = "shallow_failed"
                return result

            paper_id_str = shallow_result.get("paper_id")
            deep_decision = shallow_result.get("decision", "no_graph")

            # Phase 2.5: Neighborhood Discovery (non-blocking)
            if paper_id_str:
                pid = _UUID(paper_id_str) if isinstance(paper_id_str, str) else paper_id_str
                try:
                    disc_result = await self.discover_neighborhood(
                        pid, domain_id=domain_id,
                    )
                    result["phases"]["discover_neighborhood"] = disc_result
                except Exception as e:
                    logger.error("discover_neighborhood failed: %s", e)
                    result["phases"]["discover_neighborhood"] = {"error": str(e)}

                # Phase 2.6: Process Reference Roles
                try:
                    ref_result = await self.process_reference_roles(
                        pid, domain_id=domain_id,
                    )
                    result["phases"]["process_reference_roles"] = ref_result
                except Exception as e:
                    logger.error("process_reference_roles failed: %s", e)
                    result["phases"]["process_reference_roles"] = {"error": str(e)}

            # Phase 3: Deep Ingest (if qualified)
            if deep_decision in ("auto_full_paper", "full_review_needed") and paper_id_str:
                pid = _UUID(paper_id_str) if isinstance(paper_id_str, str) else paper_id_str
                review_needed = deep_decision == "full_review_needed"
                try:
                    deep_result = await self.deep_ingest(
                        pid, review_needed=review_needed,
                    )
                    result["phases"]["deep_ingest"] = deep_result
                except Exception as e:
                    logger.error("deep_ingest failed: %s", e)
                    result["phases"]["deep_ingest"] = {"error": str(e)}

            result["final_decision"] = deep_decision
        else:
            result["final_decision"] = decision

        await self.session.commit()
        return result
