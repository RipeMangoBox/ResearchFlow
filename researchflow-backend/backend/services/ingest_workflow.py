"""V6 Ingest Workflow — orchestrates candidate → shallow → deep → profile → report.

This is the main entry point for the V6 pipeline. Papers flow through:
  Import → DiscoveryScore → [Shallow Ingest] → DeepIngestScore → [Deep Ingest] → [Profile] → [Report]

Each phase is independently retryable. Score gates prevent low-value papers from consuming resources.
"""

import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.services import candidate_service
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

        # Run shallow agents sequentially, each with its own context pack
        shallow_agents = [
            ("shallow_paper", "paper_essence"),
            ("reference_role", "reference_role_map"),
            ("method_delta_lite", "method_delta"),
            ("score", "score_signals"),
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

        # Build deep-ingest signals from agent results
        score_signals = agent_results.get("score_signals", {})
        method_delta = agent_results.get("method_delta", {})
        paper_essence = agent_results.get("paper_essence", {})

        deep_signals = {
            "domain_fit": score_signals.get("aggregate_score", 0) * 100,
            "relation_type": "same_task_prior" if score_signals.get("same_primary_task") else "",
            "reusable_knowledge": (
                50 * int(bool(method_delta.get("should_create_method_node")))
                + 30 * int(bool(method_delta.get("changed_slots")))
                + 20 * int(bool(score_signals.get("has_ablation")))
            ),
            "evidence_quality": score_signals.get("evidence_quality", 0) * 100,
            "experiment_value": (
                40 * int(bool(score_signals.get("has_ablation")))
                + 30 * int(score_signals.get("baseline_count", 0) >= 3)
                + 30 * int(bool(score_signals.get("has_new_dataset")))
            ),
            "has_code": score_signals.get("has_code", False),
            "has_data": score_signals.get("has_new_dataset", False),
            "has_model": False,
            "has_benchmark": False,
            "novelty_freshness": score_signals.get("method_novelty", 0) * 100,
            "source_type": candidate.discovery_source or "",
            "is_direct_baseline": score_signals.get("is_direct_baseline", False),
            "in_experiment_table": score_signals.get("in_experiment_table", False),
            "same_primary_task": score_signals.get("same_primary_task", False),
            "has_changed_slots": score_signals.get("has_changed_slots", False),
            "has_ablation": score_signals.get("has_ablation", False),
            "is_baseline": score_signals.get("is_direct_baseline", False),
            "has_strong_ablation": score_signals.get("has_ablation", False),
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

        # Trigger recursive discovery from reference roles
        references_discovered = 0
        ref_map = agent_results.get("reference_role_map", {})
        classifications = ref_map.get("classifications", [])
        for ref_cls in classifications:
            ingest_level = ref_cls.get("recommended_ingest_level", "skip")
            if ingest_level in ("deep", "abstract"):
                ref_title = ref_cls.get("ref_title")
                if ref_title:
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
        }

    # ── Phase 3: Deep Ingest ──────────────────────────────────────────

    async def deep_ingest(
        self,
        paper_id: UUID,
        *,
        review_needed: bool = False,
    ) -> dict:
        """Run deep-phase agents, profile qualifying nodes/edges, generate report.

        Deep agents (sequential):
            1. MethodDeltaAgent-full → method_delta_full
            2. ExperimentAgent → experiment_matrix
            3. FormulaFigureAgent → formula_figures
            4. GraphCandidateAgent → graph_candidates

        Profile agents (for qualifying candidates):
            5. NodeProfileAgent (promotion_score >= 75)
            6. EdgeProfileAgent (confidence_score >= 70)

        Report agents:
            7. PaperReportAgent → 10-section structured report
            8. QualityAuditAgent → review items

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
            ("method_delta_full", "method_delta_full"),
            ("experiment", "experiment_matrix"),
            ("formula_figure", "formula_figures"),
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

        # ── Scoring & Profile agents ──

        graph_candidates = agent_results.get("graph_candidates", {})

        # Score and profile node candidates
        node_candidates = graph_candidates.get("nodes", [])
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
                # Run NodeProfileAgent
                try:
                    profile_context = {
                        "user_content": (
                            f"Node candidate: {node_cand}\n"
                            f"Paper: {paper.title}\n"
                            f"Promotion score: {node_score.total}"
                        ),
                        "token_budget": 4096,
                    }
                    await self.runner.run_agent(
                        "node_profile", profile_context, paper_id=paper_id,
                    )
                    agents_run.append("node_profile")
                    nodes_created += 1
                except Exception as e:
                    logger.error("node_profile agent failed: %s", e)

        # Score and profile edge candidates
        edge_candidates = graph_candidates.get("edges", [])
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
                # Run EdgeProfileAgent
                try:
                    profile_context = {
                        "user_content": (
                            f"Edge candidate: {edge_cand}\n"
                            f"Paper: {paper.title}\n"
                            f"Confidence score: {edge_score.total}"
                        ),
                        "token_budget": 4096,
                    }
                    await self.runner.run_agent(
                        "edge_profile", profile_context, paper_id=paper_id,
                    )
                    agents_run.append("edge_profile")
                    edges_created += 1
                except Exception as e:
                    logger.error("edge_profile agent failed: %s", e)

        # ── Report agents ──

        # Agent 7: PaperReportAgent
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

        # Agent 8: QualityAuditAgent
        try:
            audit_context = {
                "user_content": (
                    f"Paper: {paper.title}\n"
                    f"Agent results: {agent_results}\n"
                    f"Review needed: {review_needed}"
                ),
                "token_budget": 4096,
            }
            await self.runner.run_agent(
                "quality_audit", audit_context, paper_id=paper_id,
            )
            agents_run.append("quality_audit")
        except Exception as e:
            logger.error("quality_audit agent failed for %s: %s", paper_id, e)

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
        }

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

        async with httpx.AsyncClient(follow_redirects=True) as client:
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
