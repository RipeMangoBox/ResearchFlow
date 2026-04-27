"""Unified Ingest Workflow — single pipeline for all paper processing.

Replaces both the old pipeline_service.run_full_pipeline and V6 run_full_v6_pipeline.

Complete flow:
  import_and_score → enrich_and_prepare (metadata+pdf+parse) → shallow_ingest
  → deep_ingest (agents+graph+report) → discover_neighborhood

Each phase is independently retryable. Score gates prevent low-value papers
from consuming resources; force_ingest bypasses gates for top-conference papers.
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

# Venues that bypass the scoring gate (top conferences)
TOP_VENUES = {
    "CVPR", "ICCV", "ECCV", "NeurIPS", "ICML", "ICLR", "ACL", "EMNLP",
    "NAACL", "AAAI", "IJCAI", "SIGGRAPH", "KDD", "ICRA", "CORL",
    "SIGIR", "WWW", "ICDM", "CIKM", "WACV",
}


class IngestWorkflow:
    """V6 multi-phase, score-gated paper ingestion workflow."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.engine = ScoringEngine()
        self.runner = AgentRunner(session)
        self.ctx_builder = ContextPackBuilder(self.session)

    # ── Direct entry for existing papers (bypasses candidate layer) ──

    async def run_for_existing_paper(
        self,
        paper_id: UUID,
        *,
        skip_enrich: bool = False,
        skip_parse: bool = False,
    ) -> dict:
        """Run the full pipeline on a paper already in the papers table.

        Bypasses candidate creation/scoring — the paper already exists.
        Flow: enrich → parse → shallow agents → deep agents → materialize → report

        Args:
            paper_id: UUID of existing paper in papers table.
            skip_enrich: Skip metadata enrichment if already done.
            skip_parse: Skip L2 PDF parse if already done.

        Returns:
            Complete pipeline result dict with all phase outcomes.
        """
        from backend.models.paper import Paper
        from backend.models.analysis import PaperAnalysis
        from backend.models.enums import AnalysisLevel, PaperState

        result = {"paper_id": str(paper_id), "phases": {}}

        paper = await self.session.get(Paper, paper_id)
        if not paper:
            return {"error": f"Paper {paper_id} not found"}

        # Phase 1: Enrich metadata (abstract, authors, code_url, citations)
        if not skip_enrich:
            try:
                prep = await self.enrich_and_prepare(paper_id)
                result["phases"]["enrich_and_prepare"] = prep
                await self.session.commit()
            except Exception as e:
                logger.error("enrich_and_prepare failed for %s: %s", paper_id, e)
                result["phases"]["enrich_and_prepare"] = {"error": str(e)[:200]}
                try:
                    await self.session.rollback()
                except Exception:
                    pass
            try:
                await self.session.refresh(paper)
            except Exception:
                paper = await self.session.get(Paper, paper_id)

        # Phase 2: Run shallow agents (shallow_extractor + reference_role)
        # Build context directly from Paper (no candidate needed)
        agents_run = []
        agent_results = {}

        shallow_agents = [
            ("shallow_extractor", "shallow_extract"),
            ("reference_role", "reference_role_map"),
        ]

        for agent_name, result_key in shallow_agents:
            try:
                context = await self.ctx_builder.build(
                    agent_name, paper_id=paper_id,
                )
            except Exception as e:
                logger.warning("Context build failed for %s: %s", agent_name, e)
                context = {
                    "user_content": (
                        f"Paper: {paper.title}\n"
                        f"Abstract: {paper.abstract or 'N/A'}"
                    ),
                    "token_budget": 4096,
                }

            try:
                res = await self.runner.run_agent(
                    agent_name, context, paper_id=paper_id,
                )
                agents_run.append(agent_name)
                agent_results[result_key] = res
                await self.session.flush()
            except Exception as e:
                logger.error("%s failed for %s: %s", agent_name, paper_id, e)
                agent_results[result_key] = {}
                try:
                    await self.session.rollback()
                except Exception:
                    pass

        # Write L3-compatible PaperAnalysis from shallow output
        shallow_extract = agent_results.get("shallow_extract", {})
        paper_essence = shallow_extract.get("paper_essence", {})
        method_delta = shallow_extract.get("method_delta", {})

        if paper_essence:
            try:
                from sqlalchemy import select
                old_l3 = (await self.session.execute(
                    select(PaperAnalysis).where(
                        PaperAnalysis.paper_id == paper_id,
                        PaperAnalysis.level == AnalysisLevel.L3_SKIM,
                        PaperAnalysis.is_current.is_(True),
                    )
                )).scalar_one_or_none()
                if old_l3:
                    old_l3.is_current = False

                l3 = PaperAnalysis(
                    paper_id=paper_id,
                    level=AnalysisLevel.L3_SKIM,
                    model_provider="agent",
                    model_name="shallow_extractor",
                    prompt_version="v7_unified",
                    schema_version="v2",
                    confidence=0.75,
                    problem_summary=paper_essence.get("problem_statement"),
                    method_summary=paper_essence.get("method_summary"),
                    core_intuition=paper_essence.get("core_claim"),
                    changed_slots=[s.get("slot_name", "") for s in method_delta.get("changed_slots", [])],
                    is_plugin_patch=method_delta.get("is_plugin_patch"),
                    worth_deep_read=bool(method_delta.get("should_create_method_node")),
                    confidence_notes=paper_essence.get("evidence_refs"),
                    is_current=True,
                )
                self.session.add(l3)
                await self.session.flush()
            except Exception as e:
                logger.warning("L3 write failed: %s", e)

        result["phases"]["shallow"] = {
            "agents_run": agents_run,
            "has_paper_essence": bool(paper_essence),
            "has_method_delta": bool(method_delta),
        }

        # Link known references
        ref_map = agent_results.get("reference_role_map", {})
        classifications = ref_map.get("classifications", [])
        try:
            known_result = await self._handle_known_references(
                paper_id, classifications,
            )
            result["phases"]["known_refs"] = known_result
        except Exception as e:
            logger.warning("Known refs failed: %s", e)

        # Commit shallow results before deep phase
        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()

        # Phase 3: Deep agents + materialization
        try:
            deep_result = await self.deep_ingest(paper_id)
            result["phases"]["deep_ingest"] = deep_result
            await self.session.commit()
        except Exception as e:
            logger.error("deep_ingest failed for %s: %s", paper_id, e)
            result["phases"]["deep_ingest"] = {"error": str(e)[:200]}
            try:
                await self.session.rollback()
            except Exception:
                pass

        # Phase 4: Post-L4 — derive ring & role_in_kb from the published
        # DeltaCard's structurality_score. Without this, vault_export's
        # _paper_level() falls back to dc_struct=0 → all papers classify
        # as D-level and get filtered out of the export. (Previously this
        # only ran inside pipeline_service.run_pipeline; task_pipeline_run
        # → run_for_existing_paper bypassed it.)
        try:
            from backend.models.delta_card import DeltaCard
            await self.session.refresh(paper)
            if paper.current_delta_card_id and (
                not paper.ring or not paper.role_in_kb
            ):
                dc = await self.session.get(DeltaCard, paper.current_delta_card_id)
                if dc and dc.structurality_score is not None:
                    s = float(dc.structurality_score)
                    if not paper.ring:
                        if s >= 0.7:
                            paper.ring = "baseline"
                        elif s >= 0.4:
                            paper.ring = "structural"
                        else:
                            paper.ring = "plugin"
                    if not paper.role_in_kb:
                        paper.role_in_kb = "extension"
                    await self.session.commit()
                    result["phases"]["post_l4"] = {
                        "ring": paper.ring, "role_in_kb": paper.role_in_kb,
                    }
        except Exception as e:
            logger.warning("post_l4 ring assignment failed for %s: %s", paper_id, e)
            try:
                await self.session.rollback()
            except Exception:
                pass

        return result

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
        force_ingest: bool = False,
    ) -> dict:
        """Create a candidate and compute its DiscoveryScore.

        Routes based on score:
            >= 75 → enqueue shallow_ingest
            60-74 → candidate_pool
            40-59 → metadata_only
            < 40  → archive

        If force_ingest=True (e.g. top-conference accepted papers),
        always route to shallow_ingest regardless of score.

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

        # Auto-detect top venue → force_ingest
        if not force_ingest and isinstance(source, dict):
            venue = (source.get("venue") or "").strip()
            acceptance = (source.get("acceptance_type") or "").strip()
            if venue:
                venue_upper = venue.upper().split()[0] if venue else ""
                if venue_upper in TOP_VENUES or acceptance in ("oral", "spotlight", "poster"):
                    force_ingest = True
                    logger.info("Auto force_ingest: venue=%s acceptance=%s", venue, acceptance)

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

        # Route based on decision (force_ingest overrides scoring gate)
        decision = "shallow_ingest" if force_ingest else score_result.decision
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

    # ── Phase 1.5: Enrich & Prepare ──────────────────────────────────

    async def enrich_and_prepare(self, paper_id: UUID) -> dict:
        """Enrich metadata, download PDF, and run L2 parse.

        Merges the old pipeline_service steps into the unified workflow:
          1. Enrich metadata (arXiv, Crossref, S2, OpenAlex, DBLP, OpenReview)
          2. Venue resolution (OpenReview + DBLP + arXiv comments)
          3. Download PDF to OSS
          4. L2 Parse (PyMuPDF + section extraction)

        Returns progress dict.
        """
        from backend.models.paper import Paper
        from backend.models.analysis import PaperAnalysis
        from backend.models.enums import AnalysisLevel

        progress: dict = {}
        paper = await self.session.get(Paper, paper_id)
        if not paper:
            return {"error": f"Paper {paper_id} not found"}

        # Step 1: Enrich metadata
        if not paper.abstract or not paper.authors:
            try:
                import httpx
                from backend.services import enrich_service
                async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
                    enriched = await enrich_service.enrich_paper(self.session, paper, client)
                progress["enrich"] = enriched if enriched else "no_data"
                await self.session.flush()
                await self.session.refresh(paper)
            except Exception as e:
                logger.warning("Enrich failed for %s: %s", paper_id, e)
                progress["enrich"] = f"error: {str(e)[:100]}"
        else:
            progress["enrich"] = "skipped"

        # Step 2: Venue resolution
        if not paper.acceptance_type:
            try:
                from backend.services.venue_resolver_service import resolve_venue
                authors_list = None
                if paper.authors and isinstance(paper.authors, list):
                    authors_list = [a.get("name", "") for a in paper.authors if isinstance(a, dict)]
                venue_result = await resolve_venue(
                    self.session, paper.id,
                    title=paper.title, authors=authors_list,
                    arxiv_id=paper.arxiv_id or "",
                    current_venue=paper.venue or "",
                    current_year=paper.year or 0,
                )
                if venue_result.get("acceptance_status") and venue_result["acceptance_status"] != "unknown":
                    paper.acceptance_type = venue_result["acceptance_status"]
                    if venue_result.get("venue"):
                        paper.venue = venue_result["venue"][:100]
                progress["venue_resolve"] = venue_result.get("acceptance_status", "unknown")
                await self.session.flush()
                await self.session.refresh(paper)
            except Exception as e:
                logger.warning("Venue resolution failed for %s: %s", paper_id, e)
                progress["venue_resolve"] = f"error: {str(e)[:100]}"
        else:
            progress["venue_resolve"] = "skipped"

        # Step 3: Download PDF
        if paper.arxiv_id and not paper.pdf_path_local and not paper.pdf_object_key:
            try:
                from backend.services.pdf_download_service import download_pdf_to_oss
                ok = await download_pdf_to_oss(self.session, paper_id)
                progress["download_pdf"] = "success" if ok else "failed"
                await self.session.flush()
                await self.session.refresh(paper)
            except Exception as e:
                logger.warning("PDF download failed for %s: %s", paper_id, e)
                progress["download_pdf"] = f"error: {str(e)[:100]}"
        else:
            progress["download_pdf"] = "skipped"

        # Step 4: L2 Parse
        from sqlalchemy import select
        has_l2 = (await self.session.execute(
            select(PaperAnalysis).where(
                PaperAnalysis.paper_id == paper_id,
                PaperAnalysis.level == AnalysisLevel.L2_PARSE,
                PaperAnalysis.is_current.is_(True),
            )
        )).scalar_one_or_none()

        if not has_l2 and (paper.pdf_path_local or paper.pdf_object_key):
            try:
                from backend.services import parse_service
                l2 = await parse_service.parse_paper_pdf(self.session, paper_id)
                sections = list(l2.extracted_sections.keys()) if l2 and l2.extracted_sections else []
                progress["parse_l2"] = f"sections={sections}"
                await self.session.flush()
            except Exception as e:
                logger.warning("L2 parse failed for %s: %s", paper_id, e)
                progress["parse_l2"] = f"error: {str(e)[:100]}"
        else:
            progress["parse_l2"] = "skipped" if has_l2 else "no_pdf"

        return progress

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

        # ── Write L3-compatible PaperAnalysis from shallow_extractor output ──
        # This replaces the old L3 skim step — shallow_extractor covers the same ground.
        shallow_extract = agent_results.get("shallow_extract", {})
        paper_essence = shallow_extract.get("paper_essence", {})
        method_delta = shallow_extract.get("method_delta", {})

        if paper_essence:
            try:
                from backend.models.analysis import PaperAnalysis
                from backend.models.enums import AnalysisLevel
                # Mark old L3 as superseded
                old_l3 = await self.session.execute(
                    select(PaperAnalysis).where(
                        PaperAnalysis.paper_id == paper_id,
                        PaperAnalysis.level == AnalysisLevel.L3_SKIM,
                        PaperAnalysis.is_current.is_(True),
                    )
                )
                old = old_l3.scalar_one_or_none()
                if old:
                    old.is_current = False

                l3_compat = PaperAnalysis(
                    paper_id=paper_id,
                    level=AnalysisLevel.L3_SKIM,
                    model_provider="agent",
                    model_name="shallow_extractor",
                    prompt_version="v7_merged",
                    schema_version="v2",
                    confidence=0.75,
                    problem_summary=paper_essence.get("problem_statement"),
                    method_summary=paper_essence.get("method_summary"),
                    core_intuition=paper_essence.get("core_claim"),
                    changed_slots=[s.get("slot_name", "") for s in method_delta.get("changed_slots", [])],
                    is_plugin_patch=method_delta.get("is_plugin_patch"),
                    worth_deep_read=bool(method_delta.get("should_create_method_node")),
                    confidence_notes=paper_essence.get("evidence_refs"),
                    is_current=True,
                )
                self.session.add(l3_compat)
                await self.session.flush()
            except Exception as e:
                logger.warning("L3 compat write failed for %s: %s", paper_id, e)

        # Build deep-ingest signals deterministically
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

        # ── Link known references (对比流程，不重新分析) ──
        # Unknown refs are discovered later by discover_neighborhood (single entry)
        ref_map = agent_results.get("reference_role_map", {})
        classifications = ref_map.get("classifications", [])
        known_result = await self._handle_known_references(
            paper_id, classifications,
            domain_id=candidate.discovered_from_domain_id,
        )

        return {
            "paper_id": str(paper_id),
            "deep_ingest_score": deep_score_result.total,
            "decision": decision,
            "agents_run": agents_run,
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

        # ── Re-hydrate shallow-phase artifacts from the blackboard ──
        # `_materialize_to_graph` (called below) reads `agent_results
        # ["shallow_extract"]` to extract evidence_refs / method_delta /
        # paper_essence. If we skip this rehydration the dict is empty,
        # evidence_count stays 0, the DeltaCard never publishes, and
        # `paper.current_delta_card_id` is never set — which cascades into
        # missing ring / facets / vault export.
        from backend.models.agent import AgentBlackboardItem
        from sqlalchemy import select as _select
        for _item_type, _result_key in (
            ("shallow_extract", "shallow_extract"),
            ("reference_role_map", "reference_role_map"),
        ):
            row = (await self.session.execute(
                _select(AgentBlackboardItem)
                .where(
                    AgentBlackboardItem.paper_id == paper_id,
                    AgentBlackboardItem.item_type == _item_type,
                )
                .order_by(AgentBlackboardItem.created_at.desc())
                .limit(1)
            )).scalar_one_or_none()
            if row and row.value_json:
                agent_results[_result_key] = row.value_json

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
            figures_block = await self._build_figures_block(paper_id)
            metadata_block = self._build_paper_metadata_block(paper)

            report_context = {
                "user_content": (
                    f"Paper title: {paper.title}\n\n"
                    f"=== Paper metadata (use in section 1 metadata_overview) ===\n"
                    f"{metadata_block}\n\n"
                    f"=== Figures available (use EXACT labels in figure_placements.preferred_labels) ===\n"
                    f"{figures_block}\n\n"
                    f"=== Agent analysis artifacts ===\n"
                    f"{agent_results}\n\n"
                    f"Graph nodes created: {nodes_created}, edges created: {edges_created}"
                ),
                "token_budget": 8192,
            }
            report_result = await self.runner.run_agent(
                "paper_report", report_context, paper_id=paper_id,
            )
            agents_run.append("paper_report")
            agent_results["paper_report"] = report_result
            secs = report_result.get("sections", [])
            report_sections = [s.get("section_type", "") for s in secs] if isinstance(secs, list) else []
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
                "Materialized graph for %s: dc=%s, evidence=%d, assertions=%d",
                paper_id,
                delta_card_result.get("delta_card_id"),
                delta_card_result.get("evidence_count", 0),
                delta_card_result.get("assertion_count", 0),
            )

            # Mark blackboard items as verified after successful materialization
            from sqlalchemy import update
            from backend.models.agent import AgentBlackboardItem
            await self.session.execute(
                update(AgentBlackboardItem)
                .where(
                    AgentBlackboardItem.paper_id == paper_id,
                    AgentBlackboardItem.is_verified.is_(False),
                )
                .values(is_verified=True)
            )
        except Exception as e:
            logger.error("Graph materialization failed for %s: %s", paper_id, e)

        # ── Phase 3B: Persist KB profiles to tables ──
        if qualifying_nodes or qualifying_edges:
            try:
                await self._persist_kb_profiles(
                    paper_id, agent_results.get("graph_candidates", {}),
                )
            except Exception as e:
                logger.warning("KB profile persistence failed: %s", e)

        # ── Phase 3C: Persist paper report to tables ──
        report_result_data = agent_results.get("paper_report") if "paper_report" in agents_run else None
        if not report_result_data:
            # Try loading from blackboard
            try:
                from sqlalchemy import select as sa_select
                from backend.models.agent import AgentBlackboardItem
                bb = (await self.session.execute(
                    sa_select(AgentBlackboardItem.value_json)
                    .where(
                        AgentBlackboardItem.paper_id == paper_id,
                        AgentBlackboardItem.item_type == "paper_report",
                    )
                    .order_by(AgentBlackboardItem.created_at.desc())
                    .limit(1)
                )).scalar_one_or_none()
                if bb:
                    report_result_data = bb
            except Exception:
                pass

        if report_result_data:
            try:
                await self._persist_paper_report(paper_id, report_result_data)
            except Exception as e:
                logger.warning("Paper report persistence failed: %s", e)

        # ── Phase 3D: Materialize paper-paper relations from reference_role_map ──
        try:
            from backend.services.paper_relation_service import materialize_for_paper
            rel_stats = await materialize_for_paper(self.session, paper_id)
            logger.info("Paper relations materialized for %s: %s", paper_id, rel_stats)
        except Exception as e:
            logger.warning("Paper relation materialization failed for %s: %s", paper_id, e)

        # ── Phase 3E: Promote baseline references (multi-source search + ingest) ──
        # For each baseline-role ref the agent identified, ensure the target
        # paper is either already in `papers`, has been promoted from
        # `venue_papers`, or has a fresh `paper_candidates` row queued.
        try:
            from backend.services.baseline_promoter import promote_for_paper as _promote_baselines
            bp_stats = await _promote_baselines(self.session, paper_id, max_promote=8)
            logger.info("Baseline promote for %s: %s", paper_id, bp_stats)
        except Exception as e:
            logger.warning("Baseline promote failed for %s: %s", paper_id, e)

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
        """Convert agent outputs to DeltaCard → GraphAssertions.

        Bridges agent blackboard outputs to the delta_card_service pipeline,
        then runs evolution/concept/reconciliation steps.
        """
        from backend.models.paper import Paper
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

        # Post-delta steps — each in a SAVEPOINT so failures don't cascade.
        # expire_all() after failures to avoid greenlet errors on detached objects.
        paper_id_val = paper.id
        paper_category = paper.category

        if dc and dc.id:
            try:
                async with self.session.begin_nested():
                    await link_to_parent_baselines(self.session, dc.id, analysis_data)
            except Exception as e:
                logger.warning("link_baselines failed: %s", e)
                self.session.expire_all()

        try:
            async with self.session.begin_nested():
                await synthesize_concepts(self.session, paper_id_val, analysis_data)
        except Exception as e:
            logger.warning("synthesize_concepts failed: %s", e)
            self.session.expire_all()

        try:
            async with self.session.begin_nested():
                await reconcile_neighbors(self.session, paper_id_val, analysis_data)
        except Exception as e:
            logger.warning("reconcile_neighbors failed: %s", e)
            self.session.expire_all()

        try:
            # Refresh paper object for taxonomy write (may be expired from earlier failures)
            paper = await self.session.get(Paper, paper_id_val)
            if paper:
                async with self.session.begin_nested():
                    await self._write_taxonomy_facets(paper, paper_essence, method_delta_lite, experiment)
        except Exception as e:
            logger.warning("taxonomy_facets failed: %s", e)
            self.session.expire_all()

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

    # ── Persistence: KB profiles → tables ────────────────────────────

    async def _persist_kb_profiles(self, paper_id: UUID, graph_candidates: dict):
        """Persist kb_profiler agent output to KBNodeProfile / KBEdgeProfile tables.

        Reads profiles from the blackboard and writes to the persistent tables
        used by vault_export and the MCP server.
        """
        from sqlalchemy import select as sa_select
        from backend.models.agent import AgentBlackboardItem
        from backend.models.kb import KBNodeProfile, KBEdgeProfile

        # Load latest kb_profiles from blackboard
        bb = (await self.session.execute(
            sa_select(AgentBlackboardItem)
            .where(
                AgentBlackboardItem.paper_id == paper_id,
                AgentBlackboardItem.item_type == "kb_profiles",
            )
            .order_by(AgentBlackboardItem.created_at.desc())
            .limit(1)
        )).scalar_one_or_none()

        if not bb or not bb.value_json:
            return

        profiles_data = bb.value_json

        # Persist node profiles
        for np in profiles_data.get("node_profiles", []):
            name = np.get("node_name", "")
            if not name:
                continue
            profile = KBNodeProfile(
                entity_type="graph_node_candidate",
                entity_id=paper_id,  # linked to paper for now
                one_liner=np.get("one_liner"),
                short_intro_md=np.get("short_intro_md"),
                detailed_md=np.get("detailed_md"),
                structured_json=np.get("structured_json"),
                evidence_refs=np.get("evidence_refs"),
                generated_by_run_id=bb.run_id,
            )
            self.session.add(profile)

        # Persist edge profiles
        for ep in profiles_data.get("edge_profiles", []):
            edge_profile = KBEdgeProfile(
                source_entity_type="paper",
                source_entity_id=paper_id,
                target_entity_type="paper",
                target_entity_id=paper_id,  # placeholder
                relation_type=ep.get("relation_type", "unknown"),
                one_liner=ep.get("one_liner"),
                relation_summary=ep.get("relation_summary"),
                source_context=ep.get("source_context"),
                target_context=ep.get("target_context"),
                evidence_refs=ep.get("evidence_refs"),
                generated_by_run_id=bb.run_id,
            )
            self.session.add(edge_profile)

        await self.session.flush()
        logger.info(
            "Persisted %d node profiles + %d edge profiles for paper %s",
            len(profiles_data.get("node_profiles", [])),
            len(profiles_data.get("edge_profiles", [])),
            paper_id,
        )

    # ── Persistence: Paper report → tables ───────────────────────────

    async def _build_figures_block(self, paper_id: UUID) -> str:
        """Format figures from L2 paper_analyses for the paper_report context.

        The agent must use EXACT labels (e.g. "Figure 1", "Table 2") from this
        list when filling figure_placements.preferred_labels.
        """
        from sqlalchemy import text as sa_text

        rows = (await self.session.execute(sa_text("""
            SELECT extracted_figure_images
            FROM paper_analyses
            WHERE paper_id = :pid AND level = 'l2_parse' AND is_current = true
              AND extracted_figure_images IS NOT NULL
            ORDER BY created_at DESC LIMIT 1
        """), {"pid": str(paper_id)})).fetchall()

        if not rows or not rows[0][0]:
            return "(no figures extracted for this paper)"

        figs = rows[0][0] if isinstance(rows[0][0], list) else []
        lines = []
        for f in figs[:25]:
            label = f.get("label") or f.get("figure_num") or "?"
            role = f.get("semantic_role", "other")
            caption = (f.get("caption") or f.get("description") or "")[:140]
            lines.append(f"- {label} (role={role}): {caption}")
        return "\n".join(lines) if lines else "(no figures extracted)"

    def _build_paper_metadata_block(self, paper) -> str:
        """Format paper metadata for the metadata_overview section."""
        rows = []
        if paper.venue:
            v = paper.venue
            if getattr(paper, "acceptance_type", None):
                v = f"{v} ({paper.acceptance_type})"
            rows.append(f"venue: {v}")
        if paper.year:
            rows.append(f"year: {paper.year}")
        if getattr(paper, "arxiv_id", None):
            rows.append(f"arxiv: https://arxiv.org/abs/{paper.arxiv_id}")
        if paper.paper_link:
            rows.append(f"paper_link: {paper.paper_link}")
        if paper.code_url:
            rows.append(f"code_url: {paper.code_url}")
        if getattr(paper, "doi", None):
            rows.append(f"doi: {paper.doi}")
        if getattr(paper, "authors", None):
            authors = paper.authors if isinstance(paper.authors, list) else [paper.authors]
            rows.append(f"authors: {', '.join(str(a) for a in authors[:6])}")
        if getattr(paper, "method_family", None):
            rows.append(f"method_family: {paper.method_family}")
        if getattr(paper, "tags", None):
            tags = paper.tags if isinstance(paper.tags, list) else []
            if tags:
                rows.append(f"tags: {', '.join(tags[:8])}")
        return "\n".join(rows) if rows else "(no metadata)"

    # Standard KaTeX commands the agent may use. Macros NOT in this set are
    # likely paper-private \newcommand defs (e.g. \visionfeature) that the
    # LLM copied verbatim from the source TeX. They blow up Obsidian KaTeX
    # rendering, so we wrap them in \text{...} so they render as plain text
    # instead of breaking the whole equation.
    _KATEX_KNOWN = frozenset((
        # Greek
        "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
        "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi",
        "omicron", "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau",
        "upsilon", "phi", "varphi", "chi", "psi", "omega",
        "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
        "Phi", "Psi", "Omega",
        # Operators / structure
        "frac", "sqrt", "sum", "prod", "int", "oint", "lim", "limsup", "liminf",
        "sup", "inf", "max", "min", "argmax", "argmin", "log", "ln", "exp",
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
        "det", "dim", "ker", "deg", "gcd", "Pr", "mod",
        # Symbols
        "infty", "partial", "nabla", "forall", "exists", "in", "notin", "subset",
        "subseteq", "supset", "supseteq", "cup", "cap", "emptyset", "to", "mapsto",
        "rightarrow", "leftarrow", "leftrightarrow", "Rightarrow", "Leftarrow",
        "Leftrightarrow", "iff", "implies", "land", "lor", "lnot", "neg",
        "approx", "neq", "leq", "geq", "ll", "gg", "equiv", "sim", "simeq",
        "cong", "propto", "cdot", "cdots", "ldots", "vdots", "ddots", "dots",
        "times", "div", "pm", "mp", "circ", "bullet", "ast", "star", "oplus",
        "ominus", "otimes", "odot", "wedge", "vee", "perp", "parallel",
        # Style / wrappers
        "text", "textbf", "textit", "textrm", "texttt", "mathbf", "mathit",
        "mathrm", "mathsf", "mathtt", "mathcal", "mathbb", "mathfrak", "mathscr",
        "boldsymbol", "bm", "operatorname", "underbrace", "overbrace", "underline",
        "overline", "widehat", "widetilde", "hat", "bar", "tilde", "vec", "dot", "ddot",
        "color", "colorbox",
        # Layout
        "left", "right", "big", "Big", "bigg", "Bigg", "begin", "end",
        "quad", "qquad", "hspace", "vspace", "phantom", "vphantom", "hphantom",
        "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle",
        "label", "tag", "ref", "stackrel", "overset", "underset",
        # Brackets
        "langle", "rangle", "lceil", "rceil", "lfloor", "rfloor",
        # Newcommand we do not want to touch (frequent in legit reports)
        "begin{align}", "end{align}", "begin{pmatrix}", "end{pmatrix}",
        "begin{bmatrix}", "end{bmatrix}", "begin{cases}", "end{cases}",
        "begin{equation}", "end{equation}", "begin{aligned}", "end{aligned}",
        "begin{matrix}", "end{matrix}", "begin{Vmatrix}", "end{Vmatrix}",
        "&", "\\\\", "^", "_",
        # Common misc
        "triangleq", "triangleright", "triangleleft", "vartriangle",
        "leftrightharpoons", "rightleftharpoons",
        "rfloor", "lfloor", "vert", "Vert", "lvert", "rvert", "lVert", "rVert",
        "lbrace", "rbrace", "lbrack", "rbrack",
        "ne", "le", "ge", "lt", "gt",
        "1", "2", "3",  # \1 \2 etc capture groups (rare but possible)
        "newline", "linebreak",
    ))

    @staticmethod
    def _wrap_unknown_macros(formula: str) -> str:
        r"""Inside one ``$$...$$`` block, replace ``\X`` (X not a known KaTeX
        command) with ``\text{X}`` so KaTeX renders the name as plain text
        rather than throwing 'unknown command' and refusing the whole formula."""
        import re as _re
        def _repl(m: "_re.Match") -> str:
            name = m.group(1)
            if name in IngestWorkflow._KATEX_KNOWN:
                return m.group(0)
            # Wrap as plain text — preserves the name visibly, won't break render
            return f"\\text{{{name}}}"
        # Match \name where name is alphabetic ≥3 letters (skip short like \n, \t)
        return _re.sub(r"\\([A-Za-z]{3,})", _repl, formula)

    @staticmethod
    def _strip_placeholders(body: str) -> str:
        """Remove 'data missing' placeholder text the agent leaks into reports.

        The agent prompt says 'write 数据待补充 if missing', so it sprawls into
        bullet points, table cells, and prose ('（数据待补充具体数值）'). These
        add no information for the reader and look unprofessional in vault.

        Strategy: drop the parenthetical phrase wholesale; if the entire bullet
        / line is *only* the placeholder, drop the line.
        """
        if not body or "数据待补充" not in body:
            return body
        import re as _re
        # 1) Inline parenthetical forms: "（数据待补充xxx）" / "(数据待补充)" / "(数据待补充 …)"
        body = _re.sub(r"[（(]\s*数据待补充[^（()）]{0,80}[）)]", "", body)
        # 2) Standalone phrase
        body = _re.sub(r"数据待补充[^\n。；]{0,40}", "", body)
        # 3) Lines that are now empty bullets / table cells after stripping
        lines = body.split("\n")
        cleaned = []
        for ln in lines:
            stripped = ln.strip()
            # skip bullets that became empty (e.g. "- " or "- :")
            if _re.match(r"^[-*]\s*[:：]?\s*$", stripped):
                continue
            cleaned.append(ln)
        return "\n".join(cleaned)

    @staticmethod
    def _sanitize_latex(body: str) -> str:
        """Patch up common Kimi-K2.6 LaTeX malformations that break Obsidian KaTeX:
        1. Trailing `}}$$` (one extra closing brace before block-math delimiter).
        2. Paper-private macros (e.g. `\\visionfeature`) the LLM copied from the
           source .tex but never defined → wrap with `\\text{...}`.
        """
        body = IngestWorkflow._strip_placeholders(body)
        if not body or "$" not in body:
            return body
        import re as _re

        def _fix_block(m: "_re.Match") -> str:
            inner = m.group(1)
            opens = inner.count("{"); closes = inner.count("}")
            extra = closes - opens
            if extra > 0:
                while extra > 0 and inner.endswith("}"):
                    inner = inner[:-1]; extra -= 1
            return f"$${IngestWorkflow._wrap_unknown_macros(inner)}$$"

        def _fix_inline(m: "_re.Match") -> str:
            # Don't touch $$..$$ — those are handled by _fix_block; this only
            # matches single-$ inline math.
            return f"${IngestWorkflow._wrap_unknown_macros(m.group(1))}$"

        # 1) Block math first (so inline pass doesn't see $$ as two $)
        body = _re.sub(r"\$\$(.+?)\$\$", _fix_block, body, flags=_re.DOTALL)
        # 2) Inline math: $...$ that is NOT preceded/followed by another $.
        #    Pattern: bounded-length, no nested $, single-line.
        body = _re.sub(
            r"(?<!\$)\$(?!\$)([^\$\n]{1,400}?)\$(?!\$)",
            _fix_inline,
            body,
        )
        return body

    async def _persist_paper_report(self, paper_id: UUID, report_data: dict):
        """Persist paper_report agent output to PaperReport / PaperReportSection tables.

        Also backfills paper_analyses.full_report_md for vault export.
        figure_placements is appended as an HTML comment so the vault exporter
        can match {{FIG:xxx}} markers to actual figure labels (no schema change).
        """
        import json as _json
        from backend.models.kb import PaperReport, PaperReportSection
        from backend.models.analysis import PaperAnalysis
        from backend.models.enums import AnalysisLevel
        from sqlalchemy import select as sa_select

        title_zh = report_data.get("title_zh", "")
        title_en = report_data.get("title_en", "")
        sections = report_data.get("sections", [])
        figure_placements = report_data.get("figure_placements", []) or []

        if not sections:
            return

        # Create PaperReport
        report = PaperReport(
            paper_id=paper_id,
            title_zh=title_zh,
            title_en=title_en,
        )
        self.session.add(report)
        await self.session.flush()

        # Create sections
        for i, sec in enumerate(sections):
            section = PaperReportSection(
                report_id=report.id,
                section_type=sec.get("section_type", "unknown"),
                title=sec.get("title"),
                body_md=sec.get("body_md"),
                order_index=i,
            )
            self.session.add(section)

        # Backfill full_report_md on PaperAnalysis for vault export.
        # The metadata_overview section is rendered without a "## 概览" heading
        # because its body already contains the metadata table + TL;DR callout.
        full_md_parts = []
        for sec in sections:
            stype = sec.get("section_type", "")
            title = sec.get("title", "")
            body = sec.get("body_md", "")
            if not body:
                continue
            body = self._sanitize_latex(body)
            if stype == "metadata_overview":
                full_md_parts.append(body)
            elif title:
                full_md_parts.append(f"## {title}\n\n{body}")
            else:
                full_md_parts.append(body)

        full_report_md = "\n\n".join(full_md_parts)

        # Embed figure_placements as HTML comment for vault exporter
        if figure_placements:
            try:
                fp_json = _json.dumps(figure_placements, ensure_ascii=False)
                full_report_md += f"\n\n<!-- figure_placements: {fp_json} -->"
            except Exception as e:
                logger.debug("Failed to serialize figure_placements: %s", e)

        if full_report_md:
            l4 = (await self.session.execute(
                sa_select(PaperAnalysis).where(
                    PaperAnalysis.paper_id == paper_id,
                    PaperAnalysis.level == AnalysisLevel.L4_DEEP,
                    PaperAnalysis.is_current.is_(True),
                )
            )).scalar_one_or_none()

            if l4:
                l4.full_report_md = full_report_md
            else:
                # Create a new L4 analysis record
                l4_new = PaperAnalysis(
                    paper_id=paper_id,
                    level=AnalysisLevel.L4_DEEP,
                    model_provider="agent",
                    model_name="paper_report",
                    prompt_version="v7_unified",
                    schema_version="v2",
                    full_report_md=full_report_md,
                    is_current=True,
                )
                self.session.add(l4_new)

        await self.session.flush()
        logger.info(
            "Persisted paper report (%d sections) for paper %s",
            len(sections), paper_id,
        )

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

    # process_reference_roles — REMOVED
    # Known refs are handled by _handle_known_references in shallow_ingest.
    # Unknown refs are discovered by discover_neighborhood (single entry point).

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

    # ── Unified Pipeline ───────────────────────────────────────────

    async def run_unified_pipeline(
        self,
        source: str | dict,
        *,
        domain_id: UUID | None = None,
        force_ingest: bool = False,
    ) -> dict:
        """Run the complete unified pipeline end-to-end.

        Replaces both old pipeline_service.run_full_pipeline and run_full_v6_pipeline.

        Flow:
          1. import_and_score (candidate creation + scoring gate)
          2. shallow_ingest (promote + shallow agents)
          3. enrich_and_prepare (metadata + pdf + L2 parse)
          4. deep_ingest (deep agents + graph + report)
          5. discover_neighborhood (S2 refs/citations)

        Returns complete pipeline result dict.
        """
        result = {
            "source": source if isinstance(source, str) else str(source),
            "phases": {},
        }

        # Phase 1: Import & Score
        try:
            import_result = await self.import_and_score(
                source, domain_id=domain_id, force_ingest=force_ingest,
            )
            result["phases"]["import_and_score"] = import_result
        except Exception as e:
            logger.error("import_and_score failed: %s", e)
            result["phases"]["import_and_score"] = {"error": str(e)}
            return result

        candidate_id = import_result.get("candidate_id")
        decision = import_result.get("decision", "archive")

        if decision != "shallow_ingest":
            result["final_decision"] = decision
            return result

        if not candidate_id:
            result["final_decision"] = decision
            return result

        from uuid import UUID as _UUID
        cid = _UUID(candidate_id) if isinstance(candidate_id, str) else candidate_id

        # Phase 2: Shallow Ingest (promote + shallow agents)
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

        if not paper_id_str:
            result["final_decision"] = deep_decision
            return result

        pid = _UUID(paper_id_str) if isinstance(paper_id_str, str) else paper_id_str

        # Phase 2.5: Enrich & Prepare (metadata, pdf, L2 parse)
        try:
            prep_result = await self.enrich_and_prepare(pid)
            result["phases"]["enrich_and_prepare"] = prep_result
        except Exception as e:
            logger.error("enrich_and_prepare failed: %s", e)
            result["phases"]["enrich_and_prepare"] = {"error": str(e)}

        # Phase 3: Deep Ingest (if qualified, or force_ingest)
        if force_ingest:
            deep_decision = "auto_full_paper"

        if deep_decision in ("auto_full_paper", "full_review_needed"):
            try:
                deep_result = await self.deep_ingest(
                    pid, review_needed=(deep_decision == "full_review_needed"),
                )
                result["phases"]["deep_ingest"] = deep_result
            except Exception as e:
                logger.error("deep_ingest failed: %s", e)
                result["phases"]["deep_ingest"] = {"error": str(e)}

        # Phase 4: Neighborhood Discovery (S2 refs/citations, non-blocking)
        try:
            disc_result = await self.discover_neighborhood(
                pid, domain_id=domain_id,
            )
            result["phases"]["discover_neighborhood"] = disc_result
        except Exception as e:
            logger.error("discover_neighborhood failed: %s", e)
            result["phases"]["discover_neighborhood"] = {"error": str(e)}

        result["final_decision"] = deep_decision
        await self.session.commit()
        return result

    # ── Backward compat alias ──
    run_full_v6_pipeline = run_unified_pipeline
