"""Deterministic scoring engine for candidate papers, nodes, and edges.

LLM agents extract score_signals → this engine computes final scores.
All scores 0-100. Hard caps prevent inflation. Boosts reward important combos.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.models.domain import DomainSpec

logger = logging.getLogger(__name__)


# ── Result Container ────────────────────────────────────────────────

@dataclass
class ScoringResult:
    """Immutable output of any scoring method."""

    total: float                                  # 0-100, final score after caps/boosts
    raw_total: float                              # before caps/boosts
    breakdown: dict                               # {sub_score_name: value}
    hard_caps_applied: list[dict] = field(default_factory=list)   # [{cap_name, original, capped, reason}]
    boosts_applied: list[dict] = field(default_factory=list)      # [{boost_name, value, evidence}]
    penalties_applied: list[dict] = field(default_factory=list)
    decision: str | None = None                   # routing decision based on thresholds


# ── Lookup Tables ───────────────────────────────────────────────────

SOURCE_SIGNAL_TABLE: dict[str, int] = {
    "manual_import": 100,
    "method_section_citation": 95,
    "baseline_table": 90,
    "dataset_source": 90,
    "multi_anchor_cited": 85,
    "openreview_accepted": 80,
    "awesome_repo": 70,
    "dblp_proceedings": 65,
    "s2_reference": 60,
    "s2_reference_methodology": 80,
    "s2_reference_result": 75,
    "s2_reference_background": 45,
    "github_readme": 60,
    "s2_citation": 50,
    "hf_trending": 50,
    "s2_recommendation": 45,
    "hf_daily": 45,
    "openalex_topic": 40,
    "same_author": 40,
    "arxiv_search": 35,
    "keyword_search": 30,
}

RELATION_ROLE_TABLE: dict[str, int] = {
    "direct_baseline": 100,
    "method_source": 95,
    "formula_source": 95,
    "comparison_baseline": 90,
    "dataset_source": 90,
    "benchmark_source": 90,
    "multi_downstream": 90,
    "method_transfer": 85,
    "mechanism_proposer": 80,
    "survey_taxonomy": 75,
    "same_task_prior": 55,
    "related_work_mention": 50,
    "background_citation": 25,
    "unimportant": 10,
}


# ── Helpers ─────────────────────────────────────────────────────────

def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp *value* to [low, high]."""
    return max(low, min(high, value))


def _compute_citation_velocity(citation_count: int, months: float) -> float:
    """Map citation velocity (citations / month) to a 0-100 signal.

    Velocity brackets (per month):
        >= 20  → 100
        >= 10  → 85
        >= 5   → 70
        >= 2   → 55
        >= 1   → 40
        >= 0.3 → 25
        < 0.3  → 10
    """
    if months <= 0:
        months = 1.0  # avoid division by zero for very recent papers
    vel = citation_count / months
    if vel >= 20:
        return 100.0
    if vel >= 10:
        return 85.0
    if vel >= 5:
        return 70.0
    if vel >= 2:
        return 55.0
    if vel >= 1:
        return 40.0
    if vel >= 0.3:
        return 25.0
    return 10.0


def _compute_recency_score(months: float) -> float:
    """Exponential decay from 100 (0 months) toward 0 (very old).

    Half-life = 18 months.  Papers older than 60 months floor at 5.
    """
    if months <= 0:
        return 100.0
    score = 100.0 * math.exp(-0.693 * months / 18.0)  # ln2 ≈ 0.693
    return max(score, 5.0)


def _compute_artifact_score(
    has_code: bool,
    has_data: bool,
    has_model: bool,
    has_benchmark: bool,
) -> float:
    """Sum-of-artifacts score (each artifact contributes a fixed amount).

    code: 40, data: 25, model: 25, benchmark: 10.
    """
    total = 0.0
    if has_code:
        total += 40.0
    if has_data:
        total += 25.0
    if has_model:
        total += 25.0
    if has_benchmark:
        total += 10.0
    return _clamp(total)


# ── Scoring Engine ──────────────────────────────────────────────────

class ScoringEngine:
    """Pure-computation scoring engine.

    All methods are synchronous — they accept pre-extracted signal dicts
    and return ``ScoringResult``.  No DB / LLM access.
    """

    # ── Discovery Score ─────────────────────────────────────────────

    def compute_discovery_score(
        self,
        signals: dict,
        domain: DomainSpec | None = None,
        is_cold_start: bool = False,
    ) -> ScoringResult:
        """Score a newly-discovered paper for triage routing.

        Formula (warm start):
            0.25 * DomainMatch + 0.20 * SourceSignal + 0.20 * GraphProximity
          + 0.10 * ImpactSignal + 0.10 * ArtifactSignal + 0.10 * NoveltySignal
          + 0.05 * RecencySignal - Penalty

        Cold-start override:  DomainMatch weight = 0.35, GraphProximity = 0.05
        (remaining weights stay the same).
        """
        # ── sub-scores ──
        domain_match = self._compute_domain_match(signals)
        source_signal = self._lookup_source_signal(signals)
        graph_proximity = self._compute_graph_proximity(signals)
        months = float(signals.get("months_since_release", 0))
        citation_count = int(signals.get("citation_count", 0))
        citation_velocity = float(signals.get("citation_velocity", 0))
        impact_signal = self._compute_impact_signal(citation_count, citation_velocity, months)
        artifact_signal = _compute_artifact_score(
            signals.get("has_code", False),
            signals.get("has_data", False),
            signals.get("has_model", False),
            signals.get("has_benchmark", False),
        )
        novelty_signal = self._compute_novelty_signal(signals)
        recency_signal = _compute_recency_score(months)

        # ── weights ──
        if is_cold_start:
            w_domain, w_graph = 0.35, 0.05
        else:
            w_domain, w_graph = 0.25, 0.20

        raw_total = (
            w_domain * domain_match
            + 0.20 * source_signal
            + w_graph * graph_proximity
            + 0.10 * impact_signal
            + 0.10 * artifact_signal
            + 0.10 * novelty_signal
            + 0.05 * recency_signal
        )

        breakdown = {
            "DomainMatch": domain_match,
            "SourceSignal": source_signal,
            "GraphProximity": graph_proximity,
            "ImpactSignal": impact_signal,
            "ArtifactSignal": artifact_signal,
            "NoveltySignal": novelty_signal,
            "RecencySignal": recency_signal,
        }

        # ── penalties ──
        penalties: list[dict] = []
        penalty_value = 0.0
        redundancy = float(signals.get("redundancy", 0))
        if redundancy > 0.5:
            p = redundancy * 15.0
            penalty_value += p
            penalties.append({"name": "redundancy", "value": p, "reason": f"redundancy={redundancy:.2f}"})

        raw_total -= penalty_value

        # ── caps & boosts ──
        caps: list[dict] = []
        boosts: list[dict] = []

        # boost: baseline_table_discovered
        source_type = signals.get("source_type", "")
        if source_type == "baseline_table":
            boosts.append({"boost_name": "baseline_table_discovered", "value": 5, "evidence": "source=baseline_table"})

        total = raw_total + sum(b["value"] for b in boosts)
        total = _clamp(total)

        # ── decision ──
        decision = self.route_discovery(total)

        return ScoringResult(
            total=round(total, 2),
            raw_total=round(raw_total, 2),
            breakdown=breakdown,
            hard_caps_applied=caps,
            boosts_applied=boosts,
            penalties_applied=penalties,
            decision=decision,
        )

    # ── Deep Ingest Score ───────────────────────────────────────────

    def compute_deep_ingest_score(self, signals: dict) -> ScoringResult:
        """Score a paper for deep-ingest priority.

        Formula:
            0.22 * DomainFit + 0.28 * RelationRole + 0.18 * ReusableKnowledge
          + 0.12 * EvidenceQuality + 0.10 * ExperimentValue + 0.06 * ArtifactValue
          + 0.04 * NoveltyFreshness - Penalty
        """
        domain_fit = float(signals.get("domain_fit", 0))
        relation_role = self._lookup_relation_role(signals)
        reusable_knowledge = float(signals.get("reusable_knowledge", 0))
        evidence_quality = float(signals.get("evidence_quality", 0))
        experiment_value = float(signals.get("experiment_value", 0))
        artifact_value = _compute_artifact_score(
            signals.get("has_code", False),
            signals.get("has_data", False),
            signals.get("has_model", False),
            signals.get("has_benchmark", False),
        )
        novelty_freshness = float(signals.get("novelty_freshness", 0))

        raw_total = (
            0.22 * domain_fit
            + 0.28 * relation_role
            + 0.18 * reusable_knowledge
            + 0.12 * evidence_quality
            + 0.10 * experiment_value
            + 0.06 * artifact_value
            + 0.04 * novelty_freshness
        )

        breakdown = {
            "DomainFit": domain_fit,
            "RelationRole": relation_role,
            "ReusableKnowledge": reusable_knowledge,
            "EvidenceQuality": evidence_quality,
            "ExperimentValue": experiment_value,
            "ArtifactValue": artifact_value,
            "NoveltyFreshness": novelty_freshness,
        }

        # ── penalties ──
        penalties: list[dict] = []
        penalty_value = 0.0
        redundancy = float(signals.get("redundancy", 0))
        if redundancy > 0.5:
            p = redundancy * 15.0
            penalty_value += p
            penalties.append({"name": "redundancy", "value": p, "reason": f"redundancy={redundancy:.2f}"})

        raw_total -= penalty_value

        # ── hard caps ──
        caps: list[dict] = []
        source_type = signals.get("source_type", "")

        # low_domain_fit cap
        if domain_fit < 50 and source_type != "manual_import":
            if raw_total > 60:
                caps.append({
                    "cap_name": "low_domain_fit",
                    "original": round(raw_total, 2),
                    "capped": 60,
                    "reason": "DomainFit < 50 and not manual_import",
                })
                raw_total = 60.0

        # high_redundancy cap
        if redundancy > 0.7:
            if raw_total > 70:
                caps.append({
                    "cap_name": "high_redundancy",
                    "original": round(raw_total, 2),
                    "capped": 70,
                    "reason": f"redundancy {redundancy:.2f} > 0.7",
                })
                raw_total = 70.0

        # ── boosts ──
        boosts: list[dict] = []
        relation = signals.get("relation_type", "")

        # direct_baseline + experiment_table + same_task
        if (signals.get("is_direct_baseline") and signals.get("in_experiment_table")
                and signals.get("same_primary_task")):
            boosts.append({"boost_name": "direct_baseline_experiment_same_task", "value": 10,
                           "evidence": "direct baseline in experiment table of same task"})

        # baseline + changed_slot + ablation
        if (signals.get("is_baseline") and signals.get("has_changed_slots")
                and signals.get("has_ablation")):
            boosts.append({"boost_name": "baseline_changed_slot_ablation", "value": 12,
                           "evidence": "baseline with changed slots and ablation support"})

        # low_citation + official_code + strong_ablation + graph_gap
        if (signals.get("citation_count", 999) < 10 and signals.get("has_code")
                and signals.get("has_strong_ablation") and signals.get("fills_graph_gap")):
            boosts.append({"boost_name": "low_citation_high_quality", "value": 8,
                           "evidence": "low citation but high quality with graph gap"})

        # dataset_with_multiple_users
        if signals.get("dataset_user_count", 0) > 1 and relation in ("dataset_source", "benchmark_source"):
            boosts.append({"boost_name": "dataset_with_multiple_users", "value": 5,
                           "evidence": "dataset used by >1 paper"})

        # method_transfer_new_domain
        if relation == "method_transfer" and signals.get("cross_domain", False):
            boosts.append({"boost_name": "method_transfer_new_domain", "value": 8,
                           "evidence": "method transferred to new domain"})

        total = raw_total + sum(b["value"] for b in boosts)
        total = _clamp(total)

        decision = self.route_deep_ingest(total)

        return ScoringResult(
            total=round(total, 2),
            raw_total=round(raw_total, 2),
            breakdown=breakdown,
            hard_caps_applied=caps,
            boosts_applied=boosts,
            penalties_applied=penalties,
            decision=decision,
        )

    # ── Node Promotion Score ────────────────────────────────────────

    def compute_node_promotion_score(self, signals: dict) -> ScoringResult:
        """Score a knowledge-graph node for promotion.

        Formula:
            0.25 * EvidenceCount + 0.20 * ConnectedPaperQuality
          + 0.20 * SourceDiversity + 0.15 * StructuralImportance
          + 0.10 * NameStability + 0.10 * ProfileCompleteness
          - DuplicatePenalty
        """
        evidence_count = _clamp(float(signals.get("evidence_count", 0)))
        connected_paper_quality = _clamp(float(signals.get("connected_paper_quality", 0)))
        source_diversity = _clamp(float(signals.get("source_diversity", 0)))
        structural_importance = _clamp(float(signals.get("structural_importance", 0)))
        name_stability = _clamp(float(signals.get("name_stability", 0)))
        profile_completeness = _clamp(float(signals.get("profile_completeness", 0)))

        raw_total = (
            0.25 * evidence_count
            + 0.20 * connected_paper_quality
            + 0.20 * source_diversity
            + 0.15 * structural_importance
            + 0.10 * name_stability
            + 0.10 * profile_completeness
        )

        breakdown = {
            "EvidenceCount": evidence_count,
            "ConnectedPaperQuality": connected_paper_quality,
            "SourceDiversity": source_diversity,
            "StructuralImportance": structural_importance,
            "NameStability": name_stability,
            "ProfileCompleteness": profile_completeness,
        }

        # ── duplicate penalty ──
        penalties: list[dict] = []
        dup = float(signals.get("duplicate_score", 0))
        if dup > 0:
            p = dup * 20.0
            raw_total -= p
            penalties.append({"name": "duplicate", "value": p, "reason": f"duplicate_score={dup:.2f}"})

        total = _clamp(raw_total)
        decision = self.route_node_promotion(total)

        return ScoringResult(
            total=round(total, 2),
            raw_total=round(raw_total, 2),
            breakdown=breakdown,
            hard_caps_applied=[],
            boosts_applied=[],
            penalties_applied=penalties,
            decision=decision,
        )

    # ── Edge Confidence Score ───────────────────────────────────────

    def compute_edge_confidence_score(self, signals: dict) -> ScoringResult:
        """Score an edge for confidence / visibility.

        Formula:
            0.35 * EvidenceDirectness + 0.20 * RelationSpecificity
          + 0.15 * ExtractorAgreement + 0.15 * SourceReliability
          + 0.10 * GraphConsistency + 0.05 * DescriptionQuality
          - Penalty
        """
        evidence_directness = _clamp(float(signals.get("evidence_directness", 0)))
        relation_specificity = _clamp(float(signals.get("relation_specificity", 0)))
        extractor_agreement = _clamp(float(signals.get("extractor_agreement", 0)))
        source_reliability = _clamp(float(signals.get("source_reliability", 0)))
        graph_consistency = _clamp(float(signals.get("graph_consistency", 0)))
        description_quality = _clamp(float(signals.get("description_quality", 0)))

        raw_total = (
            0.35 * evidence_directness
            + 0.20 * relation_specificity
            + 0.15 * extractor_agreement
            + 0.15 * source_reliability
            + 0.10 * graph_consistency
            + 0.05 * description_quality
        )

        breakdown = {
            "EvidenceDirectness": evidence_directness,
            "RelationSpecificity": relation_specificity,
            "ExtractorAgreement": extractor_agreement,
            "SourceReliability": source_reliability,
            "GraphConsistency": graph_consistency,
            "DescriptionQuality": description_quality,
        }

        # ── penalties ──
        penalties: list[dict] = []
        relation = signals.get("relation_type", "")

        # related_work_only cap on relation_role
        if relation == "related_work_mention":
            penalties.append({"name": "related_work_only", "value": 0, "reason": "relation=related_work_mention"})

        # background_only cap
        if relation == "background_citation":
            penalties.append({"name": "background_only", "value": 0, "reason": "relation=background_citation"})

        # ── hard caps ──
        caps: list[dict] = []

        # no_evidence cap
        evidence_refs = signals.get("evidence_refs", [])
        if not evidence_refs:
            if raw_total > 55:
                caps.append({
                    "cap_name": "no_evidence",
                    "original": round(raw_total, 2),
                    "capped": 55,
                    "reason": "no evidence_refs provided",
                })
                raw_total = 55.0

        # no_method_evidence cap
        has_method_evidence = signals.get("has_method_evidence", False)
        edge_type = signals.get("edge_type", "")
        if not has_method_evidence and edge_type in ("modifies_slot", "extends"):
            caps.append({
                "cap_name": "no_method_evidence",
                "original": round(raw_total, 2),
                "capped": round(raw_total, 2),
                "reason": "modifies_slot/extends cannot be canonical without method evidence",
            })
            # does not numerically cap the score, but prevents canonical routing
            # (handled in route_edge via the cap record)

        total = _clamp(raw_total)
        decision = self.route_edge(total)

        # override decision if no_method_evidence cap is present and would be canonical
        if any(c["cap_name"] == "no_method_evidence" for c in caps) and decision == "canonical":
            decision = "visible_review"

        return ScoringResult(
            total=round(total, 2),
            raw_total=round(raw_total, 2),
            breakdown=breakdown,
            hard_caps_applied=caps,
            boosts_applied=[],
            penalties_applied=penalties,
            decision=decision,
        )

    # ── Anchor Score ────────────────────────────────────────────────

    def compute_anchor_score(self, signals: dict) -> ScoringResult:
        """Score a paper's suitability as an anchor node.

        Formula:
            0.25 * DownstreamCount + 0.20 * StructuralImportance
          + 0.20 * GraphCentrality + 0.15 * EvidenceConsensus
          + 0.10 * IsEstablishedBaseline + 0.10 * CommunityRecognition
        """
        downstream_count = _clamp(float(signals.get("downstream_count", 0)))
        structural_importance = _clamp(float(signals.get("structural_importance", 0)))
        graph_centrality = _clamp(float(signals.get("graph_centrality", 0)))
        evidence_consensus = _clamp(float(signals.get("evidence_consensus", 0)))
        is_established_baseline = _clamp(float(signals.get("is_established_baseline", 0)))
        community_recognition = _clamp(float(signals.get("community_recognition", 0)))

        raw_total = (
            0.25 * downstream_count
            + 0.20 * structural_importance
            + 0.20 * graph_centrality
            + 0.15 * evidence_consensus
            + 0.10 * is_established_baseline
            + 0.10 * community_recognition
        )

        breakdown = {
            "DownstreamCount": downstream_count,
            "StructuralImportance": structural_importance,
            "GraphCentrality": graph_centrality,
            "EvidenceConsensus": evidence_consensus,
            "IsEstablishedBaseline": is_established_baseline,
            "CommunityRecognition": community_recognition,
        }

        total = _clamp(raw_total)

        return ScoringResult(
            total=round(total, 2),
            raw_total=round(raw_total, 2),
            breakdown=breakdown,
        )

    # ── Decision Routing ────────────────────────────────────────────

    @staticmethod
    def route_discovery(score: float) -> str:
        """Route a discovery score to a triage bucket.

        Thresholds: 75 / 60 / 40.
        """
        if score >= 75:
            return "shallow_ingest"
        if score >= 60:
            return "candidate_pool"
        if score >= 40:
            return "metadata_only"
        return "archive"

    @staticmethod
    def route_deep_ingest(score: float) -> str:
        """Route a deep-ingest score to a processing tier.

        Thresholds: 88 / 80 / 68 / 50.
        """
        if score >= 88:
            return "auto_full_paper"
        if score >= 80:
            return "full_review_needed"
        if score >= 68:
            return "shallow_card"
        if score >= 50:
            return "candidate_card"
        return "no_graph"

    @staticmethod
    def route_node_promotion(score: float) -> str:
        """Route a node promotion score to a visibility tier.

        Thresholds: 85 / 75 / 60.
        """
        if score >= 85:
            return "canonical"
        if score >= 75:
            return "reviewed_candidate"
        if score >= 60:
            return "hidden_candidate"
        return "mention_only"

    @staticmethod
    def route_edge(score: float) -> str:
        """Route an edge confidence score to a visibility tier.

        Thresholds: 85 / 70 / 55.
        """
        if score >= 85:
            return "canonical"
        if score >= 70:
            return "visible_review"
        if score >= 55:
            return "weak"
        return "mention_only"

    # ── Internal Sub-score Computation ──────────────────────────────

    @staticmethod
    def _compute_domain_match(signals: dict) -> float:
        """Compute DomainMatch from keyword hits, task match, modality match.

        DomainMatch = 40 * keyword_ratio + 35 * task_match + 25 * modality_match
        where keyword_ratio = min(keyword_hits / 3, 1.0).
        """
        keyword_hits = int(signals.get("domain_keyword_hits", 0))
        task_match = float(signals.get("task_match", 0))         # 0-1
        modality_match = float(signals.get("modality_match", 0))  # 0-1

        keyword_ratio = min(keyword_hits / 3.0, 1.0)
        score = 40.0 * keyword_ratio + 35.0 * task_match + 25.0 * modality_match
        return _clamp(score)

    @staticmethod
    def _lookup_source_signal(signals: dict) -> float:
        """Look up source type in the SourceSignal table.

        If multiple sources, take the max.
        """
        source_type = signals.get("source_type", "")
        source_types = signals.get("source_types", [])

        # single source
        if source_type and not source_types:
            return float(SOURCE_SIGNAL_TABLE.get(source_type, 20))

        # multiple sources — take max
        if source_types:
            values = [SOURCE_SIGNAL_TABLE.get(s, 20) for s in source_types]
            return float(max(values)) if values else 20.0

        return 20.0

    @staticmethod
    def _compute_graph_proximity(signals: dict) -> float:
        """Compute GraphProximity from connected anchor count and hop distance.

        anchors contribution: min(anchor_count * 25, 70)
        hop penalty: 30 * (1 / hop_distance) if hop <= 3, else 0
        """
        anchor_count = int(signals.get("connected_anchor_count", 0))
        hop_distance = int(signals.get("hop_distance", 0))

        anchor_part = min(anchor_count * 25.0, 70.0)
        if 0 < hop_distance <= 3:
            hop_part = 30.0 * (1.0 / hop_distance)
        else:
            hop_part = 0.0

        return _clamp(anchor_part + hop_part)

    @staticmethod
    def _compute_impact_signal(
        citation_count: int,
        citation_velocity: float,
        months: float,
    ) -> float:
        """Combine citation count bracket and velocity into a 0-100 signal.

        50% from velocity, 50% from count bracket.
        """
        vel_score = _compute_citation_velocity(citation_count, months)

        # count bracket
        if citation_count >= 500:
            count_score = 100.0
        elif citation_count >= 200:
            count_score = 85.0
        elif citation_count >= 50:
            count_score = 65.0
        elif citation_count >= 10:
            count_score = 45.0
        elif citation_count >= 1:
            count_score = 25.0
        else:
            count_score = 5.0

        return _clamp(0.5 * vel_score + 0.5 * count_score)

    @staticmethod
    def _compute_novelty_signal(signals: dict) -> float:
        """Compute NoveltySignal from graph gap, new mechanism, new setting.

        fills_graph_gap: 45, new_mechanism: 35, new_setting: 20.
        """
        total = 0.0
        if signals.get("fills_graph_gap", False):
            total += 45.0
        if signals.get("new_mechanism", False):
            total += 35.0
        if signals.get("new_setting", False):
            total += 20.0
        return _clamp(total)

    @staticmethod
    def _lookup_relation_role(signals: dict) -> float:
        """Look up relation type in the RelationRole table."""
        relation = signals.get("relation_type", "")
        return float(RELATION_ROLE_TABLE.get(relation, 20))
