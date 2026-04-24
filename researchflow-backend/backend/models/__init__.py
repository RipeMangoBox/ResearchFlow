"""ResearchFlow models — 40 tables across 4 layers.

Layer A: Taxonomy — taxonomy_nodes, taxonomy_edges, paper_facets, problem_nodes, problem_claims, paradigm_templates
Layer B: Method DAG — method_nodes, method_edges, method_applications, paradigm/slot/method_candidates
Layer C: Paper — papers, paper_assets, paper_versions, paper_analyses, delta_cards, evidence_units, delta_card_lineage, graph_nodes, graph_assertions, ...
Layer D: Cross-paper — canonical_ideas, contribution_to_canonical_idea
"""
from backend.models.enums import (
    AnalysisLevel,
    AssetType,
    EvidenceBasis,
    Importance,
    JobStatus,
    PaperState,
    PeriodType,
    Tier,
)
from backend.models.paper import Paper, PaperAsset, PaperVersion
from backend.models.analysis import PaperAnalysis, ParadigmTemplate
from backend.models.evidence import EvidenceUnit
from backend.models.research import ProjectBottleneck, PaperBottleneckClaim
from backend.models.digest import Digest
from backend.models.system import Job, ModelRun
from backend.models.delta_card import DeltaCard
from backend.models.assertion import GraphNode, GraphAssertion, GraphAssertionEvidence
from backend.models.review import HumanOverride, Alias, TaxonomyVersion
from backend.models.canonical_idea import CanonicalIdea, ContributionToCanonicalIdea
from backend.models.lineage import DeltaCardLineage
from backend.models.candidates import ParadigmCandidate, SlotCandidate, MethodCandidate
from backend.models.domain import DomainSpec, DomainSourceRegistry, IncrementalCheckpoint
from backend.models.metadata import MetadataObservation, CanonicalPaperMetadata
from backend.models.taxonomy import TaxonomyNode, TaxonomyEdge, PaperFacet, ProblemNode, ProblemClaim
from backend.models.method import MethodNode, MethodEdge, MethodApplication
from backend.models.candidate import PaperCandidate, CandidateScore, ScoreSignal
from backend.models.agent import AgentRun, AgentBlackboardItem, PaperExtraction, ReferenceRoleMap
from backend.models.kb import (
    GraphNodeCandidate, GraphEdgeCandidate, KBNodeProfile, KBEdgeProfile,
    PaperReport, PaperReportSection, ReviewQueueItem, EvidenceItem,
)

__all__ = [
    # Enums
    "AnalysisLevel", "AssetType", "EvidenceBasis",
    "Importance", "JobStatus", "PaperState", "PeriodType", "Tier",
    # Layer C: Paper
    "Paper", "PaperAsset", "PaperVersion",
    "PaperAnalysis", "ParadigmTemplate",
    "EvidenceUnit",
    "DeltaCard", "DeltaCardLineage",
    "ProjectBottleneck", "PaperBottleneckClaim",
    "GraphNode", "GraphAssertion", "GraphAssertionEvidence",
    # Layer A: Taxonomy
    "TaxonomyNode", "TaxonomyEdge", "PaperFacet", "ProblemNode", "ProblemClaim",
    # Layer B: Method DAG
    "MethodNode", "MethodEdge", "MethodApplication",
    "ParadigmCandidate", "SlotCandidate", "MethodCandidate",
    # Layer D: Cross-paper
    "CanonicalIdea", "ContributionToCanonicalIdea",
    # Discovery
    "PaperCandidate", "CandidateScore", "ScoreSignal",
    # Agent
    "AgentRun", "AgentBlackboardItem", "PaperExtraction", "ReferenceRoleMap",
    # KB
    "GraphNodeCandidate", "GraphEdgeCandidate", "KBNodeProfile", "KBEdgeProfile",
    "PaperReport", "PaperReportSection", "ReviewQueueItem", "EvidenceItem",
    # Metadata & Domain
    "MetadataObservation", "CanonicalPaperMetadata",
    "DomainSpec", "DomainSourceRegistry", "IncrementalCheckpoint",
    # Review
    "HumanOverride", "Alias", "TaxonomyVersion",
    # System
    "Job", "ModelRun", "Digest",
]
