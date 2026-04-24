"""Intent-based query router — routes user queries to the best retrieval path.

Four primary intents:
  A. bottleneck — "这个方向真正卡在哪"
  B. mechanism  — "某类机制有哪些路线"
  C. lineage    — "这个方法是怎么一步步长出来的"
  D. evidence   — "证据在哪、代码在哪、改了会怎样"

Negative constraints are first-class: must_not, min_structurality, must_have_open_code, etc.
"""

import logging
import re
from uuid import UUID

from sqlalchemy import and_, desc, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.assertion import GraphAssertion, GraphNode
from backend.models.delta_card import DeltaCard
from backend.models.evidence import EvidenceUnit
from backend.models.graph import IdeaDelta, ImplementationUnit
from backend.models.method import MethodNode
from backend.models.lineage import DeltaCardLineage
from backend.models.paper import Paper
from backend.models.research import ProjectBottleneck, PaperBottleneckClaim

logger = logging.getLogger(__name__)

# ── Intent keywords ──────────────────────────────────────────────

_BOTTLENECK_SIGNALS = {
    "bottleneck", "limitation", "challenge", "problem", "obstacle",
    "卡在", "瓶颈", "难点", "挑战", "局限",
}
_MECHANISM_SIGNALS = {
    "mechanism", "approach", "method", "technique", "路线",
    "机制", "方法", "方案", "手段", "策略",
}
_LINEAGE_SIGNALS = {
    "lineage", "evolution", "history", "builds on", "based on",
    "演化", "发展", "历史", "衍生", "改进自",
}
_EVIDENCE_SIGNALS = {
    "evidence", "proof", "ablation", "experiment", "code", "implementation",
    "证据", "实验", "消融", "代码", "实现",
}


def _keyword_classify(query: str) -> tuple[str, int]:
    """Keyword-based intent classification. Returns (intent, score)."""
    q = query.lower()
    scores = {
        "bottleneck": sum(1 for w in _BOTTLENECK_SIGNALS if w in q),
        "mechanism": sum(1 for w in _MECHANISM_SIGNALS if w in q),
        "lineage": sum(1 for w in _LINEAGE_SIGNALS if w in q),
        "evidence": sum(1 for w in _EVIDENCE_SIGNALS if w in q),
    }
    best = max(scores, key=scores.get)
    return best, scores[best]


# Intent descriptions for embedding-based classification
_INTENT_DESCRIPTIONS = {
    "bottleneck": "What is the core limitation, challenge, or bottleneck blocking progress in this research direction? What problems are hard to solve?",
    "mechanism": "What methods, techniques, approaches, or mechanisms are available? What are the different routes and strategies?",
    "lineage": "How did this method evolve over time? What is it based on? What builds on it? Show the development history and inheritance.",
    "evidence": "Where is the experimental evidence? Show me ablation results, code implementations, proofs, and concrete measurements.",
}

_intent_embeddings_cache: dict[str, list[float]] | None = None


async def _get_intent_embeddings() -> dict[str, list[float]]:
    """Lazily compute and cache embeddings for intent descriptions."""
    global _intent_embeddings_cache
    if _intent_embeddings_cache is not None:
        return _intent_embeddings_cache

    from backend.services.embedding_service import embed_text
    _intent_embeddings_cache = {}
    for intent, desc in _INTENT_DESCRIPTIONS.items():
        _intent_embeddings_cache[intent] = await embed_text(desc)
    return _intent_embeddings_cache


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def classify_intent(query: str) -> str:
    """Classify a query into one of 4 intents.

    Strategy: keyword match first; if no keywords hit, use embedding similarity.
    """
    # 1. Try keyword match (fast, reliable for explicit queries)
    keyword_intent, keyword_score = _keyword_classify(query)
    if keyword_score >= 2:
        return keyword_intent

    # 2. Fall back to embedding similarity (handles mixed language, implicit queries)
    try:
        from backend.services.embedding_service import embed_text
        query_emb = await embed_text(query)
        intent_embs = await _get_intent_embeddings()

        best_intent = "bottleneck"
        best_sim = -1.0
        for intent, emb in intent_embs.items():
            sim = _cosine_sim(query_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_intent = intent

        # If keyword had a weak signal (score=1), boost that intent
        if keyword_score == 1:
            # Keyword match breaks ties
            return keyword_intent

        return best_intent
    except Exception as e:
        logger.warning(f"Embedding-based intent classification failed: {e}")
        # Final fallback: keyword result or default
        return keyword_intent if keyword_score > 0 else "bottleneck"


# ── Negative constraint application ──────────────────────────────

def _apply_paper_constraints(
    stmt,
    must_not_method_categories: list[str] | None = None,
    min_structurality_score: float | None = None,
    must_have_open_code: bool | None = None,
    must_have_evidence_count: int | None = None,
    exclude_tags: list[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
):
    """Apply negative and positive constraints to a paper query."""
    if min_structurality_score is not None:
        stmt = stmt.where(Paper.structurality_score >= min_structurality_score)
    if must_have_open_code:
        stmt = stmt.where(or_(Paper.open_code.is_(True), Paper.code_url.isnot(None)))
    if year_min:
        stmt = stmt.where(Paper.year >= year_min)
    if year_max:
        stmt = stmt.where(Paper.year <= year_max)
    if must_not_method_categories:
        for cat in must_not_method_categories:
            stmt = stmt.where(~Paper.tags.contains([f"method/{cat}"]))
    if exclude_tags:
        for tag in exclude_tags:
            stmt = stmt.where(~Paper.tags.contains([tag]))
    return stmt


def _apply_idea_constraints(
    stmt,
    min_structurality_score: float | None = None,
    must_have_evidence_count: int | None = None,
):
    """Apply constraints to IdeaDelta queries."""
    if min_structurality_score is not None:
        stmt = stmt.where(IdeaDelta.structurality_score >= min_structurality_score)
    if must_have_evidence_count is not None:
        stmt = stmt.where(IdeaDelta.evidence_count >= must_have_evidence_count)
    return stmt


# ── Route A: Bottleneck Query ────────────────────────────────────

async def query_bottleneck(
    session: AsyncSession,
    query: str,
    constraints: dict | None = None,
    limit: int = 20,
) -> dict:
    """What's really blocking progress in this direction?

    Recall priority: project_focus_bottlenecks → paper_bottleneck_claims
    → delta_cards → canonical_ideas
    """
    c = constraints or {}

    # 1. Search project_focus_bottlenecks
    from backend.models.research import ProjectFocusBottleneck
    focus_bns = await session.execute(
        select(ProjectFocusBottleneck, ProjectBottleneck)
        .join(ProjectBottleneck, ProjectFocusBottleneck.bottleneck_id == ProjectBottleneck.id)
        .where(
            ProjectFocusBottleneck.status == "active",
            or_(
                ProjectBottleneck.title.ilike(f"%{query}%"),
                ProjectBottleneck.description.ilike(f"%{query}%"),
            ),
        )
        .limit(5)
    )
    focus_results = []
    for fb, bn in focus_bns:
        focus_results.append({
            "source": "project_focus",
            "bottleneck_id": str(bn.id),
            "title": bn.title,
            "description": bn.description,
            "priority": fb.priority,
            "negative_constraints": fb.negative_constraints,
        })

    # 2. Search paper_bottleneck_claims
    claim_stmt = (
        select(PaperBottleneckClaim, ProjectBottleneck, Paper)
        .join(ProjectBottleneck, PaperBottleneckClaim.bottleneck_id == ProjectBottleneck.id)
        .join(Paper, PaperBottleneckClaim.paper_id == Paper.id)
        .where(
            or_(
                ProjectBottleneck.title.ilike(f"%{query}%"),
                PaperBottleneckClaim.claim_text.ilike(f"%{query}%"),
            ),
        )
        .order_by(desc(PaperBottleneckClaim.confidence))
        .limit(limit)
    )
    claim_stmt = _apply_paper_constraints(
        claim_stmt,
        must_not_method_categories=c.get("must_not_method_categories"),
        min_structurality_score=c.get("min_structurality_score"),
        must_have_open_code=c.get("must_have_open_code"),
        year_min=c.get("year_min"), year_max=c.get("year_max"),
        exclude_tags=c.get("exclude_tags"),
    )
    claims = await session.execute(claim_stmt)
    claim_results = []
    for claim, bn, paper in claims:
        claim_results.append({
            "source": "paper_claim",
            "paper_id": str(paper.id),
            "paper_title": paper.title,
            "bottleneck_title": bn.title,
            "claim_text": claim.claim_text[:200],
            "is_fundamental": claim.is_fundamental,
            "confidence": claim.confidence,
        })

    # 3. Search delta_cards for bottleneck context
    dc_stmt = (
        select(DeltaCard, Paper)
        .join(Paper, DeltaCard.paper_id == Paper.id)
        .where(DeltaCard.delta_statement.ilike(f"%{query}%"))
        .order_by(desc(DeltaCard.structurality_score))
        .limit(limit)
    )
    if c.get("min_structurality_score"):
        dc_stmt = dc_stmt.where(DeltaCard.structurality_score >= c["min_structurality_score"])
    dc_result = await session.execute(dc_stmt)
    delta_results = []
    for dc, paper in dc_result:
        delta_results.append({
            "source": "delta_card",
            "paper_id": str(paper.id),
            "paper_title": paper.title,
            "delta_statement": dc.delta_statement[:200],
            "structurality_score": dc.structurality_score,
        })

    return {
        "intent": "bottleneck",
        "query": query,
        "project_focus": focus_results,
        "paper_claims": claim_results[:limit],
        "delta_cards": delta_results[:limit],
    }


# ── Route B: Mechanism Query ────────────────────────────────────

async def query_mechanism(
    session: AsyncSession,
    query: str,
    constraints: dict | None = None,
    limit: int = 20,
) -> dict:
    """What approaches exist for this mechanism family?

    Recall priority: canonical_ideas → method_nodes → paper_contributions
    """
    c = constraints or {}

    # 1. Search canonical_ideas
    from backend.models.canonical_idea import CanonicalIdea
    ci_stmt = (
        select(CanonicalIdea)
        .where(
            CanonicalIdea.status != "deprecated",
            or_(
                CanonicalIdea.title.ilike(f"%{query}%"),
                CanonicalIdea.description.ilike(f"%{query}%"),
            ),
        )
        .order_by(desc(CanonicalIdea.contribution_count))
        .limit(limit)
    )
    ci_result = await session.execute(ci_stmt)
    canonical_results = [
        {
            "source": "canonical_idea",
            "id": str(ci.id),
            "title": ci.title,
            "description": ci.description[:200],
            "contribution_count": ci.contribution_count,
            "domain": ci.domain,
            "status": ci.status,
        }
        for ci in ci_result.scalars()
    ]

    # 2. Search method_nodes
    mfs = await session.execute(
        select(MethodNode).where(
            or_(
                MethodNode.name.ilike(f"%{query}%"),
                MethodNode.description.ilike(f"%{query}%"),
            )
        ).limit(10)
    )
    mechanism_results = [
        {
            "source": "method_family",
            "id": str(mf.id),
            "name": mf.name,
            "domain": mf.domain,
            "description": mf.description,
        }
        for mf in mfs.scalars()
    ]

    # 3. Search paper contributions (IdeaDeltas)
    idea_stmt = (
        select(IdeaDelta, Paper)
        .join(Paper, IdeaDelta.paper_id == Paper.id)
        .where(IdeaDelta.delta_statement.ilike(f"%{query}%"))
        .order_by(desc(IdeaDelta.structurality_score))
        .limit(limit)
    )
    idea_stmt = _apply_idea_constraints(
        idea_stmt,
        min_structurality_score=c.get("min_structurality_score"),
        must_have_evidence_count=c.get("must_have_evidence_count"),
    )
    idea_stmt = _apply_paper_constraints(
        idea_stmt,
        must_not_method_categories=c.get("must_not_method_categories"),
        must_have_open_code=c.get("must_have_open_code"),
        year_min=c.get("year_min"), year_max=c.get("year_max"),
        exclude_tags=c.get("exclude_tags"),
    )
    idea_result = await session.execute(idea_stmt)
    contribution_results = []
    for idea, paper in idea_result:
        contribution_results.append({
            "source": "paper_contribution",
            "idea_delta_id": str(idea.id),
            "paper_id": str(paper.id),
            "paper_title": paper.title,
            "delta_statement": idea.delta_statement[:200],
            "structurality_score": idea.structurality_score,
            "evidence_count": idea.evidence_count,
            "publish_status": idea.publish_status,
        })

    return {
        "intent": "mechanism",
        "query": query,
        "canonical_ideas": canonical_results,
        "method_nodes": mechanism_results,
        "paper_contributions": contribution_results[:limit],
    }


# ── Route C: Lineage Query ──────────────────────────────────────

async def query_lineage(
    session: AsyncSession,
    query: str,
    constraints: dict | None = None,
    limit: int = 20,
) -> dict:
    """How did this method evolve step by step?

    Recall priority: delta_card_lineage → builds_on assertions → established_baselines
    """
    # Find DeltaCards matching query
    dc_stmt = (
        select(DeltaCard, Paper)
        .join(Paper, DeltaCard.paper_id == Paper.id)
        .where(
            or_(
                Paper.title.ilike(f"%{query}%"),
                DeltaCard.delta_statement.ilike(f"%{query}%"),
                DeltaCard.baseline_paradigm.ilike(f"%{query}%"),
            ),
            DeltaCard.status != "deprecated",
        )
        .order_by(desc(DeltaCard.downstream_count))
        .limit(10)
    )
    dc_result = await session.execute(dc_stmt)
    cards = list(dc_result)

    lineage_trees = []
    for dc, paper in cards[:5]:
        # Get ancestors from lineage table
        ancestors = await session.execute(
            select(DeltaCardLineage, DeltaCard, Paper)
            .join(DeltaCard, DeltaCardLineage.parent_delta_card_id == DeltaCard.id)
            .join(Paper, DeltaCard.paper_id == Paper.id)
            .where(DeltaCardLineage.child_delta_card_id == dc.id)
            .limit(10)
        )
        ancestor_list = [
            {
                "relation_type": ln.relation_type,
                "confidence": ln.confidence,
                "status": ln.status,
                "parent_paper_title": p.title,
                "parent_delta_statement": pdc.delta_statement[:150],
            }
            for ln, pdc, p in ancestors
        ]

        # Get descendants
        descendants = await session.execute(
            select(DeltaCardLineage, DeltaCard, Paper)
            .join(DeltaCard, DeltaCardLineage.child_delta_card_id == DeltaCard.id)
            .join(Paper, DeltaCard.paper_id == Paper.id)
            .where(DeltaCardLineage.parent_delta_card_id == dc.id)
            .limit(10)
        )
        descendant_list = [
            {
                "relation_type": ln.relation_type,
                "child_paper_title": p.title,
                "child_delta_statement": cdc.delta_statement[:150],
            }
            for ln, cdc, p in descendants
        ]

        lineage_trees.append({
            "delta_card_id": str(dc.id),
            "paper_title": paper.title,
            "delta_statement": dc.delta_statement[:200],
            "lineage_depth": dc.lineage_depth,
            "downstream_count": dc.downstream_count,
            "is_established_baseline": dc.is_established_baseline,
            "ancestors": ancestor_list,
            "descendants": descendant_list,
        })

    return {
        "intent": "lineage",
        "query": query,
        "lineage_trees": lineage_trees,
    }


# ── Route D: Evidence / Implementation Query ────────────────────

async def query_evidence(
    session: AsyncSession,
    query: str,
    constraints: dict | None = None,
    limit: int = 20,
) -> dict:
    """Where's the evidence? Where's the code?

    Recall priority: evidence_units → implementation_units → delta_cards
    """
    c = constraints or {}

    # 1. Search evidence_units
    ev_stmt = (
        select(EvidenceUnit, Paper)
        .join(Paper, EvidenceUnit.paper_id == Paper.id)
        .where(EvidenceUnit.claim.ilike(f"%{query}%"))
        .order_by(desc(EvidenceUnit.confidence))
        .limit(limit)
    )
    ev_result = await session.execute(ev_stmt)
    evidence_results = [
        {
            "source": "evidence_unit",
            "id": str(eu.id),
            "paper_title": paper.title,
            "atom_type": eu.atom_type,
            "claim": eu.claim[:200],
            "confidence": eu.confidence,
            "basis": eu.basis.value if eu.basis else None,
            "source_section": eu.source_section,
            "conditions": eu.conditions[:100] if eu.conditions else None,
        }
        for eu, paper in ev_result
    ]

    # 2. Search implementation_units
    impl_stmt = (
        select(ImplementationUnit, Paper)
        .join(Paper, ImplementationUnit.paper_id == Paper.id)
        .where(
            or_(
                ImplementationUnit.description.ilike(f"%{query}%"),
                ImplementationUnit.class_or_function.ilike(f"%{query}%"),
            )
        )
        .limit(limit)
    )
    impl_result = await session.execute(impl_stmt)
    impl_results = [
        {
            "source": "implementation_unit",
            "paper_title": paper.title,
            "repo_url": iu.repo_url,
            "file_path": iu.file_path,
            "class_or_function": iu.class_or_function,
            "description": iu.description[:200] if iu.description else None,
        }
        for iu, paper in impl_result
    ]

    # 3. Delta cards with evidence
    dc_stmt = (
        select(DeltaCard, Paper)
        .join(Paper, DeltaCard.paper_id == Paper.id)
        .where(
            DeltaCard.evaluation_context.ilike(f"%{query}%"),
            DeltaCard.status != "deprecated",
        )
        .limit(limit)
    )
    dc_result = await session.execute(dc_stmt)
    dc_results = [
        {
            "source": "delta_card",
            "paper_title": paper.title,
            "evaluation_context": dc.evaluation_context[:200] if dc.evaluation_context else None,
            "evidence_count": len(dc.evidence_refs) if dc.evidence_refs else 0,
        }
        for dc, paper in dc_result
    ]

    return {
        "intent": "evidence",
        "query": query,
        "evidence_units": evidence_results,
        "implementation_units": impl_results,
        "delta_cards": dc_results[:limit],
    }


# ── Main router ──────────────────────────────────────────────────

async def route_query(
    session: AsyncSession,
    query: str,
    intent: str | None = None,
    constraints: dict | None = None,
    limit: int = 20,
) -> dict:
    """Route a query to the best retrieval path based on intent.

    If intent is not specified, auto-classify from query text.
    """
    if not intent:
        intent = await classify_intent(query)

    handlers = {
        "bottleneck": query_bottleneck,
        "mechanism": query_mechanism,
        "lineage": query_lineage,
        "evidence": query_evidence,
    }
    handler = handlers.get(intent, query_bottleneck)
    return await handler(session, query, constraints, limit)
