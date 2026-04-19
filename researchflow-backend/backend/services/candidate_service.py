"""Paper candidate management — the gateway to the knowledge base.

Papers are discovered → candidates → scored → selectively promoted to full Papers.
This replaces direct ingest with a candidate pool + multi-stage scoring.
"""

import re
import uuid
from uuid import UUID

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.candidate import CandidateScore, PaperCandidate, ScoreSignal
from backend.models.domain import DomainSpec
from backend.models.enums import PaperState
from backend.models.paper import Paper
from backend.utils.sanitize import sanitize_filename


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _normalize_title(title: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    t = title.lower()
    t = _PUNCT_RE.sub("", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def _extract_discovery_signals(
    candidate: PaperCandidate,
    domain: DomainSpec | None,
) -> dict:
    """Build a signals dict from candidate metadata for the scoring engine."""
    signals: dict = {}

    # Citation strength
    if candidate.citation_count is not None:
        signals["citation_count"] = candidate.citation_count

    # Venue presence
    if candidate.venue:
        signals["venue"] = candidate.venue

    # Year recency
    if candidate.year:
        signals["year"] = candidate.year

    # Has code
    if candidate.code_url:
        signals["has_code"] = True

    # Abstract presence / length
    if candidate.abstract:
        signals["abstract_length"] = len(candidate.abstract)

    # Discovery provenance — map to scoring engine keys
    signals["source_type"] = candidate.discovery_source
    # Intent-aware source type refinement
    # If discovered via S2 reference with intent info in discovery_reason,
    # map to more specific source type for better scoring differentiation
    if signals["source_type"] == "s2_reference" and candidate.discovery_reason:
        reason_lower = candidate.discovery_reason.lower()
        if "methodology" in reason_lower or "method" in reason_lower:
            signals["source_type"] = "s2_reference_methodology"
        elif "result" in reason_lower or "comparison" in reason_lower:
            signals["source_type"] = "s2_reference_result"
        elif "background" in reason_lower:
            signals["source_type"] = "s2_reference_background"
    signals["discovery_source"] = candidate.discovery_source
    if candidate.discovery_reason:
        signals["discovery_reason"] = candidate.discovery_reason
    if candidate.relation_hint:
        signals["relation_hint"] = candidate.relation_hint

    # Authors
    if candidate.authors_json:
        authors = candidate.authors_json
        if isinstance(authors, list):
            signals["author_count"] = len(authors)

    # Domain matching signals
    if domain and candidate.abstract:
        scope_keywords = []
        for field in ("scope_tasks", "scope_modalities", "scope_paradigms",
                      "scope_seed_methods", "scope_seed_models", "scope_seed_datasets"):
            vals = getattr(domain, field, None)
            if vals:
                scope_keywords.extend(vals)
        text = (candidate.title + " " + (candidate.abstract or "")).lower()
        hits = sum(1 for kw in scope_keywords if kw.lower() in text)
        signals["domain_keyword_hits"] = hits
        signals["task_match"] = any(
            t.lower() in text for t in (domain.scope_tasks or [])
        )
        signals["modality_match"] = any(
            m.lower() in text for m in (domain.scope_modalities or [])
        )

    # Graph proximity — requires DB query, set defaults here
    # (caller can override with actual values from graph query)
    signals.setdefault("connected_anchor_count", 0)
    signals.setdefault("hop_distance", 99)

    # Recency
    if candidate.year:
        from datetime import date
        months = (date.today().year - candidate.year) * 12
        signals["months_since_release"] = max(0, months)

    # Artifact signals
    signals.setdefault("has_code", bool(candidate.code_url))
    signals.setdefault("has_data", False)
    signals.setdefault("has_model", False)
    signals.setdefault("has_benchmark", False)

    # Novelty signals — defaults (overridden by agent later)
    signals.setdefault("fills_graph_gap", False)
    signals.setdefault("new_mechanism", False)
    signals.setdefault("new_setting", False)
    signals.setdefault("redundancy_score", 0.0)

    return signals


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

async def find_duplicate(
    session: AsyncSession,
    *,
    arxiv_id: str | None = None,
    doi: str | None = None,
    normalized_title: str | None = None,
) -> PaperCandidate | None:
    """Check arxiv_id first, then DOI, then normalized_title."""
    if arxiv_id:
        result = await session.execute(
            select(PaperCandidate).where(PaperCandidate.arxiv_id == arxiv_id).limit(1)
        )
        found = result.scalar_one_or_none()
        if found:
            return found

    if doi:
        result = await session.execute(
            select(PaperCandidate).where(PaperCandidate.doi == doi).limit(1)
        )
        found = result.scalar_one_or_none()
        if found:
            return found

    if normalized_title:
        result = await session.execute(
            select(PaperCandidate)
            .where(PaperCandidate.normalized_title == normalized_title)
            .limit(1)
        )
        found = result.scalar_one_or_none()
        if found:
            return found

    return None


async def find_existing_paper(
    session: AsyncSession,
    *,
    arxiv_id: str | None = None,
    doi: str | None = None,
    normalized_title: str | None = None,
) -> Paper | None:
    """检查 papers 表是否已有此论文。优先 arxiv_id → DOI → 标题。

    用于"已知论文重遇"场景：当新论文引用了 KB 中已有的论文，
    应该创建关系边而非重新入库。
    """
    if arxiv_id:
        result = await session.execute(
            select(Paper).where(Paper.arxiv_id == arxiv_id).limit(1)
        )
        found = result.scalar_one_or_none()
        if found:
            return found

    if doi:
        result = await session.execute(
            select(Paper).where(Paper.doi == doi).limit(1)
        )
        found = result.scalar_one_or_none()
        if found:
            return found

    if normalized_title:
        from sqlalchemy import func as sa_func
        result = await session.execute(
            select(Paper)
            .where(sa_func.lower(Paper.title_sanitized) == normalized_title.lower())
            .limit(1)
        )
        found = result.scalar_one_or_none()
        if found:
            return found

    return None


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

async def create_candidate(
    session: AsyncSession,
    title: str,
    discovery_source: str,
    discovery_reason: str,
    *,
    arxiv_id: str | None = None,
    doi: str | None = None,
    paper_link: str | None = None,
    abstract: str | None = None,
    authors_json: dict | None = None,
    venue: str | None = None,
    year: int | None = None,
    citation_count: int | None = None,
    code_url: str | None = None,
    metadata_json: dict | None = None,
    discovered_from_paper_id: UUID | None = None,
    discovered_from_domain_id: UUID | None = None,
    relation_hint: str | None = None,
) -> PaperCandidate:
    """Create a new candidate, deduplicating by arxiv_id / DOI / title."""
    norm_title = _normalize_title(title)

    # Check for duplicates
    existing = await find_duplicate(
        session,
        arxiv_id=arxiv_id,
        doi=doi,
        normalized_title=norm_title,
    )

    if existing:
        # Update metadata if incoming data is richer
        changed = False
        if abstract and not existing.abstract:
            existing.abstract = abstract
            changed = True
        if authors_json and not existing.authors_json:
            existing.authors_json = authors_json
            changed = True
        if venue and not existing.venue:
            existing.venue = venue
            changed = True
        if year and not existing.year:
            existing.year = year
            changed = True
        if citation_count is not None and existing.citation_count is None:
            existing.citation_count = citation_count
            changed = True
        if code_url and not existing.code_url:
            existing.code_url = code_url
            changed = True
        if paper_link and not existing.paper_link:
            existing.paper_link = paper_link
            changed = True
        if arxiv_id and not existing.arxiv_id:
            existing.arxiv_id = arxiv_id
            changed = True
        if doi and not existing.doi:
            existing.doi = doi
            changed = True
        if metadata_json and not existing.metadata_json:
            existing.metadata_json = metadata_json
            changed = True
        if changed:
            await session.flush()
            await session.refresh(existing)
        return existing

    # Create new
    candidate = PaperCandidate(
        title=title,
        normalized_title=norm_title,
        arxiv_id=arxiv_id,
        doi=doi,
        paper_link=paper_link,
        abstract=abstract,
        authors_json=authors_json,
        venue=venue,
        year=year,
        citation_count=citation_count,
        code_url=code_url,
        metadata_json=metadata_json,
        discovery_source=discovery_source,
        discovery_reason=discovery_reason,
        relation_hint=relation_hint,
        discovered_from_paper_id=discovered_from_paper_id,
        discovered_from_domain_id=discovered_from_domain_id,
        status="discovered",
        absorption_level=0,
    )
    session.add(candidate)
    await session.flush()
    await session.refresh(candidate)
    return candidate


async def get_candidate(
    session: AsyncSession,
    candidate_id: UUID,
) -> PaperCandidate | None:
    return await session.get(PaperCandidate, candidate_id)


async def list_candidates(
    session: AsyncSession,
    *,
    domain_id: UUID | None = None,
    status: str | None = None,
    min_score: float | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[PaperCandidate]:
    """List candidates filtered by domain, status, minimum score.

    Ordered by latest discovery_score descending (candidates without
    scores sort last).
    """
    # Subquery: latest score per candidate
    latest_score = (
        select(
            CandidateScore.candidate_id,
            func.max(CandidateScore.created_at).label("max_created"),
        )
        .group_by(CandidateScore.candidate_id)
        .subquery()
    )

    score_sq = (
        select(CandidateScore)
        .join(
            latest_score,
            and_(
                CandidateScore.candidate_id == latest_score.c.candidate_id,
                CandidateScore.created_at == latest_score.c.max_created,
            ),
        )
        .subquery()
    )

    stmt = (
        select(PaperCandidate)
        .outerjoin(score_sq, PaperCandidate.id == score_sq.c.candidate_id)
    )

    if domain_id is not None:
        stmt = stmt.where(PaperCandidate.discovered_from_domain_id == domain_id)
    if status is not None:
        stmt = stmt.where(PaperCandidate.status == status)
    if min_score is not None:
        stmt = stmt.where(score_sq.c.discovery_score >= min_score)

    stmt = stmt.order_by(
        desc(score_sq.c.discovery_score).nulls_last(),
        desc(PaperCandidate.created_at),
    )
    stmt = stmt.limit(limit).offset(offset)

    result = await session.execute(stmt)
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

async def score_candidate(
    session: AsyncSession,
    candidate_id: UUID,
    domain: DomainSpec | None = None,
) -> CandidateScore:
    """Score a single candidate using deterministic signals.

    Extracts signals from candidate metadata, calls the scoring engine,
    persists CandidateScore + individual ScoreSignals, and updates
    candidate status based on the decision.
    """
    from backend.services.scoring_engine import ScoringEngine

    candidate = await session.get(PaperCandidate, candidate_id)
    if not candidate:
        raise ValueError(f"Candidate {candidate_id} not found")

    # Load domain if candidate has one and none provided
    if domain is None and candidate.discovered_from_domain_id:
        domain = await session.get(DomainSpec, candidate.discovered_from_domain_id)

    signals = _extract_discovery_signals(candidate, domain)

    # Compute score via engine
    engine = ScoringEngine()
    score_result = engine.compute_discovery_score(signals)

    # Persist CandidateScore
    cs = CandidateScore(
        candidate_id=candidate_id,
        discovery_score=score_result.total,
        discovery_breakdown=score_result.breakdown,
        hard_caps_applied=score_result.hard_caps_applied,
        boosts_applied=score_result.boosts_applied,
        penalties_applied=score_result.penalties_applied,
        decision=score_result.decision,
    )
    session.add(cs)

    # Persist individual ScoreSignals
    breakdown = score_result.breakdown or {}
    for signal_name, signal_value in signals.items():
        ss = ScoreSignal(
            entity_type="candidate",
            entity_id=candidate_id,
            signal_name=signal_name,
            signal_value={"raw": signal_value} if not isinstance(signal_value, dict) else signal_value,
            signal_strength=breakdown.get(signal_name),
            producer="deterministic",
        )
        session.add(ss)

    # Update candidate status based on decision
    decision = score_result.decision or "metadata_only"
    if decision == "shallow_ingest":
        candidate.status = "scored"
    elif decision == "candidate_pool":
        candidate.status = "scored"
    elif decision == "archive":
        candidate.status = "archived"
    else:
        candidate.status = "metadata_resolved"

    await session.flush()
    await session.refresh(cs)
    return cs


async def score_batch(
    session: AsyncSession,
    *,
    limit: int = 50,
    domain_id: UUID | None = None,
) -> int:
    """Score unscored candidates (status discovered or metadata_resolved).

    Returns count of candidates scored.
    """
    stmt = select(PaperCandidate).where(
        PaperCandidate.status.in_(["discovered", "metadata_resolved"])
    )
    if domain_id is not None:
        stmt = stmt.where(PaperCandidate.discovered_from_domain_id == domain_id)
    stmt = stmt.order_by(PaperCandidate.created_at).limit(limit)

    result = await session.execute(stmt)
    candidates = list(result.scalars().all())

    count = 0
    for c in candidates:
        domain = None
        if c.discovered_from_domain_id:
            domain = await session.get(DomainSpec, c.discovered_from_domain_id)
        await score_candidate(session, c.id, domain=domain)
        count += 1

    return count


# ---------------------------------------------------------------------------
# Promotion / rejection
# ---------------------------------------------------------------------------

async def promote_candidate(
    session: AsyncSession,
    candidate_id: UUID,
    absorption_level: int = 1,
) -> Paper:
    """Promote a candidate to a full Paper in the knowledge base."""
    candidate = await session.get(PaperCandidate, candidate_id)
    if not candidate:
        raise ValueError(f"Candidate {candidate_id} not found")

    # Map absorption_level to initial PaperState
    state_map = {
        1: PaperState.L1_METADATA,
        2: PaperState.L2_PARSED,
        3: PaperState.L3_SKIMMED,
    }
    initial_state = state_map.get(absorption_level, PaperState.WAIT)

    paper = Paper(
        title=candidate.title,
        title_sanitized=sanitize_filename(candidate.title),
        venue=candidate.venue,
        year=candidate.year,
        category="uncategorized",  # will be assigned by triage
        state=initial_state,
        paper_link=candidate.paper_link,
        arxiv_id=candidate.arxiv_id,
        doi=candidate.doi,
        abstract=candidate.abstract,
        authors=candidate.authors_json,
        code_url=candidate.code_url,
        source="candidate_promotion",
        domain_id=candidate.discovered_from_domain_id,
    )
    session.add(paper)
    await session.flush()

    # Update candidate
    candidate.status = "ingested"
    candidate.ingested_paper_id = paper.id
    candidate.absorption_level = absorption_level

    await session.flush()
    await session.refresh(paper)
    return paper


async def reject_candidate(
    session: AsyncSession,
    candidate_id: UUID,
    reason: str,
) -> PaperCandidate:
    """Reject a candidate with a reason."""
    candidate = await session.get(PaperCandidate, candidate_id)
    if not candidate:
        raise ValueError(f"Candidate {candidate_id} not found")

    candidate.status = "rejected"
    candidate.reject_reason = reason
    await session.flush()
    await session.refresh(candidate)
    return candidate


async def auto_promote_batch(
    session: AsyncSession,
    *,
    threshold: float = 75.0,
    limit: int = 20,
    domain_id: UUID | None = None,
) -> list[Paper]:
    """Find scored candidates above threshold and promote each."""
    # Latest score per candidate subquery
    latest_score = (
        select(
            CandidateScore.candidate_id,
            func.max(CandidateScore.created_at).label("max_created"),
        )
        .group_by(CandidateScore.candidate_id)
        .subquery()
    )

    score_sq = (
        select(CandidateScore)
        .join(
            latest_score,
            and_(
                CandidateScore.candidate_id == latest_score.c.candidate_id,
                CandidateScore.created_at == latest_score.c.max_created,
            ),
        )
        .subquery()
    )

    stmt = (
        select(PaperCandidate)
        .join(score_sq, PaperCandidate.id == score_sq.c.candidate_id)
        .where(
            PaperCandidate.status.in_(["accepted", "scoring"]),
            score_sq.c.discovery_score >= threshold,
        )
    )
    if domain_id is not None:
        stmt = stmt.where(PaperCandidate.discovered_from_domain_id == domain_id)
    stmt = stmt.order_by(desc(score_sq.c.discovery_score)).limit(limit)

    result = await session.execute(stmt)
    candidates = list(result.scalars().all())

    papers = []
    for c in candidates:
        paper = await promote_candidate(session, c.id)
        papers.append(paper)
    return papers


# ---------------------------------------------------------------------------
# Search & reactivation
# ---------------------------------------------------------------------------

async def search_candidates(
    session: AsyncSession,
    query: str,
    *,
    domain_id: UUID | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[PaperCandidate]:
    """搜索候选论文（包括 L0 metadata-only 的论文）。

    用 ILIKE 在 title 和 abstract 上做关键词匹配。
    默认排除 archived 和 rejected 状态的候选。
    """
    from sqlalchemy import or_

    conditions = []

    # 关键词匹配 title 或 abstract
    keyword_filter = or_(
        PaperCandidate.title.ilike(f"%{query}%"),
        PaperCandidate.abstract.ilike(f"%{query}%"),
    )
    conditions.append(keyword_filter)

    if domain_id is not None:
        conditions.append(PaperCandidate.discovered_from_domain_id == domain_id)
    if status is not None:
        conditions.append(PaperCandidate.status == status)
    else:
        conditions.append(PaperCandidate.status.notin_(["archived", "rejected"]))

    stmt = (
        select(PaperCandidate)
        .where(and_(*conditions))
        .order_by(desc(PaperCandidate.created_at))
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def check_reactivation(
    session: AsyncSession,
    candidate_id: UUID,
    *,
    citation_threshold: int = 3,
) -> bool:
    """检查候选论文是否应被重新激活。

    如果一个 L0 候选被 KB 中 ≥3 篇论文引用（标题出现在 papers 的 abstract 中），
    将其 status 重置为 discovered，触发重新评分。
    """
    candidate = await session.get(PaperCandidate, candidate_id)
    if not candidate or candidate.status == "ingested":
        return False

    # 用标题的前 60 字做模糊匹配
    title_fragment = candidate.title[:60] if candidate.title else ""
    if not title_fragment:
        return False

    from sqlalchemy import func as sa_func
    count_stmt = (
        select(sa_func.count(Paper.id))
        .where(Paper.abstract.ilike(f"%{title_fragment}%"))
    )
    count = (await session.execute(count_stmt)).scalar() or 0

    if count >= citation_threshold:
        # 重置状态，让下一轮 score_batch 重新评分
        candidate.status = "discovered"
        await session.flush()
        return True
    return False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

async def get_stats(
    session: AsyncSession,
    domain_id: UUID | None = None,
) -> dict:
    """Count candidates by status and by absorption_level."""
    # By status
    status_stmt = select(
        PaperCandidate.status,
        func.count(PaperCandidate.id),
    ).group_by(PaperCandidate.status)
    if domain_id is not None:
        status_stmt = status_stmt.where(
            PaperCandidate.discovered_from_domain_id == domain_id
        )
    status_result = await session.execute(status_stmt)
    by_status = {row[0]: row[1] for row in status_result.all()}

    # By absorption_level
    abs_stmt = select(
        PaperCandidate.absorption_level,
        func.count(PaperCandidate.id),
    ).group_by(PaperCandidate.absorption_level)
    if domain_id is not None:
        abs_stmt = abs_stmt.where(
            PaperCandidate.discovered_from_domain_id == domain_id
        )
    abs_result = await session.execute(abs_stmt)
    by_absorption = {row[0]: row[1] for row in abs_result.all()}

    total = sum(by_status.values())

    return {
        "total": total,
        "by_status": by_status,
        "by_absorption_level": by_absorption,
    }
