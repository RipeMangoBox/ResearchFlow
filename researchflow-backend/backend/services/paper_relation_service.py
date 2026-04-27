"""Materialize reference_role_map blackboard items → paper_relations table.

Each agent_blackboard_items row of type `reference_role_map` carries a list
of `classifications`, each citing one reference. We:
  1. Normalize the cited title.
  2. Fuzzy-match it against papers.title_sanitized.
  3. If matched with confidence ≥ MIN_MATCH_CONF, upsert a paper_relations
     row with relation_type set from the agent's role classification.

Roles we materialize as relations (others are dropped — too speculative):
  - direct_baseline       → 'direct_baseline'
  - comparison_baseline   → 'comparison_baseline'
  - method_source         → 'method_source'
  - formula_source        → 'formula_source'
  - dataset_source        → 'dataset_source'
  - benchmark_source      → 'benchmark_source'
  - same_task_prior_work  → 'same_task_prior_work'

Also exposes `materialize_for_paper(paper_id)` (one paper) and
`materialize_all()` (all blackboard rows).
"""

from __future__ import annotations

import logging
import re
from uuid import UUID

from sqlalchemy import select, text as sa_text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.agent import AgentBlackboardItem
from backend.models.paper import Paper
from backend.models.paper_relation import PaperRelation

logger = logging.getLogger(__name__)


KEEP_ROLES = {
    "direct_baseline", "comparison_baseline",
    "method_source", "formula_source",
    "dataset_source", "benchmark_source",
    "same_task_prior_work",
}

MIN_MATCH_CONF = 0.70   # Jaccard token overlap threshold
TOP_MATCH_LIMIT = 3     # consider top-N candidates per ref


# ── Title normalization ─────────────────────────────────────────────────

_TOKEN_SPLIT_RE = re.compile(r"[^\w\u4e00-\u9fff]+")
_STOP = {"the", "a", "an", "of", "for", "with", "to", "and", "in", "on", "by",
         "via", "is", "are", "via", "based", "novel", "approach", "method"}


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def _tokens(title: str) -> set[str]:
    return {t for t in _TOKEN_SPLIT_RE.split(_normalize_title(title))
            if t and t not in _STOP and len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ── Lookup index built once per call ────────────────────────────────────

async def _load_paper_index(session: AsyncSession) -> list[tuple[UUID, str, set[str]]]:
    rows = (await session.execute(
        select(Paper.id, Paper.title)
        .where(Paper.title.isnot(None))
    )).all()
    out = []
    for pid, title in rows:
        tok = _tokens(title)
        if len(tok) >= 3:  # need enough signal
            out.append((pid, title, tok))
    return out


def _best_match(ref_title: str, index: list[tuple[UUID, str, set[str]]]) -> tuple[UUID, str, float] | None:
    """Return (paper_id, matched_title, score) for the best candidate, or None."""
    ref_tok = _tokens(ref_title)
    if len(ref_tok) < 3:
        return None
    scored = []
    for pid, title, tok in index:
        s = _jaccard(ref_tok, tok)
        if s >= MIN_MATCH_CONF:
            scored.append((pid, title, s))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[2])
    return scored[0]


# ── Upsert ──────────────────────────────────────────────────────────────

async def _upsert_relation(
    session: AsyncSession,
    *,
    source_paper_id: UUID,
    target_paper_id: UUID,
    relation_type: str,
    evidence: str,
    confidence: float,
    ref_index: str | None,
    ref_title_raw: str | None,
) -> bool:
    if source_paper_id == target_paper_id:
        return False  # never link a paper to itself
    stmt = pg_insert(PaperRelation).values(
        source_paper_id=source_paper_id,
        target_paper_id=target_paper_id,
        relation_type=relation_type,
        evidence=(evidence or "")[:1000],
        confidence=round(confidence, 2) if confidence else None,
        ref_index=ref_index,
        ref_title_raw=(ref_title_raw or "")[:500],
        match_method="title_fuzzy",
    ).on_conflict_do_nothing(
        constraint="uq_paper_relations_triple"
    )
    await session.execute(stmt)
    return True


# ── Public API ──────────────────────────────────────────────────────────

async def materialize_for_paper(
    session: AsyncSession,
    paper_id: UUID,
    *,
    paper_index: list[tuple[UUID, str, set[str]]] | None = None,
) -> dict:
    """Materialize relations for a single paper. Returns stats."""
    bb_rows = (await session.execute(
        select(AgentBlackboardItem)
        .where(
            AgentBlackboardItem.paper_id == paper_id,
            AgentBlackboardItem.item_type == "reference_role_map",
        )
        .order_by(AgentBlackboardItem.created_at.desc())
        .limit(1)
    )).scalars().all()
    if not bb_rows:
        return {"paper_id": str(paper_id), "skipped": "no_blackboard"}
    bb = bb_rows[0]
    classifications = (bb.value_json or {}).get("classifications", [])
    if not isinstance(classifications, list) or not classifications:
        return {"paper_id": str(paper_id), "skipped": "empty_classifications"}

    if paper_index is None:
        paper_index = await _load_paper_index(session)

    inserted = 0
    no_match = 0
    skipped_role = 0
    for cls in classifications:
        role = (cls.get("role") or "").strip()
        if role not in KEEP_ROLES:
            skipped_role += 1
            continue
        ref_title = (cls.get("ref_title") or "").strip()
        if not ref_title:
            no_match += 1
            continue
        match = _best_match(ref_title, paper_index)
        if not match:
            no_match += 1
            continue
        target_pid, _matched_title, score = match
        ok = await _upsert_relation(
            session,
            source_paper_id=paper_id,
            target_paper_id=target_pid,
            relation_type=role,
            evidence=cls.get("reason", ""),
            confidence=score,
            ref_index=cls.get("ref_index"),
            ref_title_raw=ref_title,
        )
        if ok:
            inserted += 1
    return {
        "paper_id": str(paper_id),
        "inserted": inserted,
        "no_match": no_match,
        "skipped_role": skipped_role,
        "total_classifications": len(classifications),
    }


async def materialize_all(session: AsyncSession) -> dict:
    """Walk every reference_role_map blackboard row once. Returns aggregate."""
    paper_ids = (await session.execute(sa_text("""
        SELECT DISTINCT paper_id FROM agent_blackboard_items
        WHERE item_type = 'reference_role_map'
        ORDER BY paper_id
    """))).scalars().all()
    if not paper_ids:
        return {"papers_processed": 0, "total_inserted": 0}

    paper_index = await _load_paper_index(session)
    logger.info("materialize_all: %d papers, index=%d entries",
                len(paper_ids), len(paper_index))

    total = {"papers_processed": 0, "total_inserted": 0,
             "total_no_match": 0, "total_skipped_role": 0}
    for pid in paper_ids:
        try:
            r = await materialize_for_paper(session, pid, paper_index=paper_index)
            total["papers_processed"] += 1
            total["total_inserted"] += r.get("inserted", 0)
            total["total_no_match"] += r.get("no_match", 0)
            total["total_skipped_role"] += r.get("skipped_role", 0)
        except Exception as e:
            logger.warning("materialize_for_paper(%s) failed: %s", pid, e)
    await session.commit()
    return total
