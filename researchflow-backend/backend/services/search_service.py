"""Hybrid search service — keyword + semantic + structured filters.

Combines:
1. PostgreSQL tsvector full-text search (BM25-like ranking)
2. pgvector cosine similarity (semantic)
3. Structured column filters (category, venue, year, tags, scores)
4. Idea-centric search via DeltaCard/IdeaDelta (v3)
5. Bottleneck search (Route 5) — keyword → bottlenecks + linked ideas
6. Mechanism search (Route 6) — entity resolution → linked ideas
7. Transfer search (Route 7) — transferable_to assertions across domains

Results are ranked by a weighted combination of text and vector scores.
"""

import logging
from uuid import UUID

from sqlalchemy import and_, desc, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.delta_card import DeltaCard
from backend.models.enums import PaperState
from backend.models.graph import IdeaDelta
from backend.models.paper import Paper
from backend.services.embedding_service import embed_text

logger = logging.getLogger(__name__)


async def hybrid_search(
    session: AsyncSession,
    query: str,
    category: str | None = None,
    venue: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    tags: list[str] | None = None,
    min_structurality: float | None = None,
    semantic: bool = True,
    limit: int = 20,
) -> list[dict]:
    """Run hybrid search combining keyword + semantic + structured filters.

    Returns list of {paper_id, title, score, text_score, vector_score, ...}.
    """
    # Build structured filter conditions
    conditions = [
        Paper.state.notin_([PaperState.ARCHIVED_OR_EXPIRED, PaperState.SKIP]),
    ]
    if category:
        conditions.append(Paper.category == category)
    if venue:
        conditions.append(Paper.venue == venue)
    if year_min:
        conditions.append(Paper.year >= year_min)
    if year_max:
        conditions.append(Paper.year <= year_max)
    if tags:
        conditions.append(Paper.tags.contains(tags))
    if min_structurality is not None:
        conditions.append(Paper.structurality_score >= min_structurality)

    filter_clause = and_(*conditions) if conditions else True

    # Strategy: run keyword and semantic searches, then merge scores

    # 1. Keyword search with ts_rank
    ts_query = func.plainto_tsquery("english", query)
    keyword_stmt = (
        select(
            Paper.id,
            Paper.title,
            Paper.venue,
            Paper.year,
            Paper.category,
            Paper.state,
            Paper.importance,
            Paper.tags,
            Paper.core_operator,
            Paper.keep_score,
            Paper.structurality_score,
            func.ts_rank(
                func.to_tsvector("english", Paper.title), ts_query
            ).label("title_rank"),
            func.ts_rank(
                func.to_tsvector("english", func.coalesce(Paper.abstract, "")), ts_query
            ).label("abstract_rank"),
        )
        .where(filter_clause)
        .order_by(text("title_rank DESC, abstract_rank DESC"))
        .limit(limit * 2)  # Over-fetch for merging
    )
    keyword_result = await session.execute(keyword_stmt)
    keyword_rows = keyword_result.fetchall()

    # Build results dict keyed by paper_id
    results: dict[UUID, dict] = {}
    for row in keyword_rows:
        text_score = float(row.title_rank or 0) * 2.0 + float(row.abstract_rank or 0)
        results[row.id] = {
            "paper_id": str(row.id),
            "title": row.title,
            "venue": row.venue,
            "year": row.year,
            "category": row.category,
            "state": row.state.value if row.state else None,
            "importance": row.importance.value if row.importance else None,
            "tags": list(row.tags) if row.tags else [],
            "core_operator": (row.core_operator or "")[:150],
            "keep_score": row.keep_score,
            "structurality_score": row.structurality_score,
            "text_score": round(text_score, 4),
            "vector_score": 0.0,
            "combined_score": 0.0,
        }

    # 2. Semantic search with pgvector (if enabled)
    if semantic:
        query_embedding = await embed_text(query)

        # Use raw SQL for pgvector cosine distance
        # Note: cast via CAST() instead of :: to avoid asyncpg $ param conflict
        vector_sql = text("""
            SELECT id, title, venue, year, category, state, importance,
                   tags, core_operator, keep_score, structurality_score,
                   1 - (embedding <=> CAST(:qvec AS vector)) AS cosine_sim
            FROM papers
            WHERE embedding IS NOT NULL
            AND state NOT IN ('archived_or_expired', 'skip')
            ORDER BY embedding <=> CAST(:qvec AS vector)
            LIMIT :lim
        """)

        params = {"qvec": str(query_embedding), "lim": limit * 2}

        if category:
            vector_sql = text("""
                SELECT id, title, venue, year, category, state, importance,
                       tags, core_operator, keep_score, structurality_score,
                       1 - (embedding <=> CAST(:qvec AS vector)) AS cosine_sim
                FROM papers
                WHERE embedding IS NOT NULL
                AND state NOT IN ('archived_or_expired', 'skip')
                AND (:cat IS NULL OR category = :cat)
                AND (:ven IS NULL OR venue = :ven)
                ORDER BY embedding <=> CAST(:qvec AS vector)
                LIMIT :lim
            """)
            params["cat"] = category
            params["ven"] = venue

        try:
            vector_result = await session.execute(vector_sql, params)
            vector_rows = vector_result.fetchall()

            for row in vector_rows:
                pid = row.id
                vscore = float(row.cosine_sim or 0)

                if pid in results:
                    results[pid]["vector_score"] = round(vscore, 4)
                else:
                    results[pid] = {
                        "paper_id": str(pid),
                        "title": row.title,
                        "venue": row.venue,
                        "year": row.year,
                        "category": row.category,
                        "state": row.state if isinstance(row.state, str) else (row.state.value if row.state else None),
                        "importance": row.importance if isinstance(row.importance, str) else (row.importance.value if row.importance else None),
                        "tags": list(row.tags) if row.tags else [],
                        "core_operator": (row.core_operator or "")[:150],
                        "keep_score": row.keep_score,
                        "structurality_score": row.structurality_score,
                        "text_score": 0.0,
                        "vector_score": round(vscore, 4),
                        "combined_score": 0.0,
                    }
        except Exception as e:
            logger.warning(f"Vector search failed (papers may lack embeddings): {e}")

    # 3. Compute combined score: 0.4 * text + 0.4 * vector + 0.2 * keep_score
    for r in results.values():
        r["combined_score"] = round(
            0.4 * r["text_score"]
            + 0.4 * r["vector_score"]
            + 0.2 * (r["keep_score"] or 0),
            4,
        )

    # Sort by combined score, return top N
    sorted_results = sorted(results.values(), key=lambda x: x["combined_score"], reverse=True)
    return sorted_results[:limit]


async def idea_search(
    session: AsyncSession,
    query: str,
    category: str | None = None,
    min_structurality: float | None = None,
    min_evidence: int | None = None,
    limit: int = 20,
) -> list[dict]:
    """Idea-centric search — search across DeltaCards and IdeaDeltas.

    Searches delta_statement, key_ideas, and assumptions for keyword matches.
    Falls back to IdeaDelta.delta_statement if no DeltaCards exist.
    """
    # Search DeltaCards by keyword in delta_statement
    dc_conditions = [DeltaCard.status != "deprecated"]
    if min_structurality is not None:
        dc_conditions.append(DeltaCard.structurality_score >= min_structurality)

    dc_stmt = (
        select(DeltaCard)
        .where(
            and_(*dc_conditions),
            DeltaCard.delta_statement.ilike(f"%{query}%"),
        )
        .order_by(desc(DeltaCard.structurality_score))
        .limit(limit)
    )
    dc_result = await session.execute(dc_stmt)
    delta_cards = list(dc_result.scalars().all())

    results = []
    seen_paper_ids = set()

    for dc in delta_cards:
        paper = await session.get(Paper, dc.paper_id)
        if category and paper and paper.category != category:
            continue
        seen_paper_ids.add(dc.paper_id)
        results.append({
            "source": "delta_card",
            "delta_card_id": str(dc.id),
            "paper_id": str(dc.paper_id),
            "title": paper.title if paper else "Unknown",
            "venue": paper.venue if paper else None,
            "year": paper.year if paper else None,
            "delta_statement": dc.delta_statement[:300],
            "structurality_score": dc.structurality_score,
            "transferability_score": dc.transferability_score,
            "status": dc.status,
            "key_ideas": dc.key_ideas_ranked[:3] if dc.key_ideas_ranked else None,
        })

    # Also search IdeaDeltas not covered by DeltaCards
    idea_conditions = [IdeaDelta.delta_statement.ilike(f"%{query}%")]
    if min_structurality is not None:
        idea_conditions.append(IdeaDelta.structurality_score >= min_structurality)
    if min_evidence is not None:
        idea_conditions.append(IdeaDelta.evidence_count >= min_evidence)

    idea_stmt = (
        select(IdeaDelta)
        .where(and_(*idea_conditions))
        .order_by(desc(IdeaDelta.structurality_score))
        .limit(limit)
    )
    idea_result = await session.execute(idea_stmt)

    for idea in idea_result.scalars():
        if idea.paper_id in seen_paper_ids:
            continue
        paper = await session.get(Paper, idea.paper_id)
        if category and paper and paper.category != category:
            continue
        results.append({
            "source": "idea_delta",
            "idea_delta_id": str(idea.id),
            "paper_id": str(idea.paper_id),
            "title": paper.title if paper else "Unknown",
            "venue": paper.venue if paper else None,
            "year": paper.year if paper else None,
            "delta_statement": idea.delta_statement[:300],
            "structurality_score": idea.structurality_score,
            "transferability_score": idea.transferability_score,
            "publish_status": idea.publish_status,
            "evidence_count": idea.evidence_count,
        })

    return results[:limit]


# ── Bottleneck search (Route 5) ──────────────────────────────────

async def bottleneck_search(
    session: AsyncSession,
    keyword: str,
    limit: int = 20,
) -> dict:
    """Search ProjectBottleneck by keyword, return bottlenecks with linked IdeaDeltas.

    Wraps graph_query_service.query_by_bottleneck as a search-service entry point.
    """
    from backend.services import graph_query_service

    return await graph_query_service.query_by_bottleneck(
        session, bottleneck_id=None, keyword=keyword, limit=limit,
    )


# ── Mechanism search (Route 6) ───────────────────────────────────

async def mechanism_search(
    session: AsyncSession,
    name: str,
    limit: int = 20,
) -> dict:
    """Resolve a mechanism via entity_resolution and return linked ideas.

    Wraps graph_query_service.query_by_mechanism as a search-service entry point.
    """
    from backend.services import graph_query_service

    return await graph_query_service.query_by_mechanism(
        session, mechanism_name=name, mechanism_id=None, limit=limit,
    )


# ── Transfer search (Route 7) ───────────────────────────────────

async def transfer_search(
    session: AsyncSession,
    source_domain: str | None = None,
    target_domain: str | None = None,
    limit: int = 20,
) -> dict:
    """Find transferable_to assertions, optionally filtered by domain.

    Wraps graph_query_service.query_transfers as a search-service entry point.
    """
    from backend.services import graph_query_service

    return await graph_query_service.query_transfers(
        session, source_domain=source_domain, target_domain=target_domain, limit=limit,
    )
