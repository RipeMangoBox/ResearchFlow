"""Hybrid search service — keyword + semantic + structured filters.

Combines:
1. PostgreSQL tsvector full-text search (BM25-like ranking)
2. pgvector cosine similarity (semantic)
3. Structured column filters (category, venue, year, tags, scores)

Results are ranked by a weighted combination of text and vector scores.
"""

import logging
from uuid import UUID

from sqlalchemy import and_, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.enums import PaperState
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
        vector_sql = text("""
            SELECT id, title, venue, year, category, state, importance,
                   tags, core_operator, keep_score, structurality_score,
                   1 - (embedding <=> :qvec::vector) AS cosine_sim
            FROM papers
            WHERE embedding IS NOT NULL
            AND state NOT IN ('archived_or_expired', 'skip')
            ORDER BY embedding <=> :qvec::vector
            LIMIT :lim
        """)

        params = {"qvec": str(query_embedding), "lim": limit * 2}

        # Add structured filters to vector query if needed
        if category:
            vector_sql = text("""
                SELECT id, title, venue, year, category, state, importance,
                       tags, core_operator, keep_score, structurality_score,
                       1 - (embedding <=> :qvec::vector) AS cosine_sim
                FROM papers
                WHERE embedding IS NOT NULL
                AND state NOT IN ('archived_or_expired', 'skip')
                AND (:cat IS NULL OR category = :cat)
                AND (:ven IS NULL OR venue = :ven)
                ORDER BY embedding <=> :qvec::vector
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
