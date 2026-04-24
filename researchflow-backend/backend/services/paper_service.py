"""Paper CRUD and query service."""

import math
import re

from sqlalchemy import Select, and_, desc, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.paper import Paper
from backend.models.analysis import PaperAnalysis
from backend.models.enums import PaperState
from backend.schemas.paper import PaperCreate, PaperFilter, PaperUpdate
from backend.utils.sanitize import sanitize_filename


async def create_paper(session: AsyncSession, data: PaperCreate) -> Paper:
    paper = Paper(
        title=data.title,
        title_sanitized=sanitize_filename(data.title),
        venue=data.venue,
        year=data.year,
        category=data.category,
        state=PaperState.EPHEMERAL_RECEIVED if data.is_ephemeral else PaperState.WAIT,
        importance=data.importance,
        tier=data.tier,
        paper_link=data.paper_link,
        project_link=data.project_link,
        arxiv_id=data.arxiv_id,
        doi=data.doi,
        abstract=data.abstract,
        tags=data.tags or [data.category],
        method_family=data.method_family,
        core_operator=data.core_operator,
        primary_logic=data.primary_logic,
        is_ephemeral=data.is_ephemeral,
        retention_days=data.retention_days,
        source="api",
    )
    session.add(paper)
    await session.flush()
    await session.refresh(paper)
    return paper


async def get_paper(session: AsyncSession, paper_id) -> Paper | None:
    return await session.get(Paper, paper_id)


async def get_paper_with_analysis(session: AsyncSession, paper_id) -> tuple[Paper | None, PaperAnalysis | None]:
    paper = await session.get(Paper, paper_id)
    if not paper:
        return None, None
    result = await session.execute(
        select(PaperAnalysis)
        .where(PaperAnalysis.paper_id == paper_id, PaperAnalysis.is_current.is_(True))
        .order_by(desc(PaperAnalysis.level))
        .limit(1)
    )
    analysis = result.scalar_one_or_none()
    return paper, analysis


async def update_paper(session: AsyncSession, paper_id, data: PaperUpdate) -> Paper | None:
    paper = await session.get(Paper, paper_id)
    if not paper:
        return None
    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(paper, key, value)
    if "title" in update_data:
        paper.title_sanitized = sanitize_filename(update_data["title"])
    await session.flush()
    await session.refresh(paper)
    return paper


async def delete_paper(session: AsyncSession, paper_id) -> bool:
    paper = await session.get(Paper, paper_id)
    if not paper:
        return False
    await session.delete(paper)
    await session.flush()
    return True


def _build_filter_query(filters: PaperFilter) -> Select:
    """Build a filtered query from PaperFilter parameters."""
    stmt = select(Paper)
    conditions = []

    if filters.state is not None:
        conditions.append(Paper.state == filters.state)
    if filters.category is not None:
        conditions.append(Paper.category == filters.category)
    if filters.venue is not None:
        conditions.append(Paper.venue == filters.venue)
    if filters.year_min is not None:
        conditions.append(Paper.year >= filters.year_min)
    if filters.year_max is not None:
        conditions.append(Paper.year <= filters.year_max)
    if filters.importance is not None:
        conditions.append(Paper.importance == filters.importance)
    if filters.tier is not None:
        conditions.append(Paper.tier == filters.tier)
    if filters.is_ephemeral is not None:
        conditions.append(Paper.is_ephemeral == filters.is_ephemeral)
    if filters.min_keep_score is not None:
        conditions.append(Paper.keep_score >= filters.min_keep_score)
    if filters.min_structurality is not None:
        conditions.append(Paper.structurality_score >= filters.min_structurality)
    if filters.min_extensionability is not None:
        conditions.append(Paper.extensionability_score >= filters.min_extensionability)
    if filters.tags:
        # Paper.tags contains ALL of the requested tags
        conditions.append(Paper.tags.contains(filters.tags))

    # Full-text search on title
    if filters.q:
        # Use PostgreSQL tsvector for full-text search
        ts_query = func.plainto_tsquery("english", filters.q)
        conditions.append(
            func.to_tsvector("english", Paper.title).op("@@")(ts_query)
        )

    if conditions:
        stmt = stmt.where(and_(*conditions))

    # Sort
    sort_col = getattr(Paper, filters.sort_by, Paper.updated_at)
    if filters.sort_order == "asc":
        stmt = stmt.order_by(sort_col.asc().nullslast())
    else:
        stmt = stmt.order_by(sort_col.desc().nullsfirst())

    return stmt


async def list_papers(
    session: AsyncSession, filters: PaperFilter
) -> tuple[list[Paper], int]:
    """Return (papers, total_count) for the given filters."""
    stmt = _build_filter_query(filters)

    # Count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await session.execute(count_stmt)).scalar() or 0

    # Paginate
    offset = (filters.page - 1) * filters.size
    stmt = stmt.offset(offset).limit(filters.size)

    result = await session.execute(stmt)
    papers = list(result.scalars().all())
    return papers, total


async def search_papers(
    session: AsyncSession, query: str, limit: int = 20
) -> list[Paper]:
    """Simple full-text search on title + abstract."""
    ts_query = func.plainto_tsquery("english", query)

    stmt = (
        select(Paper)
        .where(
            or_(
                func.to_tsvector("english", Paper.title).op("@@")(ts_query),
                func.to_tsvector("english", func.coalesce(Paper.abstract, "")).op("@@")(ts_query),
            )
        )
        .order_by(Paper.updated_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())
