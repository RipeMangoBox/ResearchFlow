"""Digest service — generate daily/weekly/monthly research summaries.

Daily:  what's new today, top 3 to read, decisions needing confirmation
Weekly: direction trends, structural vs patch ratio, branches that shifted
Monthly: strategy review, bottleneck ranking, directions to start/stop
"""

import logging
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import and_, desc, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.digest import Digest
from backend.models.enums import PaperState, PeriodType
from backend.models.paper import Paper
from backend.services.llm_service import call_llm

logger = logging.getLogger(__name__)

DIGEST_SYSTEM = """You are a research digest writer. Generate concise, actionable summaries in Chinese.
Focus on what changed, what matters, and what to do next.
Be specific — reference paper titles and concrete findings, not vague generalities."""

DAILY_PROMPT = """Based on the following research activity, generate a daily digest (今日总结).

Required sections:
## 今日新增
(list new papers added today with one-line description each)

## 推荐深读 (top 3)
(which 3 papers are most worth reading today, and why)

## 优先级变化
(any papers whose importance or priority shifted)

## 需要确认的判断
(any automated judgments that need human review)

## 后台建议任务
(what background tasks should run tonight)

Activity data:
{activity}

Output in Chinese Markdown."""

WEEKLY_PROMPT = """Based on this week's research activity, generate a weekly digest (周总结).

Required sections:
## 方向热度变化
(which topics gained or lost momentum this week)

## 结构性 vs 插件
(ratio of structural improvements to plugin patches in new papers)

## 新成形的路线
(any structural research directions that are starting to coalesce)

## 被推翻的旧判断
(any previous assessments that new evidence contradicts)

## 下周重点 (top 3)
(3 most important lines to follow next week)

Activity data:
{activity}

Output in Chinese Markdown."""

MONTHLY_PROMPT = """Based on this month's research activity, generate a monthly digest (月总结).

Required sections:
## 方向地图变化
(how the landscape of research directions shifted this month)

## 当前瓶颈排序
(rank the most pressing research bottlenecks)

## 值得立项的方向
(directions mature enough to start a project on)

## 可以停止追踪的方向
(directions that are saturated or no longer promising)

## 需要重分析的旧论文
(older papers that should be re-analyzed with newer models or taxonomy)

Activity data:
{activity}

Output in Chinese Markdown."""


async def generate_digest(
    session: AsyncSession,
    period_type: str,
    target_date: date | None = None,
) -> Digest:
    """Generate a digest for the specified period.

    Args:
        period_type: "day", "week", or "month"
        target_date: Reference date (defaults to today)
    """
    today = target_date or date.today()

    if period_type == "day":
        period_start = today
        period_end = today
        prompt_template = DAILY_PROMPT
    elif period_type == "week":
        period_start = today - timedelta(days=today.weekday())  # Monday
        period_end = period_start + timedelta(days=6)  # Sunday
        prompt_template = WEEKLY_PROMPT
    else:  # month
        period_start = today.replace(day=1)
        next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
        period_end = next_month - timedelta(days=1)
        prompt_template = MONTHLY_PROMPT

    # Gather activity data
    activity = await _gather_activity(session, period_start, period_end)

    # Generate digest via LLM
    prompt = prompt_template.format(activity=activity)
    resp = await call_llm(
        prompt=prompt,
        system=DIGEST_SYSTEM,
        max_tokens=2048,
        session=session,
        prompt_version=f"digest_{period_type}_v1",
    )

    # Check for existing digest in same period
    existing = await session.execute(
        select(Digest).where(
            Digest.period_type == PeriodType(period_type),
            Digest.period_start == period_start,
        )
    )
    old_digest = existing.scalar_one_or_none()
    if old_digest:
        # Update existing
        old_digest.rendered_text = resp.text
        old_digest.render_version = "v1"
        await session.flush()
        await session.refresh(old_digest)
        return old_digest

    # Create new digest
    # Gather source paper IDs
    paper_result = await session.execute(
        select(Paper.id).where(
            Paper.updated_at >= datetime.combine(period_start, datetime.min.time()).replace(tzinfo=timezone.utc),
            Paper.updated_at <= datetime.combine(period_end, datetime.max.time()).replace(tzinfo=timezone.utc),
        )
    )
    source_ids = [row[0] for row in paper_result.fetchall()]

    digest = Digest(
        period_type=PeriodType(period_type),
        period_start=period_start,
        period_end=period_end,
        source_paper_ids=source_ids if source_ids else None,
        rendered_text=resp.text,
        render_version="v1",
        metadata_={
            "model": f"{resp.provider}/{resp.model}",
            "tokens": {"input": resp.input_tokens, "output": resp.output_tokens},
        },
    )
    session.add(digest)
    await session.flush()
    await session.refresh(digest)
    return digest


async def get_latest_digest(session: AsyncSession, period_type: str) -> Digest | None:
    """Get the most recent digest of a given period type."""
    result = await session.execute(
        select(Digest)
        .where(Digest.period_type == PeriodType(period_type))
        .order_by(desc(Digest.period_start))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _gather_activity(
    session: AsyncSession,
    start: date,
    end: date,
) -> str:
    """Gather research activity data for the digest prompt."""
    start_dt = datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(end, datetime.max.time()).replace(tzinfo=timezone.utc)

    parts = []

    # New papers in period
    new_papers = await session.execute(
        select(Paper)
        .where(Paper.collected_at.between(start_dt, end_dt))
        .order_by(desc(Paper.keep_score))
        .limit(20)
    )
    papers = list(new_papers.scalars().all())
    if papers:
        parts.append(f"### 新增论文 ({len(papers)} 篇)")
        for p in papers:
            line = f"- {p.title} ({p.venue} {p.year}) [keep={p.keep_score:.2f}]" if p.keep_score else f"- {p.title} ({p.venue} {p.year})"
            if p.core_operator:
                line += f"\n  Core: {p.core_operator[:120]}"
            parts.append(line)

    # Analyzed papers in period
    analyzed = await session.execute(
        select(Paper)
        .where(
            Paper.analyzed_at.between(start_dt, end_dt),
            Paper.state.in_([PaperState.CHECKED, PaperState.L4_DEEP, PaperState.L3_SKIMMED]),
        )
        .limit(10)
    )
    analyzed_papers = list(analyzed.scalars().all())
    if analyzed_papers:
        parts.append(f"\n### 完成分析 ({len(analyzed_papers)} 篇)")
        for p in analyzed_papers:
            parts.append(f"- {p.title} (struct={p.structurality_score or 'N/A'})")

    # Category distribution
    cat_result = await session.execute(
        text("""
            SELECT category, count(*) as cnt,
                   avg(keep_score) as avg_keep,
                   avg(structurality_score) as avg_struct
            FROM papers
            WHERE collected_at BETWEEN :start AND :end
            GROUP BY category ORDER BY cnt DESC
        """),
        {"start": start_dt, "end": end_dt},
    )
    cats = cat_result.fetchall()
    if cats:
        parts.append("\n### 分类分布")
        for cat, cnt, avg_keep, avg_struct in cats:
            ak = f"{avg_keep:.2f}" if avg_keep else "N/A"
            ast = f"{avg_struct:.2f}" if avg_struct else "N/A"
            parts.append(f"- {cat}: {cnt} papers (avg_keep={ak}, avg_struct={ast})")

    # DeltaCards created in period
    dc_result = await session.execute(
        text("""
            SELECT count(*) as total,
                   count(CASE WHEN status = 'published' THEN 1 END) as published,
                   avg(structurality_score) as avg_struct
            FROM delta_cards
            WHERE created_at BETWEEN :start AND :end
        """),
        {"start": start_dt, "end": end_dt},
    )
    dc_row = dc_result.fetchone()
    if dc_row and dc_row[0] > 0:
        parts.append(f"\n### DeltaCard 产出")
        parts.append(f"- 新建: {dc_row[0]} 张 (已发布: {dc_row[1]})")
        if dc_row[2]:
            parts.append(f"- 平均结构性分数: {dc_row[2]:.2f}")

    # Graph assertions
    ga_result = await session.execute(
        text("""
            SELECT status, count(*) FROM graph_assertions
            WHERE created_at BETWEEN :start AND :end
            GROUP BY status
        """),
        {"start": start_dt, "end": end_dt},
    )
    ga_rows = ga_result.fetchall()
    if ga_rows:
        parts.append(f"\n### 图谱断言")
        for status, cnt in ga_rows:
            parts.append(f"- {status}: {cnt}")

    # Review queue status
    review_result = await session.execute(
        text("SELECT count(*) FROM review_tasks WHERE status IN ('pending', 'in_progress')")
    )
    pending = review_result.scalar() or 0
    if pending > 0:
        parts.append(f"\n### 待审核: {pending} 项")

    # Total KB stats
    total = await session.execute(text("SELECT count(*), count(CASE WHEN state='checked' THEN 1 END) FROM papers"))
    total_row = total.fetchone()
    dc_total = await session.execute(text("SELECT count(*) FROM delta_cards"))
    parts.append(f"\n### 知识库总量: {total_row[0]} papers, {total_row[1]} analyzed, {dc_total.scalar()} delta cards")

    return "\n".join(parts) if parts else "No activity in this period."
