"""Backfill paper.method_family from title for papers where the agent
extraction silently failed.

Root cause: ~70% of analyzed papers (`state in l4_deep / l3_skimmed`) end up
with `method_family = NULL` because the deep_analyzer agent fails for
structural-ring papers (context build fallback → JSON parse → empty dict).
The KB method graph collapses without this signal.

This script applies the same title-based fallback that
`concept_synthesizer_service.extract_method_from_title` does on the forward
path, but to existing rows. It is idempotent and only touches rows where
`method_family IS NULL`.

Run via:
    docker compose exec -T api python -m scripts.backfill_method_family
"""

from __future__ import annotations

import asyncio
import logging
import sys

from sqlalchemy import func, select, text

from backend.database import async_session
from backend.models.method import MethodNode
from backend.models.paper import Paper
from backend.services.concept_synthesizer_service import extract_method_from_title

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_method_family")


async def main(dry_run: bool = False) -> dict:
    async with async_session() as session:
        rows = (await session.execute(text("""
            SELECT id, title, category
            FROM papers
            WHERE state IN ('l4_deep', 'l3_skimmed')
              AND method_family IS NULL
              AND title IS NOT NULL
        """))).all()

        stats = {
            "examined": len(rows),
            "extracted": 0,
            "method_node_reused": 0,
            "method_node_created": 0,
            "skipped_no_match": 0,
        }
        sample: list[tuple[str, str]] = []

        for paper_id, title, category in rows:
            family_name = extract_method_from_title(title)
            if not family_name:
                stats["skipped_no_match"] += 1
                continue

            stats["extracted"] += 1
            if len(sample) < 8:
                sample.append((title[:60], family_name))

            if dry_run:
                continue

            # Find or create MethodNode (case-insensitive match on canonical name).
            mf = (await session.execute(
                select(MethodNode).where(
                    func.lower(MethodNode.name) == family_name.lower()
                ).limit(1)
            )).scalar_one_or_none()

            if mf is None:
                mf = (await session.execute(
                    select(MethodNode).where(
                        MethodNode.aliases.any(family_name)
                    ).limit(1)
                )).scalar_one_or_none()

            if mf is None:
                mf = MethodNode(
                    name=family_name,
                    type="mechanism_family",
                    domain=category,
                    description=f"Method introduced in: {title[:200]}",
                )
                session.add(mf)
                await session.flush()
                stats["method_node_created"] += 1
            else:
                stats["method_node_reused"] += 1

            await session.execute(
                text("UPDATE papers SET method_family = :mf WHERE id = :pid"),
                {"mf": mf.name, "pid": str(paper_id)},
            )

        if not dry_run:
            await session.commit()

    logger.info("Sample extractions:")
    for title, name in sample:
        logger.info("  %-60s → %s", title, name)
    logger.info("Stats: %s", stats)
    return stats


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    asyncio.run(main(dry_run=dry))
