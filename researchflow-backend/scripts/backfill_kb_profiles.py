"""Backfill kb_node_profiles for taxonomy_nodes that have none.

The kb_profiler agent was originally designed to run during deep_ingest on
fresh node candidates. For existing taxonomy nodes (loaded by an earlier
pipeline), we synthesize candidates from each node + the papers attached
via paper_facets, then batch-call the agent.

Usage:
    python -m scripts.backfill_kb_profiles                  # all unprofiled nodes
    python -m scripts.backfill_kb_profiles --batch-size 8   # tune batch
    python -m scripts.backfill_kb_profiles --limit 30       # cap total nodes
    python -m scripts.backfill_kb_profiles --dry-run        # don't write DB
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import Any
from uuid import UUID

from sqlalchemy import text

from backend.database import async_session
from backend.models.kb import KBNodeProfile
from backend.services.agent_runner import AgentRunner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


SQL_UNPROFILED_NODES = text("""
    SELECT t.id, t.name, t.name_zh, t.dimension, t.description
    FROM taxonomy_nodes t
    LEFT JOIN kb_node_profiles p
      ON p.entity_type = 'taxonomy_node' AND p.entity_id = t.id AND p.lang = 'zh'
    WHERE p.id IS NULL
    ORDER BY t.dimension, t.name
""")

SQL_NODE_PAPERS = text("""
    SELECT p.id, p.title, p.venue, p.year, dc.delta_statement
    FROM paper_facets f
    JOIN papers p ON p.id = f.paper_id
    LEFT JOIN delta_cards dc ON dc.id = p.current_delta_card_id
    WHERE f.node_id = :node_id
      AND p.state NOT IN ('skip', 'archived_or_expired')
    ORDER BY p.year DESC NULLS LAST
    LIMIT 8
""")


def _node_to_candidate(node, papers: list[Any]) -> dict:
    """Build the synthetic candidate the kb_profiler agent expects."""
    connected = []
    for p in papers:
        ds = (p.delta_statement or "").strip()
        if ds.lower().startswith("analysis of paper "):
            ds = ""
        connected.append({
            "title": (p.title or "")[:200],
            "venue": p.venue or "",
            "year": p.year,
            "claim": ds[:240] if ds else "",
        })
    return {
        "name": node.name,
        "name_zh": node.name_zh,
        "node_type": node.dimension,
        "existing_description": (node.description or "")[:400],
        "connected_papers": connected,
        "evidence_count": len(connected),
    }


async def _persist_profile(session, node_id: UUID, np: dict, run_id: UUID | None) -> None:
    """Upsert a single kb_node_profiles row for a taxonomy node."""
    profile = KBNodeProfile(
        entity_type="taxonomy_node",
        entity_id=node_id,
        lang="zh",
        one_liner=np.get("one_liner"),
        short_intro_md=np.get("short_intro_md"),
        detailed_md=np.get("detailed_md"),
        structured_json=np.get("structured_json"),
        evidence_refs=np.get("evidence_refs"),
        generated_by_run_id=run_id,
    )
    session.add(profile)


async def main(batch_size: int, limit: int | None, dry_run: bool) -> None:
    async with async_session() as session:
        nodes = (await session.execute(SQL_UNPROFILED_NODES)).fetchall()
        if limit:
            nodes = nodes[:limit]
        logger.info("Found %d unprofiled taxonomy nodes (batch=%d, dry=%s)",
                    len(nodes), batch_size, dry_run)
        if not nodes:
            return

        # Pre-load papers for every node so we batch the LLM calls
        node_with_candidates: list[tuple[Any, dict]] = []
        for n in nodes:
            papers = (await session.execute(SQL_NODE_PAPERS, {"node_id": str(n.id)})).fetchall()
            node_with_candidates.append((n, _node_to_candidate(n, papers)))

        runner = AgentRunner(session)
        total_persisted = 0
        total_failed = 0

        for i in range(0, len(node_with_candidates), batch_size):
            batch = node_with_candidates[i:i + batch_size]
            payload = {
                "node_candidates": [c for (_, c) in batch],
                "edge_candidates": [],
            }
            import json as _json
            user_content = (
                "Generate node profiles for the following taxonomy nodes.\n"
                "For each node, draw evidence ONLY from `connected_papers`.\n"
                "Match `node_name` in the output exactly to the candidate `name`.\n\n"
                f"{_json.dumps(payload, ensure_ascii=False)}"
            )
            context = {"user_content": user_content, "token_budget": 8192}

            t0 = time.monotonic()
            logger.info("Batch %d-%d / %d: calling kb_profiler",
                        i, i + len(batch), len(node_with_candidates))
            try:
                result = await runner.run_agent("kb_profiler", context)
            except Exception as e:
                logger.error("kb_profiler batch failed: %s", e)
                total_failed += len(batch)
                continue

            # Index result by node name (case-insensitive)
            np_by_name = {}
            for np in result.get("node_profiles", []) or []:
                key = (np.get("node_name") or "").strip().lower()
                if key:
                    np_by_name[key] = np

            run_id = None  # AgentRunner already created an AgentRun, but
            # the id isn't returned; leaving NULL is acceptable.

            persisted = 0
            for node, candidate in batch:
                np = np_by_name.get((node.name or "").lower())
                if not np:
                    logger.warning("kb_profiler did not return profile for %s", node.name)
                    continue
                if not (np.get("short_intro_md") or np.get("one_liner")):
                    logger.warning("Profile for %s is empty, skipping", node.name)
                    continue
                if not dry_run:
                    await _persist_profile(session, node.id, np, run_id)
                persisted += 1

            if not dry_run:
                await session.commit()
            total_persisted += persisted
            logger.info("Batch done in %.1fs — persisted %d/%d profiles",
                        time.monotonic() - t0, persisted, len(batch))

        logger.info("=== DONE: persisted=%d failed=%d total=%d ===",
                    total_persisted, total_failed, len(node_with_candidates))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.batch_size, args.limit, args.dry_run))
