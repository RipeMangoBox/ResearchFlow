"""Infer method_edges from paper_relations.

When paper A's `method_family=GRPO` cites paper B with relation_type
`direct_baseline` and paper B's `method_family=PPO`, we infer the
method_edge:  PPO  ──direct_baseline──▶  GRPO

This builds the method-evolution DAG (PPO → GRPO → DPO → ...) that the
vault export already knows how to render but has no data to render from.

Idempotent via UNIQUE(parent_method_id, child_method_id, relation_type).
No LLM calls — pure SQL inference.

Usage:
    python -m scripts.infer_method_lineage           # do it
    python -m scripts.infer_method_lineage --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from uuid import UUID, uuid4

from sqlalchemy import text as sa_text

from backend.database import async_session

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Relation types from reference_role we treat as method-evolution edges
LINEAGE_ROLES = ("direct_baseline", "method_source", "formula_source",
                 "comparison_baseline")


SQL_FETCH_RELATIONS = sa_text("""
    SELECT pr.id, pr.source_paper_id, pr.target_paper_id, pr.relation_type,
           pr.evidence,
           sp.method_family AS src_method,
           tp.method_family AS tgt_method
    FROM paper_relations pr
    JOIN papers sp ON sp.id = pr.source_paper_id
    JOIN papers tp ON tp.id = pr.target_paper_id
    WHERE pr.relation_type = ANY(:roles)
      AND sp.method_family IS NOT NULL AND sp.method_family <> ''
      AND tp.method_family IS NOT NULL AND tp.method_family <> ''
""")


def _method_canonical(name: str) -> str:
    """Strip parenthetical expansion: 'AFL (Analytic Federated Learning)' -> 'AFL'."""
    s = (name or "").strip()
    s = re.sub(r"\s*\(.*\)\s*$", "", s)
    return s.strip() or name


async def _ensure_method_node(session, name: str, sample_paper_id: UUID) -> UUID:
    """Get or create a method_nodes row for `name`. Used as fallback when the
    source/target paper's method_family doesn't yet have a method_nodes row."""
    cname = _method_canonical(name)
    row = (await session.execute(sa_text("""
        SELECT id FROM method_nodes
        WHERE lower(name) = lower(:n)
        LIMIT 1
    """), {"n": cname})).fetchone()
    if row:
        return row[0]
    new_id = uuid4()
    await session.execute(sa_text("""
        INSERT INTO method_nodes (id, name, type, maturity, domain, downstream_count,
                                  canonical_paper_id, created_at, updated_at)
        VALUES (:id, :n, 'mechanism_family', 'seed', NULL, 0,
                :canon, now(), now())
    """), {"id": str(new_id), "n": cname, "canon": str(sample_paper_id)})
    return new_id


async def main(dry_run: bool) -> None:
    async with async_session() as session:
        rows = (await session.execute(
            SQL_FETCH_RELATIONS, {"roles": list(LINEAGE_ROLES)}
        )).fetchall()
        logger.info("Found %d candidate paper_relations to project to method_edges",
                    len(rows))
        if not rows:
            return

        inserted = 0
        skipped_self = 0
        for r in rows:
            src_canon = _method_canonical(r.src_method)
            tgt_canon = _method_canonical(r.tgt_method)
            if src_canon.lower() == tgt_canon.lower():
                skipped_self += 1
                continue

            # Direction: edge goes FROM the cited (older) method TO the citing
            # (newer) one. relation_type=direct_baseline means src cited tgt
            # as a baseline → tgt is parent of src.
            parent_name = tgt_canon
            child_name = src_canon
            parent_paper = r.target_paper_id
            child_paper = r.source_paper_id

            if dry_run:
                logger.info("DRY: %s ──%s──> %s", parent_name, r.relation_type, child_name)
                continue

            parent_id = await _ensure_method_node(session, parent_name, parent_paper)
            child_id = await _ensure_method_node(session, child_name, child_paper)
            try:
                await session.execute(sa_text("""
                    INSERT INTO method_edges (
                      id, parent_method_id, child_method_id,
                      relation_type, delta_description,
                      created_at
                    ) VALUES (
                      gen_random_uuid(), :p, :c,
                      :rel, :delta,
                      now()
                    )
                    ON CONFLICT DO NOTHING
                """), {
                    "p": str(parent_id), "c": str(child_id),
                    "rel": r.relation_type,
                    "delta": (r.evidence or "")[:300],
                })
                inserted += 1
            except Exception as e:
                logger.warning("edge insert failed: %s", str(e)[:120])

        if not dry_run:
            await session.commit()
        logger.info("=== DONE: inserted=%d skipped_self=%d ===", inserted, skipped_self)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    asyncio.run(main(args.dry_run))
