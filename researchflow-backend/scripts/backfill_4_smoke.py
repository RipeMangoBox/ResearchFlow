"""Backfill reference_role + materialize_relations + baseline_promote
for the 4 CVPR 2025 smoke papers that missed the new path."""
import asyncio
import time
from uuid import UUID

from sqlalchemy import text as sa_text

from backend.database import async_session
from backend.services.agent_runner import AgentRunner
from backend.services.paper_relation_service import materialize_for_paper
from backend.services.baseline_promoter import promote_for_paper


IDS = [
    "4a7f90e4-a93f-4ba7-a05d-61b8c2cdc0fb",
    "a40f3ec1-4ada-48d6-a412-816131d91ba4",
    "fbfe19bf-44e3-49a0-a231-eeed0b196727",
    "5a8b159d-d6b4-4be3-aea0-2388cdae9752",
]


async def run_one(pid: str) -> dict:
    async with async_session() as s:
        row = (await s.execute(sa_text("""
            SELECT evidence_spans FROM paper_analyses
            WHERE paper_id = :p AND level = 'l2_parse' AND is_current
            ORDER BY created_at DESC LIMIT 1
        """), {"p": pid})).fetchone()
        if not row or not row[0]:
            return {"pid": pid, "skipped": "no_l2"}
        refs = (row[0] or {}).get("grobid_references") or []
        lines = []
        for i, r in enumerate(refs[:30], 1):
            if isinstance(r, dict) and r.get("title"):
                lines.append(f"[{i}] {r['title'][:200]}")
        if not lines:
            return {"pid": pid, "skipped": "no_titles"}
        refs_text = "\n".join(lines)

        runner = AgentRunner(s)
        try:
            await runner.run_agent(
                "reference_role",
                {"user_content": f"References:\n{refs_text}",
                 "token_budget": 8192},
                paper_id=UUID(pid),
            )
            await s.commit()
        except Exception as e:
            await s.rollback()
            return {"pid": pid, "agent_err": str(e)[:200]}

        rel = await materialize_for_paper(s, UUID(pid))
        await s.commit()
        bp = await promote_for_paper(s, UUID(pid))
        await s.commit()
        return {"pid": pid, "rel": rel, "bp": bp}


async def main() -> None:
    for pid in IDS:
        t0 = time.monotonic()
        r = await run_one(pid)
        print(f"[{pid[:8]}] in {round(time.monotonic() - t0, 1)}s -> {r}")


if __name__ == "__main__":
    asyncio.run(main())
