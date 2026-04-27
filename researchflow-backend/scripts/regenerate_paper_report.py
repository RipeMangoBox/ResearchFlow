"""Re-run only the paper_report agent for a list of paper_ids.

Skips L2/L3/L4 re-extraction. Loads existing blackboard items
(shallow_extract / deep_analysis / reference_role_map / graph_candidates),
rebuilds the report context with the new figures + metadata blocks, calls the
paper_report agent, and persists via IngestWorkflow._persist_paper_report.

Usage:
    python -m scripts.regenerate_paper_report PAPER_ID [PAPER_ID ...]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from uuid import UUID

from sqlalchemy import select

from backend.database import async_session
from backend.models.agent import AgentBlackboardItem
from backend.models.paper import Paper
from backend.services.ingest_workflow import IngestWorkflow

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Blackboard item_type → key used inside agent_results
ITEM_TYPE_TO_KEY = {
    "shallow_extract": "shallow_extract",
    "deep_analysis": "deep_analysis",
    "reference_role_map": "reference_role_map",
    "graph_candidates": "graph_candidates",
    "kb_profiles": "kb_profiles",
}


async def _load_agent_results(session, paper_id: UUID) -> dict:
    """Load latest blackboard item per type for this paper."""
    rows = (await session.execute(
        select(AgentBlackboardItem)
        .where(AgentBlackboardItem.paper_id == paper_id)
        .order_by(AgentBlackboardItem.created_at.desc())
    )).scalars().all()

    out: dict = {}
    for r in rows:
        key = ITEM_TYPE_TO_KEY.get(r.item_type, r.item_type)
        # First (most recent) row per type wins; skip if already populated
        if key in out:
            continue
        out[key] = r.value_json or {}
    return out


async def _load_existing_report(session, paper_id: UUID) -> str:
    """Pull the existing L4 full_report_md (likely in the legacy 6-section
    format) so the new agent has source material to reorganize from.
    Returns "" if no L4 row.
    """
    from sqlalchemy import text as sa_text
    row = (await session.execute(sa_text("""
        SELECT full_report_md FROM paper_analyses
        WHERE paper_id = :pid AND level = 'l4_deep' AND is_current
        ORDER BY created_at DESC LIMIT 1
    """), {"pid": str(paper_id)})).fetchone()
    if not row or not row[0]:
        return ""
    raw = row[0]
    # Drop the figure_placements trailer if it was added by an earlier run.
    import re as _re
    raw = _re.sub(r"<!--\s*figure_placements:.*?-->", "", raw, flags=_re.DOTALL)
    return raw.strip()


async def regenerate_one(paper_id: UUID) -> dict:
    async with async_session() as session:
        paper = (await session.execute(
            select(Paper).where(Paper.id == paper_id)
        )).scalar_one_or_none()
        if not paper:
            return {"paper_id": str(paper_id), "error": "paper not found"}

        wf = IngestWorkflow(session)
        agent_results = await _load_agent_results(session, paper_id)
        figures_block = await wf._build_figures_block(paper_id)
        metadata_block = wf._build_paper_metadata_block(paper)
        existing_report = await _load_existing_report(session, paper_id)

        # When the paper has no agent blackboard (most legacy papers),
        # pass the existing full_report_md as the analysis source so the
        # agent can reorganize into the new 7-section structure with
        # figure markers, instead of hallucinating from scratch.
        if existing_report:
            artifacts_block = (
                "Existing L4 report (legacy format — REORGANIZE its content "
                "into the new 7-section structure; do NOT invent new facts):\n\n"
                + existing_report[:18000]  # cap to keep prompt within budget
            )
        else:
            artifacts_block = str(agent_results) if agent_results else "(no prior artifacts)"

        report_context = {
            "user_content": (
                f"Paper title: {paper.title}\n\n"
                f"=== Paper metadata (use in section 1 metadata_overview) ===\n"
                f"{metadata_block}\n\n"
                f"=== Figures available (use EXACT labels in figure_placements.preferred_labels) ===\n"
                f"{figures_block}\n\n"
                f"=== Agent analysis artifacts ===\n"
                f"{artifacts_block}\n"
            ),
            "token_budget": 8192,
        }

        try:
            report_result = await wf.runner.run_agent(
                "paper_report", report_context, paper_id=paper_id,
            )
        except Exception as e:
            await session.rollback()
            return {"paper_id": str(paper_id), "error": f"agent failed: {e}"}

        try:
            await wf._persist_paper_report(paper_id, report_result)
            await session.commit()
        except Exception as e:
            await session.rollback()
            return {"paper_id": str(paper_id), "error": f"persist failed: {e}"}

        sections = report_result.get("sections", [])
        placements = report_result.get("figure_placements", [])
        return {
            "paper_id": str(paper_id),
            "title": paper.title[:80],
            "title_zh": report_result.get("title_zh", ""),
            "section_count": len(sections),
            "section_types": [s.get("section_type", "") for s in sections],
            "figure_placements": len(placements),
            "n_artifacts": len(agent_results),
        }


async def main(paper_ids: list[str], progress_path: str | None = None) -> None:
    import json as _json
    import time as _time
    from pathlib import Path as _Path

    results = []
    progress_file = _Path(progress_path) if progress_path else None
    if progress_file:
        progress_file.parent.mkdir(parents=True, exist_ok=True)

    n = len(paper_ids)
    started = _time.monotonic()

    for i, pid in enumerate(paper_ids, 1):
        try:
            uid = UUID(pid)
        except ValueError:
            res = {"paper_id": pid, "error": "invalid UUID"}
            results.append(res)
            continue

        t0 = _time.monotonic()
        logger.info("[%d/%d] Regenerating %s", i, n, pid)
        try:
            res = await regenerate_one(uid)
        except Exception as e:
            res = {"paper_id": pid, "error": f"unhandled: {e}"[:300]}
        dur = round(_time.monotonic() - t0, 1)
        res["duration_s"] = dur
        results.append(res)

        elapsed = _time.monotonic() - started
        eta_min = round((elapsed / i) * (n - i) / 60, 1)
        logger.info(" → ok=%s sections=%d figs=%d in %ss (eta %s min)",
                    "error" not in res,
                    res.get("section_count", 0),
                    res.get("figure_placements", 0),
                    dur, eta_min)

        if progress_file:
            progress_file.write_text(_json.dumps(results, ensure_ascii=False, indent=2))

    print("\n=== SUMMARY ===")
    ok = sum(1 for r in results if "error" not in r)
    failed = [r for r in results if "error" in r]
    print(f"OK: {ok}/{n}   FAILED: {len(failed)}")
    for r in failed[:10]:
        print(f"  {r['paper_id']}: {r['error']}")


def _read_id_file(path: str) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_ids", nargs="*")
    parser.add_argument("--from-file", help="Read paper_ids from file (one per line)")
    parser.add_argument("--progress", help="Write progress JSON here after each paper")
    args = parser.parse_args()

    ids = list(args.paper_ids)
    if args.from_file:
        ids.extend(_read_id_file(args.from_file))
    if not ids:
        parser.error("Provide paper_ids as args or via --from-file")
    asyncio.run(main(ids, progress_path=args.progress))
