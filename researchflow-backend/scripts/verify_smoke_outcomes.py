"""Comprehensive smoke verification.

Run AFTER smoke_full_ingest finishes for any set of paper_ids.
Checks 8 outcomes per paper and aggregates pass/fail.

Usage:
    python -m scripts.verify_smoke_outcomes <paper_id> [<paper_id> ...]
    python -m scripts.verify_smoke_outcomes --from-file ids.txt
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from uuid import UUID

from sqlalchemy import text as sa_text

from backend.database import async_session

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


CHECKS_SQL = sa_text("""
    SELECT
      p.id, p.title, p.state,
      EXISTS(SELECT 1 FROM paper_analyses
             WHERE paper_id=p.id AND level='l2_parse' AND is_current
               AND extracted_figure_images IS NOT NULL)            AS has_l2,
      EXISTS(SELECT 1 FROM paper_analyses
             WHERE paper_id=p.id AND level='l4_deep' AND is_current
               AND full_report_md IS NOT NULL)                      AS has_l4_report,
      (SELECT count(*) FROM paper_figures WHERE paper_id=p.id)      AS n_pfig,
      (SELECT title_zh FROM paper_reports
         WHERE paper_id=p.id ORDER BY created_at DESC LIMIT 1)      AS title_zh,
      (SELECT count(*) FROM paper_relations
         WHERE source_paper_id=p.id)                                 AS n_relations_out,
      (SELECT count(*) FROM paper_candidates
         WHERE discovered_from_paper_id=p.id)                        AS n_candidates,
      (SELECT count(*) FROM paper_analyses
         WHERE paper_id=p.id AND level='l4_deep' AND is_current
           AND full_report_md ~ '\\{\\{TBL:')                         AS report_has_tbl_marker,
      (SELECT count(*) FROM paper_analyses
         WHERE paper_id=p.id AND level='l4_deep' AND is_current
           AND full_report_md ~ '\\}\\}\\$\\$')                       AS report_has_bad_latex
    FROM papers p
    WHERE p.id = :pid
""")


CHECK_LABELS = [
    ("L2 parse exists", "has_l2"),
    ("L4 report written", "has_l4_report"),
    ("paper_figures rows ≥3", "n_pfig"),  # threshold 3
    ("title_zh present", "title_zh"),
    ("paper_relations out ≥1", "n_relations_out"),  # threshold 1
    ("paper_candidates from refs ≥1", "n_candidates"),
    ("report has {{TBL:}} marker", "report_has_tbl_marker"),
    ("LaTeX clean (no `}}$$`)", "report_has_bad_latex"),  # invert
]


def evaluate(row) -> dict:
    """Score the row against the 8 checks."""
    out = {"paper_id": str(row.id), "title": (row.title or "")[:60]}
    out["pass"] = []
    out["fail"] = []

    if row.has_l2:                         out["pass"].append("L2")
    else:                                  out["fail"].append("L2")

    if row.has_l4_report:                  out["pass"].append("L4")
    else:                                  out["fail"].append("L4")

    if (row.n_pfig or 0) >= 3:             out["pass"].append(f"figs={row.n_pfig}")
    else:                                  out["fail"].append(f"figs={row.n_pfig}")

    if row.title_zh:                       out["pass"].append("title_zh")
    else:                                  out["fail"].append("title_zh")

    if (row.n_relations_out or 0) >= 1:    out["pass"].append(f"rel={row.n_relations_out}")
    else:                                  out["fail"].append(f"rel={row.n_relations_out}")

    if (row.n_candidates or 0) >= 1:       out["pass"].append(f"cand={row.n_candidates}")
    else:                                  out["fail"].append(f"cand={row.n_candidates}")

    if (row.report_has_tbl_marker or 0):   out["pass"].append("TBL_marker")
    else:                                  out["fail"].append("TBL_marker")

    if (row.report_has_bad_latex or 0) == 0: out["pass"].append("LaTeX_clean")
    else:                                    out["fail"].append("LaTeX_dirty")

    out["score"] = f"{len(out['pass'])}/8"
    return out


async def main(paper_ids: list[str]) -> None:
    pass_count = fail_count = 0
    async with async_session() as session:
        for pid in paper_ids:
            try:
                uid = UUID(pid)
            except ValueError:
                logger.warning("invalid uuid: %s", pid); continue
            row = (await session.execute(CHECKS_SQL, {"pid": str(uid)})).one_or_none()
            if not row:
                print(f"  {pid[:8]}: NOT FOUND in papers table")
                fail_count += 1
                continue
            res = evaluate(row)
            ok = len(res["fail"]) <= 2  # allow 2 minor misses
            status = "✓" if ok else "✗"
            print(f"  {status} {res['paper_id'][:8]} [{res['score']}] {res['title']}")
            if res["fail"]:
                print(f"     fail: {', '.join(res['fail'])}")
            if ok:
                pass_count += 1
            else:
                fail_count += 1
    print(f"\n=== TOTAL: pass={pass_count} fail={fail_count} ===")


def _read_ids(path: str) -> list[str]:
    with open(path) as f:
        return [s.strip() for s in f if s.strip() and not s.startswith("#")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paper_ids", nargs="*")
    p.add_argument("--from-file")
    args = p.parse_args()
    ids = list(args.paper_ids)
    if args.from_file:
        ids.extend(_read_ids(args.from_file))
    if not ids:
        p.error("provide paper_ids or --from-file")
    asyncio.run(main(ids))
