"""End-to-end test: run the unified pipeline on all papers in DB.

Sequence:
1. List all papers (checked + wait state)
2. For each paper, run IngestWorkflow.run_for_existing_paper()
3. Verify outputs at each stage
4. Print summary report

Usage: python scripts/run_test_30.py [--dry-run] [--limit N] [--paper-id UUID]
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from uuid import UUID

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_30")


async def verify_paper(session, paper_id: UUID) -> dict:
    """Check what pipeline outputs exist for a paper."""
    from sqlalchemy import text

    checks = {}

    # Paper metadata
    row = (await session.execute(text(
        "SELECT title, abstract IS NOT NULL as has_abstract, "
        "authors IS NOT NULL as has_authors, code_url, cited_by_count, "
        "acceptance_type, ring, current_delta_card_id IS NOT NULL as has_dc, "
        "method_family "
        "FROM papers WHERE id = :pid"
    ), {"pid": paper_id})).fetchone()

    if not row:
        return {"error": "not found"}

    checks["title"] = row.title[:50]
    checks["has_abstract"] = row.has_abstract
    checks["has_authors"] = row.has_authors
    checks["code_url"] = bool(row.code_url)
    checks["cited_by_count"] = row.cited_by_count
    checks["acceptance_type"] = row.acceptance_type
    checks["ring"] = row.ring
    checks["has_delta_card"] = row.has_dc
    checks["method_family"] = row.method_family

    # L2 Parse
    row = (await session.execute(text(
        "SELECT extracted_sections IS NOT NULL as has_secs, "
        "extracted_formulas IS NOT NULL as has_formulas, "
        "figure_captions IS NOT NULL as has_figs "
        "FROM paper_analyses WHERE paper_id = :pid AND level = 'l2_parse' AND is_current = true"
    ), {"pid": paper_id})).fetchone()
    checks["l2_sections"] = row.has_secs if row else False
    checks["l2_formulas"] = row.has_formulas if row else False
    checks["l2_figures"] = row.has_figs if row else False

    # L4 Deep (new agent-generated)
    row = (await session.execute(text(
        "SELECT full_report_md IS NOT NULL as has_report, "
        "length(full_report_md) as report_len, schema_version "
        "FROM paper_analyses WHERE paper_id = :pid AND level = 'l4_deep' AND is_current = true"
    ), {"pid": paper_id})).fetchone()
    checks["l4_report"] = row.has_report if row else False
    checks["l4_report_len"] = row.report_len if row else 0
    checks["l4_schema"] = row.schema_version if row else None

    # Agent blackboard
    row = (await session.execute(text(
        "SELECT count(*) FROM agent_blackboard_items WHERE paper_id = :pid"
    ), {"pid": paper_id})).scalar()
    checks["blackboard_items"] = row or 0

    # Delta cards
    row = (await session.execute(text(
        "SELECT count(*) FROM delta_cards WHERE paper_id = :pid"
    ), {"pid": paper_id})).scalar()
    checks["delta_cards"] = row or 0

    # Evidence units
    row = (await session.execute(text(
        "SELECT count(*) FROM evidence_units WHERE paper_id = :pid"
    ), {"pid": paper_id})).scalar()
    checks["evidence_units"] = row or 0

    # Graph assertions
    row = (await session.execute(text(
        "SELECT count(*) FROM graph_assertions ga "
        "JOIN graph_nodes gn ON ga.from_node_id = gn.id "
        "WHERE gn.ref_table = 'delta_cards' AND gn.ref_id IN "
        "(SELECT id FROM delta_cards WHERE paper_id = :pid)"
    ), {"pid": paper_id})).scalar()
    checks["graph_assertions"] = row or 0

    # Taxonomy facets
    row = (await session.execute(text(
        "SELECT count(*) FROM paper_facets WHERE paper_id = :pid"
    ), {"pid": paper_id})).scalar()
    checks["taxonomy_facets"] = row or 0

    # Paper report
    row = (await session.execute(text(
        "SELECT count(*) FROM paper_reports WHERE paper_id = :pid"
    ), {"pid": paper_id})).scalar()
    checks["paper_reports"] = row or 0

    return checks


async def run_pipeline_for_paper(session, paper_id: UUID, dry_run: bool = False) -> dict:
    """Run the pipeline and return verification results."""
    from backend.services.ingest_workflow import IngestWorkflow

    if dry_run:
        return await verify_paper(session, paper_id)

    workflow = IngestWorkflow(session)
    try:
        result = await workflow.run_for_existing_paper(paper_id)
        await session.commit()
    except Exception as e:
        logger.error("Pipeline failed for %s: %s", paper_id, e)
        await session.rollback()
        return {"error": str(e)[:200]}

    # Verify after pipeline
    verification = await verify_paper(session, paper_id)
    verification["pipeline_result"] = {
        k: v for k, v in result.get("phases", {}).items()
        if not isinstance(v, dict) or "error" not in v
    }
    return verification


async def main(args):
    from backend.database import async_session
    from sqlalchemy import text

    async with async_session() as session:
        # Get papers to process
        if args.paper_id:
            paper_ids = [UUID(args.paper_id)]
        else:
            rows = (await session.execute(text(
                "SELECT id FROM papers ORDER BY year DESC NULLS LAST, title LIMIT :limit"
            ), {"limit": args.limit})).fetchall()
            paper_ids = [r.id for r in rows]

        logger.info("Processing %d papers (dry_run=%s)", len(paper_ids), args.dry_run)

        results = {}
        for i, pid in enumerate(paper_ids, 1):
            logger.info("[%d/%d] Processing %s", i, len(paper_ids), pid)
            try:
                result = await run_pipeline_for_paper(session, pid, dry_run=args.dry_run)
                results[str(pid)] = result

                # Print one-line status
                title = result.get("title", "?")[:40]
                bb = result.get("blackboard_items", 0)
                dc = result.get("delta_cards", 0)
                eu = result.get("evidence_units", 0)
                tf = result.get("taxonomy_facets", 0)
                logger.info(
                    "  %s | BB=%d DC=%d EU=%d TF=%d",
                    title, bb, dc, eu, tf,
                )
            except Exception as e:
                logger.error("  FAILED: %s", e)
                results[str(pid)] = {"error": str(e)[:200]}

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        total = len(results)
        has_abstract = sum(1 for r in results.values() if r.get("has_abstract"))
        has_l2 = sum(1 for r in results.values() if r.get("l2_sections"))
        has_bb = sum(1 for r in results.values() if r.get("blackboard_items", 0) > 0)
        has_dc = sum(1 for r in results.values() if r.get("delta_cards", 0) > 0)
        has_eu = sum(1 for r in results.values() if r.get("evidence_units", 0) > 0)
        has_ga = sum(1 for r in results.values() if r.get("graph_assertions", 0) > 0)
        has_tf = sum(1 for r in results.values() if r.get("taxonomy_facets", 0) > 0)
        has_rpt = sum(1 for r in results.values() if r.get("paper_reports", 0) > 0)
        errors = sum(1 for r in results.values() if "error" in r)

        print(f"Total papers:       {total}")
        print(f"Errors:             {errors}")
        print(f"Has abstract:       {has_abstract}/{total}")
        print(f"Has L2 parse:       {has_l2}/{total}")
        print(f"Has blackboard:     {has_bb}/{total}")
        print(f"Has delta_card:     {has_dc}/{total}")
        print(f"Has evidence:       {has_eu}/{total}")
        print(f"Has assertions:     {has_ga}/{total}")
        print(f"Has taxonomy:       {has_tf}/{total}")
        print(f"Has report:         {has_rpt}/{total}")

        # Save detailed results
        out_path = Path(__file__).parent / "test_30_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only verify, don't run pipeline")
    parser.add_argument("--limit", type=int, default=30, help="Max papers to process")
    parser.add_argument("--paper-id", type=str, help="Process a single paper by UUID")
    args = parser.parse_args()

    asyncio.run(main(args))
