"""Backfill paper_figures from PaperAnalysis.extracted_figure_images JSONB.

Idempotent — uses INSERT … ON CONFLICT DO NOTHING on (paper_id, label).
Run after migration 024 to migrate historical figures into the dedicated
table. The JSONB column is left untouched.

Usage:
    python -m scripts.backfill_paper_figures            # dry-run
    python -m scripts.backfill_paper_figures --commit   # actually write
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from sqlalchemy import text

from backend.database import async_session

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


SELECT_FIGURES = text("""
    SELECT paper_id, extracted_figure_images
    FROM paper_analyses
    WHERE level = 'l2_parse'
      AND is_current = true
      AND extracted_figure_images IS NOT NULL
      AND jsonb_typeof(extracted_figure_images) = 'array'
""")

INSERT_FIG = text("""
    INSERT INTO paper_figures (
        paper_id, label, type, semantic_role, page_num, bbox,
        object_key, public_url, caption, description,
        width, height, size_bytes, extraction_method
    )
    VALUES (
        :paper_id, :label, :type, :semantic_role, :page_num, CAST(:bbox AS jsonb),
        :object_key, :public_url, :caption, :description,
        :width, :height, :size_bytes, :extraction_method
    )
    ON CONFLICT (paper_id, label) DO NOTHING
""")


def _row_for_insert(paper_id: str, fig: dict[str, Any]) -> dict[str, Any] | None:
    object_key = fig.get("object_key") or ""
    label = (fig.get("label") or "").strip()
    if not object_key or not label:
        return None
    import json as _json
    bbox = fig.get("bbox")
    return {
        "paper_id": paper_id,
        "label": label[:64],
        "type": (fig.get("type") or "figure")[:16],
        "semantic_role": (fig.get("semantic_role") or "other")[:32],
        "page_num": fig.get("page_num"),
        "bbox": _json.dumps(bbox) if bbox is not None else None,
        "object_key": object_key[:500],
        "public_url": fig.get("public_url"),
        "caption": (fig.get("caption") or "")[:8000],
        "description": (fig.get("description") or "")[:8000],
        "width": fig.get("width"),
        "height": fig.get("height"),
        "size_bytes": fig.get("size_bytes"),
        "extraction_method": (fig.get("extraction_method") or "vlm_precise")[:32],
    }


async def main(commit: bool) -> None:
    async with async_session() as session:
        rows = (await session.execute(SELECT_FIGURES)).fetchall()
        logger.info("Inspecting %d L2 analysis rows with figures", len(rows))
        total_inserted = 0
        total_skipped = 0
        for r in rows:
            paper_id = str(r.paper_id)
            figs = r.extracted_figure_images
            if not isinstance(figs, list):
                continue
            for fig in figs:
                if not isinstance(fig, dict):
                    continue
                payload = _row_for_insert(paper_id, fig)
                if not payload:
                    total_skipped += 1
                    continue
                if commit:
                    await session.execute(INSERT_FIG, payload)
                total_inserted += 1
        if commit:
            await session.commit()
        mode = "INSERTED" if commit else "DRY-RUN candidates"
        logger.info("%s: %d rows; skipped %d", mode, total_inserted, total_skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true", help="actually write rows")
    args = parser.parse_args()
    asyncio.run(main(commit=args.commit))
