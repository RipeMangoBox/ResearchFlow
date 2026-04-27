"""Backfill paper_facets with canonical task labels via lightweight LLM.

Most papers in the DB have NO `task` dimension facet — only modality
(`Image`, `Text`) + dataset. This makes the vault graph useless for
"methods grouped by task" because the voting always falls to venue.

This script asks the LLM (1 call/paper, ~$0.003) to classify each paper
into 1-3 canonical task labels, then upserts:
  * a `taxonomy_node` (dimension='task') for each task name (or reuses one)
  * a `paper_facets` row linking paper → task node

Idempotent: skips papers that already have any task facet.

Usage:
    python -m scripts.backfill_paper_tasks
    python -m scripts.backfill_paper_tasks --limit 30
    python -m scripts.backfill_paper_tasks --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json as _json
import logging
import re
import time
from uuid import UUID, uuid4

from sqlalchemy import text as sa_text

from backend.database import async_session
from backend.services.agent_runner import call_llm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


CANONICAL_TASKS = [
    "Image Classification", "Object Detection", "Semantic Segmentation",
    "Instance Segmentation", "Depth Estimation",
    "Image Generation", "Video Generation", "Image Editing",
    "Image Captioning", "Visual Question Answering", "Visual Reasoning",
    "Cross-Modal Retrieval", "Cross-Modal Matching",
    "Video Understanding", "Action Recognition", "Action Segmentation",
    "Object Tracking", "3D Reconstruction", "Pose Estimation",
    "Speech Recognition", "Speech Synthesis", "Audio Generation",
    "Text Classification", "Text Generation", "Machine Translation",
    "Reasoning", "Code Generation", "Math Reasoning",
    "Reinforcement Learning", "Imitation Learning", "Federated Learning",
    "Continual Learning", "Domain Adaptation", "Few-Shot Learning",
    "Self-Supervised Learning", "Contrastive Learning",
    "Adversarial Robustness", "OOD Detection", "Anomaly Detection",
    "Neural Architecture Search", "Model Compression", "Pruning",
    "Quantization", "Knowledge Distillation",
    "Agent", "Embodied AI", "Robotics", "Autonomous Driving",
    "Medical Imaging", "Time Series Forecasting",
    "Recommender System", "Graph Learning", "Knowledge Graph",
    "Benchmark / Evaluation", "Fairness", "Privacy", "Interpretability",
]


SQL_PAPERS_NEEDING_TASKS = sa_text("""
    SELECT p.id, p.title, p.abstract, p.method_family, p.venue
    FROM papers p
    WHERE p.title IS NOT NULL
      AND p.state NOT IN ('skip', 'archived_or_expired')
      AND NOT EXISTS (
        SELECT 1 FROM paper_facets pf
        JOIN taxonomy_nodes tn ON tn.id = pf.node_id
        WHERE pf.paper_id = p.id AND tn.dimension = 'task'
          AND tn.name NOT LIKE '\\_\\_%' ESCAPE '\\'
      )
    ORDER BY p.created_at DESC
""")


PROMPT = """Classify the following paper into 1-3 canonical research tasks.

Pick from this list (exact spelling — use one of these strings verbatim):
{task_list}

Rules:
- A "task" is the PROBLEM being solved (e.g., "Object Detection"), NOT the
  paper's proposed method/system name.
- If the paper is purely a benchmark/dataset, use "Benchmark / Evaluation".
- Use "Reasoning" for general LLM reasoning; "Math Reasoning" for math-specific.
- Pick 1-3 tasks; choose the most central one first.
- If absolutely none fits, output a SHORT (≤4 words) canonical task name.

Paper:
  Title: {title}
  Method/system name: {method}
  Venue: {venue}
  Abstract: {abstract}

Output ONLY JSON (no markdown):
{{"tasks": ["task1", "task2", ...]}}
"""


async def _ensure_task_node(session, name: str) -> UUID:
    """Idempotent get-or-create taxonomy_node for a task."""
    row = (await session.execute(sa_text("""
        SELECT id FROM taxonomy_nodes
        WHERE dimension = 'task' AND lower(name) = lower(:n) LIMIT 1
    """), {"n": name})).fetchone()
    if row:
        return row[0]
    new_id = uuid4()
    await session.execute(sa_text("""
        INSERT INTO taxonomy_nodes (id, name, dimension, status, created_at)
        VALUES (:id, :name, 'task', 'active', now())
    """), {"id": str(new_id), "name": name})
    return new_id


async def _upsert_facet(session, paper_id: UUID, node_id: UUID) -> None:
    await session.execute(sa_text("""
        INSERT INTO paper_facets (id, paper_id, node_id, facet_role)
        VALUES (gen_random_uuid(), :p, :n, 'primary')
        ON CONFLICT DO NOTHING
    """), {"p": str(paper_id), "n": str(node_id)})


async def classify_one(session, paper, dry_run: bool) -> dict:
    abstract = (paper.abstract or "")[:1500]
    prompt = PROMPT.format(
        task_list=", ".join(CANONICAL_TASKS),
        title=paper.title or "",
        method=paper.method_family or "",
        venue=paper.venue or "",
        abstract=abstract or "(no abstract)",
    )
    try:
        resp = await call_llm(
            prompt=prompt,
            system="You are a precise research-task classifier. Output JSON only.",
            max_tokens=200, temperature=0.0,
            session=session, paper_id=paper.id,
            prompt_version="task_classify_v1",
        )
        text = (resp.text or "").strip()
        # Strip code fences if any
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n", "", text)
            text = re.sub(r"\n```$", "", text)
        obj = _json.loads(text)
        tasks = obj.get("tasks", [])
        tasks = [t.strip() for t in tasks if isinstance(t, str) and t.strip()][:3]
    except Exception as e:
        return {"paper_id": str(paper.id), "error": str(e)[:120]}

    if not tasks:
        return {"paper_id": str(paper.id), "skipped": "no_tasks_returned"}
    if dry_run:
        return {"paper_id": str(paper.id), "title": paper.title[:50], "tasks": tasks, "dry": True}

    for t in tasks:
        node_id = await _ensure_task_node(session, t)
        await _upsert_facet(session, paper.id, node_id)
    return {"paper_id": str(paper.id), "title": paper.title[:50], "tasks": tasks}


async def main(limit: int | None, dry_run: bool) -> None:
    async with async_session() as session:
        rows = (await session.execute(SQL_PAPERS_NEEDING_TASKS)).fetchall()
        if limit:
            rows = rows[:limit]
        logger.info("Found %d papers needing task facet (dry=%s)", len(rows), dry_run)
        if not rows:
            return

        n_ok = n_fail = 0
        for i, p in enumerate(rows, 1):
            t0 = time.monotonic()
            res = await classify_one(session, p, dry_run)
            if not dry_run:
                await session.commit()
            if "error" in res:
                n_fail += 1
            else:
                n_ok += 1
            logger.info("[%d/%d] %.1fs %s", i, len(rows), time.monotonic() - t0, res)

        logger.info("=== DONE: ok=%d fail=%d total=%d ===", n_ok, n_fail, len(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    asyncio.run(main(args.limit, args.dry_run))
