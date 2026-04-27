"""Post-hoc cleanup of over-tagged task facets.

Symptom: top-3 task hubs (Benchmark / Evaluation: 123 papers, Agent: 77,
Reasoning: 55) are over-populated because the agent assigned a "context" tag
to papers whose primary contribution is actually a method on a different
task. C-TPT, OnlineTAS, Unified-IO 2, FlowerFormer all got tagged
"Benchmark / Evaluation" while not being benchmark papers themselves.

Cleanup heuristic — DEMOTE the over-tagged facet if:
  - paper has another, more-specific task tag (e.g., Image Classification),
    AND the over-tagged facet is one of the bloat-prone hubs,
  - OR paper has a method_family AND the over-tagged facet is
    "Benchmark / Evaluation" AND the title doesn't contain the words
    "benchmark" / "evaluation" / "leaderboard".

This script removes ONLY paper_facets rows; taxonomy_nodes stay so the agent
can re-assign them on future analyses. Idempotent.

Run:
    docker compose exec -T api python -m scripts.cleanup_overtagged_facets [--dry-run]
"""

from __future__ import annotations

import asyncio
import logging
import sys

from sqlalchemy import text

from backend.database import async_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cleanup_facets")


# Hubs that the agent over-tags. Each hub gets its own demotion rule.
BLOAT_HUBS: dict[str, str] = {
    "Benchmark / Evaluation": (
        "Demote unless paper title contains 'benchmark', 'evaluation', "
        "'leaderboard', 'metric', or 'eval'."
    ),
    "Agent": (
        "Demote when paper has a more specific task tag (Reasoning / Code "
        "Generation / VQA / etc.) and 'agent' is not in the title."
    ),
    "Reasoning": (
        "Demote when paper has a more specific reasoning tag (Math Reasoning, "
        "Visual Reasoning, Code Generation)."
    ),
}

# Specific-task tags that, when co-tagged with the bloat hub, justify removal.
SPECIFIC_TASK_TAGS = {
    "Image Classification", "Object Detection", "Semantic Segmentation",
    "Instance Segmentation", "Depth Estimation", "Image Generation",
    "Video Generation", "Image Editing", "Image Captioning",
    "Visual Question Answering", "Visual Reasoning", "Cross-Modal Retrieval",
    "Cross-Modal Matching", "Video Understanding", "Action Recognition",
    "Action Segmentation", "Object Tracking", "3D Reconstruction",
    "Pose Estimation", "Speech Recognition", "Speech Synthesis",
    "Audio Generation", "Text Classification", "Text Generation",
    "Machine Translation", "Code Generation", "Math Reasoning",
    "Reinforcement Learning", "Imitation Learning", "Federated Learning",
    "Continual Learning", "Domain Adaptation", "Few-Shot Learning",
    "Self-Supervised Learning", "Contrastive Learning",
    "Adversarial Robustness", "OOD Detection", "Anomaly Detection",
    "Neural Architecture Search", "Model Compression", "Pruning",
    "Quantization", "Knowledge Distillation", "Embodied AI", "Robotics",
    "Autonomous Driving", "Medical Imaging", "Time Series Forecasting",
    "Recommender System", "Graph Learning", "Knowledge Graph",
}


async def main(dry_run: bool = False) -> dict:
    stats = {
        "examined_facets": 0,
        "demoted_benchmark": 0,
        "demoted_agent": 0,
        "demoted_reasoning": 0,
    }

    async with async_session() as session:
        # Find paper_facets rows that link a paper to a bloat hub.
        rows = (await session.execute(text("""
            SELECT pf.paper_id, pf.node_id, t.name AS task_name,
                   p.title, p.method_family
            FROM paper_facets pf
            JOIN taxonomy_nodes t ON t.id = pf.node_id
            JOIN papers p ON p.id = pf.paper_id
            WHERE t.name = ANY(:hubs)
              AND p.state IN ('l4_deep', 'l3_skimmed')
        """), {"hubs": list(BLOAT_HUBS.keys())})).all()
        stats["examined_facets"] = len(rows)

        # Bulk-load each paper's other task tags.
        paper_ids = list({str(r.paper_id) for r in rows})
        if not paper_ids:
            logger.info("No bloat-hub facets found.")
            return stats

        other_tags_rows = (await session.execute(text("""
            SELECT pf.paper_id, t.name
            FROM paper_facets pf
            JOIN taxonomy_nodes t ON t.id = pf.node_id
            WHERE pf.paper_id = ANY(:pids)
              AND t.dimension IN ('task','domain','learning_paradigm')
        """), {"pids": paper_ids})).all()
        other_tags: dict[str, set[str]] = {}
        for pid, name in other_tags_rows:
            other_tags.setdefault(str(pid), set()).add(name)

        to_delete: list[tuple[str, str]] = []
        for r in rows:
            pid = str(r.paper_id)
            nid = str(r.node_id)
            hub = r.task_name
            title_lower = (r.title or "").lower()
            companions = other_tags.get(pid, set()) - {hub}
            has_method = bool(r.method_family)
            has_specific = bool(companions & SPECIFIC_TASK_TAGS)

            # Conservative demotion rules — must satisfy ALL conditions.
            demote = False
            if hub == "Benchmark / Evaluation":
                title_has_eval = any(w in title_lower for w in
                                     ("benchmark", "evaluation", "leaderboard",
                                      "metric", "eval"))
                # Demote only if: (paper proposes a method) AND (has specific
                # task tag) AND (title isn't about evaluation).
                demote = has_method and has_specific and not title_has_eval
            elif hub == "Agent":
                title_has_agent = ("agent" in title_lower or
                                   "agentic" in title_lower)
                # Demote only if: (paper proposes a method) AND (has specific
                # task tag) AND (title doesn't mention agent).
                demote = has_method and has_specific and not title_has_agent
            elif hub == "Reasoning":
                more_specific = {"Math Reasoning", "Visual Reasoning",
                                 "Code Generation"} & companions
                # Demote only if a more-specific reasoning tag exists.
                demote = bool(more_specific)

            if not demote:
                continue

            to_delete.append((pid, nid))
            if hub == "Benchmark / Evaluation":
                stats["demoted_benchmark"] += 1
            elif hub == "Agent":
                stats["demoted_agent"] += 1
            elif hub == "Reasoning":
                stats["demoted_reasoning"] += 1

        if dry_run:
            logger.info("Would delete %d facet rows", len(to_delete))
        else:
            for pid, nid in to_delete:
                await session.execute(text(
                    "DELETE FROM paper_facets WHERE paper_id=:p AND node_id=:n"
                ), {"p": pid, "n": nid})
            await session.commit()

    logger.info("Stats: %s", stats)
    return stats


if __name__ == "__main__":
    asyncio.run(main(dry_run="--dry-run" in sys.argv))
