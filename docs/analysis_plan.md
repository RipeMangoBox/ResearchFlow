# Analysis Plan and Agent Execution Contract

This document is the human-readable overview for planned paper analysis runs and
the execution contract each agent must follow. The detailed architecture remains
in `researchflow-backend/ARCHITECTURE.md`; this file explains how to turn that
architecture into controlled analysis work.

## Source of Truth

- PostgreSQL is the only write target for analysis state, graph state, reports,
  and scheduling decisions.
- `paperAnalysis/`, `paperCollection/`, and `obsidian-vault/` are exports. They
  may be regenerated and must not be edited as source material.
- Planning and querying must use backend APIs, especially `/api/v1/search/*` and
  `/api/v1/pipeline/*`, rather than local Markdown exports.
- Analysis language defaults to `zh` unless a request or API payload explicitly
  sets another language.

## Planning Overview

An analysis run is planned as a batch with a declared goal, input source, and
selection rule before agents start. A valid batch records:

- Goal: the research direction, venue slice, method family, or dataset question.
- Source: candidate import, existing papers, search results, or manual IDs.
- Selection rule: deterministic filter and priority score, such as
  `DiscoveryScore >= 75`, venue/year, domain, or explicit paper IDs.
- Budget: maximum papers, expected depth, and stop conditions.
- Output target: graph materialization, paper report, profile refresh, or export.

The minimum batch lifecycle is:

1. Select candidates through backend APIs.
2. Import and score candidates.
3. Run shallow analysis for qualified papers.
4. Promote only papers that pass deterministic deep-ingest gates.
5. Materialize verified claims into DeltaCard, evidence, and graph tables.
6. Generate reports/profiles from verified blackboard items.
7. Export Obsidian/Markdown views only after DB state is complete.

## Agent Execution Constraints

Each agent owns one bounded transformation. It must consume only its declared
inputs, write structured outputs to the blackboard or DB-backed services, and
leave enough evidence for downstream agents to audit the result.

| Agent | Required constraint | Promotion gate |
|-------|---------------------|----------------|
| shallow_extractor | Extract paper essence and method delta from abstract plus method/experiment excerpts; do not infer unsupported baselines. | Writes paper_essence and method_delta. |
| reference_role | Classify references from reference list and citation contexts; separate anchor baselines from related work and datasets. | Enables deterministic DeepIngestScore. |
| deep_analyzer | Read method and experiment evidence before abstract-level claims; extract slots, pipeline, experiments, and formulas with anchors. | Runs only after deep-ingest gate. |
| graph_candidate | Propose node, edge, and lineage candidates only from prior verified items plus graph summary. | Node score >= 75, edge score >= 70. |
| kb_profiler | Profile only qualifying graph candidates; batch related nodes/edges without merging unverified concepts. | Writes Chinese node/edge profiles. |
| paper_report | Build the 10-section report from verified blackboard items; no new unsupported claims. | Requires completed prior agent outputs. |

Cross-agent rules:

- Keep steps idempotent. If a step is already complete, reruns must skip or
  update through the service layer without duplicating records.
- Preserve source anchors for claims, formulas, experiments, and baseline
  comparisons.
- Do not promote a DeltaCard unless `evidence_refs >= 2`.
- Do not let a paper self-define its comparison set; use the baseline comparator
  and DB neighbors.
- Store metadata observations append-only; let the canonical resolver choose the
  best value.
- Never write derived analysis directly into Markdown exports.

## Operator Checklist

Before a batch:

- Confirm API, worker, PostgreSQL, Redis, and object storage are reachable.
- Record the batch goal, source, selection rule, budget, and expected output.
- Check whether the same paper IDs already have completed pipeline steps.

During a batch:

- Monitor `agent_runs`, `agent_blackboard_items`, and pipeline job state.
- Stop or review papers stuck in `review_deep`, failed parsing, or low evidence
  states before forcing deeper analysis.
- Treat truncation, missing anchors, or malformed JSON as retry/repair issues,
  not as successful analysis.

After a batch:

- Verify DeltaCard publish state and evidence counts.
- Verify graph candidate thresholds and profile generation.
- Run export only from DB materialized state.
- Commit only source, docs, and reproducible config; keep generated exports,
  snapshots, backups, local storage, and symlinks out of Git.
