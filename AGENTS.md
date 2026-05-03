# Agent Guide

> Architecture & data model: [ARCHITECTURE.md](researchflow-backend/ARCHITECTURE.md)
> Deployment: [DEPLOY.md](researchflow-backend/DEPLOY.md)
> Analysis plan & execution contract: [analysis_plan.md](docs/analysis_plan.md)

## Source of truth

**PostgreSQL is the only write target.** `paperAnalysis/`, `paperCollection/`, `obsidian-vault/` are read-only exports. For queries, use `/api/v1/search/*`, not local files.

## Architecture: 4 Layers

```
Layer A: Faceted Taxonomy DAG
  taxonomy_nodes (domain/task/dataset/benchmark/modality/...)
  taxonomy_edges (is_a / part_of / uses)
  paper_facets (paper ↔ taxonomy_node with role)

Layer B: Method Evolution DAG
  method_nodes (algorithm/recipe/model_family/mechanism_family)
  method_edges (extends/modifies_slot/replaces/combines_with)
  method_applications (paper uses method with role)

Layer C: Paper Layer
  papers → delta_cards → evidence_units → graph_assertions
  Paper is the container; DeltaCard is the structured "what changed"

Layer D: Cross-paper Abstraction (Phase 2)
  canonical_ideas, bottlenecks, lineage
```

## 6-Agent Pipeline

```
Candidate → import_and_score()
    ↓ (DiscoveryScore ≥ 75 → shallow_ingest)

Shallow Phase (2 LLM calls):
  1. shallow_extractor → paper_essence + method_delta
  2. reference_role → reference classifications + anchor_baselines
    ↓ (deterministic DeepIngestScore ≥ 88 → deep_ingest)

Deep Phase (2 LLM calls):
  3. deep_analyzer → method slots/pipeline + experiment + formulas
  4. graph_candidate → node/edge/lineage candidates
    ↓ (scoring: node ≥ 75, edge ≥ 70)

Profile Phase (1 LLM call):
  5. kb_profiler → node + edge wiki profiles (batched)

Report Phase (1 LLM call):
  6. paper_report → 10-section structured report

Materialization (pure DB):
  _materialize_to_graph() → DeltaCard + EvidenceUnit + GraphAssertion
  link_to_parent_baselines() → DeltaCardLineage
  synthesize_concepts() → MethodNode + CanonicalIdea
  reconcile_neighbors() → same_family updates
```

## What each Agent needs

| Agent | Input | Output | Token Budget |
|-------|-------|--------|-------------|
| shallow_extractor | abstract + intro/method/experiment excerpts | paper_essence + method_delta | 18K |
| reference_role | reference_list + citation_contexts | classifications + anchor_baselines | 30K |
| deep_analyzer | full text + L2 formulas + shallow results | method/experiment/formulas | 40K |
| graph_candidate | all prior outputs + graph summary | node/edge/lineage candidates | 20K |
| kb_profiler | qualifying candidates | node/edge wiki profiles | 20K |
| paper_report | ALL verified blackboard items | 10-section report | 80K |

## DB Tables (40 tables)

### Core pipeline writes to:
- `papers` — metadata + state
- `paper_analyses` — L2/L3 extraction results
- `delta_cards` — structured "what changed" (truth layer)
- `evidence_units` — atomic evidence with confidence
- `graph_nodes` + `graph_assertions` — knowledge graph
- `method_nodes` — method/mechanism entities
- `agent_runs` + `agent_blackboard_items` — agent tracking
- `paper_facets` — taxonomy links
- `kb_node_profiles` + `kb_edge_profiles` — wiki pages

### Unchanged:
- `metadata_observations` — multi-source metadata ledger
- `paper_candidates` + `candidate_scores` — discovery pipeline
- `delta_card_lineage` — method evolution DAG

## Rules

1. All writes go to backend API, never edit Markdown files as source
2. For queries, prefer `/api/v1/search/*` over reading local files
3. Analysis language default: `zh` (override per request)
4. Pipeline steps are idempotent — already-completed steps are auto-skipped
5. Metadata observations are append-only — canonical resolver picks best value
6. DeltaCard publish gate: evidence_refs ≥ 2
7. Planned analysis batches must declare goal, source, selection rule, budget, and output target before agents run
8. Agents must consume only declared context and preserve source anchors in blackboard/DB outputs
9. Deep analysis runs only after deterministic DeepIngestScore promotion; graph candidates must pass node/edge score gates
10. Paper reports and profiles must be generated from verified blackboard items, not from new unsupported claims
11. Generated exports, snapshots, backups, local storage, and symlinks stay out of Git
