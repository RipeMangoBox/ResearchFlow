---
name: idea-emerge
description: Generates research ideas from ResearchFlow KB evidence, domain bottleneck diagnosis, task-specific web papers, cross-domain operators, explicit research-decision rules, implementation traces, and task constraints. Use before brainstorming or focus when the user wants idea candidates with evidence anchors, novelty checks, baselines, metrics, and kill criteria rather than paper summaries or generic inspiration.
---

# Idea Emerge

Generate research idea candidates from evidence, constraints, and reusable research operators.

Convert sources into structured cards, compose candidates from those cards, then score and revise candidates until they are testable or honestly parked.

## Evidence Priority

- Start with `paperCollection/index.jsonl` if present, then read matching `paperAnalysis/` notes.
- Extract only fields that affect generation: `core_operator`, `primary_logic`, limitation, evaluation setup, tags, baseline, metric, and missing proof.
- Diagnose domain bottlenecks from KB evidence before composing candidates.
- Add task-core web papers only when they are newer than the latest matching KB evidence, or when they fill a missing task/method/evaluation gap.
- Add implementation traces only when they change feasibility, baseline, metric, risk, data path, or code path.
- Preserve negative evidence and rejected candidates in the output note.

## Workflow

### 0. Define The Search Contract

Write a compact contract before generating:

- task core:
- domain bottleneck hypothesis:
- decision-rule source preference: KB, survey, benchmark, user-provided, or expert-material-as-context-only;
- target contribution type: method, benchmark, dataset, system, theory, application, or survey-to-experiment bridge;
- hard constraints: data, compute, timeline, code availability, venue, domain;
- desired output: 3-7 candidates by default, or one deep candidate if requested;
- minimum evidence bar: what source or experiment is required before promoting to `S3`.

If the user is vague, state one conservative assumption and proceed.

### 1. Retrieve KB Evidence

Use `papers-query-knowledge-base` silently when needed.

Retrieve:

- task-local papers: same task/domain;
- method-local papers: same operator, different task;
- cross-domain papers: same bottleneck or evaluation pattern in another field;
- negative papers or notes: failed assumptions, saturated benchmarks, missing baselines, weak reproducibility.

Extract only fields useful for idea generation:

- problem, bottleneck, operator, evidence, limitation;
- `core_operator`, `primary_logic`, tags, venue/year;
- baseline and metric;
- what the paper does not prove.
- latest matching KB evidence date/year. Prefer publication or submission date; fall back to venue/year only when no better date is available.

### 1.5 Diagnose Domain Bottlenecks

Before browsing for new papers, infer 2-3 domain bottleneck claims from KB evidence.

Prefer:

- survey papers, benchmark papers, position papers, and repeated limitations;
- negative evidence, saturated metrics, missing baselines, weak reproducibility, and evaluation failures;
- user-provided constraints when they are explicit.

Use T1b/T2 expert, lab, author, course, or talk material only as decision-rule context. Do not imitate a person, style, taste, or lab agenda.

Convert each diagnosis into one or more `Decision-rule card` entries:

- `bottleneck_claim`: what blocks progress;
- `rejection_rule`: what kind of candidate is too local, too incremental, or under-evidenced;
- `evidence_bar`: what proof is required before promotion.

Every candidate must name the domain bottleneck it attacks and the rejection rules it passes. If bottleneck evidence is thin, mark the diagnosis as an assumption and keep affected candidates below `S3`.

### 2. Retrieve Task-Core Web Evidence

Browse only after the KB pass.

Apply the newer-than-KB gate:

- record the latest matching KB evidence date/year, preferring publication or submission date over venue/year;
- keep web papers newer than that date/year;
- keep older web papers only if they fill a missing task, method, dataset, metric, implementation, or negative-evidence gap;
- do not retrieve duplicate paper summaries when the KB already has enough coverage.

Prefer primary sources:

- arXiv, OpenReview, ACL Anthology, CVF, ACM, IEEE, official conference pages;
- project pages, official repos, datasets, benchmark leaderboards;
- lab/course pages and author pages.

Filter retrieval noise before using it:

- discard sources that do not change a card, candidate, baseline, metric, or risk;
- mark claims from blogs/social posts as heuristic until backed by paper/code/data;
- record date and source type for fast-moving topics.

### 3. Build Cards

Use `references/idea-maturity.md` for schemas.

Create five required card types, plus optional contradiction cards when evidence conflicts:

- **Knowledge card**: established task/method fact from KB or primary paper.
- **Evidence card**: empirical result, ablation, dataset, benchmark, proof, code behavior, or failure report.
- **Decision-rule card**: explicit `problem_selector`, `bottleneck_claim`, `rejection_rule`, `evidence_bar`, `transfer_trigger`, or `experiment_shape` that filters candidates.
- **Operator card**: reusable mechanism that can move across tasks.
- **Gap card**: missing data, metric, setting, theory, interaction loop, or reproducibility condition.
- **Contradiction card**: conflicting evidence or assumptions that need a discriminating experiment.

Every card must have a source id and one sentence saying how it changes idea generation.

### 4. Compose Candidates

Generate candidates with this formula:

`domain bottleneck + gap + operator + evidence anchor + decision rule + smallest validation`

Use concrete composition moves:

- limitation inversion: make a known limitation the task;
- evaluation invention: turn an unmeasured property into a metric or benchmark;
- operator transfer: move a mechanism across task/domain only when preconditions match;
- implementation lift: turn repo/benchmark pain into a research question;
- contradiction test: design an experiment that resolves conflicting evidence;
- data bottleneck replacement: replace expensive labels with retrieval, simulation, weak supervision, self-checking, or human-in-loop signals;
- failure mining: turn repeated failures into training data, stress tests, or controllers.

Each candidate must include:

- title;
- maturity stage (`S1`, `S2`, or `S3`);
- core hypothesis;
- mechanism;
- evidence anchors;
- domain bottleneck attacked;
- decision rules applied;
- rejection rules passed;
- novelty boundary: what obvious baseline or existing paper it must beat;
- smallest experiment;
- data;
- baseline;
- metric;
- expected failure mode;
- kill criterion.

### 5. Score And Revise

Use `references/iteration-protocol.md`.

For each candidate:

1. Score evidence, novelty, feasibility, specificity, significance, and falsifiability with the 1-5 rubric.
2. Identify the weakest dimension.
3. Make one targeted revision only.
4. Keep the revision only if the candidate becomes more grounded, specific, or testable.
5. Kill or park candidates that lack an evidence path.

Promote to `S3` only if all hard gates pass: total score at least 24, evidence at least 4 with direct support for the mechanism or gap, novelty at least 3, feasibility at least 3, specificity at least 4, significance at least 4, falsifiability at least 4, a named domain bottleneck, at least one explicit rejection rule passed, plus visible nearest paper, baseline, metric, data path, code path or implementation path, and kill criterion.

### 6. Write The Note

Default path:

`paperIDEAs/YYYY-MM-DD_idea-emerge-<slug>.md`

For large runs, use:

`paperIDEAs/idea-emerge/YYYY-MM-DD_<slug>/emergence.md`

Use `references/output-template.md`. Follow the repo's Obsidian Markdown table rule: do not use aliased wikilinks inside tables.

### 7. Handoff

Suggest exactly one next skill:

- `research-brainstorm-from-kb`: expand one promising cluster into variants.
- `idea-focus-coach`: convert one `S3` candidate into a scoped MVP.
- `reviewer-stress-test`: attack a formed candidate.
- `papers-query-knowledge-base`: retrieve more local evidence when grounding is weak.

## Failure Modes

- If KB evidence is thin, mark the run as web-grounded and list the missing KB notes.
- If domain bottleneck diagnosis is weak, mark it as an assumption and keep affected candidates below `S3`.
- If web retrieval is noisy, stop retrieving and use only sources that alter a card or score.
- If expert/lab material yields only slogans, discard it or convert it into a testable rejection rule.
- If cross-domain transfer is speculative, keep the candidate at `S2` until preconditions, baseline, and metric are explicit.
- If no candidate reaches `S3`, output rejected clusters and the exact evidence needed next.

## References

- `references/source-protocol.md`: retrieval lanes, source tiers, and generation guardrails.
- `references/idea-maturity.md`: card schemas, composition moves, and maturity ladder.
- `references/iteration-protocol.md`: scoring rubric and revision loop.
- `references/output-template.md`: Markdown output template for `paperIDEAs/`.
