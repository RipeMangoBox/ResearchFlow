# Idea Maturity

Use this reference to convert evidence and explicit decision rules into research idea candidates.

## Maturity Ladder

| Stage | Name | Definition | Required next action |
|-------|------|------------|----------------------|
| S0 | signal | A knowledge point, result, limitation, constraint, decision rule, or reaction | Find related signals |
| S1 | pattern | Related signals recur across KB, web papers, implementation traces, or decision-rule sources | Form a falsifiable hypothesis |
| S2 | hypothesis | A claim can be tested but setup is incomplete | Identify data, baseline, metric |
| S3 | idea candidate | Mechanism, evidence, novelty boundary, baseline, metric, data path, and kill criterion are visible | Hand off to focus or review |
| S4 | project plan | Scope, experiments, resources, and stop-loss criteria are ready | Execute or write proposal |

Do not promote an idea to `S3` without T0/T1a evidence or a concrete plan to obtain it.

## Card Schemas

### Knowledge Card

- source id:
- established fact or method:
- task context:
- core operator:
- primary logic:
- limitation:
- how it changes idea generation:

### Evidence Card

- source id:
- evidence type: benchmark, ablation, theorem, user study, repo behavior, dataset, failure report
- result:
- strength:
- caveat:
- what it permits:
- what it does not prove:
- how it changes idea generation:

### Decision-Rule Card

- source id:
- rule type: problem_selector, bottleneck_claim, rejection_rule, evidence_bar, transfer_trigger, experiment_shape
- rule:
- source basis: KB, survey, benchmark, negative evidence, user-provided, or T1b/T2 context
- applicable condition:
- non-applicable condition:
- how it filters candidates:

### Operator Card

- operator name:
- source ids:
- invariant:
- preconditions:
- transformation needed for this task:
- failure modes:
- candidate applications:

### Gap Card

- gap name:
- source ids:
- missing variable:
- who cares:
- why existing work misses it:
- why now:
- smallest test:

### Contradiction Card

- contradiction:
- source A:
- source B:
- possible reconciliation:
- experiment or retrieval needed:

### Idea Candidate

- title:
- maturity: S1, S2, or S3
- core hypothesis:
- mechanism:
- evidence anchors:
- domain bottleneck attacked:
- decision rules applied:
- rejection rules passed:
- novelty boundary:
- smallest experiment:
- data path:
- code path:
- baseline:
- metric:
- expected failure mode:
- kill criterion:
- next skill:

## Composition Moves

Use these concrete moves:

1. Limitation inversion: make a paper's limitation the task.
2. Evaluation invention: turn an unmeasured property into a metric, benchmark, or protocol.
3. Operator transfer: move a mechanism across fields only when preconditions match.
4. Decision-rule filtering: reject candidates that violate explicit evidence bars or rejection rules.
5. Evidence triangulation: combine one KB fact, one primary web paper, and one implementation trace.
6. Data bottleneck replacement: replace expensive labels with retrieval, simulation, weak supervision, self-checking, active learning, or human-in-loop signals.
7. Failure mining: turn repeated failures into training data, stress tests, or controllers.
8. Implementation lift: turn repo issue, scaling trick, tool limitation, or benchmark bottleneck into a research question.
9. Contradiction test: design an experiment that decides between two plausible claims.
10. Workflow formalization: convert tacit workflow into data, reward, tool use, interaction protocol, or evaluation.

## Quality Gates

For each `S3` candidate, answer:

- Importance: who benefits if this works?
- Bottleneck: which named domain bottleneck does it attack?
- Novelty: what nearest paper or obvious baseline does it need to beat?
- Mechanism: why should the method work?
- Evidence: which T0/T1a sources support feasibility or gap?
- Rule fit: which decision rules selected this candidate, and which rejection rules does it pass?
- Testability: are data path, code path, baseline, metric, and kill criterion visible?
- Risk: what would falsify the idea quickly?

If two or more answers are weak, keep the candidate at `S2`.
