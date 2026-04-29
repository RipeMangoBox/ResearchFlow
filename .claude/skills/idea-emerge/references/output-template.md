# Output Template

Use this template for `paperIDEAs/YYYY-MM-DD_idea-emerge-<slug>.md`.

```markdown
---
created: {{ISO_DATETIME_NOW}}
updated: {{ISO_DATETIME_NOW}}
type: idea-emerge
stage: evidence-to-idea
topic: {{TOPIC}}
---

# {{YYYY-MM-DD}} Idea Emerge: {{TOPIC}}

> Goal: generate research idea candidates from KB evidence, domain bottleneck diagnosis, task-core papers, cross-domain operators, decision rules, and implementation constraints.

## 1. Search Contract

- Task core:
- Domain bottleneck hypothesis:
- Decision-rule source preference:
- Contribution type:
- Hard constraints:
- Desired output:
- Minimum evidence bar:
- Latest matching KB evidence publication/submission date or venue/year:

## 2. Domain Bottleneck Diagnosis

- Diagnosis status: KB-grounded / assumption / web-grounded

### B1. {{Bottleneck label}}

- source ids:
- bottleneck claim:
- why it blocks progress:
- evidence pattern:
- rejection rule derived:
- evidence bar derived:

## 3. Evidence Ledger

| ID | Tier | Lane | KB status | Keep reason | Gap filled | Confidence | Source | Date | Extracted use | Link or path |
|----|------|------|-----------|-------------|------------|------------|--------|------|---------------|--------------|
| S01 | T0 | KB | in_kb |  | none | high |  |  | knowledge |  |
| S02 | T1a | web-paper | newer_than_kb |  |  |  |  |  | evidence |  |
| S03 | T2 | decision-rule | not_checked |  | none |  |  |  | decision_rule |  |

## 4. Knowledge and Evidence Cards

### K1. {{Knowledge or paper title}}

- source id:
- established fact or method:
- task context:
- core operator:
- primary logic:
- limitation:
- how it changes idea generation:

### E1. {{Evidence title}}

- source id:
- evidence type:
- result:
- strength:
- caveat:
- what it permits:
- what it does not prove:
- how it changes idea generation:

## 5. Decision-Rule Cards

### D1. {{Rule label}}

- source id:
- rule type:
- rule:
- source basis:
- applicable condition:
- non-applicable condition:
- how it filters candidates:

## 6. Operator, Gap, and Contradiction Cards

### Operator Cards

- O1:

### Gap Cards

- G1:

### Contradiction Cards

- X1:

## 7. Idea Candidates

### C1. {{Candidate title}}

- maturity:
- score:
- score breakdown: evidence / novelty / feasibility / specificity / significance / falsifiability
- hard gates:
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

## 8. Rejected or Parked Ideas

| Candidate | Reason | Missing evidence |
|-----------|--------|------------------|
|  |  |  |

## 9. Iteration Log

| Iteration | Candidate | Weak dimension | Change | Old score | New score | Decision |
|-----------|-----------|----------------|--------|-----------|-----------|----------|
| I0 |  | baseline | initial score | - |  |  |

## 10. Next Step

- Recommended next skill:
- Immediate action:
```
