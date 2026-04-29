# Iteration Protocol

Use this protocol to close the loop after candidate generation.

## Candidate Scoring

Score each idea candidate from 1-5 on each dimension:

| Dimension | Meaning |
|-----------|---------|
| evidence | grounded in T0/T1a papers, code, data, benchmarks, or reproducible artifacts |
| novelty | not a trivial recombination or already-standard baseline |
| feasibility | first experiment is realistic with available resources |
| specificity | task, data, baseline, and metric are concrete |
| significance | solves a real research or practitioner pain point |
| falsifiability | has a clear kill criterion |

Use this rubric:

| Dimension | 1 | 3 | 5 |
|-----------|---|---|---|
| evidence | no T0/T1a anchor | one relevant T0/T1a anchor but weak link to mechanism | multiple direct anchors or one strong anchor plus code/data support |
| novelty | nearest work/baseline unknown | plausible gap versus named work | clear novelty boundary against nearest work and obvious baselines |
| feasibility | no data/code path | data or code path exists but setup is uncertain | data, code/implementation path, compute, and first run are concrete |
| specificity | vague task or method | task, method, and metric named | task, method, data, baseline, metric, and experiment are concrete |
| significance | local curiosity only | useful for a recognizable subcommunity | important bottleneck with clear user/research impact |
| falsifiability | no kill criterion | qualitative failure condition | measurable kill criterion tied to baseline/metric |

Interpretation:

- 24-30: possible `S3` if all hard gates pass.
- 18-23: promising `S2`, run one more retrieval or refinement iteration.
- 12-17: weak hypothesis, keep as backlog.
- below 12: kill or archive as rejected.

Hard gates for `S3`:

- total score >= 24;
- evidence >= 4;
- novelty >= 3;
- feasibility >= 3;
- specificity >= 4;
- significance >= 4;
- falsifiability >= 4;
- evidence anchor directly supports the mechanism or gap;
- domain bottleneck attacked is named;
- at least one explicit rejection rule is passed;
- nearest related paper named;
- baseline, metric, data path, code path or implementation path, and kill criterion present.

If any hard gate fails, label the candidate `S2` even if total score is high.

## Loop

For each iteration:

1. Pick one weak dimension.
2. Make one targeted change.
3. Add or remove evidence, baseline, metric, data path, code path, or kill criterion.
4. Rescore.
5. Keep the change only if the total score improves, a hard gate becomes satisfied, or uncertainty drops with no score loss.

Do not preserve changes that make the idea more vague, less testable, or less grounded.

## Independent Checks

When possible, perform one independent check:

- Retrieve local KB evidence with `papers-query-knowledge-base`.
- Check the domain bottleneck against survey, benchmark, repeated limitation, or negative-evidence sources.
- Compare against a strong baseline paper.
- Search recent primary sources for duplication.
- Ask `reviewer-stress-test` to attack only the top candidate.

## Iteration Log Format

Use this compact log inside the output note:

| Iteration | Candidate | Weak dimension | Change | Old score | New score | Decision |
|-----------|-----------|----------------|--------|-----------|-----------|----------|
| I0 | C1 | baseline | initial score | - | 21 | revise |
| I1 | C1 | specificity | added baseline and metric | 21 | 25 | keep |

Do not use aliased Obsidian wikilinks inside the table.

## Stop Conditions

Stop when:

- one or more candidates reach strong `S3`;
- all candidates are below 18 and need more evidence retrieval;
- the user asked only for source synthesis;
- the next action clearly belongs to another skill.
