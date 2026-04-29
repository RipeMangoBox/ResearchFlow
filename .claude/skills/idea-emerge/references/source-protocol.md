# Evidence Protocol

Use this protocol to add only sources that change idea generation.

## Source Tiers

| Tier | Source class | Default use |
|------|--------------|-------------|
| T0 | Local `paperAnalysis/` note, local PDF, reproducible local artifact, user-provided primary transcript | Evidence anchor |
| T1a | Paper, project page, official repo, benchmark, dataset, reproducible code, official leaderboard | S3 evidence anchor |
| T1b | Lab/course page, author page, official talk transcript | Decision-rule or context source |
| T2 | First-person researcher/engineer/professor/lab material with clear attribution | Decision-rule source |
| T3 | Interview, podcast, lecture recap, newsletter, curated thread with clear attribution | Context or decision-rule source |
| T4 | Social post, learning note, community advice, third-party summary | Discovery hint only |
| T5 | Uncited claim, anonymous screenshot, unverifiable repost | Ignore unless user asks to preserve it |

Only T0/T1a sources can directly satisfy the `S3` evidence gate. T1b-T5 sources can shape decision rules, context, or discovery hints, but cannot alone justify an `S3` idea candidate.

## Retrieval Lanes

### Local KB

Start here.

Use:

- `paperCollection/index.jsonl` for fast filtering if present;
- `paperAnalysis/` for full notes;
- frontmatter fields: `core_operator`, `primary_logic`, tags, venue, year, task path.

Retrieve:

- task-local papers: same problem/domain;
- method-local papers: same operator, different task;
- cross-domain papers: same bottleneck, metric, or workflow pattern;
- negative evidence: missing baselines, fragile assumptions, saturated benchmarks, failures.

### Task-Core Web Papers

Use only after recording the latest matching KB evidence date/year.

Keep web sources that satisfy at least one gate:

- published newer than the latest matching KB evidence;
- fills a missing task, method, dataset, metric, implementation, or negative-evidence gap;
- provides official code/project/benchmark details absent from the KB note;
- is explicitly requested by the user.

Prioritize:

- arXiv, OpenReview, ACL Anthology, CVF, ACM, IEEE, official conference pages;
- project pages, official GitHub repos, benchmark leaderboards, datasets;
- author, lab, and course pages.

Discard a source unless it changes at least one of:

- a knowledge card;
- an evidence card;
- an operator/gap card;
- baseline or metric;
- risk or kill criterion.

Do not retrieve duplicate summaries of papers already covered in `paperAnalysis/`.

### Domain Bottleneck Diagnosis

Diagnose domain bottlenecks before composing candidates.

Prefer:

- repeated limitations across KB notes;
- survey, benchmark, and position papers;
- negative evidence, saturated metrics, missing baselines, and weak reproducibility;
- user-provided constraints when explicit.

Convert diagnoses into `bottleneck_claim`, `rejection_rule`, and `evidence_bar` decision-rule cards.

T1b/T2 material can help phrase a decision rule, but it cannot replace KB, survey, benchmark, or negative-evidence support.

### Cross-Domain Frontier

Search by transferable mechanism, not only by task keyword.

Useful mechanisms:

- controllability, evaluation, synthetic data, reward guidance;
- tool use, planning, retrieval, self-verification, scientific discovery agents;
- preference learning, curriculum, exploration, active data collection;
- workflow instrumentation, human-in-loop evaluation, interaction cost;
- simulation-to-real, constraint satisfaction, embodiment, 3D/4D representation.

Extract preconditions before transferring an operator.

### Decision-Rule Sources

Use attributed T1b/T2 material only to extract explicit rules.

Valid extraction targets:

- problem selector: what problem type gets priority;
- bottleneck claim: what blocks progress;
- rejection rule: what idea/baseline/metric/framing is considered weak;
- evidence bar: what evidence is required;
- transfer trigger: when a method can move domains;
- experiment shape: smallest test setup.

Invalid extraction targets:

- style imitation;
- persona imitation;
- unsupported taste claims;
- popularity or follower counts.

### Implementation Traces

Use for feasibility and hidden constraints:

- repos, issues, PRs, discussions, release notes;
- benchmark code and leaderboards;
- demos, reproduction reports, ablations, failure reports.

Extract:

- what already works;
- what is brittle;
- what is expensive;
- what is missing from paper evaluations;
- what a first experiment can reuse.

## Generation Guardrails

- Require a nearest related paper and a named baseline.
- Require a named domain bottleneck and at least one passed rejection rule before `S3`.
- Keep only sources that change cards, scores, baseline, metric, risk, or validation plan.
- Score fixed dimensions: evidence, novelty, feasibility, specificity, significance, falsifiability.
- Require data path, code path, metric, and kill criterion before `S3`.
- Revise one weak dimension per iteration.
- Require every accepted candidate to contain a mechanism and a falsifiable experiment.
- Preserve rejected clusters with missing evidence.

## Ledger Fields

Every source entry should include:

- `id`: stable local ID such as `S01`;
- `tier`: T0, T1a, T1b, T2, T3, T4, or T5;
- `lane`: KB, web-paper, cross-domain, decision-rule, implementation, negative;
- `kb_status`: in_kb, newer_than_kb, fills_kb_gap, duplicate, not_checked;
- `keep_reason`: why this source changes generation;
- `gap_filled`: task, method, dataset, metric, implementation, negative_evidence, or none;
- `author_or_org`;
- `title`;
- `date_published`;
- `date_collected`;
- `link_or_local_path`;
- `extracted_claim`;
- `extracted_use`: knowledge, evidence, decision_rule, operator, gap, risk;
- `confidence`: high, medium, low;
- `notes`: access limitations or caveats.

## Source Score

Score each source from 0-10:

- relevance: 0-2;
- credibility: 0-2;
- novelty: 0-2;
- actionability: 0-2;
- transfer value or freshness: 0-2.

Use thresholds:

- `8-10`: use directly in synthesis.
- `5-7`: use as support.
- `1-4`: keep only if it explains a rejected path.
- `0`: discard.

## Attribution

- Cite links in Markdown.
- Prefer paraphrase over long quotation.
- Distinguish source claims from agent inference.
- Preserve uncertainty when sources conflict.
