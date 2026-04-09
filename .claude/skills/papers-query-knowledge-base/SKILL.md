---
name: papers-query-knowledge-base
description: Queries the local `paperAnalysis/` notes for research. Finds papers by title, task path, technique tags, venue, or year; summarizes or compares methods; cites core_operator and primary_logic from analysis frontmatter. `paperCollection/` is optional and mainly serves statistics, Obsidian navigation, and backlink exploration.
---

# Paper Knowledge Base (paperAnalysis-first)

Use this skill to query the local paper corpus across conversations and projects. The main agent-facing layer is **paperAnalysis** (analysis notes with TL;DR, Part I/II/III, PDF links). **paperCollection** is optional and mainly useful for statistics pages, Obsidian jumps, and backlink-friendly browsing.

**Where the knowledge base lives:** Under the current repository root that contains `paperAnalysis/` and `paperPDFs/`. `paperCollection/` may also exist as a generated companion layer.

## Paths

- Use repo-relative paths rooted at the folder that contains `paperAnalysis/` and `paperPDFs/`.
- When invoking from another workspace, replace these with the correct absolute repository path for your machine.

## Statistics (current sample values)

| Dimension | Count |
|-----------|--------|
| Papers | 254 |
| Tasks | 7 |
| Technique tags | 587 |
| Venues | 17 |

Tasks: Human_Human_Interaction, Human_Object_Interaction, Human_Scene_Interaction, Motion_Controlled_ImageVideo_Generation, Motion_Editing, Motion_Generation_Text_Speech_Music_Driven, Motion_Stylization.

## Where things live

Relative to the repository root:

- **Analysis notes**: `paperAnalysis/<Category>/<Venue_Year>/<Year>_<Title>.md`; PDF path in frontmatter `pdf_ref`.
- **Optional navigation / stats pages**: `paperCollection/README.md`, `_AllPapers.md`, `by_task/`, `by_technique/`, `by_venue/`.

All paths use forward slashes.

## How to use for research

1. **Find papers** — Search `paperAnalysis/` directly by title, task folder, tags, venue, year, `core_operator`, or `primary_logic`. Prefer frontmatter/body evidence over generated index pages.

2. **Read an analysis** — Open the matched analysis note directly. Each note has: **Frontmatter** (title, venue, year, category, tags, pdf_ref, core_operator, primary_logic); **TL;DR** (Summary, Key Performance); **Part I** (problem); **Part II** (method, "Aha!"); **Part III** (technical); **Local Reading** (PDF).

3. **Use `paperCollection/` only when helpful** — For task / technique / venue overview pages, statistics, Obsidian jumps, or backlink exploration.

4. **Cite in answers** — Use **core_operator** and **primary_logic** from frontmatter for one-line method summary; **Summary** and **Key Performance** from TL;DR for comparison. Link to note: `[[paperAnalysis/.../file.md|Title]]`; to PDF: `[[paperPDFs/.../file.pdf|PDF]]`.

## Index frontmatter (for tooling)

Collection pages: `type: paper-index`, `dimension: task | technique | venue | all`, and optional `task:` / `technique:` / `venue:`.

For exact paths and frontmatter schema, see [references/structure.md](references/structure.md).

## Regenerating the collection (optional)

If you want refreshed `paperCollection/` pages for statistics or Obsidian navigation, from the repository root:

```bash
python .claude/skills/papers-build-collection-index/scripts/build_paper_collection.py
```

Only notes with valid `pdf_ref` (path under `paperPDFs/` ending in `.pdf`) are indexed.

## Research Idea Evaluation (via knowledge base)

When the user asks to evaluate a research idea using the knowledge base, analyze along three dimensions:

### 1. Capability Ceiling

Estimate the idea's theoretical upper bound based on current performance bottlenecks in related methods from the knowledge base:

- extract core mechanisms from `core_operator` and `primary_logic`;
- identify current SOTA bottlenecks (data quality, model architecture, evaluation protocol);
- assess whether the idea addresses those bottlenecks directly or bypasses them;
- output: "What upper bound this idea can plausibly reach, and what factors constrain it."

### 2. CCF-A Reviewer Acceptance

Based on writing and evaluation patterns in top-venue papers from the KB, estimate acceptance readiness at CCF-A venues (NeurIPS / CVPR / ICCV / ECCV / ICLR / AAAI):

- compare how similar ideas were published (task framing, experiment design, ablation depth);
- assess novelty sufficiency: whether differentiated contribution is clear versus most related KB papers;
- identify likely reviewer concerns (robustness, generalization, baseline comparisons);
- output: "Estimated acceptance potential at CCF-A venues, and key areas needing reinforcement."

### 3. Literature Support Strength

Retrieve papers most related to the idea and evaluate support from prior work:

- positive support: which published works validate the idea's assumptions or submodules;
- gap confirmation: whether direct competitors already exist in the KB; if not, state the gap;
- transferable techniques: which `core_operator` patterns can be reused or adapted;
- output: "Top 3-5 most relevant papers + each paper's relation to this idea (support / competitor / reusable)."

### Output format

```markdown
## Idea Evaluation: <idea name>

### Capability Ceiling
- Upper-bound estimate: ...
- Main limiting factors: ...

### CCF-A Reviewer Acceptance
- Acceptance assessment: high / medium / low
- Main risk points: ...
- Key experiments to strengthen: ...

### Literature Support Strength
- Positive support: [[paper1]], [[paper2]]
- Direct competitors: [[paper3]] (difference: ...)
- Reusable techniques: [[paper4]] (reuse point: ...)
```
