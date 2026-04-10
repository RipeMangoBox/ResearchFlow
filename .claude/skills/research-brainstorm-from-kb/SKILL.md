---
name: research-brainstorm-from-kb
description: Explores and structures research ideas into `paperIDEAs/` notes using the local paper knowledge base and frontier techniques. Use when the user provides research questions or ideas and wants decomposed candidate directions, related-work analysis based primarily on `paperAnalysis`, and cross-domain support from image/video generation, MLLM, Agent, or RL, automatically saved as dated idea notes under `paperIDEAs/`.
---

# Research Idea Brainstorming

## 1. Use cases (When to use)

Use this skill in the following scenarios:

- **Research question / idea input**: the user describes a research question, direction, or fragmented ideas in natural language and wants systematic brainstorming plus lightweight synthesis.
- **Need local knowledge-base grounding**: rely primarily on retrieval from `paperAnalysis`, and optionally combine `paperCollection` statistics/navigation pages to produce "related-work support + research opportunity".
- **Need reusable idea notes**: automatically write outputs as structured Markdown under local `paperIDEAs/` for later writing, experiments, and project management.

## 2. Dependencies and paths (Dependencies & paths)

This skill depends on the local paper knowledge base and the `papers-query-knowledge-base` skill:

- **Paper index and analysis**: see `papers-query-knowledge-base` skill (`paperCollection/` + `paperAnalysis/`)
- **Primary paper retrieval and analysis**: see `papers-query-knowledge-base` skill (`paperAnalysis/` first, `paperCollection/` as auxiliary)
- **Idea note directory**: `paperIDEAs/`

Path convention:

- If the current workspace is the repository containing these folders, prefer relative paths.
- If the workspace does not include this KB, use the correct absolute path on your machine.

## 3. File naming and storage rules

When generating brainstorming results, **always write/update Markdown files in `paperIDEAs/`** using this naming rule:

- **Filename pattern**: `YYYY-MM-DD_<core-slug>.md`
  - `YYYY-MM-DD`: user local date (for example `2025-03-09`)
  - `<core-slug>`: topic phrase compressed into English words or pinyin, lowercase with `-` separators (for example `motion-llm-ideas`, `reactive-agent-motion`)
- **Example**: `2025-03-09_motion-llm-ideas.md`

Behavior convention:

- If a file with the same date and core slug already exists, **append a new brainstorming subsection** instead of creating a new file.
- If the user explicitly specifies a target file in `paperIDEAs`, follow the user-specified filename.

## 4. Brainstorming process and output structure

Overall goal: starting from the user's raw idea, combine the local paper KB and frontier techniques to complete one end-to-end **brainstorming-and-synthesis** cycle and leave a note structure directly usable for writing/topic selection.

### 4.1 Idea decomposition and association

Steps:

1. **Refine the problem core**: restate the user problem in 1-3 sentences and identify task/scenario/key bottleneck.
2. **Multi-dimensional decomposition**: split into sub-problems by task, data (modality/source), model (architecture), constraints (physical/semantic/interaction), and evaluation.
3. **Lateral association**: map to related themes in the local KB (for example motion generation, HSI, HOI, Agent integration), and identify possible transfer directions or variants.

In generated notes, use this section:

- **Section title**: `## 1. Idea decomposition and association`

### 4.2 Real scenarios and pain points

Steps:

1. **Lock real scenarios**: based on assumptions or user description, define representative application scenarios (for example VR interaction, rehabilitation training, animation production, robot manipulation).
2. **Enumerate needs and pain points**:
   - How current practice/systems solve this.
   - Which steps still rely heavily on manual work, low efficiency, or poor UX.
   - Hidden requirements such as safety, fairness, robustness.
3. **Map pain points back to the idea**: show where the original idea directly alleviates pain or amplifies value.

In generated notes:

- **Section title**: `## 2. Real scenarios and pain points`

### 4.3 Related-work support and research opportunities (via papers-query-knowledge-base)

Always use the local KB through `papers-query-knowledge-base`:

1. **Locate related tasks/techniques**:
   - Prioritize title, task path, tags, venue, year, `core_operator`, and `primary_logic` in `paperAnalysis/`.
   - If overview pages/statistics/Obsidian navigation aid is needed, then reference `paperCollection/by_task/`, `by_technique/`, `by_venue/`.
2. **Select representative papers**:
   - Choose 3-8 papers highly related to the idea.
   - For each paper, read `core_operator`, `primary_logic`, and TL;DR from the `paperAnalysis` note frontmatter.
   - Summarize in 1-2 sentences what it does and how it relates to the current idea.
3. **Summarize support and gaps**:
   - **Support**: where these works validate the idea's feasibility or value.
   - **Gap/opportunity**: where research space remains (scenario, data, constraints, evaluation, degree of systemization).

In generated notes:

- **Section title**: `## 3. Related-work support and research opportunities`
- Suggested content layout:
  - "Related-work overview" subsection listing papers by task/technique cluster with one-line summaries.
  - "Support points" subsection listing strong evidence for the idea.
  - "Research opportunities" subsection listing potentially top-tier innovation entry points.

### 4.4 Frontier cross-domain techniques and validation (Image / Video / MLLM / Agent / RL)

Goal: connect the current idea to a broader frontier ecosystem, propose viable cross-domain research paths, and provide links for further reading.

Steps:

1. **Identify suitable frontier directions**:
   - image/video generation (diffusion, video diffusion, 3D/4D representation)
   - multimodal LLM (MLLM), multimodal Agent
   - reinforcement learning (RL), reward/preference-guided generation
   - popular technical domains or transferable core ideas
   - other high-fit techniques (for example 3DGS, Neural ODE, Conformal prediction)
2. **Retrieve and filter**:
   - prioritize relevant existing local KB analysis notes if available.
   - for newer uncovered methods/applications, use web search for representative papers, technical blogs, or official docs.
3. **Provide integration and validation plans**:
   - explain how each technique plugs into the current idea pipeline (as backbone, module, evaluation tool, or Agent sub-skill).
   - provide simple validation plans or prototype experiment concepts.

In generated notes:

- **Section title**: `## 4. Frontier cross-domain techniques and validation ideas`
- Include a small table/list with this format:
  - technique name / direction
  - brief description (1-2 sentences)
  - related links (paper/project/blog), using Markdown links such as `[Paper Title](https://...)` (no bare URLs)

### 4.5 Summary and next actions

Finally provide a convergent summary:

1. **Summarize the core idea and its research position**: use 3-5 sentences to explain what problem is solved, key ideas used, and where novelty sits relative to existing work.
2. **List executable next steps**:
   - data/scenario: what data or scenarios need to be built/organized?
   - baseline/experiments: which existing methods can bootstrap baselines quickly?
   - metrics/evaluation: how to quantify the idea advantage?
3. **Optional: venue recommendation**: if appropriate, suggest 1-2 suitable venues and rationale.

In generated notes:

- **Section title**: `## 5. Summary and next steps`

## 5. Recommended note template

When generating `paperIDEAs` notes, you can use this template (slight adjustments allowed by context):

```markdown
---
created: {{ISO_DATETIME_NOW}}
updated: {{ISO_DATETIME_NOW}}
---

# {{YYYY-MM-DD}} {{Brief core problem description}}

> Systematic retrieval and brainstorming primarily grounded in `paperAnalysis`; optionally using `paperCollection` statistics/navigation pages, with frontier cross-domain support from image/video/MLLM/Agent/RL.

---

## 1. Idea decomposition and association
- Problem restatement:
- Key elements:
- Multi-dimensional decomposition:

## 2. Real scenarios and pain points
- Typical scenarios:
- Core needs:
- Existing solutions and pain points:

## 3. Related-work support and research opportunities
### 3.1 Related-work overview
- [Paper A]: one-line summary + relation to this idea
- [Paper B]: one-line summary + relation to this idea

### 3.2 Support points
- ...

### 3.3 Research opportunities
- ...

## 4. Frontier cross-domain techniques and validation ideas
- Technique direction A: brief description + [link](https://...)
- Technique direction B: brief description + [link](https://...)

## 5. Summary and next steps
- Core idea summary:
- Near-term executable steps:
- Potential target venue:
```

## 6. Implementation notes

- **Always prioritize `papers-query-knowledge-base`** to locate existing local analyses, then supplement with frontier web search.
- **Ensure structured Markdown output** following the section structure above for later retrieval/recomposition.
- When citing external resources, **always use Markdown links** and avoid bare URLs.
- If the user already opened a specific `paperIDEAs` file and explicitly asks to brainstorm there, append the structured subsection in that file instead of creating a new one.

## 7. Boundaries

- This skill can refine framing and next actions, but detailed scope cuts, prioritized hypotheses, and MVP planning belong to `idea-focus-coach`.
- This skill is for candidate-direction generation, not reviewer-style verdicts or acceptance-risk scoring. Use `reviewer-stress-test` for that.
