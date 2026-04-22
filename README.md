<p align="center">
  <img src="./assets/LOGO.png" alt="ResearchFlow logo" width="280"/>
</p>

<h1 align="center">ResearchFlow</h1>

<p align="center"><strong>Structured Paper Analysis and Research Memory for Knowledge-Grounded Research</strong></p>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README_CN.md">中文</a>
</p>

<p align="center">
  <img alt="Semi-automated" src="https://img.shields.io/badge/Semi--automated-Research%20Workflow-1f6feb?style=flat-square"/>
  <img alt="Automation tool extensible" src="https://img.shields.io/badge/Automation--tool-extensible-0891b2?style=flat-square"/>
  <img alt="Knowledge base" src="https://img.shields.io/badge/Local-Knowledge%20Base-0f766e?style=flat-square"/>
  <img alt="Claude Code compatible" src="https://img.shields.io/badge/Claude%20Code-compatible-d97706?style=flat-square"/>
  <img alt="Codex CLI compatible" src="https://img.shields.io/badge/Codex%20CLI-compatible-7c3aed?style=flat-square"/>
  <img alt="Zotero compatible" src="https://img.shields.io/badge/Zotero-compatible-cc2936?style=flat-square"/>
  <img alt="Obsidian optional" src="https://img.shields.io/badge/Obsidian-optional-475569?style=flat-square"/>
  <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-111827?style=flat-square"/>
</p>

---

> **Knowledge first, not execution first.** Most AI research tools focus on "help me run experiments" or "help me write the paper." ResearchFlow focuses one layer earlier: **when your agent makes a decision, does it already have enough structured, retrievable paper evidence?** Without that layer, ideas stay vague, code changes lack grounding, and reviewer concerns are harder to answer.
>
> **Structured paper analysis as reusable research memory.** ResearchFlow is a structured paper analysis framework that converts academic PDFs into metadata-rich, retrievable analysis notes, building a reusable research memory for human-in-the-loop, knowledge-grounded research.
>
> **Pure Markdown, zero lock-in.** The whole system is just local folders + Markdown + CSV. No database, no Docker, no backend service. Every skill is a `SKILL.md` file that Claude Code, Codex CLI, Cursor, or your own agent can read.
>
> *ResearchFlow is a methodology, not a platform. The valuable part is the research memory you accumulate.*

---

## 🔭 Current Goals

- [ ] Release a stronger paper analysis template for more structured, comparable, and reusable paper understanding.
- [ ] Release structured paper-analysis metadata that supports efficient retrieval, filtering, and cross-paper comparison.
- [ ] Release a high-quality paper analysis knowledge base that supports human-in-the-loop research.

---

## 🎯 Not Just Prompts, But a Full Knowledge Pipeline

Give ResearchFlow a research topic and it can help you build the knowledge base step by step:

```text
collect → download → analyze → build index → research assist
```

You can use it in four modes:

**Build mode**: collect candidate papers from webpages, conference lists, or GitHub awesome repositories.

```text
/papers-collect-from-github-awesome
Collect motion generation papers from https://github.com/Foruck/Awesome-Human-Motion
and keep only diffusion, controllability, and real-time related entries.
```

**Query mode**: once the knowledge base exists, retrieve papers by task, method, tags, or title.

```text
/papers-query-knowledge-base
Which papers use diffusion + spatial control, and what are their core_operator fields?
```

**Decision mode**: compare methods before changing a design, choosing baselines, or writing related work.

```text
/papers-query-knowledge-base
Compare the representation design of DART, OmniControl, and MoMask,
with a focus on whether they can support long-horizon generation.
```

**Idea mode**: generate candidate directions grounded in the local knowledge base.

```text
/research-brainstorm-from-kb
I want to study text-driven reactive motion generation.
Please propose 3 directions grounded in the local knowledge base.
```

> Already have PDFs? Jump straight to analyze. Already have an idea? Jump to focus or reviewer stress testing. Not sure which skill to use? Start with `research-workflow`.

> Analysis language: the default note language is **Chinese** (`analysis_language: zh`). To switch new analysis notes to English, change `analysis_language` to `en` in `AGENTS.md`, or explicitly ask for English output in the current prompt.

---

## ✨ Core Capabilities

| Capability              | What it does                                                                                                                             | Skill                                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Workflow entry          | Detects whether you are in sync / collect / download / analyze / build / query / ideate / focus / review and recommends the next step | `research-workflow`                                                 |
| Paper collection        | Collects candidate papers from web pages or GitHub awesome repositories and writes them into `analysis_log.csv`                        | `papers-collect-from-web` · `papers-collect-from-github-awesome` |
| PDF download            | Downloads, deduplicates, and repairs PDFs from triage lists                                                                              | `papers-download-from-list`                                         |
| Structured analysis     | Converts PDFs into Markdown notes with structured frontmatter                                                                            | `papers-analyze-pdf`                                                |
| Index building          | Builds `paperCollection/index.jsonl` (agent index) and Obsidian navigation pages from analysis notes                                    | `papers-build-collection-index`                                     |
| Knowledge retrieval     | Retrieves papers by title, task, technique tag, venue, or year and summarizes the evidence; includes code-context mode for pre-coding retrieval | `papers-query-knowledge-base`                                       |
| Idea generation         | Produces research directions grounded in the knowledge base                                                                              | `research-brainstorm-from-kb`                                       |
| Idea focusing           | Narrows a broad idea into a scoped, executable plan with MVPs                                                                            | `idea-focus-coach`                                                  |
| Reviewer stress test    | Challenges an idea from a reviewer perspective and surfaces major risks                                                                  | `reviewer-stress-test`                                              |
| Metadata audit          | Checks title / venue / year / link consistency and produces a quality report                                                             | `papers-audit-metadata-consistency`                                 |
| Share-ready export      | Converts internal notes into outward-facing share versions                                                                               | `notes-export-share-version`                                        |
| Domain migration        | Migrates the architecture into another professional domain                                                                               | `domain-fork`                                                       |

> For the full skill map and usage guide, see `[.claude/skills/User_README.md](.claude/skills/User_README.md)`.

---

## 🏗️ Three-Layer Architecture

```text
┌─────────────────────────────────────────────────────────┐
│  Output layer   paperIDEAs/                             │
│                 ideas, plans, review notes               │
├─────────────────────────────────────────────────────────┤
│  Index layer    paperCollection/                        │
│                 index.jsonl (agent) + Obsidian pages    │
├─────────────────────────────────────────────────────────┤
│  Retrieval      paperAnalysis/ + paperPDFs/             │
│  layer          structured notes + raw PDFs             │
└─────────────────────────────────────────────────────────┘
```

- `**paperAnalysis/**` is the core of the system. Each paper gets one `.md` note with structured fields such as `core_operator`, `primary_logic`, `dataset`, `metrics`, and `venue`. Retrieval, comparison, ideation, and code-context steps all start here.
- `**paperCollection/**` is the index and navigation layer, serving both agents and humans:
  - `index.jsonl` — one JSONL line per paper for fast agent retrieval at scale (5 000 notes → 20–50 candidates), generated after the first `papers-build-collection-index` run.
  - `by_task/`, `by_technique/`, `by_venue/` — Obsidian navigation pages for graph view, backlinks, and human exploration.
  - Both are generated by `papers-build-collection-index`.
- `**paperPDFs/**` keeps the raw papers for deeper reading when the structured evidence is not enough.
- `**paperIDEAs/**` stores downstream outputs such as brainstorm notes, focused plans, and reviewer-style critiques.

A practical rule of thumb:

- At scale, after the first build, agents start from `paperCollection/index.jsonl` → filter → read matching `paperAnalysis/` notes
- Humans browse `paperCollection/` in Obsidian for overview and graph exploration
- Open `paperPDFs/` when you need deeper verification
- Write outputs into `paperIDEAs/` or your linked code repository

---

## 🤖 Claude Code, Codex CLI, and Other Agents

ResearchFlow supports three access modes:

- **Claude Code / Cursor**: use the built-in `.claude/skills` directly.
- **Codex CLI**: after clone, run `scripts/setup_shared_skills.py` locally to generate `.codex/skills` and `.codex/skills-config.json` from the same `.claude/skills` source of truth.
- **Other AI agents**: even without skill support, they can still use the repository as a local knowledge base by reading `paperAnalysis/`, `paperPDFs/`, and `paperIDEAs/`.

This lets multiple agents share the same research memory instead of rebuilding the literature context separately in each tool.

---

## 🚀 Quick Start

### Prerequisites

- Claude Code, Codex CLI, or any LLM agent that can read `SKILL.md`
- Python 3.10+ for PDF download, analysis, and index scripts
- Obsidian if you want visual browsing of the knowledge base

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ResearchFlow.git
cd ResearchFlow
```

### 2. Prepare the skill entry

- Claude Code / Cursor: use the existing `.claude/skills`
- Codex CLI: after clone, generate the local `.codex` compatibility layer; see [Codex CLI compatibility](#codex-cli-compat)

### 3. Start using it

You can describe the task directly, or explicitly name a skill.

The most general entry point is:

```text
/research-workflow
I want to build a knowledge base for controllable motion generation from scratch.
```

Or jump straight to a concrete task:

```text
Collect motion generation papers from https://github.com/Foruck/Awesome-Human-Motion
```

### 4. Optional: Obsidian visualization

If you want to browse the knowledge base in Obsidian, see [Obsidian setup](#obsidian-config).

---

<a id="usage-examples"></a>
## 📖 Usage Examples

A few practical rules before the examples:

- Prefer one main skill per prompt.
- For staged work, separate prompts are usually more stable than a giant combined prompt.
- Paths, venues, and topics in the examples are templates; replace them with your own.
- Longer prompts are not always better. Usually the best prompt clearly states the goal, scope, input path, and expected output.
- For skills that do not require extra input, calling the skill alone is often enough, for example `/papers-build-collection-index`.

### A. Build and maintain the knowledge base

<details>
<summary>1. Build a topic-specific knowledge base from scratch</summary>

The best first step is often the unified entry:

```text
/research-workflow
I want to build a knowledge base for controllable motion generation from scratch.
Please start from awesome repositories and the ICLR 2026 accepted list,
then tell me whether the next best step is collect, download, analyze, or build.
```

If you prefer to control each stage yourself, split the process manually.

Optional pre-step without naming a skill directly:

```text
I want to build a knowledge base for controllable human motion generation.
Please search for relevant GitHub awesome repositories or curated paper lists first,
and recommend the 2 most suitable options.
```

Prompt 1: collect candidate papers

```text
/papers-collect-from-github-awesome
Collect papers related to controllable human motion generation from this GitHub awesome repository: <URL>
Keep only items related to diffusion, controllability, real-time generation, or long-form motion.
Organize the result into a candidate list suitable for the downstream download workflow, and mark which entries should be kept or skipped.
```

Prompt 2: download local PDFs

```text
/papers-download-from-list
Download the papers that are still marked as Wait in the current candidate list.
If a link is broken or a download fails, mark it as Missing and provide a short failure reason.
After finishing, return the counts of successful downloads, failures, and skipped items.
```

Prompt 3: analyze PDFs into `paperAnalysis/`

```text
/papers-analyze-pdf
Analyze the PDFs under paperPDFs/ that were newly downloaded but do not yet have corresponding analysis notes.
Write the analysis results into paperAnalysis/, and make sure the frontmatter includes title, venue, year, tags, core_operator, and primary_logic.
When finished, tell me which papers were newly added into the knowledge base in this round.
```

Prompt 4: rebuild indexes

```text
/papers-build-collection-index
```

</details>

<details>
<summary>2. Import papers from Zotero</summary>

ResearchFlow includes a registered `papers-sync-from-zotero` skill and a runnable sync script at `.claude/skills/papers-sync-from-zotero/scripts/zotero_to_rf.py`. The script filters to paper-like Zotero items, deduplicates against `analysis_log.csv` + Zotero manifest, copies PDFs into RF's canonical layout, and can append Zotero highlights/notes into existing analysis notes.

```text
/papers-sync-from-zotero
Please guide me through connecting Zotero to ResearchFlow.
```

Local Zotero API example:

```bash
python .claude/skills/papers-sync-from-zotero/scripts/zotero_to_rf.py sync \
  --repo-root /path/to/isolated/Zotero \
  --local-api \
  --library-type user \
  --library-id 0 \
  --append-annotations-to-md
```

Web API example:

```bash
python .claude/skills/papers-sync-from-zotero/scripts/zotero_to_rf.py sync \
  --repo-root /path/to/isolated/Zotero \
  --library-type user \
  --library-id <YOUR_LIBRARY_ID> \
  --api-key <READ_ONLY_KEY> \
  --collection "Video Generation"
```

If PDFs have already been analyzed and you only want to backfill Zotero highlights/comments into the Markdown notes:

```bash
python .claude/skills/papers-sync-from-zotero/scripts/zotero_to_rf.py append-annotations
```

See `.claude/skills/papers-sync-from-zotero/README.md` for workflow details and category mapping notes.

</details>

<details>
<summary>3. Incrementally refresh the knowledge base after new PDFs arrive</summary>

Minimal version:

```text
/papers-analyze-pdf
Analyze newly added PDFs under paperPDFs/Human_Motion_Generation/CVPR_2026/
```

```text
/papers-build-collection-index
```

If you want the agent to summarize the changes as well:

```text
/papers-analyze-pdf
Process the items in paperAnalysis/analysis_log.csv that still need analysis work.
Write new analysis notes into paperAnalysis/ and report which papers were added in this round.
```

`/papers-build-collection-index` can usually run by itself. If you care about what changed in this round, ask a follow-up question after it finishes about new method tags, technique tags, and venue / task pages.

</details>

<details>
<summary>4. Run a knowledge base health check</summary>

```text
/papers-audit-metadata-consistency
```

This is especially useful after large collection or analysis batches. By default it scans the current `paperAnalysis/` and outputs a quality report.

</details>

### B. Make knowledge-driven research decisions

<details>
<summary>5. Compare papers for a design decision</summary>

When you need to choose between multiple design alternatives, write a Related Work table, pick baselines, or present a method overview:

```text
/papers-query-knowledge-base
Compare DART, OmniControl, MoMask, and ReactDance, with a focus on their representation design.
Based on the core thinking, applicable scenarios, and capability characteristics of each design,
analyze whether they can support <XXX>.
Save the comparison result into paperIDEAs/compare_motion_designs.md.
```

</details>

<details>
<summary>6. Run targeted retrieval from the knowledge base</summary>

```text
/papers-query-knowledge-base
Find all papers in the local knowledge base related to reactive motion generation, diffusion, and spatial control.
Prioritize papers relevant to controllability and long-horizon consistency,
and summarize their core_operator, primary_logic, and main limitations.
```

This is a good entry point when you want to understand the evidence landscape before deciding whether to compare or brainstorm.

</details>

<details>
<summary>7. Generate candidate ideas from the knowledge base</summary>

Short version:

```text
/research-brainstorm-from-kb
I want to study text-driven reactive motion generation.
Please propose 3 candidate directions grounded in the local knowledge base and frontier evidence when needed.
```

If you want more execution-ready output:

```text
/research-brainstorm-from-kb
I want to study long-audio-driven reactive motion generation.
Please identify the first-principles challenges of this task, ground the discussion in the local knowledge base,
add external search only when needed, and propose 3 viable directions.
For each direction, include:
- the problem entry point
- why it is feasible
- which existing papers provide technical support
- the minimum viable experiment
Write the final result into paperIDEAs/.
```

</details>

<details>
<summary>8. Turn a broad idea into an executable plan</summary>

If you want to narrow the scope interactively:

```text
/idea-focus-coach
My idea is to use a diffusion model for reactive motion generation,
but I am not sure how large the scope should be or what the first experiment should be.
Please narrow it into an executable MVP.
```

If you already have an idea note:

```text
/idea-focus-coach
I designed a new motion representation in @paperIDEAs/2026-10-08_new_motion_representation_raw.md,
and the goal is to support high generation quality, strong instruction following, and long-horizon consistency at the same time.
Please narrow it into a focused plan with:
- scope cuts
- prioritized hypotheses
- 1-2 MVP experiments for this week
Write the result back into the markdown file.
```

</details>

<details>
<summary>9. Pressure-test an idea like a reviewer</summary>

Short version:

```text
/reviewer-stress-test
Review my idea from the perspective of an ICLR reviewer:
[paste the idea description or point to a file under paperIDEAs/]
Focus on novelty, experimental design, and differentiation from SOTA.
```

If you already have a more detailed plan:

```text
/reviewer-stress-test
I designed a new motion representation in @paperIDEAs/2026-10-10_new_motion_representation.md,
and the goal is to support high generation quality, strong instruction following, and long-horizon consistency at the same time.
Please act like a top-tier conference reviewer, point out the main weaknesses of the current plan,
its overlap with related work, the core evaluation criteria, and the major rejection risks.
Write the result back into the markdown file.
```

</details>

### C. Work with code and multi-agent systems

<details>
<summary>10. Retrieve paper support before coding</summary>

Short version:

```text
/code-context-paper-retrieval
I am about to modify the attention mechanism in the motion decoder.
Please retrieve relevant attention designs and experimental conclusions from the knowledge base first.
```

If you want the output to directly support implementation planning:

```text
/code-context-paper-retrieval
Before editing the implementation for @paperIDEAs/2026-10-12_new_motion_representation_roadmap.md,
retrieve relevant papers from the local knowledge base first.
Summarize:
- transferable operators
- key implementation constraints
- points that may conflict with the current code structure
- which designs best match the current roadmap
If the current knowledge base does not provide enough evidence, point that out clearly.
```

After this retrieval step, continue the actual code change in a separate coding prompt. That is usually more stable.

</details>

<details>
<summary>11. Use ResearchFlow as a shared knowledge layer for automation</summary>

ResearchFlow is a plain file-based knowledge layer, so any agent that can read files can use it:

```text
ResearchFlow/              ← shared knowledge layer
├── paperCollection/
│   └── index.jsonl        ← agent index (generated after first build; read first to filter)
├── paperAnalysis/         ← evidence source for all agents
├── paperPDFs/             ← raw papers
└── paperIDEAs/            ← research outputs written by agents

Agent A (Claude Code)  ── reads index.jsonl → paperAnalysis/ ──→ generates ideas
Agent B (Codex CLI)    ── reads index.jsonl → paperAnalysis/ ──→ assists code changes
Agent C (custom)       ── reads index.jsonl → paperAnalysis/ ──→ designs automated experiments
```

If you want to give explicit coordination rules to an agent, you can paste this:

```text
Treat ResearchFlow/ as the shared research memory and evidence layer for multiple agents in this project.
Follow these conventions while working:
- if `paperCollection/index.jsonl` exists, start retrieval there to narrow candidates; otherwise search `paperAnalysis/` directly, then read matching notes
- use paperCollection/ Obsidian pages only when you want navigation / backlink exploration
- write new research ideas into paperIDEAs/
- if the local knowledge base already covers the relevant papers, do not restate the whole literature chain from scratch
- before proposing a new idea or code change, first check whether the local knowledge base already contains reusable operators, baselines, or reviewer concerns
- when returning results to the automation system, prioritize concise, traceable, evidence-based summaries instead of generic brainstorming
```

</details>

---

## 📁 Repository Structure

```text
ResearchFlow/
├── .claude/skills/             # skill definitions (source of truth)
├── .claude/skills-config.json  # skill routing metadata
├── .codex/
│   └── README.md               # local Codex aliases are generated here by setup_shared_skills.py
├── paperAnalysis/              # structured paper notes (core data source)
│   └── analysis_log.csv        # candidate list and status tracking
├── paperCollection/            # index & navigation layer
│   ├── index.jsonl             # agent index: generated after the first build
│   ├── by_task/                # Obsidian navigation by task
│   ├── by_technique/           # Obsidian navigation by technique
│   └── by_venue/               # Obsidian navigation by venue
├── paperPDFs/                  # raw PDFs
├── paperIDEAs/                 # downstream outputs: ideas, plans, reviews
├── scripts/                    # helper scripts
└── .obsidian/                  # optional Obsidian setup
```

- `paperAnalysis/` is the primary agent-facing evidence layer.
- `paperCollection/` is the generated index and navigation layer — `index.jsonl` for agent retrieval after build, Obsidian pages for human browsing.
- `paperIDEAs/` stores downstream research outputs.
- `.claude/skills` is the only source of truth.
- `.codex/` is optional local generated state for Codex CLI compatibility and is not tracked by git.
- For the fastest bilingual skill overview, see `.claude/skills/User_README.md` and `.claude/skills/User_README_CN.md`.

---

<a id="advanced-config"></a>
## 🔧 Advanced Config

<a id="codex-cli-compat"></a>
<details>
<summary>Codex CLI compatibility</summary>

Claude Code / Cursor does not need this step. Codex CLI does.

The repository does not track `.codex/`. After clone, generate `.codex/skills` and `.codex/skills-config.json` locally on the current platform:

macOS / Linux:

```bash
python3 scripts/setup_shared_skills.py
```

Windows:

```powershell
py -3 scripts\setup_shared_skills.py
```

If you only want to verify that the compatibility aliases are present:

```bash
python3 scripts/setup_shared_skills.py --check
```

If `.codex/skills` or `.codex/skills-config.json` disappears, rerun the same script.

</details>

<a id="obsidian-config"></a>
<details>
<summary>Obsidian setup</summary>

- Obsidian is optional. It is only a visualization layer.
- If you have a shared Obsidian package, extract it at the repository root and rename it to `.obsidian`.
- If the extracted folder is named `obsidian setting`, rename it to `.obsidian`.

</details>

<details>
<summary>Domain Fork</summary>

ResearchFlow is not limited to academic research. With the `domain-fork` skill, you can migrate the architecture into another professional domain:

```text
/domain-fork
I want to migrate ResearchFlow into a frontend engineering workflow
for tracking framework evolution, design patterns, and performance optimization knowledge.
```

If you want to keep the methodology but use a more neutral framing, you can also treat it as a general local knowledge workflow template.

</details>

---

## 📝 Notes

- The main state flow in `analysis_log.csv` is `Wait → Downloaded → checked`. Abnormal states: `analysis_mismatch` (incomplete analysis template) and `too_large` (PDF exceeds size limit). Side states: `Missing` and `Skip`. See `.claude/skills/STATE_CONVENTION.md` for full definitions.
- If analysis notes are added or modified and you want refreshed indexes, rebuild with `papers-build-collection-index` (generates or updates both `paperCollection/index.jsonl` and Obsidian navigation pages).
- Obsidian is optional; the repository still works as a normal local folder for agents.
- `.claude/skills` is the only maintained source.
- `.codex/` is local generated state; Codex users should run `scripts/setup_shared_skills.py` after clone.
- `build index` is optional for agents when the knowledge base is small. At scale (1 000+ notes), run one build first, then read `paperCollection/index.jsonl` for efficient retrieval.

## Citation

If ResearchFlow helps your research, please cite the repository directly:

```bibtex
@misc{lin2026researchflow,
  title        = {{ResearchFlow}: A Structured Paper Analysis Framework for Knowledge-Grounded Research},
  author       = {Jingzhong Lin and Ziheng Huang},
  year         = {2026},
  howpublished = {\url{https://github.com/RipeMangoBox/ResearchFlow}},
  note         = {GitHub repository}
}
```

## License

MIT
