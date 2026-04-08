# ResearchFlow

ResearchFlow turns paper lists and PDFs into an agent-ready research knowledge base.

```text
collect paper list -> paper analysis -> build index -> research assist
```

It is designed to work in three modes:

- Claude Code / Cursor via `.claude/skills`
- Codex CLI via `AGENTS.md` and `.agents/skills`
- Any other AI research tool that can read local Markdown, PDFs, and indexes

The main point is not just "paper management". The point is to build a local knowledge base that agents can query, compare, reuse for ideation, and connect to a codebase before implementation.

## What You Can Do With It

- Build a topic-specific paper base from conference pages, blogs, or GitHub awesome lists
- Turn local PDFs into structured analysis notes with reusable fields such as `core_operator`, `primary_logic`, tags, venue, and year
- Rebuild searchable indexes by task, technique, and venue
- Use the indexed knowledge base for comparison, question-bank generation, idea generation, reviewer-style stress tests, and paper-aware coding assistance
- Reuse one ResearchFlow folder across multiple agents and multiple research/code projects

## High-Quality Usage Examples

Each block below is intended to be sent to an agent as a single prompt.

- Text outside the blocks is for human readers and should not be mixed into the prompt.
- The first line of each block explicitly names the recommended skills.
- For knowledge-base construction, it is better to run collect / download / analyze / build as separate prompts instead of one large request.

### 1. Build a topic knowledge base from scratch

If you do not yet have a good source list, start by asking the agent to find a suitable awesome repo or curated list.

```text
Skills: research-workflow
I want to build a knowledge base for controllable human motion generation.
Please search for strong GitHub awesome repositories or curated paper lists for this topic.
Return 3-5 candidate sources and explain:
- whether each source is suitable for ResearchFlow intake
- whether it is better as a survey hub, project index, or paper list
- which one is the best first source to ingest
Finish with one recommended option.
```

After selecting a source, run the next four prompts step by step.

```text
Skills: papers-collect-from-github-awesome
Collect controllable human motion generation papers from this GitHub awesome repository: <URL>
Keep only entries related to diffusion, controllability, real-time generation, or long-form motion.
Organize the result into a candidate list that is ready for the downstream download workflow, and mark which entries should likely be kept or skipped.
```

```text
Skills: papers-download-from-list
Download the papers in the current candidate list that are still marked `Wait`.
If a link is broken or a paper cannot be downloaded, mark it as `Missing` and summarize the failure reason.
When finished, report the number of successful downloads, failures, and skipped entries.
```

```text
Skills: papers-analyze-pdf
Analyze the PDFs under `paperPDFs/` that do not yet have corresponding analysis notes.
Write the results into `paperAnalysis/`, and ensure the frontmatter includes title, venue, year, tags, core_operator, and primary_logic.
When finished, tell me which papers were newly added into the knowledge base.
```

```text
Skills: papers-build-collection-index
Rebuild `paperCollection/` from the latest `paperAnalysis/`.
When finished, give me a short coverage summary by task, technique, and venue.
```

### 2. Refresh the knowledge base after new PDFs arrive

```text
Skills: papers-analyze-pdf, papers-build-collection-index, papers-query-knowledge-base
Scan `paperPDFs/` for PDFs that do not yet have analysis notes, and analyze only the missing ones.
Then rebuild `paperCollection/`.
Finally summarize:
- which papers were newly added
- which method tags and technique tags were newly introduced
- which task or venue coverage became noticeably stronger
```

### 3. Compare papers for a design decision

```text
Skills: papers-compare-table, papers-query-knowledge-base
I want to design a new motion representation that supports high generation quality, strong instruction following, and long-horizon consistency.
Compare DART, OmniControl, MoMask, and ReactDance with a focus on representation design.
Return:
- the core representation unit of each paper
- how each representation supports temporal modeling, control interfaces, and long-range consistency
- the most important strengths, limitations, and best-fit task settings
- which design elements are compatible and worth combining into a new representation
- one minimum ablation plan to validate the new representation
Finish with your recommended representation mix.
```

### 4. Use the knowledge base for idea generation instead of generic brainstorming

```text
Skills: research-question-bank, research-brainstorm-from-kb, reviewer-stress-test
I want to study long-audio temporal localization.
First identify the first-principles challenges of this task.
Then combine the local knowledge base with external search when useful, borrow relevant recent ideas from video understanding, and run a multi-round discussion between an author persona and a CVPR/ICLR reviewer persona.
Produce 3 viable research ideas.
For each idea, include:
- the problem entry point
- why the idea is feasible
- which existing papers provide technical support and evidence
- whether the supporting papers have open-source code, how credible that support is, and how strongly it matches the idea
- the minimum viable experiment
- the most likely reviewer objection
Write the final result into `paperIDEAs/`.
```

### 5. Link the knowledge base to a code repository

Once `./ResearchFlow` is linked into a code repository, the local knowledge base can guide code changes more faithfully to the target idea or roadmap.

```text
Skills: code-context-paper-retrieval
This code repository is linked to the local knowledge base through `./ResearchFlow`.
Before editing `src/models/diffusion/`, retrieve the most relevant papers from the local knowledge base.
Summarize:
- the transferable operators
- the key implementation constraints
- the parts that may conflict with the current code structure
- which design choices best match our current idea or roadmap
Then propose a concrete code plan and start implementation.
If the current knowledge base does not provide strong enough evidence, say so before continuing.
```

### 6. Use ResearchFlow as shared memory for multiple agents

```text
Skills: papers-query-knowledge-base, research-question-bank, research-brainstorm-from-kb
Treat `ResearchFlow/` as shared research memory for multiple agents working on this project.
Follow these rules:
- use `paperCollection/` as the retrieval entry layer
- use `paperAnalysis/` as the evidence layer
- write new question lists into `QuestionBank/`
- write new research directions into `paperIDEAs/`
- if the local knowledge base already covers the relevant papers, do not rebuild the literature chain from scratch
- before proposing a new idea or code change, check whether the local knowledge base already contains reusable operators, baselines, or reviewer concerns
```

## Core Workflow

| Phase | Main goal | Main outputs |
| --- | --- | --- |
| Collect paper list | Gather candidates from web pages or GitHub lists | candidate list aligned with local workflow |
| Paper analysis | Download PDFs and convert them into structured notes | `paperPDFs/`, `paperAnalysis/`, `analysis_log.csv` |
| Build index | Rebuild retrieval views from analysis notes | `paperCollection/` by task, technique, and venue |
| Research assist | Query, compare, ideate, critique, and connect papers to code | answers, tables, `QuestionBank/`, `paperIDEAs/`, implementation plans |

`Download` is part of the intake path between collection and analysis, but the repository is best understood through the four-step loop above.

## How Research Assist Works After `build index`

Once `paperCollection/` exists, ResearchFlow becomes more than a reading archive.

| Layer | Main folders | Role |
| --- | --- | --- |
| Retrieval layer | `paperCollection/` | Fast entry by task, technique, and venue |
| Evidence layer | `paperAnalysis/` | Deep structured notes, operators, logic, links, and tags |
| Output layer | `QuestionBank/`, `paperIDEAs/`, linked code repos | Where agents write questions, ideas, plans, and code-grounded decisions |

This enables several useful linkage patterns:

- `KB -> comparison`: choose which operator, training strategy, or control mechanism is worth borrowing
- `KB -> question bank`: map open problems, missing evaluations, baselines, and reviewer concerns
- `KB -> idea generation`: produce ideas grounded in the current literature instead of generic brainstorming
- `KB -> reviewer simulation`: test a direction before committing to a project or paper draft
- `KB -> codebase`: retrieve paper evidence before changing model, loss, network, control, or training code
- `KB -> multi-agent workflow`: let Claude Code, Codex CLI, and other agents reuse the same local research memory

## Claude Code, Codex CLI, and Other Agents

### Claude Code / Cursor

ResearchFlow ships with `.claude/skills`, so Claude-style skill routing works directly in tools that support that convention.

### Codex CLI

ResearchFlow also supports Codex CLI in two layers:

- `AGENTS.md` gives repo-wide instructions about the workflow and source-of-truth folders
- `.agents/skills` is linked to `.claude/skills`, so the same skill library can be discovered by Codex without duplicating the workflow definitions

In practice, this means ResearchFlow can support Claude Code and Codex CLI from the same repository instead of maintaining two separate prompt systems.

### Other AI Research Tools

Even without skill support, the repository still works as a local knowledge base:

- start retrieval from `paperCollection/`
- open `paperAnalysis/` for detailed evidence
- read PDFs from `paperPDFs/` when deeper verification is needed
- write downstream outputs into `QuestionBank/`, `paperIDEAs/`, or your linked codebase

## Quick Setup

```bash
git clone https://github.com/<your-username>/ResearchFlow.git
cd ResearchFlow
```

Optional Obsidian setup:

```bash
unzip .obsidian.zip -d .obsidian
```

Optional codebase linkage:

```bash
ln -s /path/to/ResearchFlow ./ResearchFlow
```

One codebase can link to one ResearchFlow folder, and multiple codebases can share the same ResearchFlow folder.

## Repository Structure

```text
ResearchFlow/
├── AGENTS.md
├── .claude/skills/
├── .agents/skills -> ../.claude/skills
├── paperAnalysis/
│   └── analysis_log.csv
├── paperCollection/
├── paperPDFs/
├── QuestionBank/
├── paperIDEAs/
├── scripts/
└── .obsidian.zip
```

- `paperCollection/` is the retrieval layer
- `paperAnalysis/` is the main knowledge layer
- `QuestionBank/` and `paperIDEAs/` are downstream research outputs
- `scripts/` contains maintenance and automation helpers
- `.claude/skills/User_README.md` is the quickest way to browse the workflow map

## Domain Migration

ResearchFlow's architecture is not limited to academic research. The `domain-fork` skill can migrate the entire framework to any professional domain:

```
"帮我 fork 一个前端开发版本的 ResearchFlow"
"把 ResearchFlow 迁移到会计领域"
```

The skill walks you through an interactive session:
1. Confirm target domain and repo name (e.g. `FrontendFlow`, `AccountingFlow`)
2. Review concept mapping (paper → article, venue → blog/framework, etc.)
3. Confirm which skills to keep, rename, or drop
4. Generate the complete adapted repo with skills, folders, and README

Default output location is `ResearchFlow/<RepoName>/` — you'll be asked if you want a different path.

## Notes

- If you add or update analysis notes, rebuild the collection index before asking broad KB-level questions
- `analysis_log.csv` tracks the paper state flow: `Wait -> Downloaded -> checked`, with `Missing` and `Skip` as side states
- Obsidian is optional; the repository still works fine as a plain local folder for agents

## License

MIT
