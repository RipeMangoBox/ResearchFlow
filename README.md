

# ResearchFlow

**Semi-automated local research knowledge base and research assistant for paper analysis, ideation, coding, experiments, writing, and publication workflows.**

[English](README.md) | [Chinese](README_ZN.md)



---

ResearchFlow turns paper lists, PDFs, and structured analysis notes into shared local research memory that Claude Code, Codex CLI, and other AI research tools can reuse, and provides a coordinated skill layer that helps across ideation, framework design, coding, experimentation, writing, and publication.

```text
collect paper list -> paper analysis -> build index -> research assist
```

It can serve three usage modes at the same time:

- Claude Code / Cursor: via `.claude/skills`
- Codex CLI: via `AGENTS.md` plus generated `.agents/skills` / `.codex/skills` compatibility aliases
- Other AI research tools: by directly reading local Markdown, PDFs, and indexes

Its focus is not just "paper management". The real goal is to turn local literature accumulation into an agent-ready knowledge base that can be retrieved, compared, reused for ideation, and linked with local codebases while keeping ResearchFlow itself as the active workspace.

## Domain Fork

ResearchFlow is not limited to academic research. With the `domain-fork` skill, you can migrate the same architecture into other professional domains while preserving the ideas of local knowledge accumulation, structured indexing, workflow skills, and downstream execution.

Typical examples include frontend development, accounting, journalism, policy analysis, legal workflows, or any domain that benefits from a shared local memory plus semi-automated assistance.

If you want a neutral starter name for your own fork, you are also welcome to try `YourAnyFlow`.

```text
/domain-fork
Fork ResearchFlow into a frontend engineering version.
Keep the local knowledge base + workflow architecture, but remap papers, venues, and analysis notes to frontend references, implementation patterns, bug cases, and framework comparisons.
Generate the adapted repository structure, renamed skills, and README.
You are also welcome to try YourAnyFlow!
```

## What It Can Do

- Collect candidate papers for a topic from conference pages, blogs, and GitHub awesome repositories
- Turn local PDFs into structured paper analysis notes with reusable fields such as `core_operator`, `primary_logic`, tags, venue, and year
- Automatically rebuild `paperCollection/` pages for statistics, Obsidian navigation, and backlink-friendly browsing
- Use `paperAnalysis/` as the main agent-facing retrieval and evidence layer for comparison, question lists, idea generation, reviewer-style critique, and paper retrieval before implementation
- Let multiple agents, research projects, and code repositories reuse the same local knowledge base

## Usage Examples

Below are several prompt examples.

- The first line of each block explicitly names the recommended skill.
- For the knowledge-base construction workflow, it is better to run the stages one by one instead of merging collect / download / analyze / build into one giant prompt.

### 1. Build a topic-specific knowledge base from scratch

If you do not yet have a good paper source, you can first ask the agent to find a better awesome repository or curated list for intake.

```text
I want to build a knowledge base for controllable human motion generation.
Please search for relevant GitHub awesome repositories or curated paper lists first, and recommend the 2 most suitable options.
```

After selecting the source, run the following four prompts stage by stage.

(1) Build the paper candidate list

```text
/papers-collect-from-github-awesome
Collect papers related to controllable human motion generation from this GitHub awesome repository: <URL>
Keep only items related to diffusion, controllability, real-time generation, or long-form motion.
Organize the result into a candidate list suitable for the downstream download workflow, and mark which entries should be kept or skipped.
```

(2) Download papers

```text
/papers-download-from-list
Download the papers that are still marked as `Wait` in the current candidate list.
If a link is broken or a download fails, mark it as `Missing` and provide a short failure reason.
After finishing, return the counts of successful downloads, failures, and skipped items.
```

(3) Analyze papers

Because each session has limited context, it is recommended to process no more than 8 papers at a time, and continue in a new session when the context is getting full.

```text
/papers-analyze-pdf
Analyze the PDFs under `paperPDFs/` that were newly downloaded but do not yet have corresponding analysis notes.
Write the analysis results into `paperAnalysis/`, and make sure the frontmatter includes title, venue, year, tags, core_operator, and primary_logic.
When finished, tell me which papers were newly added into the knowledge base in this round.
```

(4) Build indexes

Each analyzed paper carries tags. Index building organizes the tags across all papers so later research-assist steps can retrieve relevant papers more quickly.

```text
/papers-build-collection-index
Rebuild `paperCollection/` based on the latest `paperAnalysis/`.
When finished, briefly summarize the current coverage of the knowledge base by task, technique, and venue.
```

### 2. Incrementally refresh the knowledge base after new PDFs arrive

```text
/papers-analyze-pdf, /papers-build-collection-index
Process the items in `paperAnalysis/analysis_log.csv` that still need work.
After the analysis is done, rebuild `paperCollection/`.
Finally summarize:
- which papers were newly added this time
- which new method tags and technique tags were introduced
```

### 3. Compare papers for a design decision

```text
/papers-compare-table
Compare DART, OmniControl, MoMask, and ReactDance, with a focus on their representation design.
Based on the core thinking, applicable scenarios, and capability characteristics of each design, analyze whether they can support <XXX>.
Save the analysis result into `paperIDEAs/`.
```

### 4. Use the knowledge base for idea generation instead of generic brainstorming

(1) Use the agent to generate initial ideas from a rough direction

```text
/research-brainstorm-from-kb, /reviewer-stress-test
I want to study long-audio temporal localization.
Please first identify the first-principles challenges of this task, then combine the local knowledge base with necessary external search, borrow recent developments from video understanding, and organize several rounds of discussion between an author perspective and a CVPR/ICLR reviewer perspective.
Finally propose 3 viable ideas.
Each idea should include:
- the problem entry point
- why it is feasible
- which existing papers provide technical support and evidence
- whether the supporting papers have open-source code, how credible that support is, and how strongly it matches the current idea
- the minimum viable experiment
- the most likely reviewer objection
Write the final result into `paperIDEAs/`.
```

(2) Refine an initial idea with the agent

```text
/idea-focus-coach
I designed a new motion representation in @paperIDEAs/2026-10-08_new_motion_representation_raw.md, and the goal is to support high generation quality, strong instruction following, and long-horizon consistency at the same time. I still need to further determine the following parts:
1. XXXX
2. XXXXX
3. ...
Write the result back into the markdown file.
```

(3) Once the idea is more mature, let the agent act as a reviewer for a pre-mortem

```text
/reviewer-stress-test /research-question-bank
I designed a new motion representation in @paperIDEAs/2026-10-10_new_motion_representation.md, and the goal is to support high generation quality, strong instruction following, and long-horizon consistency at the same time.
Please act like a top-tier conference reviewer, point out the main weaknesses of the current plan, its overlap with related work, the core evaluation criteria, and major risks, and write the result back into the markdown file.
```

### 5. Link a code repository into the knowledge-base workspace

Link the target codebase under `linkedCodebases/` inside ResearchFlow. This keeps ResearchFlow as the active workspace, so Claude Code and Codex can still discover the shared skills while the agent retrieves relevant papers before changing code.

```text
/code-context-paper-retrieval
Please implement the code changes according to @paperIDEAs/2026-10-12_new_motion_representation_roadmap.md. Before editing, retrieve relevant KB materials if needed.
Summarize:
- transferable operators
- key implementation constraints
- points that may conflict with the current code structure
- which designs best match our current idea / roadmap
Then provide the code modification plan and start implementing.
If the current knowledge base does not provide enough evidence, please point that out clearly before continuing.
Output a handling summary, and if there is progress, update @paperIDEAs/2026-10-12_new_motion_representation_roadmap.md.
```

### 6. Link the knowledge base with automation tools

ResearchFlow can serve as a high-quality local knowledge base and be combined with existing automated research tools, such as [Deep Researcher Agent](https://github.com/Xiangyue-Zhang/auto-deep-researcher-24x7).

```text
/papers-query-knowledge-base
Treat `ResearchFlow/` as the shared research memory and evidence layer for multiple agents in this project.
Follow these conventions while working:
- start retrieval from `paperAnalysis/`
- use `paperCollection/` only when you want statistics, overview pages, or Obsidian navigation / backlink exploration
- write new question lists into `QuestionBank/`
- write new research ideas into `paperIDEAs/`
- if the local knowledge base already covers the relevant papers, do not restate the whole literature chain from scratch
- before proposing a new idea or code change, first check whether the local knowledge base already contains reusable operators, baselines, or reviewer concerns
- when returning results to the automation system, prioritize concise, traceable, evidence-based summaries instead of generic brainstorming
```

## Core Workflow


| Phase              | Goal                                                                           | Main outputs                                                          |
| ------------------ | ------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Collect paper list | Collect candidate papers from web pages or GitHub lists                        | candidate list aligned with the local workflow                        |
| Paper analysis     | Download PDFs and convert them into structured analysis notes                  | `paperPDFs/`, `paperAnalysis/`, `analysis_log.csv`                    |
| Build index        | Rebuild statistics / navigation pages from analysis notes                       | `paperCollection/` organized by task, technique, and venue            |
| Research assist    | Query, compare, generate ideas, run reviewer stress tests, and connect to code | answers, tables, `QuestionBank/`, `paperIDEAs/`, implementation plans |


`Download` belongs to the intake stage between collect and analysis. For agent retrieval, `paperAnalysis/` is the primary source; `build index` mainly refreshes the generated `paperCollection/` pages for statistics and Obsidian-side navigation.

## How `paperAnalysis` and `paperCollection` Work Together

Once `paperAnalysis/` accumulates structured notes, ResearchFlow is no longer just a paper archive directory. `paperCollection/` can then be regenerated as a companion layer for statistics and Obsidian-friendly navigation.


| Layer           | Main directories                                   | Role                                                                                 |
| --------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Agent-facing evidence layer | `paperAnalysis/`                       | Primary retrieval and evidence layer for agents, with structured fields such as operators, logic, links, and tags |
| Statistics / navigation layer | `paperCollection/`                   | Generated overview pages for task / technique / venue statistics, Obsidian jumps, and backlink exploration |
| Output layer    | `QuestionBank/`, `paperIDEAs/`, `linkedCodebases/` | Turns knowledge into question lists, ideas, implementation plans, and code decisions |


Based on the interaction of these three layers, several forms of research assist naturally emerge:

- `KB -> comparison decision`: decide which operator, control mechanism, or training strategy is worth borrowing
- `KB -> question list`: organize open problems, evaluation gaps, baseline gaps, and reviewer risks
- `KB -> idea generation`: ground ideas in the literature network instead of generic brainstorming
- `KB -> reviewer simulation`: run a strict challenge before project kickoff or writing
- `KB -> codebase linkage`: retrieve paper evidence before modifying model, loss, network, control, or training code
- `KB -> multi-agent collaboration`: let Claude Code, Codex CLI, and other agents share the same local research memory

## Claude Code, Codex CLI, and Other Agents

### Claude Code / Cursor

ResearchFlow ships with `.claude/skills`, so tools that support this convention can use skill routing directly.

### Codex CLI

ResearchFlow also supports Codex CLI in two layers:

- `AGENTS.md` is the stable Codex entry point for repo-level instructions
- `scripts/setup_shared_skills.py` creates `.agents/skills`, `.agents/skills-config.json`, `.codex/skills`, and `.codex/skills-config.json` as compatibility aliases pointing to the same `.claude/skills` source of truth
- on macOS/Linux the script uses symlinks; on Windows it uses directory junctions plus a hard-linked config file

This keeps one canonical skill library while still giving Claude Code and Codex the compatibility paths they expect on different platforms.

### Other AI Research Tools

Even without skill support, this repository still works directly as a local knowledge base:

- start from `paperAnalysis/` for agent-facing retrieval and evidence
- use `paperCollection/` only if you want overview pages, statistics, or Obsidian-style navigation
- open `paperPDFs/` when deeper verification is needed
- write outputs into `QuestionBank/`, `paperIDEAs/`, or code repositories linked under `linkedCodebases/`

## Quick Start

For the minimum working setup, complete Step 1 and Step 2 first. Step 3 and Step 4 are optional add-ons.

### 1. Clone the repository

Clone ResearchFlow and enter the workspace root:

```bash
git clone https://github.com/<your-username>/ResearchFlow.git
cd ResearchFlow
```

### 2. Initialize shared skills for Claude Code and Codex

This step keeps `.claude/skills` as the only maintained skill library while automatically exposing the same files to Codex-compatible paths.

macOS / Linux:

```bash
python3 scripts/setup_shared_skills.py
```

Windows:

```powershell
py -3 scripts\setup_shared_skills.py
```

### 3. Optional: Add the shared Obsidian setup

Use this if you want the recommended Obsidian workspace configuration.

1. Download the shared Obsidian package from [Google Drive](https://drive.google.com/file/d/1tSEfV6kVI5dViojqZjDU42AYY8viZ0M4/view?usp=sharing).
2. Unzip it locally.
3. Rename the extracted folder to `.obsidian`.
4. Place that `.obsidian` folder at the repository root.

### 4. Optional: Link an external code repository

Use this when you want to keep ResearchFlow as the active workspace while making one or more local codebases available under `linkedCodebases/`.

Recommended automated setup:

macOS / Linux:

```bash
python3 scripts/link_codebase.py /path/to/your-codebase
```

Windows:

```powershell
py -3 scripts\link_codebase.py C:\path\to\your-codebase
```

Custom alias under `linkedCodebases/`:

```bash
python3 scripts/link_codebase.py /path/to/your-codebase --name your-codebase
```

Manual fallback:

```bash
ln -s /path/to/your-codebase linkedCodebases/your-codebase
```

Windows PowerShell fallback:

```powershell
New-Item -ItemType Junction -Path .\linkedCodebases\your-codebase -Target C:\path\to\your-codebase
```

One ResearchFlow can host links to multiple codebases while staying the workspace that exposes Claude/Codex skills.

## Repository Structure

```text
ResearchFlow/
├── AGENTS.md
├── LOGO.png
├── .claude/skills/
├── .claude/skills-config.json
├── .agents/
│   └── skills/                  # generated by scripts/setup_shared_skills.py
├── .codex/
│   └── skills/                  # generated by scripts/setup_shared_skills.py
├── paperAnalysis/
│   └── analysis_log.csv
├── paperCollection/
├── paperPDFs/
├── QuestionBank/
├── paperIDEAs/
├── linkedCodebases/
├── scripts/
└── obsidian setting/
```

- `paperCollection/` is the generated statistics / navigation layer
- `paperAnalysis/` is the main agent-facing knowledge and retrieval layer
- `QuestionBank/` and `paperIDEAs/` are downstream research output layers
- `linkedCodebases/` stores local symlinks or junctions to external code repositories
- `scripts/` stores maintenance and automation utilities
- `.claude/skills/User_README.md` and `.claude/skills/User_README_ZN.md` provide the quickest bilingual skill map

## Notes

- If analysis notes are added or modified and you want refreshed `paperCollection/` pages, rebuild the index
- The main state flow in `analysis_log.csv` is `Wait -> Downloaded -> checked`, with `Missing` and `Skip` as side states
- Obsidian is optional; the repository still works as a normal local folder for agents
- If `.agents/skills` or `.codex/skills` is missing, rerun `scripts/setup_shared_skills.py`
- The shared Obsidian package is public-safe and has had private workspace state and share tokens removed
- If the extracted folder is named `obsidian setting`, rename it to `.obsidian` before placing it in the repository root

## Skill Usage Details

### How to invoke skills

- Describe the task directly in natural language
- Name the skill explicitly, such as `papers-download-from-list`
- Use slash-style invocation, such as `/papers-analyze-pdf`

### Recommended entry points

- If you are unsure where to start, begin with `research-workflow`
- For KB construction, the usual path is `papers-collect-* -> papers-download-from-list -> papers-analyze-pdf -> papers-build-collection-index`
- For retrieval and design work, start from `papers-query-knowledge-base` or `papers-compare-table`
- For ideation, use `research-brainstorm-from-kb`, `research-question-bank`, `idea-focus-coach`, or `reviewer-stress-test`
- For code-grounded implementation, use `code-context-paper-retrieval` before editing model- or method-related code
- For cross-domain adaptation of the full repository architecture, use `domain-fork`

### Detailed skill references

- `.claude/skills/User_README.md`: English user-facing skill selection guide
- `.claude/skills/User_README_ZN.md`: Chinese user-facing skill selection guide
- `.claude/skills/README.md`: compact skill map by workflow family
- `.claude/skills-config.json`: registered skill descriptions and routing metadata

## License

MIT
