# ResearchFlow Agent Guide

ResearchFlow is an agent-ready research knowledge base, not just a paper folder.

Core loop:

```text
collect paper list -> paper analysis -> build index -> research assist
```

`Download` sits inside the intake path between collection and analysis.

## What This Repository Is For

- Build a local literature knowledge base from web pages, GitHub awesome lists, and PDFs
- Convert papers into structured notes that can be queried by agents
- Reuse the knowledge base for comparison, idea generation, question banks, reviewer-style critique, and code-grounded implementation planning

## Source of Truth

- `paperAnalysis/`: authoritative structured evidence and the primary retrieval layer for agents
- `paperCollection/`: generated statistics / navigation pages for Obsidian, overview browsing, and backlink exploration
- `paperPDFs/`: local PDF storage
- `QuestionBank/`: question and challenge outputs
- `paperIDEAs/`: idea outputs
- `scripts/`: maintenance and automation helpers

## Working Rules

- For broad research questions, start from `paperAnalysis/` and use its frontmatter/body as the main evidence source
- Use `paperCollection/` only when you want overview pages, statistics, or Obsidian navigation help
- When analysis notes are added or updated, rebuild `paperCollection/` only if you want refreshed statistics / navigation pages
- Prefer answers grounded in local note structure such as tags, venue, year, `core_operator`, and `primary_logic`
- Treat this repository as shared memory that can support Claude Code, Codex CLI, and other agents
- When code and KB need to work together, keep ResearchFlow as the active workspace and link external repositories under `linkedCodebases/` instead of linking ResearchFlow into a code repo

## Skill Routing

- `AGENTS.md` is the stable repo-level entry for Codex
- The canonical skill library lives in `.claude/skills/`
- Run `python3 scripts/setup_shared_skills.py` on macOS/Linux or `py -3 scripts\setup_shared_skills.py` on Windows to create `.agents/skills` and `.codex/skills` compatibility aliases without copying
- When a task matches a workflow, Codex should:
  1. open `.claude/skills/User_README.md` or `.claude/skills/User_README_ZN.md` for quick routing
  2. open the matching `.claude/skills/<skill>/SKILL.md`
  3. follow that skill file as the workflow definition
- Use `.claude/skills/User_README.md` for the English quick skill map or `.claude/skills/User_README_ZN.md` for the Chinese version
- Main workflow families:
  - paper collection
  - PDF download and repair
  - paper analysis
  - collection index rebuild
  - knowledge-base query and comparison
  - question-bank and idea generation
  - reviewer-style stress testing
  - code-context paper retrieval

## Typical Agent Tasks

- Build or refresh a topic-specific knowledge base
- Compare multiple papers and extract transferable operators or design patterns
- Generate question banks and idea notes grounded in the current literature
- In a codebase linked under `linkedCodebases/`, retrieve relevant papers before editing model- or method-related code
