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

- `paperCollection/`: first retrieval layer by task, technique, and venue
- `paperAnalysis/`: authoritative structured evidence
- `paperPDFs/`: local PDF storage
- `QuestionBank/`: question and challenge outputs
- `paperIDEAs/`: idea outputs
- `scripts/`: maintenance and automation helpers

## Working Rules

- For broad research questions, start from `paperCollection/`, then open relevant files in `paperAnalysis/`
- When analysis notes are added or updated, rebuild `paperCollection/` before answering KB-wide questions
- Prefer answers grounded in local note structure such as tags, venue, year, `core_operator`, and `primary_logic`
- Treat this repository as shared memory that can support Claude Code, Codex CLI, and other agents

## Skill Routing

- Reusable workflows live in `.agents/skills` and `.claude/skills`
- Use `.claude/skills/User_README.md` when you need a quick map of which workflow to invoke
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
- In a linked code repository, retrieve relevant papers before editing model- or method-related code
