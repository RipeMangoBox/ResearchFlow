---
name: code-context-paper-retrieval
description: Retrieves code-context-relevant papers from the local knowledge base in two modes (brief/deep), using `paperAnalysis/` as the primary retrieval source and `paperPDFs/` for optional deeper reading. `paperCollection/` is optional and mainly serves statistics, Obsidian navigation, and backlink exploration. Environment detection prioritizes codebase environment files; asks user if none found. Trigger timing: BEFORE code modification, not after.
---

# Code Context Paper Retrieval

## What this skill does

For code implementation/refactor/planning tasks, this skill provides the most relevant paper evidence from the local knowledge base for the current code context.

Fixed retrieval order:

1. `paperAnalysis/` (primary retrieval and evidence layer)
2. `paperPDFs/` (recommend deeper reading only when necessary)
3. `paperCollection/` (reference only when statistics, overview pages, Obsidian jumps, or backlink support are needed)

Two output levels:

- `brief`: 3-5 must-read papers + analysis note path + whether PDF reading is recommended
- `deep`: broader candidate list + comparison rationale + practical implementation suggestions

## Environment Detection (must be completed before execution)

Before running any script, the Python environment must be confirmed. Check in this priority order:

1. **Environment files in the target codebase** (high to low):
   - `environment.yml` / `environment.yaml` (conda)
   - `requirements.txt` / `requirements*.txt`
   - `pyproject.toml` (check `[project.dependencies]` or `[tool.poetry.dependencies]`)
   - `setup.py` / `setup.cfg`
2. **Infer conda env name from environment files**: read `name:` in `environment.yml` and check whether that env exists (`conda env list`).
3. **If none found or still uncertain**: **proactively ask the user**, stating no environment file was found, and request one of:
   - conda environment name, or
   - Python interpreter path, or
   - confirmation to use system default Python

**Prohibited behavior**: do not guess and run scripts with `python3`/`python` before environment confirmation.

## Triggering

### Core principle: retrieve before code changes, not after

When the agent receives code-modification tasks (for example "implement module X", "refactor loss in Y", "add mechanism Z"), it should decide whether paper support is needed **before** writing code.

### Suggestive trigger (recommended)

In the following scenarios, the agent should proactively ask **before code edits**:

> "This task appears related to [motion diffusion / attention mechanism / ...], and the local KB may contain supporting papers. Retrieve first? (brief / deep / skip)"

Trigger when any of these is true:
- user task text contains task/technique keywords likely appearing in `paperAnalysis` paths, frontmatter, or body
- target file paths involve core model modules (for example `model/`, `network/`, `loss/`, `train/`)
- user explicitly mentions a method or paper name

Do not trigger (avoid disturbance):
- pure bug fixes, formatting changes, comment-only edits
- config/script-argument tweaks
- user already skipped retrieval once in the current conversation turn

### Explicit trigger

- direct call: `/code-context-paper-retrieval`
- or user asks "help me find related papers", "any related paper", etc.

### Script helper (optional)

```bash
# Use the confirmed conda environment
conda run -n <env_name> python ".claude/skills/code-context-paper-retrieval/scripts/code_context_paper_retrieval/query_code_context_papers.py" --mode brief --query "motion diffusion training pipeline"
```

## Inputs

- `--mode brief|deep`
- `--query "..."` (optional, coding task description)
- `--changed-file <path>` (repeatable)
- `--top-k` (candidate upper bound for deep mode)

## Minimal workflow

1. **Environment detection** (see section above).
2. Extract keywords from task description / target files.
3. Run first-pass recall in `paperAnalysis/**/*.md` by title/path/tags/venue/year/`core_operator`/`primary_logic`.
4. Read `core_operator`, `primary_logic`, and TL;DR summary from matched papers.
5. If needed, reference `paperCollection/` for statistics/overview/Obsidian jump support.
6. Output brief/deep results and recommend PDF reading only when evidence is weak.
7. **Start code changes only after user confirmation.**

## Output contract

### brief

- 3-5 must-read papers
- for each paper: analysis path, method cue, whether PDF reading is recommended (with one-line reason)

### deep

- broader candidate list (adjustable via `--top-k`)
- for each paper: analysis path, core_operator/primary_logic cue, evidence summary, PDF suggestion
- end with compact comparison and implementation advice

## Non-goals

- do not trigger only after code edits are finished (too late)
- do not trigger for non-core changes (bug fix/format/config)
- do not repeatedly suggest retrieval after the user already skipped in the same conversation turn
