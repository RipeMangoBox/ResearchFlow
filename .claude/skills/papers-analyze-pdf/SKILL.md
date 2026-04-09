---
name: papers-analyze-pdf
description: Analyzes academic PDFs into structured Markdown under `paperAnalysis/`. Reads local PDFs only (no network, no image generation). Use when the user asks to analyze one PDF or a folder of PDFs, or to run paper analysis for the local knowledge base. Handles path resolution so that if the PDF is already under `paperPDFs/<Category>/<Venue_Year>/`, the `.md` is written to `paperAnalysis/` in the same structure; otherwise infers category and venue (e.g. `CVPR_2025`), copies the PDF into `paperPDFs/`, and writes the `.md` to `paperAnalysis/`.
---

# Paper Analysis

Analyze one or more PDFs into structured analysis notes under `paperAnalysis/`, following a fixed template. **Read only local PDFs**; do not fetch from the network or generate figures.

## Scope

- **Input**: A single PDF path, a directory, or a batch list from `paperAnalysis/analysis_log.csv` (process `Downloaded` entries in order).
- **Output**: One `.md` per PDF at the correct path under `paperAnalysis/`. After each batch, update `analysis_log.csv` state to `checked`.

Assume repository root is the folder that contains `paperPDFs/` and `paperAnalysis/`.

---

## Path and storage rules

### Filename convention

- **Title part** (`<Year>_<SanitizedTitle>`): Use a single `_` between each pair of words. No spaces, commas, hyphens, or other separators. Example: `High_Fidelity` not `High-Fidelity`, `Long_Form` not `Long-Form`.
- **Rename requirement**: When ingesting or analyzing a new paper, **always rename both the PDF and its corresponding `.md`** to the `<Year>_<ShortEnglishTitle>` pattern before writing/updating analysis notes.

### When the PDF is already under paperPDFs

- PDF path pattern: `paperPDFs/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.pdf`
- **MD path**: Replace `paperPDFs` with `paperAnalysis`; change extension to `.md`. Keep the rest unchanged.
- Do **not** modify an existing `.md` if the corresponding PDF was not found or could not be read.

### When the PDF is not under paperPDFs

1. **Infer or ask for**: **category** and **venue + year** (for example CVPR, 2025 → `CVPR_2025`).
2. **Categories** (must choose one): `Human_Human_Interaction`, `Human_Object_Interaction`, `Human_Scene_Interaction`, `Motion_Controlled_ImageVideo_Generation`, `Motion_Editing`, `Motion_Generation_Text_Speech_Music_Driven`, `Motion_Stylization`.
3. **Place PDF**: Copy (or move) the PDF to `paperPDFs/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.pdf`. Use colocated ingest script if available:
   `python .claude/skills/papers-analyze-pdf/scripts/ingest_pdfs.py <path> --category <cat> --venue <Venue> --year <Year> [--title <Title>]`
4. **Place MD**: Write to `paperAnalysis/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.md`.
5. If category or venue/year cannot be inferred, ask user before copying/writing.

---

## Writing principles

All analysis body text (except YAML frontmatter, hyperlinks, and formulas) is written in **Chinese** for consistency inside the local vault.

### The Three Questions — every analysis must answer these

1. **What/Why**: What is the true bottleneck? Why solve it now? (data/representation/optimization/controllability/consistency/physical constraints/generalization/evaluation...)
2. **How**: What key causal knob did the authors introduce? (conditioning / constraint / architecture / objective / sampling / training recipe). What changed → which distribution/constraint/information bottleneck changed → what capability changed?
3. **So what**: Compared with prior work, where is the capability jump? Which experiment signals best support that claim?

### Expression priority

**Intuition / structured explanation > mechanism-level abstraction > minimal symbolic notation**

- For mechanism, prioritize **system-level causality**: what changed → which distribution/constraint/information bottleneck changed → what capability changed.
- For evidence, prioritize **signal type + conclusion** (more controllable / more stable / better long-term consistency / less drift / stronger generalization / less data dependence), rather than piling up numbers/tables.
- Useful high-level framing objects: **spaces** (input/condition/latent/action/constraint/evaluation space), **objects** (representation, prior, control variable, decomposition, alignment, retrieval, planning, feedback), **processes** (what distributions/constraints changed in training; how controllability/stability is injected at inference).

### Formula rules

- **Do not derive formulas**; do not explain loss term-by-term; do not explain every symbol one by one.
- If formulas must be mentioned: keep at most **1-2 verbalized objective/constraint descriptions**, attached under the information-flow explanation.
- Only minimal placeholders are allowed (optional): \(x\) (condition/observation), \(z\) (latent), \(f_\theta\) (model), \(\mathcal{L}\) (objective). They must serve mechanism→capability narration, not derivation.

### Content scope

- Keep only the most critical high-level information needed to understand the paper.
- Omit unnecessary detail such as table-by-table restatement, exhaustive hyperparameters, and ablation numbers unrelated to core idea.
- Any needed detail should stay attached to the core thread of "problem–method–capability change" to keep writing short and strong.

---

## Required analysis structure

Each `.md` **must** follow this exact layout. If required sections are missing or clearly insufficient, regenerate once. If the paper genuinely cannot fit the structure, keep best attempt, add a one-line note, and tag `analysis_mismatch`.

### 1. YAML frontmatter

- **Flat only**: all keys are top-level scalars or lists; no nested maps.
- For values with colons or long text (for example `primary_logic`), use quoted strings or `|` / `>` multiline scalars.
- Required fields: `title`, `venue`, `year`, `tags` (list), `core_operator`, `primary_logic`, `pdf_ref`, `category`.

```yaml
---
title: "Short Title of the Paper"
venue: CVPR
year: 2025
tags:
  - Motion_Generation_Text_Speech_Music_Driven
  - status/analyzed
core_operator: One-line description of the core mechanism
primary_logic: |
  Input condition → key transformation steps → output
pdf_ref: paperPDFs/Category/Venue_Year/Year_Title.pdf
category: Motion_Generation_Text_Speech_Music_Driven
---
```

### 2. Title and Quick Links & TL;DR

- Level-1 heading: paper title (must match PDF).
- Callout `> [!abstract] **Quick Links & TL;DR**` with:
  - **Links**: arXiv/project if known.
  - **Summary**: one-sentence Chinese summary of core contribution.
  - **Key Performance**: 1-2 bullets on most important metrics/results.

### 3. Part I — Problem and Challenge / The "Skill" Signature

**Semantics: what hard problem does this paper really solve, and where is the bottleneck?**

- core capability definition (what the method can and cannot do)
- true source of challenge (data/representation/optimization/controllability/consistency/physical constraints/generalization/evaluation...)
- input/output interface (concise; enough to understand method, not API documentation)
- boundary conditions (where the method works vs fails)

Keep compact. Avoid implementation details.

### 4. Part II — Method and Insight / High-Dimensional Insight

**Semantics: what causal knob did the authors introduce, and what is the core mechanism?**

- overall design philosophy (paradigm shift/innovation point)
- **Must include explicit subsection: The "Aha!" Moment**
  - core intuition: what changed → which distributions/constraints/information bottlenecks changed → what capability changed
  - why this design works (causal explanation, not restatement)
  - strategic trade-offs (advantages and limits)

### 5. Part III — Experiments and Evidence / Technical Deep Dive

**Semantics: where is the evidence for capability jump, and how well does it work?**

- core pipeline overview (flow/module relations; text description is enough)
- key experiment signals: **prioritize signal type and conclusion**, not number dumping. For example: better coherence under OOD instructions; less drift in long sequences...
- a few key numbers (1-2 metrics) that best support the capability-jump narrative
- implementation constraints (dataset/backbone/hardware; one line is enough)

### 6. Local Reading / Local PDF reference

```
![[paperPDFs/<Category>/<Venue_Year>/<filename>.pdf]]
```

Must match `pdf_ref` in frontmatter.

---

## Execution steps (per PDF)

1. **Resolve paths**
   - **Primary**: if processing from `analysis_log.csv`, read the 8th column (`pdf_path`) directly as PDF path. This field already uses `paperPDFs/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.pdf`; use as-is.
   - **Fallback** (if `pdf_path` empty or not from log): build path from `paperPDFs/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.pdf` using inferred category/venue.
   - If not under paperPDFs, infer/obtain category+venue+year; run `ingest_pdfs.py` (or equivalent); write MD to resolved path.
   - MD path is always derived from final PDF path: `paperPDFs` → `paperAnalysis`, `.pdf` → `.md`.
2. **Read PDF** (local only). Extract along challenge → causal knob → capability evidence.
   - **PDF size threshold**: if PDF > 20 MB, compress in place first:
     ```
     gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
        -dNOPAUSE -dQUIET -dBATCH \
        -dColorImageResolution=150 -dGrayImageResolution=150 \
        -sOutputFile=<tmp>.pdf <input>.pdf && mv <tmp>.pdf <input>.pdf
     ```
     If still > 20 MB, retry with `/screen`. If still > 20 MB after both, continue analysis anyway.
   - **Supplementary material**: skip by default. Read supplement only if main PDF defers critical method details required to answer the Three Questions.
3. **Generate `.md`** with required structure and writing principles.
4. **Structural check**: if Part I/II/III or "Aha!" missing/thin, regenerate once. If still not fitting, keep best attempt + note + `analysis_mismatch`.
5. **Write file** to resolved MD path. Ensure `pdf_ref` and final `![[...]]` point to the same PDF.
6. **Update log** after each batch in `paperAnalysis/analysis_log.csv`:
   - success and structure-complete → `checked`
   - PDF missing/unreadable → leave as `Wait`
   - structure mismatch after retry → `analysis_mismatch`

---

## Batch behavior

- Process **4 PDFs per batch** in log order (top-to-bottom, `Wait` entries only).
- **Context isolation**: each paper must be analyzed only from its own PDF. Do not cross-reference or carry information across papers in the same batch.
- **Parallelism**: PDFs in one batch can be read/analyzed concurrently. Resolve paths and read all PDFs first, then generate each `.md` independently. Write files and update log after all analyses finish.
- At batch end, output: list of written MD files, any `analysis_mismatch` with reason, and log status update preview.
- **Next step (suggest only, no auto-run)**: suggest "If index refresh is needed, call `papers-build-collection-index` next." Do not auto-run build inside `papers-analyze-pdf` to avoid large index file churn on every analysis.
- Skip non-`Wait` entries.

---

## Reference examples (structure only)

- **Primary reference (gold standard)**: `paperAnalysis/Motion_Generation_Text_Speech_Music_Driven/ICLR_2026/2026_Motion_R1_Enhancing_Motion_Generation_Decomposed_CoT_RL_Binding.md`
- **Additional structure reference**: `paperAnalysis/Motion_Generation_Text_Speech_Music_Driven/AAAI_2025/2025_ALERT_Motion_Autonomous_LLM_Enhanced_Adversarial_Attack_for_Text_to_Motion.md`

Use only as layout references; do not copy content. Paths are relative to repository root.
