---
name: papers-analyze-pdf
description: Analyzes academic PDFs into structured Markdown under `paperAnalysis/`. Reads local PDFs only (no network, no image generation). Use when the user asks to analyze one PDF or a folder of PDFs, or to run paper analysis for the local knowledge base. Handles path resolution so that if the PDF is already under `paperPDFs/<Category>/<Venue_Year>/`, the `.md` is written to `paperAnalysis/` in the same structure; otherwise infers category and venue (e.g. `CVPR_2025`), copies the PDF into `paperPDFs/`, and writes the `.md` to `paperAnalysis/`.
---

# Paper Analysis

Analyze one or more PDFs into structured analysis notes under `paperAnalysis/`, following a fixed template. **Read only local PDFs**; do not fetch from the network or generate figures.

## Scope

- **Input**: A single PDF path, a directory, or a batch list from `paperAnalysis/analysis_log.csv` (process `Downloaded` entries in order).
- **Output**: One `.md` per PDF at the correct path under `paperAnalysis/`. After each batch, update `analysis_log.csv` state to `checked`.

Assume the repository root is the directory that contains `paperPDFs/` and `paperAnalysis/`.

---

## Path and storage rules

### Filename convention

- **Title part** (`<Year>_<SanitizedTitle>`): Use a single `_` between every two words. No spaces, commas, hyphens, or other separators. Example: `High_Fidelity` not `High-Fidelity`, `Long_Form` not `Long-Form`.
- **Rename requirement**: When ingesting or analyzing a new paper, **always rename both the PDF and the corresponding `.md`** to follow the `<Year>_<ShortEnglishTitle>` pattern before writing or updating analysis notes.

### When the PDF is already under paperPDFs

- PDF path pattern: `paperPDFs/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.pdf`
- **MD path**: Replace `paperPDFs` with `paperAnalysis` in the PDF path; change extension to `.md`. Keep the rest of the path and filename (suffix only changes).
- Do **not** modify an existing `.md` if the corresponding PDF was not found or not read.

### When the PDF is not under paperPDFs

1. **Infer or ask for**: **category** and **venue + year** (e.g. CVPR, 2025 → `CVPR_2025`).
2. **Categories** (use one of): `Human_Human_Interaction`, `Human_Object_Interaction`, `Human_Scene_Interaction`, `Motion_Controlled_ImageVideo_Generation`, `Motion_Editing`, `Motion_Generation_Text_Speech_Music_Driven`, `Motion_Stylization`.
3. **Place PDF**: Copy (or move) the PDF to `paperPDFs/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.pdf`. Use the colocated ingest script if available:
   `python .claude/skills/papers-analyze-pdf/scripts/ingest_pdfs.py <path> --category <cat> --venue <Venue> --year <Year> [--title <Title>]`
4. **Place MD**: Write to `paperAnalysis/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.md`.
5. If category or venue/year cannot be inferred, ask the user before copying or writing.

---

## Writing principles（写作总原则）

All analysis body text (except YAML frontmatter, hyperlinks, and formulas) is written in **Chinese** for reading consistency inside the local vault.

### The Three Questions — every analysis must answer these

1. **What/Why**：真正的卡点是什么？为什么现在要解决？（数据/表示/优化/可控性/一致性/物理约束/泛化/评测……）
2. **How**：作者引入了什么关键"因果旋钮"？（conditioning / constraint / architecture / objective / sampling / training recipe）改了哪里 → 影响了哪些分布/约束/信息瓶颈 → 产生了什么能力变化？
3. **So what**：相比前人，能力跃迁在哪里？哪些实验信号最能支撑这个结论？

### Expression priority

**直觉 / 结构化解释 > 机制抽象描述 > 必要的少量符号**

- 写"机制"时优先描述**系统层面的因果关系**：改了哪里 → 影响了哪些分布/约束/信息瓶颈 → 带来什么能力变化。
- 写"证据"时优先描述**信号类型与结论**（更可控 / 更稳 / 更长时一致 / 更少漂移 / 更强泛化 / 更少数据依赖），而非堆砌数字与表格。
- 用来组织语言的高维对象：**空间**（输入/条件/潜空间/动作/约束/评测空间）、**对象**（表示、先验、控制量、分解、对齐、检索、规划、反馈）、**流程**（训练时改变了什么分布/约束；推理时如何注入可控性与稳定性）。

### Formula rules

- **不展开推导**；不逐项解释 loss；不逐符号解读每个变量。
- 如必须提及公式：最多保留 **1-2 条"口头化"的目标/约束描述**，挂在信息流解释之下。
- 只允许极简占位符（可选）：\(x\)（条件/观测）、\(z\)（潜表征）、\(f_\theta\)（模型）、\(\mathcal{L}\)（目标）。必须服务于"机制→能力"叙事，不做公式推导。

### Content scope

- 只保留对理解论文**最关键的高维信息**。不必展开的内容：逐表逐图抄细节、完整超参数列表、与核心思想无关的所有 ablation 数字。
- 少量必要细节**挂靠在"问题–方法–能力变化"主线之下**，保证整体文字尽量短而有力。

---

## Required analysis structure

Each `.md` **must** follow this exact layout. If any required part is missing or clearly insufficient, regenerate once; if the paper genuinely cannot fit the structure, keep the best attempt, add a one-sentence note, and tag `analysis_mismatch`.

### 1. YAML frontmatter

- **Flat only**: all keys are top-level scalars or lists; no nested maps.
- For values with colons or long text (e.g. `primary_logic`), use quoted strings or `|` / `>` multiline scalars.
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

- Level-1 heading: paper title (match the PDF).
- Callout `> [!abstract] **Quick Links & TL;DR**` with:
  - **Links**: arXiv / project link if known.
  - **Summary**: 一句中文概括主要贡献。
  - **Key Performance**: 1-2 bullet points on the most important metrics or results.

### 3. Part I — 问题与挑战 / The "Skill" Signature

**语义：这篇论文真正想解决什么难题？卡点在哪里？**

- 核心能力定义（这个方法能做什么，做不到什么）
- 真正的挑战来源（数据/表示/优化/可控性/一致性/物理约束/泛化/评测……）
- 输入/输出接口（简洁，不做 API 文档，只要够理解方法即可）
- 边界条件（方法在什么场景下有效，在什么条件下失效）

Keep compact. Avoid implementation details.

### 4. Part II — 方法与洞察 / High-Dimensional Insight

**语义：作者引入了什么因果旋钮？核心机制是什么？**

- 方法的整体设计哲学（范式转变/创新点）
- **必须包含显式小节：The "Aha!" Moment**
  - 核心直觉：作者改变了什么 → 影响了哪些分布/约束/信息瓶颈 → 带来什么能力变化
  - 这个设计为什么有效（因果解释，不是描述）
  - 战略权衡（优势与局限）

### 5. Part III — 实验与证据 / Technical Deep Dive

**语义：能力跃迁的证据在哪里？方法的实际效果如何？**

- 核心 pipeline 概述（流程图 / 模块关系，文字描述即可）
- 关键实验信号：**优先描述信号类型与结论**，而不是堆砌数字。例如：在 OOD 指令下生成连贯性显著高于 baseline；在长序列条件下不发生漂移……
- 少量关键数字（1-2 个最能支撑主张的 metric），支撑"能力跃迁"叙事
- 实现约束（数据集、骨干模型、硬件，一行即可）

### 6. Local Reading / 本地 PDF 引用

```
![[paperPDFs/<Category>/<Venue_Year>/<filename>.pdf]]
```

Use the same path as `pdf_ref` in frontmatter.

---

## Execution steps (per PDF)

1. **Resolve paths**
   - **Primary**: if processing from `analysis_log.csv`, read the 8th column (`pdf_path`) of the matching row directly as the PDF path. This field uses the format `paperPDFs/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.pdf` and is already resolved — use it as-is.
   - **Fallback** (when `pdf_path` field is empty or the PDF is not from the log): construct the path from `paperPDFs/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.pdf` using the category and venue inferred from the paper title/venue.
   - Not under paperPDFs → infer/obtain category+venue+year; run `ingest_pdfs.py` (or equivalent); write MD to resolved path.
   - MD path is always derived from the final PDF path: replace `paperPDFs` → `paperAnalysis`, `.pdf` → `.md`.
2. **Read the PDF** (local only). Extract along: challenge → causal knob → capability evidence.
   - **PDF size threshold**: if the PDF is larger than 20 MB, first compress it in-place using:
     ```
     gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
        -dNOPAUSE -dQUIET -dBATCH \
        -dColorImageResolution=150 -dGrayImageResolution=150 \
        -sOutputFile=<tmp>.pdf <input>.pdf && mv <tmp>.pdf <input>.pdf
     ```
     If the compressed file is still > 20 MB, try `/screen` instead of `/ebook`. If still > 20 MB after both attempts, proceed with analysis anyway (large file size does not block analysis).
   - **Supplementary materials**: skip by default. Read supplementary only if the main PDF explicitly defers critical method details (e.g. algorithm pseudocode, key ablation design) to the supplement and those details are necessary to answer the Three Questions.
3. **Generate the `.md`** with the structure above. Follow Writing Principles throughout.
4. **Structural check**: If any of Part I / Part II / Part III or the "Aha!" subsection is missing or clearly thin, regenerate once. If still not fitting, keep best attempt + note + tag `analysis_mismatch`.
5. **Write the file** to the resolved MD path. Ensure `pdf_ref` and final `![[...]]` point to the same PDF path.
6. **Update log**: After processing each batch, update `paperAnalysis/analysis_log.csv`:
   - Successful, structure-complete → `checked`
   - PDF not found / unreadable → leave as `Wait`
   - Structure mismatch after retry → `analysis_mismatch`

---

## Batch behavior

- Process **4 PDFs per batch**, in log order (top-to-bottom, `Wait` entries only).
- **Context isolation**: each paper is analyzed solely from its own PDF. Do not reference, compare, or carry over information from other PDFs in the same batch. Treat every paper as an independent task.
- **Parallelism**: all PDFs in a batch can be read and analyzed concurrently. Resolve paths and read all batch PDFs first, then generate each `.md` independently. Write files and update the log only after all analyses in the batch are complete.
- At batch end, output: list of MDs written, any `analysis_mismatch` with reason, log status update preview.
- **Next step (suggest only, do not auto-run)**: 提示用户“如需刷新索引，可下一步调用 `papers-build-collection-index`”；不要在 `papers-analyze-pdf` 内自动执行 build，以避免每次分析都产生大批索引文件改动。
- Skip non-`Wait` entries without processing.

---

## Reference examples (structure only)

- **Primary reference (gold standard)**: `paperAnalysis/Motion_Generation_Text_Speech_Music_Driven/ICLR_2026/2026_Motion_R1_Enhancing_Motion_Generation_Decomposed_CoT_RL_Binding.md`
- **Additional structure reference**: `paperAnalysis/Motion_Generation_Text_Speech_Music_Driven/AAAI_2025/2025_ALERT_Motion_Autonomous_LLM_Enhanced_Adversarial_Attack_for_Text_to_Motion.md`

Use only as layout references; do not copy content. Paths are relative to the repository root.
