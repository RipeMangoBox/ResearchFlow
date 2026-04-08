---
name: papers-query-knowledge-base
description: Queries the local paper collection and analysis notes for research. Finds papers by task, technique, or venue; summarizes or compares methods; cites core_operator and primary_logic from analysis frontmatter. Use when the user asks to find papers, list papers by topic/technique/venue, compare motion or interaction methods, cite from their paper analyses, or when the workspace contains the paper knowledge base.
---

# Paper Knowledge Base (paperCollection + paperAnalysis)

Use this skill to query the local paper corpus across conversations and projects. The corpus has two layers: **paperCollection** (index by task / technique / venue) and **paperAnalysis** (analysis notes with TL;DR, Part I/II/III, PDF links).

**Where the knowledge base lives:** Under the current repository root that contains `paperCollection/` and `paperAnalysis/`. When used from another project, open or reference the repository that contains those folders.

## Paths

- Use repo-relative paths rooted at the folder that contains `paperCollection/`, `paperAnalysis/`, and `paperPDFs/`.
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

- **Entry**: `paperCollection/README.md` — links to All papers, by task, by technique, by venue.
- **All papers**: `paperCollection/_AllPapers.md` — by task → venue+year; each line `[[analysis|Title (Venue Year)]]` · `[[PDF]]`.
- **By task**: `paperCollection/by_task/<Task>.md`.
- **By technique**: `paperCollection/by_technique/_Index.md` and `paperCollection/by_technique/<tag>.md`.
- **By venue**: `paperCollection/by_venue/_Index.md` and `paperCollection/by_venue/<Venue>.md`.
- **Analysis notes**: `paperAnalysis/<Category>/<Venue_Year>/<Year>_<Title>.md`; PDF path in frontmatter `pdf_ref`.

All paths use forward slashes.

## How to use for research

1. **Find papers** — By task: `paperCollection/by_task/<Task>.md`. By technique: `by_technique/_Index.md` → `by_technique/<tag>.md`. By venue: `by_venue/_Index.md` → `by_venue/<Venue>.md`. Full list: `paperCollection/_AllPapers.md`.

2. **Read an analysis** — Follow collection links to the analysis note. Each note has: **Frontmatter** (title, venue, year, category, tags, pdf_ref, core_operator, primary_logic); **TL;DR** (Summary, Key Performance); **Part I** (problem); **Part II** (method, "Aha!"); **Part III** (technical); **Local Reading** (PDF).

3. **Cite in answers** — Use **core_operator** and **primary_logic** from frontmatter for one-line method summary; **Summary** and **Key Performance** from TL;DR for comparison. Link to note: `[[paperAnalysis/.../file.md|Title]]`; to PDF: `[[paperPDFs/.../file.pdf|PDF]]`.

## Index frontmatter (for tooling)

Collection pages: `type: paper-index`, `dimension: task | technique | venue | all`, and optional `task:` / `technique:` / `venue:`.

For exact paths and frontmatter schema, see [references/structure.md](references/structure.md).

## Regenerating the collection

After adding or changing analysis notes, from the repository root:

```bash
python .claude/skills/papers-build-collection-index/scripts/build_paper_collection.py
```

Only notes with valid `pdf_ref` (path under `paperPDFs/` ending in `.pdf`) are indexed.

## Research Idea Evaluation (via knowledge base)

When the user asks to evaluate a research idea using the knowledge base, analyze along three dimensions:

### 1. 能力上限（Capability Ceiling）

基于知识库中现有方法的性能瓶颈，评估该 idea 的理论上限：

- 从 `core_operator` 和 `primary_logic` 提取现有方法的核心机制；
- 识别当前 SOTA 的性能瓶颈（数据质量、模型结构、评估协议）；
- 判断该 idea 的改进是否触及这些瓶颈，还是绕过它们；
- 输出：「该 idea 能达到的上限是什么，受限于哪些因素」。

### 2. CCF-A 审稿人认可度（Reviewer Acceptance）

基于知识库中顶会论文的成文惯例，评估该 idea 在 CCF-A 场馆（NeurIPS / CVPR / ICCV / ECCV / ICLR / AAAI）的可接受性：

- 对比同类 idea 在顶会的发表形式（任务设定、实验设计、消融研究）；
- 判断 novelty 是否足够：相比知识库中最相关论文，差异化贡献是否清晰；
- 识别审稿人可能质疑的点（鲁棒性、泛化性、与 baseline 的对比）；
- 输出：「该 idea 在 CCF-A 的接受可能性，以及需要补强的关键点」。

### 3. 现有工作支持力度（Literature Support）

从知识库中检索与该 idea 最相关的论文，评估文献支撑：

- 正向支撑：哪些已发表工作验证了该 idea 的前提假设或子模块；
- 空白确认：知识库中是否存在直接竞品，若不存在则说明空白；
- 技术借鉴：哪些工作的方法（core_operator）可直接复用或改造；
- 输出：「最相关的 3-5 篇论文 + 各自与该 idea 的关系（支撑 / 竞品 / 可借鉴）」。

### 输出格式

```markdown
## Idea 评估：<idea 名称>

### 能力上限
- 上限估计：...
- 主要限制因素：...

### CCF-A 审稿人认可度
- 认可度评估：高 / 中 / 低
- 主要风险点：...
- 需补强的关键实验：...

### 现有工作支持力度
- 正向支撑：[[paper1]], [[paper2]]
- 直接竞品：[[paper3]]（差异：...）
- 可借鉴技术：[[paper4]]（借鉴点：...）
```
