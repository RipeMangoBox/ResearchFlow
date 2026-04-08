# Share Note Export Reference

## Detailed workflow

### Step 1: Read the source note

Read the full note first.
Identify:

- all internal paper references
- any internal-path leakage
- any sections that assume the reader knows the local knowledge base
- the central topic for filename generation

### Step 2: Generate the share filename

Generate the new filename from the content topic, not from the original date or working-copy suffix.

Rules:

- remove leading dates like `2026-04-02_`
- remove temporary suffixes like `copy`, `final`, `v2`, `draft`
- keep only the topic-defining slug
- use lowercase words joined by `-`
- end with `_share.md`

Examples:

- `2026-04-02_vq-codebook-motion-text-alignment-survey.md` → `vq-codebook-motion-text-alignment-survey_share.md`
- `2026-04-02_motion-unified-llm-open-source-survey copy.md` → `motion-unified-llm-open-source-survey_share.md`

If the title is ambiguous, infer the slug from the main topic headings and repeated keywords.

### Step 3: Replace internal paper references with public links

#### Body citation format

Use standard Markdown links with short visible names:

```md
[MoMask](URL)
```

Visible text should be the paper abbreviation or short paper name only.
Do not expose local wiki paths.

#### Link resolution priority

Resolve paper links in this order:

1. `paperAnalysis/analysis_log.csv`
2. existing public links already present in the relevant analysis note
3. web search as fallback

#### URL preference order

Use the most stable public paper URL available:

1. official paper page / venue open-access page / AAAI paper page / OpenReview
2. arXiv
3. project page only when a stable paper URL is unavailable

### Step 4: Remove internal traces

Remove or rewrite anything that reveals the internal knowledge base or local note organization.

Typical items to remove or rewrite:

- `[[paperAnalysis/...|...]]`
- internal folder paths
- `Local Reading`
- `本地 PDF`
- `paperAnalysis`
- `paperCollection`
- `knowledge base`
- `知识库`
- wording like `路径：`, `本地索引：`, `对应本地论文：`

Rewrite them into external-reader-friendly phrasing, for example:

- `对应论文：` → `代表论文：`
- `路径：` → `论文：`
- `知识库里可见` → `相关工作中可以看到`
- `我这里把它归类为` → `可以将其归入`

### Step 5: Rewrite wording for external readers

The share version should sound like a polished research note, not a personal working memo.

Rewrite when needed to:

- remove references to personal workflow
- remove references to local collections or folders
- make transitions smoother
- reduce “我这里/我会/我在知识库里” style phrasing
- keep the analysis content and conclusions intact

Preferred tone:

- concise
- professional
- shareable
- reader-oriented

### Step 6: Add a short abstract

Unless the user says otherwise, add a short 3-sentence abstract near the top.

Use this structure:

```md
> 这篇笔记聚焦……
> 核心整理了……
> 如果只想快速把握主线，优先看……
```

Keep it short and useful for forwarding.

### Step 7: Add a unified References section

Append a final `## References` section if one does not already exist.

Format each item as:

```md
1. [MoMask: Generative Masked Modeling of 3D Human Motions](URL)
```

Rules:

- numbered list
- abbreviation first, then full title
- use the same public URL as the body citation when possible
- deduplicate repeated papers
- preserve reading order when reasonable; otherwise group by first appearance

### Step 8: Validate markdown formatting

After edits, run a markdown lint check on the exported file.
Fix any introduced formatting problems.

## Editing policy

When transforming a note:

- prefer editing a generated share copy rather than the source note
- keep the original structure unless sharing readability clearly improves with light restructuring
- do not remove substantive analytical content unless it only exists to describe internal tooling or note organization

## Decision rules

### If the note already looks shareable

Still do the following:

- verify filename matches the share rule
- verify all paper links are public
- verify a `References` section exists
- verify no internal traces remain

### If a paper link cannot be found

Use a stable fallback in this order:

1. venue paper page
2. arXiv
3. project page

Never leave an internal wiki link in the share version.

## Example transformations

### Internal citation → share citation

Before:

```md
[[paperAnalysis/Motion_Generation_Text_Speech_Music_Driven/CVPR_2024/2024_MoMask_Generative_Masked_Modeling_of_3D_Human_Motions|MoMask]]
```

After:

```md
[MoMask](https://arxiv.org/abs/2312.00063)
```

### Internal naming → share naming

Before:

```text
2026-04-02_vq-codebook-motion-text-alignment-survey copy.md
```

After:

```text
vq-codebook-motion-text-alignment-survey_share.md
```

### Internal phrasing → share phrasing

Before:

```md
路径：[[...|MoMask]]
```

After:

```md
论文：[MoMask](https://arxiv.org/abs/2312.00063)
```

## Notes

- This skill is for **share export**, not for building or exposing the internal knowledge base.
- Internal resources may be used to resolve links, but must not be exposed in the final exported note.
- Keep the final note Obsidian-friendly, but ensure all citations are standard shareable Markdown links.
