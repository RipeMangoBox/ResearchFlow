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
- `local PDF`
- `paperAnalysis`
- `paperCollection`
- `knowledge base`
- wording like `path:`, `local index:`, `local paper mapping:`

Rewrite them into external-reader-friendly phrasing, for example:

- `mapped paper:` → `representative paper:`
- `path:` → `paper:`
- `visible in the knowledge base` → `seen in related work`
- `I classify this as` → `this can be categorized as`

### Step 5: Rewrite wording for external readers

The share version should read like a polished research note, not a personal working memo.

Rewrite where needed to:

- remove references to personal workflow
- remove references to local collections/folders
- make transitions smoother
- reduce first-person workflow phrasing
- keep analysis content and conclusions intact

Preferred tone:

- concise
- professional
- shareable
- reader-oriented

### Step 6: Add a short abstract

Unless the user says otherwise, add a short 3-sentence abstract near the top.

Use this structure:

```md
> This note focuses on ...
> It summarizes ...
> If you only want the main thread quickly, start with ...
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
- reuse the same public URL as body citation when possible
- deduplicate repeated papers
- preserve reading order when practical; otherwise group by first appearance

### Step 8: Validate markdown formatting

After edits, run markdown lint on the exported file.
Fix any introduced formatting issues.

## Editing policy

When transforming a note:

- prefer editing a generated share copy rather than the source note
- keep original structure unless readability clearly improves with light restructuring
- do not remove substantive analysis unless it only describes internal tooling or note organization

## Decision rules

### If the note already looks shareable

Still do the following:

- verify filename follows share rule
- verify all paper links are public
- verify a `References` section exists
- verify no internal traces remain

### If a paper link cannot be found

Use this fallback order:

1. venue paper page
2. arXiv
3. project page

Never leave internal wiki links in the share version.

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
path: [[...|MoMask]]
```

After:

```md
paper: [MoMask](https://arxiv.org/abs/2312.00063)
```

## Notes

- This skill is for **share export**, not for building or exposing the internal knowledge base.
- Internal resources may be used to resolve links, but must not be exposed in final exported notes.
- Keep the final note Obsidian-friendly, while ensuring all citations are standard shareable Markdown links.
