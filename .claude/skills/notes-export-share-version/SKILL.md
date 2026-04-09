---
name: notes-export-share-version
description: Converts internal research notes into shareable Obsidian-friendly Markdown files. Use when the user explicitly asks to make a share version, export a note for sharing, hide knowledge base traces, replace internal wiki links with public paper links, rewrite internal wording for external readers, or generate a content-driven share filename.
---

# Share Note Export

## What this skill does

Converts an internal research note into a shareable Markdown note for external readers.

Default behavior:

- keep the source note unchanged
- create a share version in the same directory
- remove internal knowledge-base traces
- rewrite paper mentions as public Markdown links with short visible names
- add a short abstract at the top
- append a unified `References` section at the end
- run a markdown lint check after edits

## When to use

Use this skill only when the user explicitly asks for a share version, such as:

- "make a share version"
- "export a share draft"
- "version for external readers"
- "hide knowledge base traces"
- "clean internal notes into a shareable version"

Do not apply automatically for normal note editing.

## Quick workflow

Copy this checklist and follow it in order:

```text
Share Export Progress:
- [ ] Step 1: Read the source note
- [ ] Step 2: Generate a content-driven `_share.md` filename
- [ ] Step 3: Replace internal paper references with public Markdown links
- [ ] Step 4: Remove internal traces and rewrite internal wording
- [ ] Step 5: Add a short abstract
- [ ] Step 6: Add or refresh a unified `References` section
- [ ] Step 7: Validate markdown formatting
```

## Core output rules

### 1. Never overwrite the source note by default

Treat the source note as the internal working version.
Create a separate share version unless the user explicitly asks to overwrite.

### 2. Output location

Write the share version to the **same directory** as the source file.

### 3. Filename rule

Use a content-driven filename:

- remove leading dates like `2026-04-02_`
- remove temporary suffixes like `copy`, `final`, `v2`, `draft`
- keep only the topic-defining slug
- use lowercase words joined by `-`
- end with `_share.md`

Example:

- `2026-04-02_vq-codebook-motion-text-alignment-survey copy.md` → `vq-codebook-motion-text-alignment-survey_share.md`

### 4. Body citation format

In the body, use standard Markdown links with short visible names:

```md
[MoMask](URL)
```

Do not expose local wiki paths in the exported note.

### 5. References format

Append a final `## References` section using numbered Markdown links:

```md
1. [MoMask: Generative Masked Modeling of 3D Human Motions](URL)
```

## Additional resources

- For detailed rules, fallback logic, phrasing cleanup, and examples, see [reference.md](reference.md)
