---
name: reviewer-stress-test
description: Runs a strict-but-fair ICLR/CVPR/SIGGRAPH reviewer challenge on idea, roadmap, or full paper, with major-risk diagnostics and actionable repair paths. Use when the user wants high-pressure review questioning instead of brainstorming.
---

# Reviewer Stress Test

## Purpose

Provide a strong reviewer-mode stress test for:

- idea
- roadmap
- full paper draft

Default strictness is **strong review**.
The skill must still be fair and must provide repair paths.

## Positioning

This is an **evaluative** mode, not a co-creation ideation mode.

- Focus: attack assumptions, expose rejection-level risks, test evidence sufficiency.
- Output style: verdict-like risk assessment with fixes.

## Evidence protocol

1. **KB-first**: first ground analysis via `papers-query-knowledge-base`.
2. **On-demand web checks**: search web only for high-risk or uncertain claims (e.g., novelty/SOTA/parallel work).
3. Separate missing evidence from actual invalidity.

## Mandatory user clarification

Before final verdict, require the user to state:

- The key difference between this work and the most similar prior works.
- Preferably top-3 nearest works with one-line distinctions.

If this is missing, mark novelty confidence as limited.

## Review dimensions

Challenge from multiple angles:

- Problem framing and significance
- Novelty and non-triviality
- Technical soundness
- Experimental rigor (baselines, ablations, statistics)
- Reproducibility and implementation feasibility
- Failure modes, robustness, ethics/bias scope

## Output contract

Return structured sections:

1. **Overall risk verdict** (acceptance risk band + confidence)
2. **Major concerns** (rejection-level)
3. **Minor concerns**
4. **Nearest-work gap check**
5. **Repair paths** (for each major concern: minimum actionable steps)
6. **Re-review checklist** (what evidence would change verdict)

## Tone and fairness guardrails

- Be rigorous, specific, and evidence-based.
- Do not use dismissive or insulting phrasing.
- Every major concern should include at least one feasible repair action.

## Non-goals

- Not for open-ended idea expansion.
- Not for broad roadmap generation from scratch.

## Independence

本 skill 可独立使用，不依赖 `idea-focus-coach` 或 `research-brainstorm-from-kb` 的输出。用户可以直接带着已成型的 idea、roadmap 或 paper draft 进入 stress-test。
