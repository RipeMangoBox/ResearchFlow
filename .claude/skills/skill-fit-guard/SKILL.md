---
name: skill-fit-guard
description: Detects post-invocation skill mismatch patterns, summarizes why they may recur, and asks whether to revise the skill now. Use when a called .claude skill output is clearly unfit for the user intent and similar mismatch is likely to happen again.
---

# Skill Fit Guard

## What this skill does

This skill handles recurring skill mismatch after a `.claude` skill call.

It does **not** auto-modify any skill.
It asks for user consent before any skill spec change.

## Triggering conditions

Use this skill only when both conditions are true:

- A just-called skill is not aligned with user intent (scope/stage/input/output mismatch).
- The same mismatch is likely to recur (pattern-level issue, not one-off noise).

## Inputs

- Invoked skill name
- User's actual intent
- Observed mismatch signals
- Recurrence evidence (if any)

## Diagnosis framework

Diagnose mismatch in one or more buckets:

- Intent mismatch: chosen skill solves a different goal.
- Stage mismatch: correct capability, wrong pipeline stage.
- Input contract mismatch: required inputs differ from user-provided context.
- Output contract mismatch: user expects different granularity/format.
- Strictness mismatch: too harsh/too shallow for the requested mode.

## Output contract

Always return four blocks:

1. **Observed mismatch**: short, evidence-based summary.
2. **Why recurrence is likely**: what in trigger/spec causes repeated misfire.
3. **Revision options**: 2-4 options (light/medium/heavy).
4. **Consent question**: ask whether to modify the skill now.

## Option menu guidance

Offer options similar to:

- Light: tune trigger wording and non-goals.
- Medium: refine workflow and output contract.
- Heavy: split one skill into two focused skills.
- No change: keep current skill and add usage note only.

## Boundaries

- No autonomous edits to skill files.
- No forced workflow switch.
- No repeated prompting after user declines.
