#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


DEFAULT_PATTERNS = [
    "src/",
    "model",
    "models/",
    "train",
    "trainer",
    "inference",
    "infer",
    "motion",
    "pipeline",
    "eval",
    "benchmark",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Low-disturbance suggestion: prompt paper retrieval only when changed files are highly relevant."
    )
    p.add_argument("--changed-file", action="append", default=[], help="Changed file path (repeatable)")
    p.add_argument("--from-git-diff", action="store_true", help="Read changed files from `git diff --name-only`")
    return p.parse_args()


def changed_from_git() -> List[str]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-only"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def is_relevant(path: str) -> bool:
    low = path.lower()
    return any(p in low for p in DEFAULT_PATTERNS)


def main() -> None:
    args = parse_args()

    changed = list(args.changed_file)
    if args.from_git_diff:
        changed.extend(changed_from_git())

    # dedup keep order
    uniq: List[str] = []
    seen = set()
    for p in changed:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    if not uniq:
        return

    hits = [p for p in uniq if is_relevant(p)]
    if not hits:
        return

    preview = ", ".join(hits[:3])
    suffix = " ..." if len(hits) > 3 else ""
    print("Detected highly relevant code changes. Run /code-context-paper-retrieval (brief)?")
    print(f"Matched files: {preview}{suffix}")


if __name__ == "__main__":
    main()
