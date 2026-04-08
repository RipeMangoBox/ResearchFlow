from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VAULT_ROOT = Path(__file__).resolve().parents[4]
PAPER_ANALYSIS_DIR = VAULT_ROOT / "paperAnalysis"

SUMMARY_PATH = VAULT_ROOT / "paperCollection" / "_tag_normalization_summary.json"

# Technique tag synonym map (apply after formatting normalization).
# Keyed by norm_key(tag) so that variants like underscores/spaces/case unify.
TECH_SYNONYMS: Dict[str, str] = {
    # Inbetweening variants → canonical
    "inbetweening": "motion-inbetweening",
    "in-betweening": "motion-inbetweening",
    "motion-in-betweening": "motion-inbetweening",

    # Diffusion family
    "diffusion": "diffusion",
    "diffusion-model": "diffusion",

    # Mixture-of-experts / MoE
    "mixture-of-experts": "mixture-of-experts",
    "transformer/moe": "mixture-of-experts",

    # LLM family
    "llm": "LLM",
    "large-language-model": "LLM",
    "language-model": "LLM",
    "multimodal-llm": "multimodal-LLM",
    "multimodal-large-language-models": "multimodal-LLM",

    # Reinforcement learning
    "rl": "reinforcement-learning",
    "reinforcement-learning": "reinforcement-learning",

    # Real-time / online
    "real-time": "real-time",
    "realtime": "real-time",
    "online/real-time": "real-time",

    # Mocap
    "mocap": "motion-capture",
    "motion-capture": "motion-capture",

    # VR / AR
    "vr": "VR/AR",
    "ar/vr": "VR/AR",

    # Body models
    "smpl": "SMPL-X",
    "smpl-x": "SMPL-X",

    # Text-driven motion
    "text-to-motion": "text-to-motion",
    "text2motion": "text-to-motion",
    "language-to-motion": "text-to-motion",
    "motion-generation/text-to-motion": "text-to-motion",

    # Motion editing & stylization (technique-level; task tags handled separately)
    "pose-editing": "motion-editing",
    "editing": "motion-editing",
    "animation-editing": "motion-editing",
    "stylized-motion": "motion-stylization",
    "style-transfer": "motion-stylization",

    # Audio / music driven
    "music-driven": "music-driven-generation",
    "music-to-dance": "music-driven-generation",
    "music-to-motion": "music-driven-generation",
    "music-driven-dance": "music-driven-generation",
    "audio-driven": "audio-driven",
    "audio-driven-animation": "audio-driven",

    # Dance generation
    "dance-generation": "dance-generation",
    "dance": "dance-generation",

    # Video generation
    "text-to-video": "video-generation",
    "motion-video-generation": "video-generation",
    "video-diffusion": "video-generation",

    # Physics-related
    "physics-based": "physics-based",
    "physics": "physics-based",
    "physics-informed": "physics-based",

    # Benchmarking / evaluation
    "benchmark": "benchmark",
    "benchmarking": "benchmark",
    "metrics": "benchmark",
    "evaluation": "benchmark",
}


FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$")


def list_task_dirs() -> List[str]:
    if not PAPER_ANALYSIS_DIR.exists():
        return []
    out: List[str] = []
    for p in PAPER_ANALYSIS_DIR.iterdir():
        if not p.is_dir():
            continue
        if p.name == "__pycache__":
            continue
        out.append(p.name)
    out.sort(key=lambda x: x.lower())
    return out


def norm_key(tag: str) -> str:
    t = (tag or "").strip()
    t = t.replace("—", "-").replace("–", "-")
    t = t.replace("_", "-").replace(" ", "-")
    t = re.sub(r"-{2,}", "-", t)
    t = t.lower()
    segs: List[str] = []
    for seg in t.split("/"):
        seg = re.sub(r"[^a-z0-9\-\.\+]+", "-", seg)
        seg = re.sub(r"-{2,}", "-", seg).strip("-")
        segs.append(seg)
    return "/".join(segs)


def canonicalize_status(tag: str) -> str:
    # keep slash, normalize to lowercase
    return norm_key(tag)


def canonicalize_tech(tag: str) -> str:
    return norm_key(tag)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def iter_md_files() -> List[Path]:
    files = [p for p in PAPER_ANALYSIS_DIR.rglob("*.md") if "__pycache__" not in p.parts]
    files.sort(key=lambda p: p.as_posix().lower())
    return files


def extract_frontmatter(lines: List[str]) -> Tuple[Optional[int], Optional[int]]:
    if not lines or not FRONTMATTER_BOUNDARY.match(lines[0]):
        return (None, None)
    for i in range(1, len(lines)):
        if FRONTMATTER_BOUNDARY.match(lines[i]):
            return (0, i)
    return (0, None)


def find_tags_block(fm_lines: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (start_idx, end_idx) in fm_lines slice for tags block:
    - start_idx points to the 'tags:' line
    - end_idx is exclusive, the first line after the list items
    """
    start = None
    for i, line in enumerate(fm_lines):
        if line.strip() == "tags:":
            start = i
            break
    if start is None:
        return (None, None)
    end = start + 1
    while end < len(fm_lines):
        l = fm_lines[end]
        # list item lines are typically indented: "  - xxx"
        if re.match(r"^\s*-\s+.+$", l):
            end += 1
            continue
        # allow blank lines inside tags block (rare) and keep them as part of block
        if l.strip() == "":
            end += 1
            continue
        break
    return (start, end)


def parse_tags_from_block(block_lines: List[str]) -> List[str]:
    tags: List[str] = []
    for l in block_lines:
        m = re.match(r"^\s*-\s+(.+?)\s*$", l)
        if not m:
            continue
        tags.append(m.group(1))
    return tags


def build_tags_block(tags: List[str]) -> List[str]:
    out = ["tags:"]
    for t in tags:
        out.append(f"  - {t}")
    return out


@dataclass
class Change:
    path: str
    before: List[str]
    after: List[str]


def normalize_tags(raw_tags: List[str], task_tag: str, task_norm_map: Dict[str, str]) -> List[str]:
    # Ensure task tag present as the first tag.
    canonical_task = task_tag
    out: List[str] = []
    seen = set()

    def add(tag: str) -> None:
        if not tag:
            return
        if tag in seen:
            return
        seen.add(tag)
        out.append(tag)

    add(canonical_task)

    status_tags: List[str] = []
    for t in raw_tags:
        t = str(t).strip()
        if not t:
            continue
        nk = norm_key(t)
        if nk in task_norm_map:
            # map any variant to canonical task tag
            add(task_norm_map[nk])
            continue
        if nk.startswith("status/"):
            status_tags.append(canonicalize_status(t))
            continue
        tech = canonicalize_tech(t)
        tech = TECH_SYNONYMS.get(norm_key(tech), tech)
        add(tech)

    # Keep status tags at the end, stable unique
    for st in status_tags:
        add(st)

    return out


def main() -> int:
    task_dirs = list_task_dirs()
    task_norm_map = {norm_key(t): t for t in task_dirs}

    files = iter_md_files()
    changes: List[Change] = []
    tag_rewrites = 0

    for md in files:
        rel = md.relative_to(VAULT_ROOT).as_posix()
        parts = rel.split("/")
        task_tag = parts[1] if len(parts) >= 2 else "Uncategorized"

        lines = read_text(md).splitlines()
        fm_start, fm_end = extract_frontmatter(lines)
        if fm_start is None or fm_end is None:
            continue

        fm_lines = lines[fm_start + 1 : fm_end]
        start, end = find_tags_block(fm_lines)
        if start is None or end is None:
            # no tags block; skip (we won't inject new tags automatically)
            continue

        raw_tags = parse_tags_from_block(fm_lines[start + 1 : end])
        new_tags = normalize_tags(raw_tags, task_tag=task_tag, task_norm_map=task_norm_map)

        if raw_tags == new_tags:
            continue

        new_fm = fm_lines[:start] + build_tags_block(new_tags) + fm_lines[end:]
        new_lines = lines[: fm_start + 1] + new_fm + lines[fm_end:]

        write_text(md, "\n".join(new_lines) + "\n")
        changes.append(Change(path=rel, before=raw_tags, after=new_tags))
        tag_rewrites += 1

    summary = {
        "generated": datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "files_scanned": len(files),
        "files_rewritten": tag_rewrites,
        "task_dirs": task_dirs,
        "top_changes": [
            {
                "path": c.path,
                "before": c.before,
                "after": c.after,
            }
            for c in changes[:50]
        ],
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] scanned: {len(files)}")
    print(f"[OK] rewritten: {tag_rewrites}")
    print(f"[OK] summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

