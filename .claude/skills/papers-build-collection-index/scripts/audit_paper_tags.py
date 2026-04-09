from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


VAULT_ROOT = Path(__file__).resolve().parents[4]
PAPER_ANALYSIS_DIR = VAULT_ROOT / "paperAnalysis"
REPORT_PATH = VAULT_ROOT / "paperCollection" / "_tag_unification_report.md"
RAW_COUNTS_PATH = VAULT_ROOT / "paperCollection" / "_tag_raw_counts.json"


FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$")
KEY_LINE = re.compile(r"^([A-Za-z0-9_\-]+):(?:\s*(.*))?$")


def parse_frontmatter(md_text: str) -> Dict[str, object]:
    lines = md_text.splitlines()
    if not lines or not FRONTMATTER_BOUNDARY.match(lines[0]):
        return {}

    end_idx: Optional[int] = None
    for i in range(1, len(lines)):
        if FRONTMATTER_BOUNDARY.match(lines[i]):
            end_idx = i
            break
    if end_idx is None:
        return {}

    fm_lines = lines[1:end_idx]
    data: Dict[str, object] = {}

    i = 0
    while i < len(fm_lines):
        raw = fm_lines[i]
        if not raw.strip():
            i += 1
            continue

        m = KEY_LINE.match(raw)
        if not m:
            i += 1
            continue

        key = m.group(1)
        rest = (m.group(2) or "").rstrip()

        # list form: key: then "- item"
        if rest == "" and i + 1 < len(fm_lines) and fm_lines[i + 1].lstrip().startswith("- "):
            items: List[str] = []
            i += 1
            while i < len(fm_lines) and fm_lines[i].lstrip().startswith("- "):
                items.append(fm_lines[i].lstrip()[2:].strip())
                i += 1
            data[key] = items
            continue

        # multi-line scalar: key: | or >
        if rest in ("|", ">"):
            block: List[str] = []
            i += 1
            while i < len(fm_lines):
                li = fm_lines[i]
                if li.startswith("  "):
                    block.append(li[2:])
                    i += 1
                elif li.strip() == "":
                    block.append("")
                    i += 1
                else:
                    break
            data[key] = "\n".join(block).rstrip("\n")
            continue

        val = rest.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        data[key] = val
        i += 1

    return data


def iter_analysis_mds() -> Iterable[Path]:
    for p in PAPER_ANALYSIS_DIR.rglob("*.md"):
        # skip any nested tooling caches
        if "__pycache__" in p.parts:
            continue
        yield p


def norm_key(tag: str) -> str:
    """
    Aggressive normalization for grouping near-duplicates:
    - lower
    - unify separators: space/_/dash -> dash
    - normalize unicode dashes
    - normalize each slash segment
    """
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


@dataclass(frozen=True)
class Group:
    norm: str
    total: int
    variants: List[Tuple[str, int]]


def build_report(raw_counts: Counter, groups: List[Group], generated: str) -> str:
    lines: List[str] = []
    lines.append("---")
    lines.append("type: tag-audit")
    lines.append("tags:")
    lines.append("  - paperCollection")
    lines.append("  - maintenance/tags")
    lines.append(f"generated: {generated}")
    lines.append("---")
    lines.append("")
    lines.append("# Tag unification report (paperAnalysis → paperCollection)")
    lines.append("")
    lines.append("This report identifies **synonymous or near-variant** tags in `paperAnalysis` as a basis for later unification.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- raw unique tags: {len(raw_counts)}")
    lines.append(f"- near-duplicate groups (variants>1): {len(groups)}")
    lines.append("")

    lines.append("## Top near-duplicate groups (by total frequency)")
    lines.append("")
    for g in groups[:80]:
        variants_str = ", ".join([f"`{v}` ({c})" for v, c in g.variants[:12]])
        lines.append(f"- **{g.norm}** (total {g.total}): {variants_str}")
    lines.append("")

    lines.append("## Raw tag frequency (top 120)")
    lines.append("")
    for tag, c in raw_counts.most_common(120):
        lines.append(f"- `{tag}`: {c}")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `norm` is a coarse normalization key for clustering, not necessarily the final canonical tag.")
    lines.append("- Final unification should explicitly separate: task tags (directory-aligned), status tags (`status/...`), and technique tags (normalized case/separators).")
    lines.append("")

    lines.append("## Canonical tag policy (current standard)")
    lines.append("")
    lines.append("### 1) Task tags (task/domain categories)")
    lines.append("")
    lines.append("- Canonical task tag = `paperAnalysis/<top-level-directory-name>` (must match the directory name).")
    lines.append("- Example: `Motion_Editing`, `Human_Object_Interaction`.")
    lines.append("- Hyphenated/lowercase variants (for example `motion-editing`) are normalized back to the directory form.")
    lines.append("")
    lines.append("### 2) Status tags")
    lines.append("")
    lines.append("- Use lowercase hierarchical tags: `status/...` (for example `status/analyzed`, `status/checked`).")
    lines.append("")
    lines.append("### 3) Technique / keyword tags")
    lines.append("")
    lines.append("- Standard rule: lowercase; spaces/underscores -> `-`; keep `/` hierarchy and normalize each segment the same way.")
    lines.append("- Example: `VQ-VAE` -> `vq-vae`, `dataset/HumanML3D` -> `dataset/humanml3d`.")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    raw_counts: Counter = Counter()
    by_norm: Dict[str, Counter] = defaultdict(Counter)

    for md in iter_analysis_mds():
        text = md.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        tags = fm.get("tags")
        if not isinstance(tags, list):
            continue
        for t in tags:
            tag = str(t).strip()
            if not tag:
                continue
            raw_counts[tag] += 1
            by_norm[norm_key(tag)][tag] += 1

    groups: List[Group] = []
    for nk, variants in by_norm.items():
        if len(variants) <= 1:
            continue
        total = sum(variants.values())
        var_list = sorted(variants.items(), key=lambda kv: (-kv[1], kv[0].lower()))
        groups.append(Group(norm=nk, total=total, variants=var_list))

    groups.sort(key=lambda g: (-g.total, g.norm))
    generated = datetime.now().strftime("%Y-%m-%dT%H:%M")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(raw_counts, groups, generated), encoding="utf-8")
    RAW_COUNTS_PATH.write_text(json.dumps(raw_counts, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] scanned: {PAPER_ANALYSIS_DIR}")
    print(f"[OK] raw unique tags: {len(raw_counts)}")
    print(f"[OK] near-duplicate groups: {len(groups)}")
    print(f"[OK] report: {REPORT_PATH}")
    print(f"[OK] raw counts: {RAW_COUNTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
