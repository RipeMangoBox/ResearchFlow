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
    lines.append("本报告用于发现 `paperAnalysis` 中 **同义/写法微差** 的 tags，作为后续统一规范的依据。")
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
    lines.append("- `norm` 是为了聚类用的“粗归一化 key”，不等于最终要采用的 canonical tag。")
    lines.append("- 后续统一会明确：task tags（与目录一致）、status tags（`status/...`）、technique tags（统一大小写与分隔符）。")
    lines.append("")

    lines.append("## Canonical tag policy (当前采用的统一规范)")
    lines.append("")
    lines.append("### 1) Task tags（任务/大类）")
    lines.append("")
    lines.append("- canonical task tag = `paperAnalysis/<顶层目录名>`（与目录一致）")
    lines.append("- 例如 `Motion_Editing`、`Human_Object_Interaction` 等")
    lines.append("- task 的连字符/小写变体（如 `motion-editing`）会被统一回目录名写法")
    lines.append("")
    lines.append("### 2) Status tags（状态）")
    lines.append("")
    lines.append("- 统一为小写层级 tag：`status/...`（例如 `status/analyzed`、`status/checked`）")
    lines.append("")
    lines.append("### 3) Technique / keyword tags（技术与关键词）")
    lines.append("")
    lines.append("- 统一规则：全小写；空格/下划线 → `-`；保留 `/` 层级并对每个 segment 做同样归一化")
    lines.append("- 示例：`VQ-VAE` → `vq-vae`，`dataset/HumanML3D` → `dataset/humanml3d`")
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

