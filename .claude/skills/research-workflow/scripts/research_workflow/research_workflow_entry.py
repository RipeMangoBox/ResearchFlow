#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[5]
PAPER_ANALYSIS = REPO_ROOT / "paperAnalysis"
PAPER_COLLECTION = REPO_ROOT / "paperCollection"
PAPER_PDFS = REPO_ROOT / "paperPDFs"

STAGES = ["collect", "download", "analyze", "build", "query", "ideate"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified research workflow entry (advisor): detect/choose stage and print suggested next commands."
    )
    p.add_argument("--stage", choices=STAGES + ["auto"], default="auto")
    p.add_argument("--log-file", default="", help="Optional triage/log file under paperAnalysis (e.g. ICLR_2026.txt)")
    p.add_argument("--mode", choices=["brief", "deep"], default="brief", help="Query mode hint")
    return p.parse_args()


def resolve_log_file(log_file: str) -> Path | None:
    if not log_file:
        return None
    p = Path(log_file)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p if p.exists() else None


def count_wait_entries(log_path: Path) -> int:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    c = 0
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        parts = [x.strip() for x in line.split("|")]
        if parts and parts[0].lower() == "wait":
            c += 1
    return c


def discover_logs() -> List[Path]:
    if not PAPER_ANALYSIS.exists():
        return []
    logs: List[Path] = []
    for p in PAPER_ANALYSIS.glob("*.txt"):
        if p.name.startswith("missing") or p.name.startswith("paper_analysis_check_task"):
            continue
        logs.append(p)
    return sorted(logs, key=lambda x: x.name.lower())


def detect_stage(preferred_log: Path | None) -> str:
    logs = discover_logs()
    target_logs = [preferred_log] if preferred_log else logs
    target_logs = [p for p in target_logs if p and p.exists()]

    if not target_logs:
        return "collect"

    total_wait = sum(count_wait_entries(p) for p in target_logs)
    if total_wait > 0:
        return "download"

    # no wait entries; if collection index missing, build next
    if not (PAPER_COLLECTION / "README.md").exists() or not (PAPER_COLLECTION / "_AllPapers.md").exists():
        return "build"

    # index exists; query is the natural default
    return "query"


def stage_spec(stage: str, mode: str, log_hint: str) -> Dict[str, object]:
    if stage == "collect":
        return {
            "inputs": "URLs 或 GitHub repo URL + venue/year + include/exclude",
            "outputs": "paperAnalysis/*.txt 或 paperAnalysis/github_awesome_*.xlsx",
            "commands": [
                "使用 /papers-collect-from-web 或 /papers-collect-from-github-awesome",
            ],
            "next": "download",
        }
    if stage == "download":
        log = log_hint or "paperAnalysis/ICLR_2026.txt"
        return {
            "inputs": "triage/log 文件（含 Wait 条目）",
            "outputs": "paperPDFs/** + log 状态更新",
            "commands": [
                "使用 /papers-download-from-list",
                f'或执行: python3 ".claude/skills/papers-download-from-list/scripts/paper_download_tools/download_wait_papers.py" --log "{log}" --out-root "paperPDFs"',
            ],
            "next": "analyze",
        }
    if stage == "analyze":
        return {
            "inputs": "PDF 路径或 Wait 队列",
            "outputs": "paperAnalysis/**/*.md",
            "commands": [
                "使用 /papers-analyze-pdf",
                "注意: analyze 结束后仅提示下一步 build，不自动 build",
            ],
            "next": "build",
        }
    if stage == "build":
        return {
            "inputs": "paperAnalysis 已有结构化 md",
            "outputs": "paperCollection/README.md, _AllPapers.md, by_task/, by_technique/, by_venue/",
            "commands": [
                "使用 /papers-build-collection-index",
                '或执行: python3 "scripts/build_paper_collection.py"',
            ],
            "next": "query",
        }
    if stage == "query":
        return {
            "inputs": "任务描述/关键词（可选 changed files）",
            "outputs": f"检索结果（{mode}）+ 论文路径/analysis 路径/PDF 建议",
            "commands": [
                "使用 /papers-query-knowledge-base",
                f"或使用 /code-context-paper-retrieval（mode={mode}）",
            ],
            "next": "ideate",
        }
    return {
        "inputs": "研究问题",
        "outputs": "paperIDEAs/YYYY-MM-DD_<topic>.md",
        "commands": [
            "使用 /research-brainstorm-from-kb",
        ],
        "next": "(end)",
    }


def render(stage: str, mode: str, log_hint: str) -> str:
    spec = stage_spec(stage, mode, log_hint)
    lines: List[str] = []
    lines.append("## Research Workflow Entry")
    lines.append("")
    lines.append(f"- 当前阶段: {stage}")
    lines.append(f"- 输入要求: {spec['inputs']}")
    lines.append(f"- 产出路径: {spec['outputs']}")
    lines.append("")
    lines.append("### 推荐执行")
    for i, cmd in enumerate(spec["commands"], start=1):
        lines.append(f"{i}. {cmd}")
    lines.append("")
    lines.append(f"- 下一阶段建议: {spec['next']}")
    lines.append("- 状态集合: collect / download / analyze / build / query / ideate")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    log_path = resolve_log_file(args.log_file)
    log_hint = args.log_file or (log_path.as_posix() if log_path else "")

    stage = args.stage
    if stage == "auto":
        stage = detect_stage(log_path)

    print(render(stage, args.mode, log_hint))


if __name__ == "__main__":
    main()
