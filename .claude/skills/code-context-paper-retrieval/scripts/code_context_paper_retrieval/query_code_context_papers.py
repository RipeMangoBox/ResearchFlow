#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[5]
PAPER_COLLECTION_DIR = REPO_ROOT / "paperCollection"
PAPER_ANALYSIS_DIR = REPO_ROOT / "paperAnalysis"

ANALYSIS_LINK_RE = re.compile(r"\[\[(paperAnalysis/[^\]|]+\.md)(?:\|([^\]]+))?\]\]")
PDF_LINK_RE = re.compile(r"\[\[(paperPDFs/[^\]|]+\.pdf)(?:\|[^\]]+)?\]\]")
KEY_LINE_RE = re.compile(r"^([A-Za-z0-9_\-]+):(?:\s*(.*))?$")
BOUNDARY_RE = re.compile(r"^---\s*$")
WORD_RE = re.compile(r"[A-Za-z0-9_]{3,}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieve code-context-related papers from local KB (paperCollection -> paperAnalysis -> optional paperPDFs)."
    )
    p.add_argument("--mode", choices=["brief", "deep"], default="brief")
    p.add_argument("--query", default="", help="Natural-language coding task or context")
    p.add_argument("--changed-file", action="append", default=[], help="Changed file path (repeatable)")
    p.add_argument("--top-k", type=int, default=12, help="Candidate count for deep mode")
    return p.parse_args()


def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in WORD_RE.findall(text)]
    stop = {
        "with",
        "from",
        "that",
        "this",
        "into",
        "about",
        "using",
        "use",
        "code",
        "file",
        "files",
        "model",
        "models",
        "train",
        "training",
        "infer",
        "inference",
        "python",
        "script",
    }
    return [t for t in toks if t not in stop]


def keywords_from_context(query: str, changed_files: List[str]) -> List[str]:
    bag: List[str] = []
    if query:
        bag.extend(tokenize(query))
    for f in changed_files:
        bag.extend(tokenize(f.replace("/", " ").replace(".", " ")))

    ordered: List[str] = []
    seen = set()
    for t in bag:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered[:30]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def parse_frontmatter(md_text: str) -> Dict[str, object]:
    lines = md_text.splitlines()
    if not lines or not BOUNDARY_RE.match(lines[0]):
        return {}

    end = None
    for i in range(1, len(lines)):
        if BOUNDARY_RE.match(lines[i]):
            end = i
            break
    if end is None:
        return {}

    fm = lines[1:end]
    data: Dict[str, object] = {}
    i = 0
    while i < len(fm):
        raw = fm[i]
        if not raw.strip():
            i += 1
            continue
        m = KEY_LINE_RE.match(raw)
        if not m:
            i += 1
            continue

        key = m.group(1)
        rest = (m.group(2) or "").rstrip()

        if rest == "" and i + 1 < len(fm) and fm[i + 1].lstrip().startswith("- "):
            vals: List[str] = []
            i += 1
            while i < len(fm) and fm[i].lstrip().startswith("- "):
                vals.append(fm[i].lstrip()[2:].strip())
                i += 1
            data[key] = vals
            continue

        if rest in ("|", ">"):
            block: List[str] = []
            i += 1
            while i < len(fm):
                li = fm[i]
                if li.startswith("  "):
                    block.append(li[2:])
                    i += 1
                elif li.strip() == "":
                    block.append("")
                    i += 1
                else:
                    break
            data[key] = "\n".join(block).strip()
            continue

        val = rest.strip().strip('"').strip("'")
        data[key] = val
        i += 1

    return data


def extract_summary_line(md_text: str) -> str:
    for ln in md_text.splitlines():
        s = ln.strip()
        if "**Summary**:" in s:
            return s.split("**Summary**:", 1)[1].strip()
    return ""


def score_collection(keywords: List[str]) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str]]:
    score: Dict[str, int] = defaultdict(int)
    title_hint: Dict[str, str] = {}
    pdf_hint: Dict[str, str] = {}

    if not PAPER_COLLECTION_DIR.exists():
        return score, title_hint, pdf_hint

    collection_files = list(PAPER_COLLECTION_DIR.glob("_AllPapers.md"))
    collection_files += sorted((PAPER_COLLECTION_DIR / "by_task").glob("*.md"))
    collection_files += sorted((PAPER_COLLECTION_DIR / "by_technique").glob("*.md"))

    for fp in collection_files:
        text = read_text(fp)
        if not text:
            continue
        for line in text.splitlines():
            links = ANALYSIS_LINK_RE.findall(line)
            if not links:
                continue

            low = line.lower()
            kw_hits = sum(1 for kw in keywords if kw in low)
            for rel, alias in links:
                base = 1
                if kw_hits:
                    base += kw_hits * 2
                if "by_task" in fp.as_posix():
                    base += 1
                if "by_technique" in fp.as_posix():
                    base += 1
                score[rel] += base
                if alias and rel not in title_hint:
                    title_hint[rel] = alias.strip()
                pdf_match = PDF_LINK_RE.search(line)
                if pdf_match and rel not in pdf_hint:
                    pdf_hint[rel] = pdf_match.group(1)

    return score, title_hint, pdf_hint


def enrich_from_analysis(
    ranked_rel: List[str], keywords: List[str], title_hint: Dict[str, str], pdf_hint: Dict[str, str]
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for rel in ranked_rel:
        ap = REPO_ROOT / rel
        txt = read_text(ap)
        if not txt:
            continue
        fm = parse_frontmatter(txt)
        title = str(fm.get("title") or title_hint.get(rel) or Path(rel).stem)
        venue = str(fm.get("venue") or "Unknown")
        year = str(fm.get("year") or "Unknown")
        core_operator = str(fm.get("core_operator") or "").replace("\n", " ").strip()
        primary_logic = str(fm.get("primary_logic") or "").replace("\n", " ").strip()
        summary = extract_summary_line(txt)
        pdf_ref = str(fm.get("pdf_ref") or pdf_hint.get(rel) or "")

        blob = " ".join([title, venue, year, core_operator, primary_logic, summary]).lower()
        a_score = sum(1 for kw in keywords if kw in blob)
        need_pdf = "Yes" if (a_score <= 1 and (core_operator or primary_logic)) else "No"
        reason = (
            "analysis 信息已足够支撑实现判断"
            if need_pdf == "No"
            else "analysis 证据偏弱，建议快速核对 PDF 方法细节"
        )

        out.append(
            {
                "analysis_rel": rel,
                "title": title,
                "venue_year": f"{venue} {year}".strip(),
                "core_operator": core_operator,
                "primary_logic": primary_logic,
                "summary": summary,
                "pdf_ref": pdf_ref,
                "recommend_pdf": need_pdf,
                "recommend_reason": reason,
                "analysis_score": str(a_score),
            }
        )
    return out


def render_brief(items: List[Dict[str, str]], keywords: List[str], query: str, changed_files: List[str]) -> str:
    chosen = items[:5]
    if len(chosen) > 3:
        chosen = chosen[: max(3, min(5, len(chosen)))]

    lines: List[str] = []
    lines.append("## Code Context Paper Retrieval (brief)")
    lines.append("")
    lines.append("- 检索顺序: paperCollection -> paperAnalysis -> paperPDFs(按需)")
    lines.append(f"- 关键词: {', '.join(keywords) if keywords else 'N/A'}")
    if query:
        lines.append(f"- Query: {query}")
    if changed_files:
        lines.append(f"- Changed files: {', '.join(changed_files)}")
    lines.append("")
    lines.append("### 必看论文 (3-5)")

    for i, it in enumerate(chosen, start=1):
        method = it["core_operator"] or it["primary_logic"] or it["summary"]
        if len(method) > 120:
            method = method[:120] + "..."
        lines.append(f"{i}. **{it['title']}** ({it['venue_year']})")
        lines.append(f"   - analysis: `{it['analysis_rel']}`")
        lines.append(f"   - method cue: {method if method else 'N/A'}")
        lines.append(
            f"   - 建议读 PDF: {it['recommend_pdf']}（{it['recommend_reason']}）"
            + (f"，`{it['pdf_ref']}`" if it["pdf_ref"] else "")
        )

    if not chosen:
        lines.append("- 未检索到足够匹配条目，请补充 query 或 changed-file。")

    return "\n".join(lines)


def render_deep(items: List[Dict[str, str]], keywords: List[str], query: str, changed_files: List[str], top_k: int) -> str:
    chosen = items[: max(5, top_k)]
    lines: List[str] = []
    lines.append("## Code Context Paper Retrieval (deep)")
    lines.append("")
    lines.append("- 检索顺序: paperCollection -> paperAnalysis -> paperPDFs(按需)")
    lines.append(f"- 关键词: {', '.join(keywords) if keywords else 'N/A'}")
    if query:
        lines.append(f"- Query: {query}")
    if changed_files:
        lines.append(f"- Changed files: {', '.join(changed_files)}")
    lines.append("")
    lines.append("### 候选论文与理由")

    for i, it in enumerate(chosen, start=1):
        method = it["core_operator"] or it["primary_logic"] or "N/A"
        if len(method) > 180:
            method = method[:180] + "..."
        lines.append(f"{i}. **{it['title']}** ({it['venue_year']})")
        lines.append(f"   - analysis: `{it['analysis_rel']}`")
        lines.append(f"   - core operator / logic: {method}")
        if it["summary"]:
            sm = it["summary"]
            if len(sm) > 180:
                sm = sm[:180] + "..."
            lines.append(f"   - evidence: {sm}")
        lines.append(
            f"   - PDF decision: {it['recommend_pdf']}（{it['recommend_reason']}）"
            + (f"，`{it['pdf_ref']}`" if it["pdf_ref"] else "")
        )

    lines.append("")
    lines.append("### 对比与落地建议")
    lines.append("- 先看前 3 篇，抽取可复用 operator（训练目标/控制信号/约束方式）。")
    lines.append("- 若当前代码改动涉及训练范式切换或评测协议变化，优先阅读标记 Yes 的 PDF。")
    lines.append("- 若只做工程实现细化，analysis 信息通常足够，可先不读 PDF。")

    if not chosen:
        lines.append("- 未检索到足够匹配条目，请补充 query 或 changed-file。")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    keywords = keywords_from_context(args.query, args.changed_file)

    score_map, title_hint, pdf_hint = score_collection(keywords)
    ranked_rel = [k for k, _ in sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)]

    enriched = enrich_from_analysis(ranked_rel, keywords, title_hint, pdf_hint)
    enriched.sort(key=lambda x: int(x.get("analysis_score", "0")), reverse=True)

    if args.mode == "brief":
        print(render_brief(enriched, keywords, args.query, args.changed_file))
    else:
        print(render_deep(enriched, keywords, args.query, args.changed_file, args.top_k))


if __name__ == "__main__":
    main()
