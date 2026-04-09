#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[5]
PAPER_ANALYSIS_DIR = REPO_ROOT / "paperAnalysis"

KEY_LINE_RE = re.compile(r"^([A-Za-z0-9_\-]+):(?:\s*(.*))?$")
BOUNDARY_RE = re.compile(r"^---\s*$")
WORD_RE = re.compile(r"[A-Za-z0-9_]{3,}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieve code-context-related papers from local KB (paperAnalysis -> optional paperPDFs; paperCollection optional for navigation)."
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


def as_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value or "")


def score_analysis_notes(keywords: List[str]) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str]]:
    score: Dict[str, int] = {}
    title_hint: Dict[str, str] = {}
    pdf_hint: Dict[str, str] = {}

    if not PAPER_ANALYSIS_DIR.exists():
        return score, title_hint, pdf_hint

    for ap in sorted(PAPER_ANALYSIS_DIR.rglob("*.md")):
        rel = ap.relative_to(REPO_ROOT).as_posix()
        txt = read_text(ap)
        if not txt:
            continue

        fm = parse_frontmatter(txt)
        title = as_text(fm.get("title")) or ap.stem
        venue = as_text(fm.get("venue"))
        year = as_text(fm.get("year"))
        tags = as_text(fm.get("tags"))
        core_operator = as_text(fm.get("core_operator")).replace("\n", " ").strip()
        primary_logic = as_text(fm.get("primary_logic")).replace("\n", " ").strip()
        summary = extract_summary_line(txt)
        pdf_ref = as_text(fm.get("pdf_ref"))

        title_hint[rel] = title
        if pdf_ref:
            pdf_hint[rel] = pdf_ref

        if not keywords:
            score[rel] = 0
            continue

        rel_blob = rel.replace("/", " ").lower()
        title_blob = title.lower()
        method_blob = " ".join([tags, core_operator, primary_logic]).lower()
        meta_blob = " ".join([venue, year, summary]).lower()

        total = 0
        for kw in keywords:
            if kw in title_blob:
                total += 4
            if kw in method_blob:
                total += 3
            if kw in meta_blob:
                total += 2
            if kw in rel_blob:
                total += 1

        if total > 0:
            score[rel] = total

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
            "analysis information is sufficient for implementation decisions"
            if need_pdf == "No"
            else "analysis evidence is weak; quick PDF check is recommended"
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
    lines.append("- Retrieval order: paperAnalysis -> paperPDFs (as needed); paperCollection is for statistics/navigation support only")
    lines.append(f"- Keywords: {', '.join(keywords) if keywords else 'N/A'}")
    if query:
        lines.append(f"- Query: {query}")
    if changed_files:
        lines.append(f"- Changed files: {', '.join(changed_files)}")
    lines.append("")
    lines.append("### Must-read papers (3-5)")

    for i, it in enumerate(chosen, start=1):
        method = it["core_operator"] or it["primary_logic"] or it["summary"]
        if len(method) > 120:
            method = method[:120] + "..."
        lines.append(f"{i}. **{it['title']}** ({it['venue_year']})")
        lines.append(f"   - analysis: `{it['analysis_rel']}`")
        lines.append(f"   - method cue: {method if method else 'N/A'}")
        lines.append(
            f"   - Suggested PDF reading: {it['recommend_pdf']} ({it['recommend_reason']})"
            + (f"，`{it['pdf_ref']}`" if it["pdf_ref"] else "")
        )

    if not chosen:
        lines.append("- Not enough matching items found. Please provide additional query or changed-file context.")

    return "\n".join(lines)


def render_deep(items: List[Dict[str, str]], keywords: List[str], query: str, changed_files: List[str], top_k: int) -> str:
    chosen = items[: max(5, top_k)]
    lines: List[str] = []
    lines.append("## Code Context Paper Retrieval (deep)")
    lines.append("")
    lines.append("- Retrieval order: paperAnalysis -> paperPDFs (as needed); paperCollection is for statistics/navigation support only")
    lines.append(f"- Keywords: {', '.join(keywords) if keywords else 'N/A'}")
    if query:
        lines.append(f"- Query: {query}")
    if changed_files:
        lines.append(f"- Changed files: {', '.join(changed_files)}")
    lines.append("")
    lines.append("### Candidate papers and rationale")

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
    lines.append("### Comparison and implementation suggestions")
    lines.append("- Start with the top 3 papers and extract reusable operators (training objectives/control signals/constraint forms).")
    lines.append("- If code changes involve training paradigm shifts or evaluation protocol changes, prioritize PDFs marked Yes.")
    lines.append("- For engineering-level implementation refinements only, analysis notes are often sufficient and PDF reading can be skipped initially.")

    if not chosen:
        lines.append("- Not enough matching items found. Please provide additional query or changed-file context.")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    keywords = keywords_from_context(args.query, args.changed_file)

    score_map, title_hint, pdf_hint = score_analysis_notes(keywords)
    ranked_rel = [k for k, _ in sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)]

    enriched = enrich_from_analysis(ranked_rel, keywords, title_hint, pdf_hint)
    enriched.sort(key=lambda x: int(x.get("analysis_score", "0")), reverse=True)

    if args.mode == "brief":
        print(render_brief(enriched, keywords, args.query, args.changed_file))
    else:
        print(render_deep(enriched, keywords, args.query, args.changed_file, args.top_k))


if __name__ == "__main__":
    main()
