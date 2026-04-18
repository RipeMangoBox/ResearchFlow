"""Export service — async DB → paperAnalysis/ + paperCollection/ sync.

Called after pipeline completion to keep Markdown exports in sync with DB.
Also generates paperCollection index from DB on demand.
"""

import logging
from pathlib import Path
from uuid import UUID

import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings

logger = logging.getLogger(__name__)


def _render_frontmatter(data: dict) -> str:
    yaml_str = yaml.dump(
        data, default_flow_style=False, allow_unicode=True,
        sort_keys=False, width=200,
    )
    return f"---\n{yaml_str}---\n"


async def export_paper_analysis(
    session: AsyncSession,
    paper_id: UUID,
    out_dir: str | None = None,
) -> str | None:
    """Export a single paper's analysis to paperAnalysis/ Markdown.

    Called automatically after L4 analysis completes.
    Returns the output file path or None if no analysis exists.
    """
    root = Path(out_dir or settings.paper_analysis_dir)

    row = (await session.execute(text("""
        SELECT
            p.id, p.title, p.venue, p.year, p.category, p.tags,
            p.core_operator, p.primary_logic, p.claims,
            p.title_sanitized, p.paper_link, p.code_url,
            pa.full_report_md, pa.problem_summary, pa.method_summary,
            pa.evidence_summary, pa.core_intuition,
            dc.delta_statement, dc.baseline_paradigm,
            dc.structurality_score AS dc_struct,
            dc.transferability_score AS dc_transfer,
            dc.key_ideas_ranked, dc.assumptions, dc.failure_modes,
            dc.status AS dc_status
        FROM papers p
        LEFT JOIN paper_analyses pa ON pa.paper_id = p.id
            AND pa.is_current = true AND pa.level = 'l4_deep'
        LEFT JOIN delta_cards dc ON dc.id = p.current_delta_card_id
        WHERE p.id = :pid
    """), {"pid": paper_id})).fetchone()

    if not row:
        return None

    category = row.category or "Uncategorized"
    venue_year = f"{row.venue}_{row.year}" if row.venue and row.year else "Unknown"
    filename = f"{row.title_sanitized or str(row.id)}.md"

    out_path = root / category / venue_year / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build frontmatter
    fm = {
        "title": row.title,
        "venue": row.venue,
        "year": row.year,
        "category": category,
        "tags": list(row.tags) if row.tags else [],
        "core_operator": row.core_operator,
        "primary_logic": row.primary_logic,
    }
    if row.dc_struct is not None:
        fm["structurality_score"] = round(float(row.dc_struct), 3)
    if row.dc_transfer is not None:
        fm["transferability_score"] = round(float(row.dc_transfer), 3)
    if row.baseline_paradigm:
        fm["paradigm"] = row.baseline_paradigm
    if row.dc_status:
        fm["delta_card_status"] = row.dc_status
    if row.paper_link:
        fm["paper_link"] = row.paper_link
    if row.code_url:
        fm["code_url"] = row.code_url

    # Build body
    body_parts = []
    if row.full_report_md:
        body_parts.append(row.full_report_md)
    else:
        if row.problem_summary:
            body_parts.append(f"## Part I: 问题与挑战\n\n{row.problem_summary}\n")
        if row.method_summary:
            body_parts.append(f"## Part II: 方法与洞察\n\n{row.method_summary}\n")
        if row.core_intuition:
            body_parts.append(f"### 核心直觉\n\n{row.core_intuition}\n")
        if row.evidence_summary:
            body_parts.append(f"## Part III: 证据与局限\n\n{row.evidence_summary}\n")

    if row.delta_statement:
        body_parts.append(f"\n## Delta Statement\n\n{row.delta_statement}\n")

    content = _render_frontmatter(fm) + "\n" + "\n".join(body_parts)
    out_path.write_text(content, encoding="utf-8")
    logger.info(f"Exported analysis: {out_path}")
    return str(out_path)


async def export_analysis_log_csv(
    session: AsyncSession,
    out_path: str | None = None,
) -> int:
    """Export analysis_log.csv from DB. Returns row count."""
    import csv
    import io

    target = Path(out_path or settings.paper_analysis_dir) / "analysis_log.csv"
    target.parent.mkdir(parents=True, exist_ok=True)

    rows = (await session.execute(text("""
        SELECT p.title, p.venue, p.year, p.category, p.state,
               p.paper_link, p.code_url, p.title_sanitized
        FROM papers p
        WHERE p.state NOT IN ('archived_or_expired', 'skip')
        ORDER BY p.category, p.venue, p.year
    """))).fetchall()

    with open(target, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "venue", "year", "category", "state", "paper_link", "code_url", "filename"])
        for r in rows:
            writer.writerow([r.title, r.venue, r.year, r.category, r.state, r.paper_link, r.code_url, r.title_sanitized])

    return len(rows)


async def build_collection_index(
    session: AsyncSession,
    out_dir: str | None = None,
) -> dict:
    """Build paperCollection/index.jsonl + navigation pages from DB.

    Returns stats about what was generated.
    """
    import json

    root = Path(out_dir or settings.paper_collection_dir)
    root.mkdir(parents=True, exist_ok=True)

    rows = (await session.execute(text("""
        SELECT p.id, p.title, p.venue, p.year, p.category, p.tags,
               p.mechanism_family, p.structurality_score,
               p.keep_score, p.importance, p.state,
               dc.delta_statement, dc.baseline_paradigm
        FROM papers p
        LEFT JOIN delta_cards dc ON dc.id = p.current_delta_card_id
        WHERE p.state NOT IN ('archived_or_expired', 'skip')
        ORDER BY p.category, p.venue, p.year
    """))).fetchall()

    # Write index.jsonl
    index_path = root / "index.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for r in rows:
            entry = {
                "id": str(r.id),
                "title": r.title,
                "venue": r.venue,
                "year": r.year,
                "category": r.category,
                "tags": list(r.tags) if r.tags else [],
                "mechanism_family": r.mechanism_family,
                "structurality_score": float(r.structurality_score) if r.structurality_score else None,
                "keep_score": float(r.keep_score) if r.keep_score else None,
                "importance": r.importance,
                "state": r.state,
                "delta_statement": r.delta_statement[:200] if r.delta_statement else None,
                "paradigm": r.baseline_paradigm,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Generate by_venue navigation page
    venues: dict[str, list] = {}
    for r in rows:
        key = f"{r.venue}_{r.year}" if r.venue else "Unknown"
        venues.setdefault(key, []).append(r.title)

    venue_page = root / "by_venue.md"
    with open(venue_page, "w", encoding="utf-8") as f:
        f.write("# Papers by Venue\n\n")
        for venue, titles in sorted(venues.items()):
            f.write(f"## {venue} ({len(titles)})\n\n")
            for t in titles:
                f.write(f"- {t}\n")
            f.write("\n")

    # Generate by_category navigation page
    categories: dict[str, list] = {}
    for r in rows:
        categories.setdefault(r.category, []).append(r.title)

    cat_page = root / "by_category.md"
    with open(cat_page, "w", encoding="utf-8") as f:
        f.write("# Papers by Category\n\n")
        for cat, titles in sorted(categories.items()):
            f.write(f"## {cat} ({len(titles)})\n\n")
            for t in titles:
                f.write(f"- {t}\n")
            f.write("\n")

    return {
        "index_entries": len(rows),
        "venues": len(venues),
        "categories": len(categories),
        "files": ["index.jsonl", "by_venue.md", "by_category.md"],
    }


# ── Obsidian Vault Export (Karpathy-style) ───────────────────────

def _sanitize_wikilink(title: str) -> str:
    """Sanitize a title for use as Obsidian wikilink target."""
    return title.replace("|", "-").replace("[", "(").replace("]", ")").replace("#", "")


async def export_obsidian_vault(
    session: AsyncSession,
    out_dir: str | None = None,
) -> dict:
    """Export full knowledge base as Obsidian-ready vault with wikilinks + concept pages.

    Generates:
    - papers/{Category}/{Venue_Year}/{title}.md — per-paper pages with [[wikilinks]]
    - concepts/mechanisms/{name}.md — hub page per mechanism family
    - concepts/bottlenecks/{title}.md — hub page per bottleneck
    - concepts/paradigms/{name}.md — hub page per paradigm
    - _Index.md — vault root with Dataview queries
    - _Graph.md — Obsidian graph view instructions

    All pages use Dataview-compatible YAML frontmatter and [[wikilinks]].
    """
    root = Path(out_dir or settings.paper_analysis_dir).parent / "obsidian-vault"
    root.mkdir(parents=True, exist_ok=True)

    stats = {"papers": 0, "mechanisms": 0, "bottlenecks": 0, "paradigms": 0, "lineage_edges": 0}

    # ── 1. Fetch all data ─────────────────────────────────────
    papers = (await session.execute(text("""
        SELECT p.id, p.title, p.title_sanitized, p.venue, p.year, p.category,
               p.tags, p.core_operator, p.primary_logic, p.abstract,
               p.paper_link, p.code_url, p.mechanism_family,
               p.structurality_score, p.extensionability_score, p.keep_score,
               p.open_code, p.open_data, p.importance, p.state,
               dc.delta_statement, dc.baseline_paradigm,
               dc.structurality_score AS dc_struct,
               dc.key_ideas_ranked, dc.assumptions, dc.failure_modes,
               dc.key_equations, dc.key_figures, dc.same_family_paper_ids,
               dc.changed_slot_ids, dc.unchanged_slot_ids,
               pa.problem_summary, pa.method_summary, pa.evidence_summary,
               pa.core_intuition, pa.full_report_md, pa.changed_slots
        FROM papers p
        LEFT JOIN delta_cards dc ON dc.id = p.current_delta_card_id
        LEFT JOIN paper_analyses pa ON pa.paper_id = p.id
            AND pa.is_current = true AND pa.level = 'l4_deep'
        WHERE p.state NOT IN ('archived_or_expired', 'skip')
        ORDER BY p.category, p.venue, p.year
    """))).fetchall()

    mechanisms = (await session.execute(text(
        "SELECT id, name, domain, description, aliases FROM mechanism_families ORDER BY name"
    ))).fetchall()

    bottlenecks = (await session.execute(text(
        "SELECT id, title, description, domain FROM project_bottlenecks WHERE status = 'active' ORDER BY title"
    ))).fetchall()

    paradigms = (await session.execute(text(
        "SELECT id, name, version, domain, slots FROM paradigm_templates ORDER BY name"
    ))).fetchall()

    lineage = (await session.execute(text("""
        SELECT dcl.relation_type, dcl.confidence, dcl.status,
               child_p.title AS child_title, parent_p.title AS parent_title
        FROM delta_card_lineage dcl
        JOIN delta_cards child_dc ON child_dc.id = dcl.child_delta_card_id
        JOIN papers child_p ON child_p.id = child_dc.paper_id
        JOIN delta_cards parent_dc ON parent_dc.id = dcl.parent_delta_card_id
        JOIN papers parent_p ON parent_p.id = parent_dc.paper_id
    """))).fetchall()

    # Build lookup maps
    paper_title_set = {r.title for r in papers}
    mechanism_names = {r.name for r in mechanisms}
    bottleneck_titles = {r.title for r in bottlenecks}
    paradigm_names = {r.name for r in paradigms}

    # Lineage lookup: child_title → [(relation, parent_title), ...]
    lineage_map: dict[str, list] = {}
    reverse_lineage: dict[str, list] = {}
    for ln in lineage:
        lineage_map.setdefault(ln.child_title, []).append((ln.relation_type, ln.parent_title))
        reverse_lineage.setdefault(ln.parent_title, []).append((ln.relation_type, ln.child_title))
        stats["lineage_edges"] += 1

    # Paper → mechanism mapping
    paper_mechanisms: dict[str, str] = {}
    for p in papers:
        if p.mechanism_family:
            paper_mechanisms[p.title] = p.mechanism_family

    # Build id→title lookup for same_family resolution
    id_to_title = {str(p.id): p.title for p in papers}

    # ── 2. Generate paper pages (structured modification card) ─
    for p in papers:
        cat_dir = root / "papers" / (p.category or "Uncategorized")
        venue_dir = cat_dir / (f"{p.venue}_{p.year}" if p.venue and p.year else "Unknown")
        venue_dir.mkdir(parents=True, exist_ok=True)

        safe_title = _sanitize_wikilink(p.title)
        filename = f"{p.title_sanitized or str(p.id)}.md"

        # Resolve changed/unchanged slots
        changed_slots = list(p.changed_slots) if p.changed_slots else []
        is_structural = p.dc_struct is not None and float(p.dc_struct) >= 0.5

        # Frontmatter (Dataview-compatible)
        fm = {
            "title": p.title,
            "type": "paper",
            "venue": p.venue,
            "year": p.year,
            "category": p.category,
            "tags": list(p.tags) if p.tags else [],
            "importance": p.importance,
            "state": p.state,
            "structurality_score": round(float(p.dc_struct), 3) if p.dc_struct else None,
            "keep_score": round(float(p.keep_score), 3) if p.keep_score else None,
            "open_code": p.open_code,
            "open_data": p.open_data,
            "mechanism_family": p.mechanism_family,
            "paradigm": p.baseline_paradigm,
            "changed_slots": changed_slots,
        }
        if p.paper_link:
            fm["paper_link"] = p.paper_link
        if p.code_url:
            fm["code_url"] = p.code_url

        body = [f"# {p.title}\n"]

        # ── One-line summary (callout) ──
        slot_str = ", ".join(f"`{s}`" for s in changed_slots[:3]) if changed_slots else "unknown slots"
        type_str = "structural" if is_structural else "plugin"
        ancestors = lineage_map.get(p.title, [])
        ancestor_links = ", ".join(f"[[{_sanitize_wikilink(pt)}]]" for _, pt in ancestors[:3])
        if ancestor_links:
            body.append(f"> 基于 {ancestor_links}，改了 {slot_str}；类型：**{type_str}**\n")
        elif p.baseline_paradigm:
            body.append(f"> 基于 `{p.baseline_paradigm}` 范式，改了 {slot_str}；类型：**{type_str}**\n")

        # ── Baseline comparison table ──
        body.append("## Baseline 对照\n")
        body.append("| 项 | 内容 |")
        body.append("|---|---|")
        # Baselines
        if ancestors:
            body.append(f"| 基于 | {', '.join(f'[[{_sanitize_wikilink(pt)}]]' for _, pt in ancestors)} |")
        elif p.baseline_paradigm:
            body.append(f"| 基于 | `{p.baseline_paradigm}` (标准范式) |")
        # Changed slots
        if changed_slots:
            body.append(f"| 改了 | {', '.join(changed_slots)} |")
        # Type
        body.append(f"| 类型 | {type_str} |")
        # Mechanism
        if p.mechanism_family:
            if p.mechanism_family in mechanism_names:
                body.append(f"| 机制 | [[{p.mechanism_family}]] |")
            else:
                body.append(f"| 机制 | {p.mechanism_family} |")
        # Structurality
        if p.dc_struct is not None:
            body.append(f"| 结构性 | {float(p.dc_struct):.2f} |")
        body.append("")

        # ── Key equations ──
        key_eqs = p.key_equations if isinstance(p.key_equations, list) else []
        if key_eqs:
            body.append("## 关键公式\n")
            for eq in key_eqs[:4]:
                if isinstance(eq, dict):
                    latex = eq.get("latex", "")
                    slot = eq.get("slot_affected", "")
                    explanation = eq.get("explanation", "")
                    body.append(f"- $${latex}$$")
                    if slot or explanation:
                        body.append(f"  - {slot}：{explanation}")
            body.append("")

        # ── Key figures ──
        key_figs = p.key_figures if isinstance(p.key_figures, list) else []
        if key_figs:
            body.append("## 关键图表\n")
            for fig in key_figs[:4]:
                if isinstance(fig, dict):
                    ref = fig.get("fig_ref", "")
                    caption = fig.get("caption", "")
                    evidence = fig.get("evidence_for", "")
                    body.append(f"- **{ref}**: {caption}")
                    if evidence:
                        body.append(f"  - 证据：{evidence}")
            body.append("")

        # ── Same-family papers ──
        same_fam_ids = p.same_family_paper_ids if p.same_family_paper_ids else []
        same_fam_titles = [id_to_title.get(str(sid)) for sid in same_fam_ids if str(sid) in id_to_title]
        if same_fam_titles:
            body.append("## 同类型论文\n")
            for t in same_fam_titles[:8]:
                body.append(f"- [[{_sanitize_wikilink(t)}]]")
            body.append("")

        # ── Lineage position ──
        descendants = reverse_lineage.get(p.title, [])
        if ancestors or descendants:
            body.append("## 演化位置\n")
            for rel, parent in ancestors:
                body.append(f"- 上游 ({rel}): [[{_sanitize_wikilink(parent)}]]")
            for rel, child in descendants:
                body.append(f"- 下游 ({rel}): [[{_sanitize_wikilink(child)}]]")
            body.append("")

        # ── Detailed analysis ──
        body.append("## 详细分析\n")
        if p.full_report_md:
            body.append(p.full_report_md)
        else:
            if p.problem_summary:
                body.append(f"### Part I: 问题与挑战\n\n{p.problem_summary}\n")
            if p.method_summary:
                body.append(f"### Part II: 方法与洞察\n\n{p.method_summary}\n")
            if p.core_intuition:
                body.append(f"#### 核心直觉\n\n{p.core_intuition}\n")
            if p.evidence_summary:
                body.append(f"### Part III: 证据与局限\n\n{p.evidence_summary}\n")

        if p.delta_statement:
            body.append(f"\n### Delta Statement\n\n{p.delta_statement}\n")

        content = _render_frontmatter(fm) + "\n".join(body)
        (venue_dir / filename).write_text(content, encoding="utf-8")
        stats["papers"] += 1

    # ── 3. Generate mechanism hub pages ───────────────────────
    mech_dir = root / "concepts" / "mechanisms"
    mech_dir.mkdir(parents=True, exist_ok=True)

    for mf in mechanisms:
        fm = {
            "title": mf.name,
            "type": "mechanism",
            "domain": mf.domain,
            "aliases": list(mf.aliases) if mf.aliases else [],
        }
        linked_papers = [p.title for p in papers if p.mechanism_family == mf.name]
        body = [f"# {mf.name}\n"]
        if mf.description:
            body.append(f"{mf.description}\n")
        if mf.aliases:
            body.append(f"**Aliases:** {', '.join(mf.aliases)}\n")
        if linked_papers:
            body.append(f"## Papers ({len(linked_papers)})\n")
            for t in linked_papers:
                body.append(f"- [[{_sanitize_wikilink(t)}]]")
            body.append("")

        content = _render_frontmatter(fm) + "\n".join(body)
        (mech_dir / f"{mf.name}.md").write_text(content, encoding="utf-8")
        stats["mechanisms"] += 1

    # ── 4. Generate bottleneck hub pages ──────────────────────
    bn_dir = root / "concepts" / "bottlenecks"
    bn_dir.mkdir(parents=True, exist_ok=True)

    # Get paper-bottleneck claims
    claims = (await session.execute(text("""
        SELECT pbc.bottleneck_id, pbc.claim_text, p.title AS paper_title
        FROM paper_bottleneck_claims pbc
        JOIN papers p ON p.id = pbc.paper_id
    """))).fetchall()

    bn_claims: dict[str, list] = {}
    for c in claims:
        bn_claims.setdefault(str(c.bottleneck_id), []).append((c.paper_title, c.claim_text))

    for bn in bottlenecks:
        fm = {
            "title": bn.title,
            "type": "bottleneck",
            "domain": bn.domain,
        }
        body = [f"# {bn.title}\n"]
        if bn.description:
            body.append(f"{bn.description}\n")

        paper_claims = bn_claims.get(str(bn.id), [])
        if paper_claims:
            body.append(f"## Papers addressing this ({len(paper_claims)})\n")
            for title, claim in paper_claims:
                body.append(f"- [[{_sanitize_wikilink(title)}]] — {claim[:100]}")
            body.append("")

        content = _render_frontmatter(fm) + "\n".join(body)
        safe_name = bn.title.replace("/", "-").replace(":", "-")[:80]
        (bn_dir / f"{safe_name}.md").write_text(content, encoding="utf-8")
        stats["bottlenecks"] += 1

    # ── 5. Generate paradigm hub pages ────────────────────────
    para_dir = root / "concepts" / "paradigms"
    para_dir.mkdir(parents=True, exist_ok=True)

    for pt in paradigms:
        fm = {
            "title": pt.name,
            "type": "paradigm",
            "version": pt.version,
            "domain": pt.domain,
        }
        slots = pt.slots if isinstance(pt.slots, dict) else {}
        linked_papers = [p.title for p in papers if p.baseline_paradigm == pt.name]

        body = [f"# {pt.name} ({pt.version})\n"]
        if pt.domain:
            body.append(f"**Domain:** {pt.domain}\n")
        if slots:
            body.append("## Slots\n")
            for name, desc in slots.items():
                body.append(f"- **{name}**: {desc if isinstance(desc, str) else ''}")
            body.append("")
        if linked_papers:
            body.append(f"## Papers ({len(linked_papers)})\n")
            for t in linked_papers:
                body.append(f"- [[{_sanitize_wikilink(t)}]]")
            body.append("")

        content = _render_frontmatter(fm) + "\n".join(body)
        (para_dir / f"{pt.name}.md").write_text(content, encoding="utf-8")
        stats["paradigms"] += 1

    # ── 6. Generate vault index page ──────────────────────────
    index_content = """---
title: ResearchFlow Knowledge Base
type: index
---

# ResearchFlow Knowledge Base

## Quick Stats

```dataview
TABLE length(rows) AS Count
FROM ""
GROUP BY type
```

## Recent Papers

```dataview
TABLE venue, year, structurality_score, mechanism_family
FROM "papers"
SORT year DESC
LIMIT 20
```

## Bottlenecks

```dataview
TABLE domain
FROM "concepts/bottlenecks"
SORT title ASC
```

## Mechanisms

```dataview
TABLE domain, length(file.inlinks) AS "Paper Count"
FROM "concepts/mechanisms"
SORT title ASC
```

## Paradigms

```dataview
TABLE version, domain
FROM "concepts/paradigms"
SORT title ASC
```
"""
    (root / "_Index.md").write_text(index_content, encoding="utf-8")

    # ── 7. Graph view instructions ────────────────────────────
    graph_content = """---
title: Graph View Guide
type: meta
---

# Knowledge Graph

Open Obsidian's **Graph View** (Ctrl/Cmd + G) to see the full knowledge network.

## Filters

- **Papers**: `path:papers`
- **Mechanisms**: `path:concepts/mechanisms`
- **Bottlenecks**: `path:concepts/bottlenecks`
- **Paradigms**: `path:concepts/paradigms`

## Color Groups (Graph Settings → Groups)

| Query | Color | Meaning |
|-------|-------|---------|
| `path:concepts/mechanisms` | Green | Mechanism families |
| `path:concepts/bottlenecks` | Red | Research bottlenecks |
| `path:concepts/paradigms` | Blue | Paradigm templates |
| `tag:#structural` | Orange | Structural changes |

## Tips

- Orphan nodes (no links) = papers not yet analyzed or categorized
- Hub nodes = important mechanisms or bottlenecks many papers reference
- Clusters = research sub-communities
"""
    (root / "_Graph.md").write_text(graph_content, encoding="utf-8")

    # ── 8. Domain Overview entry page ─────────────────────────
    analyzed = [p for p in papers if p.state == "l4_deep"]
    categories = {}
    for p in papers:
        categories.setdefault(p.category, []).append(p)

    overview_body = ["# 方向总览\n"]
    overview_body.append(f"**知识库规模**: {len(papers)} 篇论文, {len(analyzed)} 篇已深度分析\n")
    overview_body.append("## 领域分布\n")
    for cat, cat_papers in sorted(categories.items()):
        overview_body.append(f"- **{cat}**: {len(cat_papers)} 篇")
    overview_body.append("")

    overview_body.append("## 范式框架\n")
    for pt in paradigms:
        slots_dict = pt.slots if isinstance(pt.slots, dict) else {}
        slot_names = ", ".join(slots_dict.keys()) if slots_dict else "N/A"
        linked = sum(1 for p in papers if p.baseline_paradigm == pt.name)
        overview_body.append(f"- **[[{pt.name}]]** ({pt.domain}): {slot_names} — {linked} 篇论文")
    overview_body.append("")

    overview_body.append("## 核心瓶颈\n")
    for bn in bottlenecks[:10]:
        claim_count = len(bn_claims.get(str(bn.id), []))
        safe_bn = bn.title.replace("/", "-").replace(":", "-")[:80]
        overview_body.append(f"- [[{safe_bn}]] — {claim_count} 篇论文声称在解决")
    overview_body.append("")

    overview_body.append("## 建议阅读顺序\n")
    overview_body.append("1. 先看范式定义页，理解标准框架")
    overview_body.append("2. 看 [[_MethodEvolution]] 了解方法演化脉络")
    overview_body.append("3. 看 [[_BottleneckMap]] 了解当前瓶颈分布")
    overview_body.append("4. 按 structurality_score 从高到低阅读论文笔记")
    overview_body.append("")

    overview_body.append("## 按结构性排序的论文\n")
    sorted_papers = sorted(analyzed, key=lambda x: float(x.dc_struct or 0), reverse=True)
    for i, p in enumerate(sorted_papers[:20], 1):
        score = f"{float(p.dc_struct):.2f}" if p.dc_struct else "?"
        body_type = "structural" if p.dc_struct and float(p.dc_struct) >= 0.5 else "plugin"
        overview_body.append(f"{i}. [[{_sanitize_wikilink(p.title)}]] — struct={score}, {body_type}")

    (root / "_DomainOverview.md").write_text(
        _render_frontmatter({"title": "方向总览", "type": "index"}) + "\n".join(overview_body),
        encoding="utf-8",
    )

    # ── 9. Method Evolution entry page ────────────────────────
    evo_body = ["# 方法演化脉络\n"]
    evo_body.append("本页展示论文之间的方法继承和改进关系。\n")

    # Group by paradigm
    paradigm_papers: dict[str, list] = {}
    for p in analyzed:
        key = p.baseline_paradigm or "unknown"
        paradigm_papers.setdefault(key, []).append(p)

    for paradigm, pps in sorted(paradigm_papers.items()):
        evo_body.append(f"## {paradigm}\n")

        # Find roots (no ancestors) and build chains
        roots = [p for p in pps if p.title not in lineage_map]
        non_roots = [p for p in pps if p.title in lineage_map]

        if roots:
            evo_body.append("**基础方法 (Baseline)**:")
            for p in roots:
                dc = float(p.dc_struct) if p.dc_struct else 0
                evo_body.append(f"- [[{_sanitize_wikilink(p.title)}]] (struct={dc:.2f})")
            evo_body.append("")

        if non_roots:
            evo_body.append("**改进方法**:")
            for p in non_roots:
                ancestors = lineage_map.get(p.title, [])
                parent_str = ", ".join(f"[[{_sanitize_wikilink(pt)}]]" for _, pt in ancestors)
                slots = list(p.changed_slots) if p.changed_slots else []
                slot_str = ", ".join(slots[:3]) if slots else "?"
                evo_body.append(f"- [[{_sanitize_wikilink(p.title)}]] ← {parent_str} (改了: {slot_str})")
            evo_body.append("")

        # Show all papers in this paradigm if no lineage data
        if not roots and not non_roots:
            for p in pps:
                slots = list(p.changed_slots) if p.changed_slots else []
                slot_str = ", ".join(slots[:3]) if slots else "?"
                evo_body.append(f"- [[{_sanitize_wikilink(p.title)}]] — 改了: {slot_str}")
            evo_body.append("")

    (root / "_MethodEvolution.md").write_text(
        _render_frontmatter({"title": "方法演化脉络", "type": "index"}) + "\n".join(evo_body),
        encoding="utf-8",
    )

    # ── 10. Bottleneck Map entry page ─────────────────────────
    bn_body = ["# 瓶颈地图\n"]
    bn_body.append("研究方向中的核心瓶颈，以及哪些论文在攻克它们。\n")

    for bn in bottlenecks:
        claim_list = bn_claims.get(str(bn.id), [])
        safe_bn = bn.title.replace("/", "-").replace(":", "-")[:80]
        bn_body.append(f"## [[{safe_bn}]] ({len(claim_list)} 篇)\n")
        if bn.description:
            bn_body.append(f"> {bn.description[:200]}\n")
        for title, claim in claim_list[:5]:
            bn_body.append(f"- [[{_sanitize_wikilink(title)}]] — {claim[:80]}")
        bn_body.append("")

    if not bottlenecks:
        bn_body.append("*暂无瓶颈数据。运行更多论文的 L4 分析后会自动提取。*")

    (root / "_BottleneckMap.md").write_text(
        _render_frontmatter({"title": "瓶颈地图", "type": "index"}) + "\n".join(bn_body),
        encoding="utf-8",
    )

    logger.info(f"Obsidian vault exported to {root}: {stats}")
    return {"vault_path": str(root), **stats}
