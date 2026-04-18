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
               pa.problem_summary, pa.method_summary, pa.evidence_summary,
               pa.core_intuition, pa.full_report_md
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

    # ── 2. Generate paper pages ───────────────────────────────
    for p in papers:
        cat_dir = root / "papers" / (p.category or "Uncategorized")
        venue_dir = cat_dir / (f"{p.venue}_{p.year}" if p.venue and p.year else "Unknown")
        venue_dir.mkdir(parents=True, exist_ok=True)

        safe_title = _sanitize_wikilink(p.title)
        filename = f"{p.title_sanitized or str(p.id)}.md"

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
        }
        if p.paper_link:
            fm["paper_link"] = p.paper_link
        if p.code_url:
            fm["code_url"] = p.code_url

        # Body with wikilinks
        body = [f"# {p.title}\n"]

        # Links section
        links = []
        if p.mechanism_family and p.mechanism_family in mechanism_names:
            links.append(f"- Mechanism: [[{p.mechanism_family}]]")
        if p.baseline_paradigm and p.baseline_paradigm in paradigm_names:
            links.append(f"- Paradigm: [[{p.baseline_paradigm}]]")

        # Lineage wikilinks
        ancestors = lineage_map.get(p.title, [])
        for rel, parent in ancestors:
            links.append(f"- {rel}: [[{_sanitize_wikilink(parent)}]]")
        descendants = reverse_lineage.get(p.title, [])
        for rel, child in descendants:
            links.append(f"- Extended by: [[{_sanitize_wikilink(child)}]]")

        if links:
            body.append("## Links\n")
            body.extend(links)
            body.append("")

        # Analysis content
        if p.full_report_md:
            body.append(p.full_report_md)
        else:
            if p.problem_summary:
                body.append(f"## Part I: 问题与挑战\n\n{p.problem_summary}\n")
            if p.method_summary:
                body.append(f"## Part II: 方法与洞察\n\n{p.method_summary}\n")
            if p.core_intuition:
                body.append(f"### 核心直觉\n\n{p.core_intuition}\n")
            if p.evidence_summary:
                body.append(f"## Part III: 证据与局限\n\n{p.evidence_summary}\n")

        if p.delta_statement:
            body.append(f"\n## Delta\n\n{p.delta_statement}\n")

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

    logger.info(f"Obsidian vault exported to {root}: {stats}")
    return {"vault_path": str(root), **stats}
