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


# ── Obsidian Vault Export v4.0 ────────────────────────────────────
#
# 5 note types: Paper (P__) / Concept (C__) / Bottleneck (B__) /
#               Lineage (L__) / Overview (00_Home)
#
# Paper Note wikilink budget: 6-8 in body.
# Paradigm / Domain Overview are NOT linked — only in frontmatter.
# ─────────────────────────────────────────────────────────────────


def _safe_name(raw: str, max_len: int = 80) -> str:
    """Turn an arbitrary string into a filesystem + wikilink-safe slug."""
    return (
        raw.replace(" ", "_").replace("/", "-").replace(":", "-")
        .replace("|", "-").replace("[", "(").replace("]", ")")
        .replace("#", "")[:max_len]
    )


def _paper_level(p) -> str:
    """Assign A/B/C/D level from ring or structurality_score.

    Priority: ring field → dc.structurality_score → paper.structurality_score.
    """
    ring = getattr(p, "ring", None)
    if ring == "baseline":
        return "A"
    if ring == "structural":
        return "B"
    if ring == "plugin":
        return "C"
    # Try delta_card score first, fall back to paper-level score
    s = None
    if getattr(p, "dc_struct", None) is not None:
        s = float(p.dc_struct)
    elif getattr(p, "structurality_score", None) is not None:
        s = float(p.structurality_score)
    if s is not None:
        if s >= 0.7:
            return "A"
        if s >= 0.5:
            return "B"
        if s >= 0.3:
            return "C"
    return "D"


_LEVEL_DIRS = {
    "A": "A__Baselines",
    "B": "B__Structural",
    "C": "C__Plugins",
    "D": "D__Peripheral",
}


async def export_obsidian_vault(
    session: AsyncSession,
    out_dir: str | None = None,
) -> dict:
    """Export knowledge base as Obsidian vault — v4.0 redesign.

    Only 5 note types:
        P__ Paper / C__ Concept / B__ Bottleneck / L__ Lineage / Overview
    Directory layout:
        00_Home / 10_Lineages / 20_Concepts / 30_Bottlenecks /
        40_Papers / 80_Assets / 90_Views
    Paper body wikilinks capped at 6-8; no links to Domain Overview or Paradigm.
    """
    import shutil

    root = Path(out_dir or settings.paper_analysis_dir).parent / "obsidian-vault"
    if root.exists():
        # Clear contents without removing root (may be a Docker mount point)
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)

    stats = {"papers": 0, "concepts": 0, "bottlenecks": 0, "lineages": 0}

    # ── 1. Fetch all data ─────────────────────────────────────────
    papers = (await session.execute(text("""
        SELECT p.id, p.title, p.title_sanitized, p.venue, p.year, p.category,
               p.tags, p.core_operator, p.primary_logic, p.abstract,
               p.paper_link, p.code_url, p.mechanism_family, p.ring,
               p.structurality_score, p.keep_score,
               p.open_code, p.open_data, p.importance, p.state,
               dc.delta_statement, dc.baseline_paradigm,
               dc.structurality_score AS dc_struct,
               dc.transferability_score AS dc_transfer,
               dc.key_ideas_ranked, dc.assumptions, dc.failure_modes,
               dc.key_equations, dc.key_figures, dc.same_family_paper_ids,
               dc.changed_slot_ids, dc.unchanged_slot_ids,
               dc.baseline_paper_ids,
               pa.problem_summary, pa.method_summary, pa.evidence_summary,
               pa.core_intuition, pa.full_report_md, pa.changed_slots,
               pa.extracted_figure_images
        FROM papers p
        LEFT JOIN delta_cards dc ON dc.id = COALESCE(
            p.current_delta_card_id,
            (SELECT id FROM delta_cards dc2
             WHERE dc2.paper_id = p.id AND dc2.status != 'deprecated'
             ORDER BY dc2.created_at DESC LIMIT 1)
        )
        LEFT JOIN paper_analyses pa ON pa.paper_id = p.id
            AND pa.is_current = true AND pa.level = 'l4_deep'
        WHERE p.state NOT IN ('archived_or_expired', 'skip')
        ORDER BY p.category, p.venue, p.year
    """))).fetchall()

    mechanisms = (await session.execute(text(
        "SELECT id, name, domain, description, aliases "
        "FROM mechanism_families ORDER BY name"
    ))).fetchall()

    canonical_ideas = (await session.execute(text("""
        SELECT ci.id, ci.title, ci.description, ci.domain,
               ci.mechanism_family_id, ci.aliases, ci.tags,
               ci.contribution_count
        FROM canonical_ideas ci
        WHERE ci.status IN ('candidate', 'established')
        ORDER BY ci.title
    """))).fetchall()

    paper_idea_links = (await session.execute(text(
        "SELECT canonical_idea_id, paper_id, contribution_type "
        "FROM contribution_to_canonical_idea"
    ))).fetchall()

    bottlenecks = (await session.execute(text(
        "SELECT id, title, description, domain, symptom_query, latent_need "
        "FROM project_bottlenecks WHERE status = 'active' ORDER BY title"
    ))).fetchall()

    claims = (await session.execute(text("""
        SELECT pbc.bottleneck_id, pbc.claim_text, pbc.is_primary,
               p.title AS paper_title, p.title_sanitized
        FROM paper_bottleneck_claims pbc
        JOIN papers p ON p.id = pbc.paper_id
    """))).fetchall()

    lineage_rows = (await session.execute(text("""
        SELECT dcl.relation_type, dcl.confidence,
               child_p.title AS child_title,
               child_p.title_sanitized AS child_sanitized,
               child_p.mechanism_family AS child_mech,
               parent_p.title AS parent_title,
               parent_p.title_sanitized AS parent_sanitized
        FROM delta_card_lineage dcl
        JOIN delta_cards child_dc ON child_dc.id = dcl.child_delta_card_id
        JOIN papers child_p ON child_p.id = child_dc.paper_id
        JOIN delta_cards parent_dc ON parent_dc.id = dcl.parent_delta_card_id
        JOIN papers parent_p ON parent_p.id = parent_dc.paper_id
        WHERE dcl.status != 'rejected'
    """))).fetchall()

    # ── 2. Build lookup maps ──────────────────────────────────────
    id_to_san: dict[str, str] = {str(p.id): p.title_sanitized for p in papers}
    san_to_title: dict[str, str] = {p.title_sanitized: p.title for p in papers}
    title_to_san: dict[str, str] = {p.title: p.title_sanitized for p in papers}

    # Lineage maps  (keyed by title_sanitized)
    lineage_map: dict[str, list[tuple[str, str]]] = {}   # child_san → [(rel, parent_san)]
    reverse_lineage: dict[str, list[tuple[str, str]]] = {}  # parent_san → [(rel, child_san)]
    for ln in lineage_rows:
        lineage_map.setdefault(ln.child_sanitized, []).append(
            (ln.relation_type, ln.parent_sanitized))
        reverse_lineage.setdefault(ln.parent_sanitized, []).append(
            (ln.relation_type, ln.child_sanitized))

    # Mechanism → papers
    mech_papers: dict[str, list] = {}
    for p in papers:
        if p.mechanism_family:
            mech_papers.setdefault(p.mechanism_family, []).append(p)

    mech_id_to_name = {str(m.id): m.name for m in mechanisms}

    # Canonical‑idea ↔ paper mapping
    idea_to_paper_sans: dict[str, list[str]] = {}
    for link in paper_idea_links:
        san = id_to_san.get(str(link.paper_id))
        if san:
            idea_to_paper_sans.setdefault(str(link.canonical_idea_id), []).append(san)

    # Bottleneck helpers
    bn_safe_name: dict[str, str] = {}  # bn.title → safe slug
    for bn in bottlenecks:
        bn_safe_name[bn.title] = _safe_name(bn.title, 60)

    bn_claims_map: dict[str, list[tuple[str, str, str, bool]]] = {}
    for c in claims:
        bn_claims_map.setdefault(str(c.bottleneck_id), []).append(
            (c.paper_title, c.title_sanitized, c.claim_text,
             getattr(c, "is_primary", False)))

    # Paper → primary bottleneck slug
    paper_to_bn: dict[str, str] = {}  # san → bn slug
    for bn in bottlenecks:
        for _title, _san, _claim, _primary in bn_claims_map.get(str(bn.id), []):
            if _primary and _san:
                paper_to_bn.setdefault(_san, bn_safe_name[bn.title])

    # ── Build Concept list (merge Mechanism + CanonicalIdea) ──────
    concepts: list[dict] = []
    used_idea_ids: set[str] = set()

    for mf in mechanisms:
        slug = _safe_name(mf.name)
        linked_ideas = [ci for ci in canonical_ideas
                        if ci.mechanism_family_id
                        and str(ci.mechanism_family_id) == str(mf.id)]
        used_idea_ids.update(str(ci.id) for ci in linked_ideas)

        concepts.append({
            "slug": slug,
            "display": mf.name,
            "desc": mf.description,
            "domain": mf.domain,
            "aliases": list(mf.aliases) if mf.aliases else [],
            "papers": mech_papers.get(mf.name, []),
            "ideas": linked_ideas,
        })

    # Standalone canonical ideas (not linked to any mechanism)
    for ci in canonical_ideas:
        if str(ci.id) in used_idea_ids:
            continue
        slug = _safe_name(ci.title)
        idea_sans = idea_to_paper_sans.get(str(ci.id), [])
        idea_papers = [p for p in papers if p.title_sanitized in idea_sans]
        concepts.append({
            "slug": slug,
            "display": ci.title,
            "desc": ci.description,
            "domain": ci.domain,
            "aliases": list(ci.aliases) if ci.aliases else [],
            "papers": idea_papers,
            "ideas": [ci],
        })

    # Paper → concept slug
    paper_to_concept: dict[str, str] = {}  # san → concept slug
    for c in concepts:
        for p in c["papers"]:
            paper_to_concept.setdefault(p.title_sanitized, c["slug"])

    # ── Build Lineage groups (by mechanism family) ────────────────
    lineage_groups: dict[str, dict] = {}  # group_key → {papers, edges}
    for p in papers:
        san = p.title_sanitized
        if san in lineage_map or san in reverse_lineage:
            group = p.mechanism_family or "Mixed"
            grp = lineage_groups.setdefault(group, {"papers": set(), "edges": []})
            grp["papers"].add(san)
            for rel, parent_san in lineage_map.get(san, []):
                grp["papers"].add(parent_san)
                grp["edges"].append((parent_san, san, rel))

    lineage_slug: dict[str, str] = {}  # group_key → safe slug
    for gk in lineage_groups:
        lineage_slug[gk] = _safe_name(gk)

    paper_to_lineage: dict[str, str] = {}  # san → lineage slug
    for gk, gd in lineage_groups.items():
        for san in gd["papers"]:
            paper_to_lineage.setdefault(san, lineage_slug[gk])

    # ── 3. Generate Paper Notes (40_Papers/) ──────────────────────
    for p in papers:
        level = _paper_level(p)
        paper_dir = root / "40_Papers" / _LEVEL_DIRS[level]
        paper_dir.mkdir(parents=True, exist_ok=True)

        san = p.title_sanitized or str(p.id)
        filename = f"P__{san}.md"
        changed_slots = list(p.changed_slots) if p.changed_slots else []

        # Resolve wikilink targets  (budget: 6-8)
        ancestors = lineage_map.get(san, [])
        baseline_links = [f"[[P__{psan}]]" for _, psan in ancestors[:2]]

        same_fam_ids = p.same_family_paper_ids or []
        same_fam = [id_to_san[str(s)] for s in same_fam_ids
                     if str(s) in id_to_san][:2]

        concept_slug = paper_to_concept.get(san)
        bn_slug = paper_to_bn.get(san)
        lin_slug = paper_to_lineage.get(san)

        concept_link = f"[[C__{concept_slug}]]" if concept_slug else None
        bn_link = f"[[B__{bn_slug}]]" if bn_slug else None
        lin_link = f"[[L__{lin_slug}]]" if lin_slug else None

        # ── Frontmatter ──
        fm: dict = {
            "title": p.title,
            "type": "paper",
            "paper_id": f"P__{san}",
            "aliases": [san],
            "year": p.year,
            "venue": p.venue,
            "paper_level": level,
            "frame": p.baseline_paradigm,          # property only, never wikilink
            "changed_slots": changed_slots,
            "structurality_score": round(float(p.dc_struct), 3) if p.dc_struct else None,
            "keep_score": round(float(p.keep_score), 3) if p.keep_score else None,
            "open_code": p.open_code,
            "baseline_papers": baseline_links or None,
            "concepts": [concept_link] if concept_link else [],
            "bottleneck": [bn_link] if bn_link else [],
            "lineage": [lin_link] if lin_link else [],
            "same_family_papers": [f"[[P__{s}]]" for s in same_fam] or [],
        }
        fm = {k: v for k, v in fm.items() if v is not None}
        if p.paper_link:
            fm["paper_link"] = p.paper_link
        if p.code_url:
            fm["code_url"] = p.code_url

        # ── Body ──
        body: list[str] = [f"# {p.title}\n"]

        # 一眼看懂 callout
        slot_str = ", ".join(f"`{s}`" for s in changed_slots[:3]) if changed_slots else "?"
        base_str = (", ".join(baseline_links[:2])
                    if baseline_links
                    else (f"`{p.baseline_paradigm}`" if p.baseline_paradigm else "?"))
        callout = [f"> 基于 {base_str}，改了 {slot_str}"]
        if concept_link:
            callout.append(f"> 属于 {concept_link}")
        if bn_link:
            callout.append(f"> 目标是缓解 {bn_link}")
        body.append(",\n".join(callout) + "\n")

        # 相对 baseline 改了什么
        body.append("## 相对 baseline 改了什么\n")
        if ancestors and changed_slots:
            body.append("| 相比 | 改动 slot | 收益 | 代价 |")
            body.append("|------|---------|------|------|")
            parent_ref = f"[[P__{ancestors[0][1]}]]"
            for slot in changed_slots[:4]:
                body.append(f"| {parent_ref} | {slot} | — | — |")
        elif p.delta_statement:
            body.append(f"> {p.delta_statement[:300]}\n")
        body.append("")

        # 关键公式
        key_eqs = p.key_equations if isinstance(p.key_equations, list) else []
        if key_eqs:
            body.append("## 关键公式\n")
            for eq in key_eqs[:4]:
                if isinstance(eq, dict):
                    body.append(f"- $${eq.get('latex', '')}$$")
                    slot = eq.get("slot_affected", "")
                    expl = eq.get("explanation", "")
                    if slot or expl:
                        body.append(f"  - {slot}：{expl}")
            body.append("")

        # 关键图表
        key_figs = p.key_figures if isinstance(p.key_figures, list) else []
        if key_figs:
            body.append("## 关键图表\n")
            for fig in key_figs[:4]:
                if isinstance(fig, dict):
                    body.append(f"- **{fig.get('fig_ref', '')}**: {fig.get('caption', '')}")
                    ev = fig.get("evidence_for", "")
                    if ev:
                        body.append(f"  - 证据：{ev}")
            body.append("")

        # 同类型工作 (max 2 links)
        if same_fam:
            body.append("## 同类型工作\n")
            for s in same_fam[:2]:
                body.append(f"- [[P__{s}]]")
            body.append("")

        # 在主线中的位置 (1 lineage link + immediate neighbors)
        if lin_link:
            body.append("## 在主线中的位置\n")
            body.append(f"→ 见 {lin_link}\n")
            for rel, psan in ancestors[:2]:
                body.append(f"- 上游 ({rel}): [[P__{psan}]]")
            for rel, csan in reverse_lineage.get(san, [])[:2]:
                body.append(f"- 下游 ({rel}): [[P__{csan}]]")
            body.append("")

        # 阅读建议
        body.append("## 阅读建议\n")
        advice = {
            "A": "> **必读 baseline**。先理解此论文建立的标准框架，再看后续改进。\n",
            "B": "> **结构性改进**。建议先读 baseline，再看本文如何修改核心 slot。\n",
            "C": "> **插件型改进**。可快速浏览，重点看改了哪个 slot 和实验对比。\n",
            "D": "> **外围参考**。按需阅读。\n",
        }
        body.append(advice[level])

        # 详细分析
        body.append("## 详细分析\n")
        if p.full_report_md:
            body.append(p.full_report_md)
        else:
            if p.problem_summary:
                body.append(f"### 问题与挑战\n\n{p.problem_summary}\n")
            if p.method_summary:
                body.append(f"### 方法与洞察\n\n{p.method_summary}\n")
            if p.core_intuition:
                body.append(f"#### 核心直觉\n\n{p.core_intuition}\n")
            if p.evidence_summary:
                body.append(f"### 证据与局限\n\n{p.evidence_summary}\n")
        if p.delta_statement:
            body.append(f"\n### Delta Statement\n\n{p.delta_statement}\n")

        content = _render_frontmatter(fm) + "\n".join(body)
        (paper_dir / filename).write_text(content, encoding="utf-8")

        # Export figure images to 80_Assets/
        fig_images = p.extracted_figure_images if isinstance(p.extracted_figure_images, list) else []
        if fig_images:
            fig_dir = root / "80_Assets" / "figures" / san
            fig_dir.mkdir(parents=True, exist_ok=True)
            from backend.services.object_storage import get_storage
            storage = get_storage()
            for fig_rec in fig_images[:6]:
                obj_key = fig_rec.get("object_key", "")
                fig_num = fig_rec.get("figure_num", 0)
                ext = obj_key.rsplit(".", 1)[-1] if "." in obj_key else "png"
                try:
                    img_data = await storage.get(obj_key)
                    if img_data:
                        (fig_dir / f"fig_{fig_num}.{ext}").write_bytes(img_data)
                except Exception:
                    pass

        stats["papers"] += 1

    # ── 4. Generate Concept Notes (20_Concepts/) ─────────────────
    concept_dir = root / "20_Concepts"
    concept_dir.mkdir(parents=True, exist_ok=True)

    for c in concepts:
        filename = f"C__{c['slug']}.md"
        fm = {
            "title": c["display"],
            "type": "concept",
            "concept_id": f"C__{c['slug']}",
            "aliases": c["aliases"],
            "domain": c["domain"],
        }

        body = [f"# {c['display']}\n"]

        if c["desc"]:
            body.append(f"{c['desc']}\n")

        # Sub-concepts from canonical ideas
        for ci in c["ideas"]:
            if ci.description and ci.description != c["desc"]:
                body.append(f"### {ci.title}\n\n{ci.description}\n")

        # Which slots does this concept change?
        all_slots: set[str] = set()
        for p in c["papers"]:
            if p.changed_slots:
                all_slots.update(p.changed_slots)
        if all_slots:
            body.append("## 改动的 Slot\n")
            body.append(", ".join(f"`{s}`" for s in sorted(all_slots)) + "\n")

        # Representative papers comparison table
        if c["papers"]:
            body.append(f"## 代表论文 ({len(c['papers'])})\n")
            body.append("| 论文 | Year | Venue | 结构性 | 改动 |")
            body.append("|------|------|-------|--------|------|")
            sorted_cp = sorted(c["papers"],
                               key=lambda x: float(x.dc_struct or 0), reverse=True)
            for p in sorted_cp[:10]:
                psan = p.title_sanitized or str(p.id)
                score = f"{float(p.dc_struct):.2f}" if p.dc_struct else "?"
                slots = ", ".join(list(p.changed_slots)[:2]) if p.changed_slots else "—"
                body.append(f"| [[P__{psan}]] | {p.year} | {p.venue} | {score} | {slots} |")
            body.append("")

        content = _render_frontmatter(fm) + "\n".join(body)
        (concept_dir / filename).write_text(content, encoding="utf-8")
        stats["concepts"] += 1

    # ── 5. Generate Bottleneck Notes (30_Bottlenecks/) ────────────
    bn_dir = root / "30_Bottlenecks"
    bn_dir.mkdir(parents=True, exist_ok=True)

    for bn in bottlenecks:
        slug = bn_safe_name[bn.title]
        filename = f"B__{slug}.md"
        paper_claims = bn_claims_map.get(str(bn.id), [])

        # Classify solutions: structural vs plugin
        structural_sols: list[tuple[str, str]] = []
        plugin_sols: list[tuple[str, str]] = []
        for _title, _san, _claim, _ in paper_claims:
            p_obj = next((p for p in papers if p.title == _title), None)
            if p_obj and _paper_level(p_obj) in ("A", "B"):
                structural_sols.append((_san, _claim))
            else:
                plugin_sols.append((_san, _claim))

        fm = {
            "title": bn.title,
            "type": "bottleneck",
            "bottleneck_id": f"B__{slug}",
            "domain": bn.domain,
            "paper_count": len(paper_claims),
        }
        body = [f"# {bn.title}\n"]

        if bn.symptom_query or bn.description:
            body.append("## 症状\n")
            body.append(f"{bn.symptom_query or bn.description}\n")

        if bn.latent_need:
            body.append("## 根因\n")
            body.append(f"{bn.latent_need}\n")

        if structural_sols:
            body.append("## 结构性解法\n")
            for _san, _claim in structural_sols[:5]:
                body.append(f"- [[P__{_san}]] — {_claim[:100]}")
            body.append("")

        if plugin_sols:
            body.append("## 插件型解法\n")
            for _san, _claim in plugin_sols[:5]:
                body.append(f"- [[P__{_san}]] — {_claim[:100]}")
            body.append("")

        if bn.description and not bn.symptom_query:
            pass  # already shown above
        elif bn.description:
            body.append("## 当前判断\n")
            body.append(f"{bn.description}\n")

        content = _render_frontmatter(fm) + "\n".join(body)
        (bn_dir / filename).write_text(content, encoding="utf-8")
        stats["bottlenecks"] += 1

    # ── 6. Generate Lineage Notes (10_Lineages/) ─────────────────
    lin_dir = root / "10_Lineages"
    lin_dir.mkdir(parents=True, exist_ok=True)

    for gk, gd in lineage_groups.items():
        slug = lineage_slug[gk]
        filename = f"L__{slug}.md"
        edges = gd["edges"]
        all_sans = gd["papers"]

        # Build parent→children map for tree rendering
        parent_children: dict[str, list[tuple[str, str]]] = {}
        child_set: set[str] = set()
        for par, chi, rel in edges:
            parent_children.setdefault(par, []).append((chi, rel))
            child_set.add(chi)
        roots = [s for s in all_sans if s not in child_set]
        if not roots:
            roots = sorted(all_sans)

        fm = {
            "title": f"{gk} 方法演化",
            "type": "lineage",
            "lineage_id": f"L__{slug}",
            "paper_count": len(all_sans),
        }
        body = [f"# {gk} 方法演化\n"]

        # ASCII evolution tree
        body.append("## 演化链\n")
        body.append("```")
        visited: set[str] = set()

        def _tree(san: str, depth: int = 0) -> None:
            if san in visited:
                return
            visited.add(san)
            prefix = "  " * depth + ("└─ " if depth else "")
            title = san_to_title.get(san, san)
            body.append(f"{prefix}{title}")
            for ch, _rel in parent_children.get(san, []):
                _tree(ch, depth + 1)

        for r in sorted(roots):
            _tree(r)
        for san in sorted(all_sans - visited):
            body.append(f"{san_to_title.get(san, san)} (独立)")
        body.append("```\n")

        # Per-step diff
        body.append("## 每步改了什么\n")
        for par, chi, rel in sorted(edges, key=lambda x: x[0]):
            child_p = next((p for p in papers if p.title_sanitized == chi), None)
            slots = (", ".join(list(child_p.changed_slots)[:3])
                     if child_p and child_p.changed_slots else "?")
            body.append(f"- [[P__{par}]] → [[P__{chi}]] ({rel})")
            body.append(f"  - 改了: {slots}")
        body.append("")

        # Fork points
        forks = {p: cs for p, cs in parent_children.items() if len(cs) > 1}
        if forks:
            body.append("## 分叉点\n")
            for par, children in forks.items():
                body.append(f"- [[P__{par}]] 分叉为:")
                for chi, rel in children:
                    body.append(f"  - [[P__{chi}]] ({rel})")
            body.append("")

        content = _render_frontmatter(fm) + "\n".join(body)
        (lin_dir / filename).write_text(content, encoding="utf-8")
        stats["lineages"] += 1

    # ── 7. Generate 00_Home (navigation only, not in main graph) ─
    home_dir = root / "00_Home"
    home_dir.mkdir(parents=True, exist_ok=True)

    analyzed = [p for p in papers if p.state == "l4_deep"]
    level_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for p in papers:
        level_counts[_paper_level(p)] += 1

    ov = ["# 方向总览\n"]
    ov.append(f"**知识库规模**: {len(papers)} 篇论文, {len(analyzed)} 篇已深度分析\n")

    if lineage_groups:
        ov.append("## 方法主线\n")
        for gk in sorted(lineage_groups):
            cnt = len(lineage_groups[gk]["papers"])
            ov.append(f"- [[L__{lineage_slug[gk]}]] — {cnt} 篇论文")
        ov.append("")

    if concepts:
        ov.append("## 核心概念\n")
        for c in sorted(concepts, key=lambda x: len(x["papers"]), reverse=True)[:10]:
            ov.append(f"- [[C__{c['slug']}]] — {len(c['papers'])} 篇论文")
        ov.append("")

    if bottlenecks:
        ov.append("## 研究瓶颈\n")
        for bn in bottlenecks[:10]:
            cnt = len(bn_claims_map.get(str(bn.id), []))
            ov.append(f"- [[B__{bn_safe_name[bn.title]}]] — {cnt} 篇论文在解决")
        ov.append("")

    ov.append("## 论文分布\n")
    ov.append(f"- **A** (必读 baseline): {level_counts['A']}")
    ov.append(f"- **B** (结构性改进): {level_counts['B']}")
    ov.append(f"- **C** (插件型): {level_counts['C']}")
    ov.append(f"- **D** (外围): {level_counts['D']}")
    ov.append("")

    (home_dir / "00_方向总览.md").write_text(
        _render_frontmatter({"title": "方向总览", "type": "overview"}) + "\n".join(ov),
        encoding="utf-8",
    )

    # 阅读顺序
    rd = ["# 阅读顺序建议\n"]
    rd.append("## 第一层：理解框架\n")
    rd.append("1. 读 [[00_方向总览]]，了解方向全貌")
    if lineage_groups:
        first_slug = list(lineage_slug.values())[0]
        rd.append(f"2. 读任一 Lineage 笔记（如 [[L__{first_slug}]]），理解方法演化主线")
    rd.append("")

    rd.append("## 第二层：读 Baseline\n")
    for p in [p for p in papers if _paper_level(p) == "A"][:5]:
        rd.append(f"- [[P__{p.title_sanitized}]]")
    rd.append("")

    rd.append("## 第三层：读结构性改进\n")
    struct_papers = sorted(
        [p for p in papers if _paper_level(p) == "B"],
        key=lambda x: float(x.dc_struct or 0), reverse=True)
    for p in struct_papers[:8]:
        score = f"{float(p.dc_struct):.2f}" if p.dc_struct else "?"
        rd.append(f"- [[P__{p.title_sanitized}]] (struct={score})")
    rd.append("")

    rd.append("## 第四层：按需查阅\n")
    rd.append("- C/D 类论文按需查阅")
    rd.append("- 关注 Concept 和 Bottleneck 笔记中的综合判断\n")

    (home_dir / "01_阅读顺序.md").write_text(
        _render_frontmatter({"title": "阅读顺序", "type": "overview"}) + "\n".join(rd),
        encoding="utf-8",
    )

    # ── 8. Generate 90_Views (Dataview queries) ──────────────────
    views_dir = root / "90_Views"
    views_dir.mkdir(parents=True, exist_ok=True)

    (views_dir / "papers_by_structurality.md").write_text("\n".join([
        "# 论文 — 按结构性排序\n",
        "```dataview",
        'TABLE year, venue, paper_level, structurality_score, concepts, bottleneck',
        'FROM "40_Papers"',
        'WHERE type = "paper"',
        'SORT structurality_score DESC',
        "```",
    ]), encoding="utf-8")

    (views_dir / "papers_by_year.md").write_text("\n".join([
        "# 论文 — 按年份\n",
        "```dataview",
        'TABLE venue, paper_level, structurality_score, concepts',
        'FROM "40_Papers"',
        'WHERE type = "paper"',
        'SORT year DESC',
        "```",
    ]), encoding="utf-8")

    (views_dir / "concept_map.md").write_text("\n".join([
        "# 概念地图\n",
        "```dataview",
        'TABLE domain, length(file.inlinks) AS "引用数"',
        'FROM "20_Concepts"',
        'WHERE type = "concept"',
        'SORT length(file.inlinks) DESC',
        "```",
    ]), encoding="utf-8")

    (views_dir / "bottleneck_overview.md").write_text("\n".join([
        "# 瓶颈一览\n",
        "```dataview",
        'TABLE domain, paper_count',
        'FROM "30_Bottlenecks"',
        'WHERE type = "bottleneck"',
        'SORT paper_count DESC',
        "```",
    ]), encoding="utf-8")

    logger.info(f"Obsidian vault v4.0 exported to {root}: {stats}")
    return {"vault_path": str(root), **stats}
