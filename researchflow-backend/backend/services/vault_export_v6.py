"""Obsidian Vault Export v6 — full vault export with profiles + edge one-liners.

Structure:
  10_Tasks/    — T__ nodes, Common Problems embedded
  20_Methods/  — M__ nodes with evolution DAG (Mermaid)
  30_Mechanisms/ — C__ reusable technique concepts
  40_Papers/   — P__ with A/B/C/D tiers
  50_Datasets/ — D__ benchmarks and datasets
  60_Labs/     — Lab__ research team nodes
  80_Assets/   — figures from OSS
  90_Views/    — tables (by task, by year, method evolution)

Enhancements:
- Node profiles from kb_node_profiles
- Edge one-liners from kb_edge_profiles
- Lab pages with paper associations

Paper wikilink budget: 6-10 (T + M + C + D + P links only)
Other facets go in frontmatter YAML, not as wikilinks.
"""

import logging
import re
import shutil
from pathlib import Path

import yaml
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings

logger = logging.getLogger(__name__)


def _fm(data: dict) -> str:
    """Render YAML frontmatter."""
    s = yaml.dump(data, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=200)
    return f"---\n{s}---\n\n"


def _slug(raw: str, max_len: int = 80) -> str:
    """Filesystem + wikilink safe slug."""
    return (raw.replace(" ", "_").replace("/", "-").replace(":", "-")
            .replace("|", "-").replace("[", "(").replace("]", ")")
            .replace("#", "").replace("'", "").replace('"', '')[:max_len])


def _level(ring, dc_struct, paper_struct) -> str:
    if ring == "baseline": return "A"
    if ring == "structural": return "B"
    if ring == "plugin": return "C"
    s = dc_struct or paper_struct
    if s is not None:
        s = float(s)
        if s >= 0.7: return "A"
        if s >= 0.5: return "B"
        if s >= 0.3: return "C"
    return "D"


_TIER_DIRS = {"A": "A_Baselines", "B": "B_Structural", "C": "C_Adaptations", "D": "D_Peripheral"}
_TIER_ADVICE = {
    "A": "> **必读 baseline**。先理解此论文建立的标准框架。",
    "B": "> **结构性改进**。先读 baseline，再看本文修改了哪些核心组件。",
    "C": "> **适配/插件型**。可快速浏览，看改了哪个 slot 和效果。",
    "D": "> **外围参考**。按需阅读。",
}


# Prefix → entity_type mapping for node profile lookups
_PREFIX_TO_ENTITY = {
    "T__": ("taxonomy_node", "task"),
    "M__": ("method_node", "method"),
    "C__": ("taxonomy_node", "mechanism"),
    "D__": ("taxonomy_node", "dataset"),
    "P__": ("paper", "paper"),
    "Lab__": ("taxonomy_node", "lab"),
}

# Wikilink regex: captures prefix and slug inside [[ ]]
_WIKILINK_RE = re.compile(r'\[\[(T|M|C|D|P|Lab)__([^\]]+)\]\]')


def _detect_node_key(filename: str) -> tuple[str, str, str] | None:
    """From a filename like T__foo.md, return (prefix, entity_type, slug) or None."""
    stem = Path(filename).stem
    for prefix, (etype, _) in _PREFIX_TO_ENTITY.items():
        if stem.startswith(prefix):
            slug = stem[len(prefix):]
            return prefix, etype, slug
    return None


async def _export_vault_base(session: AsyncSession, out_dir: str | None = None) -> dict:
    """Export base vault structure (tasks, methods, mechanisms, papers, datasets, views)."""

    root = Path(out_dir or (str(Path(settings.paper_analysis_dir).parent / "obsidian-vault")))
    if root.exists():
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)

    stats = {"tasks": 0, "methods": 0, "mechanisms": 0, "papers": 0, "datasets": 0}

    # ═══════════════════════════════════════════════════════════
    # Fetch all data
    # ═══════════════════════════════════════════════════════════

    papers = (await session.execute(text("""
        SELECT p.id, p.title, p.title_sanitized, p.venue, p.year, p.category,
               p.tags, p.core_operator, p.primary_logic, p.abstract,
               p.paper_link, p.code_url, p.method_family, p.ring,
               p.structurality_score, p.keep_score, p.acceptance_type,
               p.open_code, p.open_data, p.importance, p.state,
               p.cited_by_count, p.authors,
               dc.delta_statement, dc.baseline_paradigm,
               dc.structurality_score AS dc_struct,
               dc.transferability_score AS dc_transfer,
               dc.key_ideas_ranked, dc.key_equations, dc.key_figures,
               dc.changed_slot_ids, dc.baseline_paper_ids,
               dc.same_family_paper_ids,
               pa.problem_summary, pa.method_summary, pa.evidence_summary,
               pa.core_intuition, pa.full_report_md, pa.changed_slots,
               pa.extracted_figure_images
        FROM papers p
        LEFT JOIN delta_cards dc ON dc.id = COALESCE(
            p.current_delta_card_id,
            (SELECT id FROM delta_cards dc2 WHERE dc2.paper_id = p.id
             AND dc2.status != 'deprecated' ORDER BY dc2.created_at DESC LIMIT 1)
        )
        LEFT JOIN paper_analyses pa ON pa.paper_id = p.id
            AND pa.is_current = true AND pa.level = 'l4_deep'
        WHERE p.state NOT IN ('archived_or_expired', 'skip')
        ORDER BY p.category, p.venue, p.year
    """))).fetchall()

    id_to_san = {str(p.id): p.title_sanitized for p in papers}
    san_to_paper = {p.title_sanitized: p for p in papers}

    # Taxonomy data
    tax_nodes = []
    tax_edges = []
    paper_facets = []
    try:
        tax_nodes = (await session.execute(text(
            "SELECT id, name, name_zh, dimension, description, aliases, status "
            "FROM taxonomy_nodes WHERE status != 'archived' ORDER BY dimension, name"
        ))).fetchall()
        tax_edges = (await session.execute(text(
            "SELECT parent_id, child_id, relation_type FROM taxonomy_edges"
        ))).fetchall()
        paper_facets = (await session.execute(text(
            "SELECT paper_id, node_id, facet_role FROM paper_facets"
        ))).fetchall()
    except Exception:
        logger.info("Taxonomy tables not yet created, skipping taxonomy export")

    # Method data
    method_nodes = []
    method_edges = []
    method_apps = []
    method_slots = []
    try:
        method_nodes = (await session.execute(text(
            "SELECT id, name, name_zh, type, maturity, description, "
            "downstream_count, canonical_paper_id "
            "FROM method_nodes ORDER BY downstream_count DESC, name"
        ))).fetchall()
        method_edges = (await session.execute(text(
            "SELECT parent_method_id, child_method_id, relation_type, "
            "delta_description, status FROM method_edges WHERE status != 'rejected'"
        ))).fetchall()
        method_apps = (await session.execute(text(
            "SELECT paper_id, method_id, role FROM method_applications"
        ))).fetchall()
        method_slots = (await session.execute(text(
            "SELECT method_id, slot_name, default_description FROM method_slots ORDER BY sort_order"
        ))).fetchall()
    except Exception:
        logger.info("Method tables not yet created, skipping method export")

    # Problem data
    problems = []
    problem_claims = []
    try:
        problems = (await session.execute(text(
            "SELECT id, name, name_zh, parent_task_id, symptom, root_cause, "
            "solution_families, status FROM problem_nodes WHERE status != 'archived'"
        ))).fetchall()
        problem_claims = (await session.execute(text(
            "SELECT paper_id, problem_id, claim_type FROM problem_claims"
        ))).fetchall()
    except Exception:
        logger.info("Problem tables not yet created, skipping problem export")

    # Legacy data (mechanisms, lineage)
    mechanisms = []
    try:
        mechanisms = (await session.execute(text(
            "SELECT id, name, domain, description FROM method_nodes ORDER BY name"
        ))).fetchall()
    except Exception:
        pass

    lineage_rows = []
    try:
        lineage_rows = (await session.execute(text("""
            SELECT dcl.relation_type,
                   child_p.title_sanitized AS child_san,
                   parent_p.title_sanitized AS parent_san
            FROM delta_card_lineage dcl
            JOIN delta_cards child_dc ON child_dc.id = dcl.child_delta_card_id
            JOIN papers child_p ON child_p.id = child_dc.paper_id
            JOIN delta_cards parent_dc ON parent_dc.id = dcl.parent_delta_card_id
            JOIN papers parent_p ON parent_p.id = parent_dc.paper_id
            WHERE dcl.status != 'rejected'
        """))).fetchall()
    except Exception:
        pass

    evidence_paper_links = []
    try:
        evidence_paper_links = (await session.execute(text("""
            SELECT DISTINCT
                eu.paper_id,
                p.title_sanitized AS paper_san,
                dc.paper_id AS dc_paper_id,
                dp.title_sanitized AS dc_paper_san
            FROM evidence_units eu
            JOIN delta_cards dc ON dc.id = eu.delta_card_id
            JOIN papers p ON p.id = eu.paper_id
            JOIN papers dp ON dp.id = dc.paper_id
            WHERE eu.paper_id != dc.paper_id
              AND p.title_sanitized IS NOT NULL
              AND dp.title_sanitized IS NOT NULL
        """))).fetchall()
    except Exception:
        pass

    # ═══════════════════════════════════════════════════════════
    # Build lookup maps
    # ═══════════════════════════════════════════════════════════

    node_by_id = {str(n.id): n for n in tax_nodes}
    children_of = {}
    for e in tax_edges:
        children_of.setdefault(str(e.parent_id), []).append(str(e.child_id))

    paper_facet_map = {}
    for pf in paper_facets:
        node = node_by_id.get(str(pf.node_id))
        if node:
            paper_facet_map.setdefault(str(pf.paper_id), []).append({
                "role": pf.facet_role, "name": node.name, "name_zh": node.name_zh,
                "dimension": node.dimension,
            })

    method_by_id = {str(m.id): m for m in method_nodes}
    method_slot_map = {}
    for s in method_slots:
        method_slot_map.setdefault(str(s.method_id), []).append(s.slot_name)
    method_app_map = {}
    for a in method_apps:
        m = method_by_id.get(str(a.method_id))
        if m:
            method_app_map.setdefault(str(a.paper_id), []).append({
                "method": m.name, "role": a.role,
            })

    problem_claim_map = {}
    for pc in problem_claims:
        problem_claim_map.setdefault(str(pc.problem_id), []).append(str(pc.paper_id))

    lineage_parents = {}
    lineage_children = {}
    for lr in lineage_rows:
        lineage_parents.setdefault(lr.child_san, []).append((lr.relation_type, lr.parent_san))
        lineage_children.setdefault(lr.parent_san, []).append((lr.relation_type, lr.child_san))

    evidence_links_out = {}
    evidence_links_in = {}
    for el in evidence_paper_links:
        if el.paper_san and el.dc_paper_san:
            evidence_links_out.setdefault(el.paper_san, set()).add(el.dc_paper_san)
            evidence_links_in.setdefault(el.dc_paper_san, set()).add(el.paper_san)

    mech_family_papers = {}
    for p in papers:
        if p.method_family:
            mech_family_papers.setdefault(p.method_family, []).append(p.title_sanitized)

    mech_papers = {}
    for p in papers:
        if p.method_family:
            mech_papers.setdefault(p.method_family, []).append(p)

    # ═══════════════════════════════════════════════════════════
    # 10_Tasks/
    # ═══════════════════════════════════════════════════════════

    task_dir = root / "10_Tasks"
    task_dir.mkdir(parents=True, exist_ok=True)

    task_nodes = [n for n in tax_nodes if n.dimension in ("task", "subtask")]
    for tn in task_nodes:
        slug = _slug(tn.name)
        body = [f"# {tn.name}\n"]
        if tn.name_zh:
            body[0] = f"# {tn.name} ({tn.name_zh})\n"
        if tn.description:
            body.append(f"{tn.description}\n")

        child_ids = children_of.get(str(tn.id), [])
        subtasks = [node_by_id[cid] for cid in child_ids if cid in node_by_id
                    and node_by_id[cid].dimension in ("subtask", "task")]
        if subtasks:
            body.append("## 子任务\n")
            for st in subtasks:
                body.append(f"- [[T__{_slug(st.name)}]]")
            body.append("")

        task_problems = [p for p in problems if str(p.parent_task_id) == str(tn.id)]
        if task_problems:
            body.append("## Common Problems\n")
            for prob in task_problems:
                body.append(f"### {prob.name}")
                if prob.symptom:
                    body.append(f"- **Symptoms**: {prob.symptom}")
                if prob.root_cause:
                    body.append(f"- **Root cause**: {prob.root_cause}")
                claim_pids = problem_claim_map.get(str(prob.id), [])
                if claim_pids:
                    body.append("- **Solutions**:")
                    for pid in claim_pids[:5]:
                        san = id_to_san.get(pid)
                        if san:
                            body.append(f"  - [[P__{san}]]")
                body.append("")

        task_paper_ids = [str(pf.paper_id) for pf in paper_facets
                         if str(pf.node_id) == str(tn.id)]
        task_keywords = set(tn.name.lower().replace("_", " ").replace("-", " ").split())
        task_keywords.discard("")
        if tn.aliases:
            for alias in (tn.aliases if isinstance(tn.aliases, list) else []):
                task_keywords.update(alias.lower().replace("_", " ").replace("-", " ").split())
        task_keywords -= {"a", "an", "the", "for", "of", "in", "on", "and", "or", "to", "with"}

        if task_keywords and len(task_paper_ids) < 5:
            for pp in papers:
                pid = str(pp.id)
                if pid in task_paper_ids:
                    continue
                title_words = set((pp.title or "").lower().split())
                overlap = task_keywords & title_words
                if len(overlap) >= 2:
                    task_paper_ids.append(pid)
                    continue
                if pp.category:
                    cat_words = set(pp.category.lower().replace("_", " ").split())
                    if task_keywords & cat_words:
                        task_paper_ids.append(pid)

        if task_paper_ids:
            body.append(f"## 相关论文 ({len(task_paper_ids)})\n")
            for pid in task_paper_ids[:20]:
                san = id_to_san.get(pid)
                if san:
                    pp = san_to_paper.get(san)
                    venue = f" ({pp.venue} {pp.year})" if pp and pp.venue else ""
                    body.append(f"- [[P__{san}]]{venue}")
            body.append("")

        fm = {"title": tn.name, "type": "task", "dimension": tn.dimension,
              "name_zh": tn.name_zh}
        (task_dir / f"T__{slug}.md").write_text(_fm(fm) + "\n".join(body), encoding="utf-8")
        stats["tasks"] += 1

    # ═══════════════════════════════════════════════════════════
    # 20_Methods/
    # ═══════════════════════════════════════════════════════════

    meth_dir = root / "20_Methods"
    meth_dir.mkdir(parents=True, exist_ok=True)

    for mn in method_nodes:
        slug = _slug(mn.name)
        body = [f"# {mn.name}\n"]
        if mn.name_zh:
            body[0] = f"# {mn.name} ({mn.name_zh})\n"

        body.append(f"**Type**: {mn.type} | **Maturity**: {mn.maturity} | "
                     f"**Downstream**: {mn.downstream_count}\n")

        if mn.description:
            body.append(f"{mn.description}\n")

        slots = method_slot_map.get(str(mn.id), [])
        if slots:
            body.append("## Slots\n")
            for s in slots:
                body.append(f"- `{s}`")
            body.append("")

        parent_edges = [e for e in method_edges if str(e.child_method_id) == str(mn.id)]
        child_edges = [e for e in method_edges if str(e.parent_method_id) == str(mn.id)]

        if parent_edges or child_edges:
            body.append("## 演化关系\n")
            body.append("```mermaid")
            body.append("graph TD")
            for e in parent_edges:
                parent = method_by_id.get(str(e.parent_method_id))
                if parent:
                    body.append(f"    {_slug(parent.name)}[{parent.name}] -->|{e.relation_type}| {slug}[{mn.name}]")
            for e in child_edges:
                child = method_by_id.get(str(e.child_method_id))
                if child:
                    body.append(f"    {slug}[{mn.name}] -->|{e.relation_type}| {_slug(child.name)}[{child.name}]")
            body.append("```\n")

            for e in parent_edges:
                parent = method_by_id.get(str(e.parent_method_id))
                if parent and e.delta_description:
                    body.append(f"- ← [[M__{_slug(parent.name)}]]: {e.delta_description}")
            for e in child_edges:
                child = method_by_id.get(str(e.child_method_id))
                if child and e.delta_description:
                    body.append(f"- → [[M__{_slug(child.name)}]]: {e.delta_description}")
            body.append("")

        apps = [a for a in method_apps if str(a.method_id) == str(mn.id)]
        if apps:
            body.append(f"## 使用论文 ({len(apps)})\n")
            body.append("| Paper | Role |")
            body.append("|-------|------|")
            for a in apps[:15]:
                san = id_to_san.get(str(a.paper_id))
                if san:
                    body.append(f"| [[P__{san}]] | {a.role} |")
            body.append("")

        fm = {"title": mn.name, "type": "method", "maturity": mn.maturity,
              "method_type": mn.type, "downstream_count": mn.downstream_count}
        (meth_dir / f"M__{slug}.md").write_text(_fm(fm) + "\n".join(body), encoding="utf-8")
        stats["methods"] += 1

    # ═══════════════════════════════════════════════════════════
    # 30_Mechanisms/
    # ═══════════════════════════════════════════════════════════

    mech_dir = root / "30_Mechanisms"
    mech_dir.mkdir(parents=True, exist_ok=True)

    mech_tax_nodes = [n for n in tax_nodes if n.dimension == "mechanism"]
    if mech_tax_nodes:
        for mn in mech_tax_nodes:
            slug = _slug(mn.name)
            body = [f"# {mn.name}"]
            if mn.name_zh:
                body[0] += f" ({mn.name_zh})"
            body[0] += "\n"
            if mn.description:
                body.append(f"{mn.description}\n")

            mech_pids = [str(pf.paper_id) for pf in paper_facets
                         if str(pf.node_id) == str(mn.id)]
            mech_name_lower = mn.name.lower().replace("_", " ").replace("-", " ")
            for pp in papers:
                pid = str(pp.id)
                if pid in mech_pids:
                    continue
                if pp.method_family:
                    mf_lower = pp.method_family.lower().replace("_", " ")
                    if mech_name_lower in mf_lower or mf_lower in mech_name_lower:
                        mech_pids.append(pid)

            if mech_pids:
                body.append(f"## 相关论文 ({len(mech_pids)})\n")
                for pid in mech_pids[:15]:
                    san = id_to_san.get(pid)
                    if san:
                        body.append(f"- [[P__{san}]]")
                body.append("")

            fm = {"title": mn.name, "type": "mechanism", "name_zh": mn.name_zh}
            (mech_dir / f"C__{slug}.md").write_text(_fm(fm) + "\n".join(body), encoding="utf-8")
            stats["mechanisms"] += 1
    else:
        for mf in mechanisms:
            slug = _slug(mf.name)
            body = [f"# {mf.name}\n"]
            if mf.description:
                body.append(f"{mf.description}\n")
            mf_papers = mech_papers.get(mf.name, [])
            if mf_papers:
                body.append(f"## 代表论文 ({len(mf_papers)})\n")
                body.append("| 论文 | Year | Venue |")
                body.append("|------|------|-------|")
                for p in sorted(mf_papers, key=lambda x: x.year or 0, reverse=True)[:10]:
                    body.append(f"| [[P__{p.title_sanitized}]] | {p.year} | {p.venue or ''} |")
                body.append("")

            fm = {"title": mf.name, "type": "mechanism", "domain": mf.domain}
            (mech_dir / f"C__{slug}.md").write_text(_fm(fm) + "\n".join(body), encoding="utf-8")
            stats["mechanisms"] += 1

    # ═══════════════════════════════════════════════════════════
    # 40_Papers/
    # ═══════════════════════════════════════════════════════════

    for p in papers:
        lvl = _level(p.ring, p.dc_struct, p.structurality_score)
        paper_dir = root / "40_Papers" / _TIER_DIRS[lvl]
        paper_dir.mkdir(parents=True, exist_ok=True)

        san = p.title_sanitized or str(p.id)
        pid_str = str(p.id)

        links = []

        facets = paper_facet_map.get(pid_str, [])
        task_facets = [f for f in facets if f["dimension"] in ("task", "subtask")]
        for tf in task_facets[:2]:
            links.append(f"- Task: [[T__{_slug(tf['name'])}]]")

        apps = method_app_map.get(pid_str, [])
        for a in apps[:2]:
            links.append(f"- {a['role'].replace('_', ' ').title()}: [[M__{_slug(a['method'])}]]")

        mech_facets = [f for f in facets if f["dimension"] == "mechanism"]
        for mf in mech_facets[:2]:
            links.append(f"- Mechanism: [[C__{_slug(mf['name'])}]]")

        ancestors = lineage_parents.get(san, [])
        for rel, parent_san in ancestors[:2]:
            links.append(f"- builds on: [[P__{parent_san}]]")
        descendants = lineage_children.get(san, [])
        for rel, child_san in descendants[:1]:
            links.append(f"- extended by: [[P__{child_san}]]")

        ev_out = evidence_links_out.get(san, set())
        for related_san in list(ev_out)[:3]:
            links.append(f"- evidence links to: [[P__{related_san}]]")
        ev_in = evidence_links_in.get(san, set())
        for related_san in list(ev_in)[:3]:
            links.append(f"- evidence from: [[P__{related_san}]]")

        if not mech_facets and p.method_family:
            links.append(f"- Mechanism: [[C__{_slug(p.method_family)}]]")

        fm = {
            "title": p.title,
            "type": "paper",
            "paper_level": lvl,
            "venue": p.venue,
            "year": p.year,
            "acceptance": p.acceptance_type,
            "cited_by": p.cited_by_count,
        }
        facet_fm = {}
        for f in facets:
            facet_fm.setdefault(f["dimension"], []).append(f["name"])
        if facet_fm:
            fm["facets"] = facet_fm
        if p.core_operator:
            fm["core_operator"] = p.core_operator
        if p.paper_link:
            fm["paper_link"] = p.paper_link
        if p.code_url:
            fm["code_url"] = p.code_url
        if p.dc_struct:
            fm["structurality_score"] = round(float(p.dc_struct), 3)

        body = [f"# {p.title}\n"]

        if links:
            body.append("## Links\n")
            body.extend(links)
            body.append("")

        if p.delta_statement:
            body.append(f"> {p.delta_statement[:300]}\n")

        body.append(_TIER_ADVICE[lvl] + "\n")

        eqs = p.key_equations if isinstance(p.key_equations, list) else []
        if eqs:
            body.append("## 核心公式\n")
            for eq in eqs:
                if isinstance(eq, dict):
                    latex = eq.get("latex", "")
                    expl = eq.get("explanation", "")
                    slot = eq.get("slot_affected", "")
                    if latex:
                        body.append(f"$$\n{latex}\n$$\n")
                        if expl:
                            body.append(f"> {expl}")
                        if slot:
                            body.append(f"> *Slot*: {slot}")
                        body.append("")

        figs_meta = p.key_figures if isinstance(p.key_figures, list) else []
        if figs_meta:
            body.append("## 关键图表\n")
            for fig in figs_meta:
                if isinstance(fig, dict):
                    ref = fig.get("fig_ref", "")
                    caption = fig.get("caption", "")
                    evidence = fig.get("evidence_for", "")
                    if ref:
                        body.append(f"**{ref}**")
                        if caption:
                            body.append(f": {caption}")
                        if evidence:
                            body.append(f"> 证据支持: {evidence}")
                        body.append("")

        if p.full_report_md:
            body.append("## 详细分析\n")
            body.append(p.full_report_md)
        else:
            if p.problem_summary:
                body.append(f"## 问题与挑战\n\n{p.problem_summary}\n")
            if p.method_summary:
                body.append(f"## 方法与洞察\n\n{p.method_summary}\n")
            if p.core_intuition:
                body.append(f"### 核心直觉\n\n{p.core_intuition}\n")
            if p.evidence_summary:
                body.append(f"## 证据与局限\n\n{p.evidence_summary}\n")

        content = _fm(fm) + "\n".join(body)
        (paper_dir / f"P__{san}.md").write_text(content, encoding="utf-8")

        figs = p.extracted_figure_images if isinstance(p.extracted_figure_images, list) else []
        if figs:
            fig_dir = root / "80_Assets" / "figures" / san
            fig_dir.mkdir(parents=True, exist_ok=True)
            try:
                from backend.services.object_storage import get_storage
                storage = get_storage()
                for fig in figs[:8]:
                    obj_key = fig.get("object_key", "")
                    if obj_key:
                        img_data = await storage.get(obj_key)
                        if img_data:
                            ext = obj_key.rsplit(".", 1)[-1] if "." in obj_key else "png"
                            label = fig.get("label", f"fig_{fig.get('figure_num', 0)}")
                            (fig_dir / f"{_slug(label)}.{ext}").write_bytes(img_data)
            except Exception as e:
                logger.warning(f"Figure export failed for {san}: {e}")

        stats["papers"] += 1

    # ═══════════════════════════════════════════════════════════
    # 50_Datasets_Benchmarks/
    # ═══════════════════════════════════════════════════════════

    ds_dir = root / "50_Datasets_Benchmarks"
    ds_dir.mkdir(parents=True, exist_ok=True)

    dataset_nodes = [n for n in tax_nodes if n.dimension in ("dataset", "benchmark")]
    for dn in dataset_nodes:
        slug = _slug(dn.name)
        body = [f"# {dn.name}\n"]
        if dn.description:
            body.append(f"{dn.description}\n")

        ds_pids = [str(pf.paper_id) for pf in paper_facets if str(pf.node_id) == str(dn.id)]
        if ds_pids:
            body.append(f"## 使用论文 ({len(ds_pids)})\n")
            for pid in ds_pids[:10]:
                san = id_to_san.get(pid)
                if san:
                    body.append(f"- [[P__{san}]]")
            body.append("")

        fm = {"title": dn.name, "type": "dataset", "name_zh": dn.name_zh}
        (ds_dir / f"D__{slug}.md").write_text(_fm(fm) + "\n".join(body), encoding="utf-8")
        stats["datasets"] += 1

    # ═══════════════════════════════════════════════════════════
    # 00_Home/
    # ═══════════════════════════════════════════════════════════

    home_dir = root / "00_Home"
    home_dir.mkdir(parents=True, exist_ok=True)

    level_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for p in papers:
        level_counts[_level(p.ring, p.dc_struct, p.structurality_score)] += 1

    ov = [f"# 方向总览\n",
          f"**知识库**: {len(papers)} 篇论文, {stats['tasks']} 个任务, "
          f"{stats['methods']} 个方法, {stats['mechanisms']} 个机制\n"]

    if task_nodes:
        ov.append("## 任务\n")
        top_tasks = [n for n in task_nodes if n.dimension == "task"]
        for t in top_tasks[:10]:
            ov.append(f"- [[T__{_slug(t.name)}]]")
        ov.append("")

    if method_nodes:
        ov.append("## 方法\n")
        for m in method_nodes[:10]:
            ov.append(f"- [[M__{_slug(m.name)}]] ({m.maturity}, ↓{m.downstream_count})")
        ov.append("")

    ov.append("## 论文分布\n")
    ov.append(f"- **A** Baselines: {level_counts['A']}")
    ov.append(f"- **B** Structural: {level_counts['B']}")
    ov.append(f"- **C** Adaptations: {level_counts['C']}")
    ov.append(f"- **D** Peripheral: {level_counts['D']}\n")

    (home_dir / "00_方向总览.md").write_text(
        _fm({"title": "方向总览", "type": "overview"}) + "\n".join(ov), encoding="utf-8")

    # ═══════════════════════════════════════════════════════════
    # 90_Views/
    # ═══════════════════════════════════════════════════════════

    views_dir = root / "90_Views"
    views_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# 论文 — 按年份\n",
             "| 论文 | Year | Venue | Level | Task |",
             "|------|------|-------|-------|------|"]
    for p in sorted(papers, key=lambda x: x.year or 0, reverse=True):
        san = p.title_sanitized or str(p.id)
        lvl = _level(p.ring, p.dc_struct, p.structurality_score)
        facets = paper_facet_map.get(str(p.id), [])
        task = next((f["name"] for f in facets if f["dimension"] == "task"), p.category or "")
        lines.append(f"| [[P__{san}]] | {p.year} | {p.venue or ''} | {lvl} | {task} |")
    (views_dir / "papers_by_year.md").write_text("\n".join(lines), encoding="utf-8")

    if method_nodes:
        lines = ["# 方法演化\n",
                 "| 方法 | Type | Maturity | Downstream | Canonical Paper |",
                 "|------|------|----------|-----------|----------------|"]
        for m in method_nodes:
            cp_san = id_to_san.get(str(m.canonical_paper_id), "") if m.canonical_paper_id else ""
            cp_link = f"[[P__{cp_san}]]" if cp_san else "—"
            lines.append(f"| [[M__{_slug(m.name)}]] | {m.type} | {m.maturity} | {m.downstream_count} | {cp_link} |")
        (views_dir / "method_evolution.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"Obsidian vault base exported to {root}: {stats}")
    return {"vault_path": str(root), **stats}


async def export_vault_v6(session: AsyncSession, out_dir: str | None = None) -> dict:
    """Export knowledge base as Obsidian vault v6 (base + profiles + labs)."""

    # ── Step 1: Generate base vault ──
    stats = await _export_vault_base(session, out_dir)
    root = Path(stats["vault_path"])

    profiles_injected = 0

    # ── Step 2: Load profiles from DB ──
    try:
        node_profiles = (await session.execute(text(
            "SELECT entity_type, entity_id, one_liner, short_intro_md, detailed_md, "
            "structured_json FROM kb_node_profiles WHERE lang = 'zh' AND profile_kind = 'page'"
        ))).fetchall()
    except Exception:
        logger.info("kb_node_profiles table not available, skipping profile injection")
        node_profiles = []

    try:
        edge_profiles = (await session.execute(text(
            "SELECT source_entity_type, source_entity_id, target_entity_type, target_entity_id, "
            "relation_type, one_liner, display_priority FROM kb_edge_profiles WHERE lang = 'zh' "
            "ORDER BY display_priority ASC"
        ))).fetchall()
    except Exception:
        logger.info("kb_edge_profiles table not available, skipping edge one-liners")
        edge_profiles = []

    # ── Step 3: Build lookup maps ──
    node_profile_map: dict[tuple[str, str], object] = {}
    for np in node_profiles:
        key = (np.entity_type, str(np.entity_id))
        node_profile_map[key] = np

    edge_profile_map: dict[tuple[str, str, str, str], str] = {}
    for ep in edge_profiles:
        key = (ep.source_entity_type, str(ep.source_entity_id),
               ep.target_entity_type, str(ep.target_entity_id))
        edge_profile_map[key] = ep.one_liner

    # ── Step 2b: Load taxonomy + paper data for Lab pages ──
    tax_nodes = []
    paper_facets = []
    id_to_san: dict[str, str] = {}
    try:
        tax_nodes = (await session.execute(text(
            "SELECT id, name, name_zh, dimension, description, aliases, status "
            "FROM taxonomy_nodes WHERE status != 'archived' ORDER BY dimension, name"
        ))).fetchall()
        paper_facets = (await session.execute(text(
            "SELECT paper_id, node_id, facet_role FROM paper_facets"
        ))).fetchall()
        papers_rows = (await session.execute(text(
            "SELECT id, title_sanitized FROM papers WHERE state NOT IN ('archived_or_expired', 'skip')"
        ))).fetchall()
        id_to_san = {str(r.id): r.title_sanitized for r in papers_rows}
    except Exception:
        logger.info("Could not load taxonomy/paper data for Lab pages")

    # Build node name → (entity_type, entity_id) reverse map for wikilink matching
    node_by_slug: dict[str, tuple[str, str]] = {}
    for tn in tax_nodes:
        slug = _slug(tn.name)
        dim = tn.dimension
        if dim in ("task", "subtask"):
            node_by_slug[f"T__{slug}"] = ("taxonomy_node", str(tn.id))
        elif dim == "mechanism":
            node_by_slug[f"C__{slug}"] = ("taxonomy_node", str(tn.id))
        elif dim in ("dataset", "benchmark"):
            node_by_slug[f"D__{slug}"] = ("taxonomy_node", str(tn.id))
        elif dim == "lab":
            node_by_slug[f"Lab__{slug}"] = ("taxonomy_node", str(tn.id))

    # Method nodes for slug mapping
    try:
        method_nodes = (await session.execute(text(
            "SELECT id, name FROM method_nodes ORDER BY name"
        ))).fetchall()
        for mn in method_nodes:
            node_by_slug[f"M__{_slug(mn.name)}"] = ("method_node", str(mn.id))
    except Exception:
        pass

    # Paper slug mapping
    for pid, san in id_to_san.items():
        if san:
            node_by_slug[f"P__{san}"] = ("paper", pid)

    # ── Step 4 & 5: Enhance existing pages with profiles + edge one-liners ──
    if node_profile_map or edge_profile_map:
        md_files = list(root.rglob("*.md"))
        for md_path in md_files:
            try:
                stem = md_path.stem
                info = _detect_node_key(md_path.name)
                if not info:
                    continue

                prefix, entity_type, slug = info
                content = md_path.read_text(encoding="utf-8")
                original = content

                # --- Identify current node for edge lookups ---
                current_node_key = node_by_slug.get(stem)  # (entity_type, entity_id)

                # --- Inject short_intro_md after first heading ---
                node_key_candidates = []
                if current_node_key:
                    node_key_candidates.append(current_node_key)
                # Also try entity_type from prefix
                node_key_candidates.append((entity_type, slug))

                profile = None
                for nk in node_key_candidates:
                    profile = node_profile_map.get(nk)
                    if profile:
                        break

                if profile and profile.short_intro_md:
                    # Insert after the first "# heading" line
                    lines = content.split("\n")
                    insert_idx = None
                    for i, line in enumerate(lines):
                        if line.startswith("# ") and not line.startswith("## "):
                            insert_idx = i + 1
                            break
                    if insert_idx is not None:
                        # Skip any existing blank lines right after heading
                        while insert_idx < len(lines) and lines[insert_idx].strip() == "":
                            insert_idx += 1
                        lines.insert(insert_idx, "")
                        lines.insert(insert_idx + 1, profile.short_intro_md)
                        lines.insert(insert_idx + 2, "")
                        content = "\n".join(lines)

                # --- Inject edge one-liners into wikilinks ---
                if current_node_key and edge_profile_map:
                    lines = content.split("\n")
                    new_lines = []
                    for line in lines:
                        # Skip lines that already have " — " (already annotated)
                        if " \u2014 " in line:
                            new_lines.append(line)
                            continue

                        # Find wikilinks in this line
                        match = _WIKILINK_RE.search(line)
                        if match:
                            wikilink_stem = f"{match.group(1)}__{match.group(2)}"
                            target_key = node_by_slug.get(wikilink_stem)
                            if target_key:
                                src_type, src_id = current_node_key
                                tgt_type, tgt_id = target_key
                                edge_key = (src_type, src_id, tgt_type, tgt_id)
                                one_liner = edge_profile_map.get(edge_key)
                                if not one_liner:
                                    # Try reverse direction
                                    edge_key_rev = (tgt_type, tgt_id, src_type, src_id)
                                    one_liner = edge_profile_map.get(edge_key_rev)
                                if one_liner:
                                    line = f"{line} \u2014 {one_liner}"
                        new_lines.append(line)
                    content = "\n".join(new_lines)

                if content != original:
                    md_path.write_text(content, encoding="utf-8")
                    profiles_injected += 1

            except Exception as e:
                logger.warning(f"Profile injection failed for {md_path.name}: {e}")

    # ── Step 6: Generate Lab__ pages ──
    lab_dir = root / "60_Labs"
    lab_dir.mkdir(parents=True, exist_ok=True)

    lab_nodes = [n for n in tax_nodes if n.dimension == "lab"]
    stats["labs"] = 0
    for ln in lab_nodes:
        try:
            slug = _slug(ln.name)
            profile = node_profile_map.get(("taxonomy_node", str(ln.id)))

            body = [f"# {ln.name}"]
            if ln.name_zh:
                body[0] += f" ({ln.name_zh})"
            body[0] += "\n"

            if profile and profile.short_intro_md:
                body.append(profile.short_intro_md + "\n")
            elif ln.description:
                body.append(ln.description + "\n")

            # Papers from this lab
            lab_paper_ids = [str(pf.paper_id) for pf in paper_facets
                            if str(pf.node_id) == str(ln.id)]
            if lab_paper_ids:
                body.append(f"## \u4ee3\u8868\u8bba\u6587 ({len(lab_paper_ids)})\n")
                for pid in lab_paper_ids[:15]:
                    san = id_to_san.get(pid)
                    if san:
                        edge_key = ("taxonomy_node", str(ln.id), "paper", pid)
                        one_liner = edge_profile_map.get(edge_key, "")
                        suffix = f" \u2014 {one_liner}" if one_liner else ""
                        body.append(f"- [[P__{san}]]{suffix}")
                body.append("")

            fm_data = {"title": ln.name, "type": "lab", "name_zh": ln.name_zh}
            (lab_dir / f"Lab__{slug}.md").write_text(
                _fm(fm_data) + "\n".join(body), encoding="utf-8")
            stats["labs"] += 1
        except Exception as e:
            logger.warning(f"Lab page generation failed for {ln.name}: {e}")

    # ── Step 7: Update 00_Home with Labs section ──
    if stats["labs"] > 0:
        home_path = root / "00_Home" / "00_\u65b9\u5411\u603b\u89c8.md"
        try:
            if home_path.exists():
                home_content = home_path.read_text(encoding="utf-8")
                if "## \u7814\u7a76\u56e2\u961f" not in home_content:
                    lab_section = ["\n## \u7814\u7a76\u56e2\u961f\n"]
                    for ln in lab_nodes[:10]:
                        lab_section.append(f"- [[Lab__{_slug(ln.name)}]]")
                    lab_section.append("")
                    home_content += "\n".join(lab_section)
                    home_path.write_text(home_content, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Home page update failed: {e}")

    # ── Step 8: Return updated stats ──
    stats["profiles_injected"] = profiles_injected

    logger.info(f"Obsidian vault v6 exported to {root}: {stats}")
    return stats
