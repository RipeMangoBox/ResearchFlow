"""Obsidian Vault Export v5 — Task/Method/Mechanism/Paper/Dataset structure.

Replaces v4's Lineage/Concept/Bottleneck with:
  10_Tasks/    — T__ nodes, Common Problems embedded
  20_Methods/  — M__ nodes with evolution DAG (Mermaid)
  30_Mechanisms/ — C__ reusable technique concepts
  40_Papers/   — P__ with A/B/C/D tiers (unchanged)
  50_Datasets/ — D__ benchmarks and datasets (new)
  80_Assets/   — figures from OSS
  90_Views/    — tables (by task, by year, method evolution)

Paper wikilink budget: 6-10 (T + M + C + D + P links only)
Other facets go in frontmatter YAML, not as wikilinks.
"""

import logging
import shutil
from pathlib import Path
from uuid import UUID

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


async def export_vault_v5(session: AsyncSession, out_dir: str | None = None) -> dict:
    """Export knowledge base as Obsidian vault v5."""

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
               p.paper_link, p.code_url, p.mechanism_family, p.ring,
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

    # Legacy data (mechanisms, bottlenecks, lineage) for backward compat
    mechanisms = []
    try:
        mechanisms = (await session.execute(text(
            "SELECT id, name, domain, description FROM mechanism_families ORDER BY name"
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

    # Evidence units → build paper-to-paper links via shared evidence themes
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
    children_of = {}  # parent_id → [child_id]
    for e in tax_edges:
        children_of.setdefault(str(e.parent_id), []).append(str(e.child_id))

    # Paper → facets
    paper_facet_map = {}  # paper_id → [{role, node_name, node_name_zh}]
    for pf in paper_facets:
        node = node_by_id.get(str(pf.node_id))
        if node:
            paper_facet_map.setdefault(str(pf.paper_id), []).append({
                "role": pf.facet_role, "name": node.name, "name_zh": node.name_zh,
                "dimension": node.dimension,
            })

    # Method lookups
    method_by_id = {str(m.id): m for m in method_nodes}
    method_slot_map = {}  # method_id → [slot_name]
    for s in method_slots:
        method_slot_map.setdefault(str(s.method_id), []).append(s.slot_name)
    method_app_map = {}  # paper_id → [{method_name, role}]
    for a in method_apps:
        m = method_by_id.get(str(a.method_id))
        if m:
            method_app_map.setdefault(str(a.paper_id), []).append({
                "method": m.name, "role": a.role,
            })

    # Problem → claims
    problem_claim_map = {}  # problem_id → [paper_id]
    for pc in problem_claims:
        problem_claim_map.setdefault(str(pc.problem_id), []).append(str(pc.paper_id))

    # Lineage maps
    lineage_parents = {}  # child_san → [(rel, parent_san)]
    lineage_children = {}
    for lr in lineage_rows:
        lineage_parents.setdefault(lr.child_san, []).append((lr.relation_type, lr.parent_san))
        lineage_children.setdefault(lr.parent_san, []).append((lr.relation_type, lr.child_san))

    # Evidence-based inter-paper links
    evidence_links_out = {}  # paper_san → [related_paper_san]
    evidence_links_in = {}
    for el in evidence_paper_links:
        if el.paper_san and el.dc_paper_san:
            evidence_links_out.setdefault(el.paper_san, set()).add(el.dc_paper_san)
            evidence_links_in.setdefault(el.dc_paper_san, set()).add(el.paper_san)

    # Build same-mechanism paper groups for cross-linking
    mech_family_papers = {}  # mechanism_family → [title_sanitized]
    for p in papers:
        if p.mechanism_family:
            mech_family_papers.setdefault(p.mechanism_family, []).append(p.title_sanitized)

    # Paper → mechanism (legacy)
    mech_papers = {}
    for p in papers:
        if p.mechanism_family:
            mech_papers.setdefault(p.mechanism_family, []).append(p)

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

        # Subtasks
        child_ids = children_of.get(str(tn.id), [])
        subtasks = [node_by_id[cid] for cid in child_ids if cid in node_by_id
                    and node_by_id[cid].dimension in ("subtask", "task")]
        if subtasks:
            body.append("## 子任务\n")
            for st in subtasks:
                body.append(f"- [[T__{_slug(st.name)}]]")
            body.append("")

        # Common Problems (embedded, not separate notes)
        task_problems = [p for p in problems if str(p.parent_task_id) == str(tn.id)]
        if task_problems:
            body.append("## Common Problems\n")
            for prob in task_problems:
                body.append(f"### {prob.name}")
                if prob.symptom:
                    body.append(f"- **Symptoms**: {prob.symptom}")
                if prob.root_cause:
                    body.append(f"- **Root cause**: {prob.root_cause}")
                # Papers addressing this problem
                claim_pids = problem_claim_map.get(str(prob.id), [])
                if claim_pids:
                    body.append("- **Solutions**:")
                    for pid in claim_pids[:5]:
                        san = id_to_san.get(pid)
                        if san:
                            body.append(f"  - [[P__{san}]]")
                body.append("")

        # Papers tagged with this task (from paper_facets)
        task_paper_ids = [str(pf.paper_id) for pf in paper_facets
                         if str(pf.node_id) == str(tn.id)]
        # Also match papers by keyword overlap with task name / aliases
        task_keywords = set(tn.name.lower().replace("_", " ").replace("-", " ").split())
        task_keywords.discard("")
        # Add alias keywords
        if tn.aliases:
            for alias in (tn.aliases if isinstance(tn.aliases, list) else []):
                task_keywords.update(alias.lower().replace("_", " ").replace("-", " ").split())
        # Remove common stop words
        task_keywords -= {"a", "an", "the", "for", "of", "in", "on", "and", "or", "to", "with"}

        if task_keywords and len(task_paper_ids) < 5:
            for pp in papers:
                pid = str(pp.id)
                if pid in task_paper_ids:
                    continue
                # Check title keyword overlap (at least 2 keywords match)
                title_words = set((pp.title or "").lower().split())
                overlap = task_keywords & title_words
                if len(overlap) >= 2:
                    task_paper_ids.append(pid)
                    continue
                # Check category match
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

        # Slots
        slots = method_slot_map.get(str(mn.id), [])
        if slots:
            body.append("## Slots\n")
            for s in slots:
                body.append(f"- `{s}`")
            body.append("")

        # Evolution DAG (Mermaid)
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

            # Text description of each edge
            for e in parent_edges:
                parent = method_by_id.get(str(e.parent_method_id))
                if parent and e.delta_description:
                    body.append(f"- ← [[M__{_slug(parent.name)}]]: {e.delta_description}")
            for e in child_edges:
                child = method_by_id.get(str(e.child_method_id))
                if child and e.delta_description:
                    body.append(f"- → [[M__{_slug(child.name)}]]: {e.delta_description}")
            body.append("")

        # Papers using this method
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
    # 30_Mechanisms/ (from legacy mechanism_families + concepts)
    # ═══════════════════════════════════════════════════════════

    mech_dir = root / "30_Mechanisms"
    mech_dir.mkdir(parents=True, exist_ok=True)

    # Use taxonomy mechanism nodes if available, else legacy mechanism_families
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

            # Papers with this mechanism as facet
            mech_pids = [str(pf.paper_id) for pf in paper_facets
                         if str(pf.node_id) == str(mn.id)]
            # Also match papers by mechanism_family name
            mech_name_lower = mn.name.lower().replace("_", " ").replace("-", " ")
            for pp in papers:
                pid = str(pp.id)
                if pid in mech_pids:
                    continue
                if pp.mechanism_family:
                    mf_lower = pp.mechanism_family.lower().replace("_", " ")
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
        # Fallback: use legacy mechanism_families
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

        # ── Build wikilinks (6-10 budget) ──
        links = []

        # Task links
        facets = paper_facet_map.get(pid_str, [])
        task_facets = [f for f in facets if f["dimension"] in ("task", "subtask")]
        for tf in task_facets[:2]:
            links.append(f"- Task: [[T__{_slug(tf['name'])}]]")

        # Method links
        apps = method_app_map.get(pid_str, [])
        for a in apps[:2]:
            links.append(f"- {a['role'].replace('_', ' ').title()}: [[M__{_slug(a['method'])}]]")

        # Mechanism links
        mech_facets = [f for f in facets if f["dimension"] == "mechanism"]
        for mf in mech_facets[:2]:
            links.append(f"- Mechanism: [[C__{_slug(mf['name'])}]]")

        # Lineage links (from delta_card_lineage)
        ancestors = lineage_parents.get(san, [])
        for rel, parent_san in ancestors[:2]:
            links.append(f"- builds on: [[P__{parent_san}]]")
        descendants = lineage_children.get(san, [])
        for rel, child_san in descendants[:1]:
            links.append(f"- extended by: [[P__{child_san}]]")

        # Evidence-based inter-paper links
        ev_out = evidence_links_out.get(san, set())
        for related_san in list(ev_out)[:3]:
            links.append(f"- evidence links to: [[P__{related_san}]]")
        ev_in = evidence_links_in.get(san, set())
        for related_san in list(ev_in)[:3]:
            links.append(f"- evidence from: [[P__{related_san}]]")

        # Fallback: mechanism_family as concept link
        if not mech_facets and p.mechanism_family:
            links.append(f"- Mechanism: [[C__{_slug(p.mechanism_family)}]]")

        # ── Frontmatter ──
        fm = {
            "title": p.title,
            "type": "paper",
            "paper_level": lvl,
            "venue": p.venue,
            "year": p.year,
            "acceptance": p.acceptance_type,
            "cited_by": p.cited_by_count,
        }
        # Facets as frontmatter (not wikilinks)
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

        # ── Body ──
        body = [f"# {p.title}\n"]

        # Links section
        if links:
            body.append("## Links\n")
            body.extend(links)
            body.append("")

        # Core insight callout
        if p.delta_statement:
            body.append(f"> {p.delta_statement[:300]}\n")

        # Reading advice
        body.append(_TIER_ADVICE[lvl] + "\n")

        # Key equations from DeltaCard
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

        # Key figures from DeltaCard
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

        # Analysis content
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

        # Export figures to 80_Assets/
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

        # Papers using this dataset
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

    # Papers by year
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

    # Method evolution
    if method_nodes:
        lines = ["# 方法演化\n",
                 "| 方法 | Type | Maturity | Downstream | Canonical Paper |",
                 "|------|------|----------|-----------|----------------|"]
        for m in method_nodes:
            cp_san = id_to_san.get(str(m.canonical_paper_id), "") if m.canonical_paper_id else ""
            cp_link = f"[[P__{cp_san}]]" if cp_san else "—"
            lines.append(f"| [[M__{_slug(m.name)}]] | {m.type} | {m.maturity} | {m.downstream_count} | {cp_link} |")
        (views_dir / "method_evolution.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"Obsidian vault v5 exported to {root}: {stats}")
    return {"vault_path": str(root), **stats}
