"""Obsidian Vault Export — 4-layer knowledge graph → Obsidian wiki.

Directory structure:
  00_Home/        — overview dashboard
  domain/         — domain directories with T__ task pages inside
  method/         — M__ method/mechanism pages (flat)
  dataset/        — D__ dataset pages
  paper/          — P__ papers organized by venue_year/
  assets/figures/ — extracted figure images
  views/          — aggregated tables

Page types (first phase):
  T__xxx.md — Task page (definition, challenges, methods, datasets, papers)
  M__xxx.md — Method page (core idea, slots, variants, papers)
  D__xxx.md — Dataset page (scale, modalities, papers)
  P__xxx.md — Paper page (10-section structured report)

Data flow: DB → vault_export → Obsidian (read-only projection)
"""

import logging
import shutil
from pathlib import Path

import yaml
from sqlalchemy import text, select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────

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


def _venue_year(venue: str | None, year: int | None) -> str:
    """Build venue_year directory name."""
    v = _slug(venue or "Unknown", 30)
    y = str(year) if year else "Unknown"
    return f"{v}_{y}"


# ── Main Export ──────────────────────────────────────────────────

async def export_vault(session: AsyncSession, vault_dir: str | None = None) -> dict:
    """Export the full Obsidian vault from DB state.

    Returns stats dict with counts per section.
    """
    vault = Path(vault_dir or settings.obsidian_vault_dir)
    if vault.exists():
        shutil.rmtree(vault)
    vault.mkdir(parents=True, exist_ok=True)

    stats = {"domains": 0, "tasks": 0, "methods": 0, "datasets": 0, "papers": 0}

    # Load all data first
    data = await _load_all_data(session)

    # Generate pages
    await _export_domains_and_tasks(vault, data, stats)
    await _export_methods(vault, data, stats)
    await _export_datasets(vault, data, stats)
    await _export_papers(vault, session, data, stats)
    _export_views(vault, data, stats)
    _export_home(vault, data, stats)

    logger.info(
        "Vault exported to %s: %d domains, %d tasks, %d methods, %d datasets, %d papers",
        vault, stats["domains"], stats["tasks"], stats["methods"],
        stats["datasets"], stats["papers"],
    )
    return stats


# ── Data Loading ─────────────────────────────────────────────────

async def _load_all_data(session: AsyncSession) -> dict:
    """Load all data needed for vault export in bulk queries."""

    # Papers with DeltaCard + Analysis
    papers_raw = (await session.execute(text("""
        SELECT p.id, p.title, p.title_sanitized, p.venue, p.year, p.category,
               p.tags, p.paper_link, p.code_url, p.method_family, p.ring,
               p.acceptance_type, p.cited_by_count, p.authors, p.abstract,
               p.current_delta_card_id,
               dc.delta_statement, dc.baseline_paradigm,
               dc.structurality_score AS dc_struct, dc.transferability_score,
               dc.key_ideas_ranked, dc.key_equations, dc.key_figures,
               dc.publish_status, dc.changed_slots_json, dc.is_structural,
               a.problem_summary, a.method_summary, a.evidence_summary,
               a.core_intuition, a.full_report_md, a.changed_slots
        FROM papers p
        LEFT JOIN delta_cards dc ON dc.id = p.current_delta_card_id
        LEFT JOIN paper_analyses a ON a.paper_id = p.id
            AND a.is_current = true AND a.level = 'L4_DEEP'
        WHERE p.state != 'wait'
        ORDER BY p.year DESC NULLS LAST, p.title
    """))).fetchall()

    # Taxonomy nodes
    tax_nodes = (await session.execute(text(
        "SELECT id, name, name_zh, dimension, description, status FROM taxonomy_nodes ORDER BY dimension, name"
    ))).fetchall()

    # Taxonomy edges
    tax_edges = (await session.execute(text(
        "SELECT parent_id, child_id, relation_type FROM taxonomy_edges"
    ))).fetchall()

    # Paper facets
    facets = (await session.execute(text(
        "SELECT paper_id, node_id, facet_role FROM paper_facets"
    ))).fetchall()

    # Method nodes
    methods = (await session.execute(text("""
        SELECT id, name, name_zh, type, domain, description, maturity,
               downstream_count, parent_method_id, canonical_paper_id, aliases
        FROM method_nodes ORDER BY downstream_count DESC NULLS LAST, name
    """))).fetchall()

    # Method edges
    method_edges = (await session.execute(text(
        "SELECT parent_method_id, child_method_id, relation_type, delta_description FROM method_edges"
    ))).fetchall()

    # KB profiles
    profiles = (await session.execute(text(
        "SELECT entity_type, entity_id, one_liner, short_intro_md FROM kb_node_profiles WHERE lang = 'zh'"
    ))).fetchall()

    # Build lookup dicts
    tax_by_id = {str(n.id): n for n in tax_nodes}
    tax_by_dim = {}
    for n in tax_nodes:
        tax_by_dim.setdefault(n.dimension, []).append(n)

    # Paper facets: paper_id → [{node_id, role}]
    paper_facets = {}
    for f in facets:
        paper_facets.setdefault(str(f.paper_id), []).append({
            "node_id": str(f.node_id), "role": f.facet_role,
        })

    # Facet reverse: node_id → [paper_ids]
    node_papers = {}
    for f in facets:
        node_papers.setdefault(str(f.node_id), []).append(str(f.paper_id))

    # Papers by ID
    papers_by_id = {str(p.id): p for p in papers_raw}

    # Profiles by entity
    profile_map = {}
    for pr in profiles:
        profile_map[f"{pr.entity_type}:{pr.entity_id}"] = pr

    # Methods by ID
    methods_by_id = {str(m.id): m for m in methods}

    # Domain → tasks mapping (from taxonomy_edges)
    domain_tasks = {}
    for e in tax_edges:
        parent = tax_by_id.get(str(e.parent_id))
        child = tax_by_id.get(str(e.child_id))
        if parent and child and parent.dimension == "domain" and child.dimension == "task":
            domain_tasks.setdefault(str(e.parent_id), []).append(str(e.child_id))

    return {
        "papers": papers_raw,
        "papers_by_id": papers_by_id,
        "tax_nodes": tax_nodes,
        "tax_by_id": tax_by_id,
        "tax_by_dim": tax_by_dim,
        "paper_facets": paper_facets,
        "node_papers": node_papers,
        "methods": methods,
        "methods_by_id": methods_by_id,
        "method_edges": method_edges,
        "profile_map": profile_map,
        "domain_tasks": domain_tasks,
    }


# ── Domain & Task Pages ─────────────────────────────────────────

async def _export_domains_and_tasks(vault: Path, data: dict, stats: dict):
    """Export domain/_overview.md + domain/T__task.md pages."""
    domain_dir = vault / "domain"
    domain_dir.mkdir(exist_ok=True)

    domains = data["tax_by_dim"].get("domain", [])
    tasks = data["tax_by_dim"].get("task", [])

    # Group tasks by domain (via taxonomy_edges or paper.category fallback)
    domain_task_map = {}  # domain_name → [task_nodes]

    # From explicit edges
    for domain in domains:
        child_ids = data["domain_tasks"].get(str(domain.id), [])
        for cid in child_ids:
            task = data["tax_by_id"].get(cid)
            if task and task.dimension == "task":
                domain_task_map.setdefault(domain.name, []).append(task)

    # Orphan tasks (no parent edge): assign to "Uncategorized"
    assigned_task_ids = {str(t.id) for ts in domain_task_map.values() for t in ts}
    for task in tasks:
        if str(task.id) not in assigned_task_ids:
            domain_task_map.setdefault("Uncategorized", []).append(task)

    # Also create domains from paper.category if no taxonomy domains exist
    if not domains:
        categories = set()
        for p in data["papers"]:
            if p.category:
                categories.add(p.category)
        for cat in sorted(categories):
            domain_task_map.setdefault(cat, [])

    for domain_name, domain_tasks in sorted(domain_task_map.items()):
        dslug = _slug(domain_name)
        dpath = domain_dir / dslug
        dpath.mkdir(exist_ok=True)

        # Domain overview
        overview = _fm({"title": domain_name, "type": "domain"})
        overview += f"# {domain_name}\n\n"

        # Count papers in this domain
        domain_paper_count = sum(
            1 for p in data["papers"] if p.category == domain_name
        )
        overview += f"论文数: {domain_paper_count}\n\n"

        if domain_tasks:
            overview += "## 任务列表\n\n"
            for t in domain_tasks:
                tslug = _slug(t.name)
                paper_count = len(data["node_papers"].get(str(t.id), []))
                overview += f"- [[T__{tslug}]] ({paper_count} 篇)\n"

        (dpath / "_overview.md").write_text(overview, encoding="utf-8")
        stats["domains"] += 1

        # Task pages
        for task in domain_tasks:
            _write_task_page(dpath, task, domain_name, data)
            stats["tasks"] += 1


def _write_task_page(parent_dir: Path, task, domain_name: str, data: dict):
    """Write a T__xxx.md task page."""
    tslug = _slug(task.name)
    profile = data["profile_map"].get(f"taxonomy_node:{task.id}")

    fm = _fm({
        "title": task.name,
        "type": "task",
        "domain": domain_name,
        "name_zh": task.name_zh,
    })

    body = f"# {task.name}"
    if task.name_zh:
        body += f" ({task.name_zh})"
    body += "\n\n"

    # Profile intro
    if profile and profile.short_intro_md:
        body += profile.short_intro_md + "\n\n"
    elif task.description:
        body += task.description + "\n\n"

    # Papers
    paper_ids = data["node_papers"].get(str(task.id), [])
    if paper_ids:
        body += f"## 论文列表 ({len(paper_ids)})\n\n"
        body += "| 论文 | 会议 | 年份 | 核心贡献 |\n|------|------|------|----------|\n"
        for pid in paper_ids[:50]:
            p = data["papers_by_id"].get(pid)
            if p:
                pslug = _slug(p.title_sanitized or p.title)
                venue = p.venue or ""
                year = p.year or ""
                contribution = (p.delta_statement or "")[:60]
                body += f"| [[P__{pslug}]] | {venue} | {year} | {contribution} |\n"

    (parent_dir / f"T__{tslug}.md").write_text(fm + body, encoding="utf-8")


# ── Method Pages ─────────────────────────────────────────────────

async def _export_methods(vault: Path, data: dict, stats: dict):
    """Export method/M__xxx.md pages."""
    method_dir = vault / "method"
    method_dir.mkdir(exist_ok=True)

    for m in data["methods"]:
        mslug = _slug(m.name)
        profile = data["profile_map"].get(f"method_node:{m.id}")

        # Find parent method
        parent_name = None
        if m.parent_method_id:
            parent = data["methods_by_id"].get(str(m.parent_method_id))
            if parent:
                parent_name = parent.name

        fm = _fm({
            "title": m.name,
            "type": "method",
            "method_type": m.type,
            "maturity": m.maturity,
            "domain": m.domain,
            "parent_method": parent_name,
            "downstream_count": m.downstream_count or 0,
        })

        body = f"# {m.name}"
        if m.name_zh:
            body += f" ({m.name_zh})"
        body += "\n\n"

        # Profile intro
        if profile and profile.short_intro_md:
            body += profile.short_intro_md + "\n\n"
        elif m.description:
            body += m.description + "\n\n"

        # Lineage
        if parent_name:
            body += "## 所属谱系\n\n"
            body += f"- 父方法: [[M__{_slug(parent_name)}]]\n"
            if m.domain:
                body += f"- 领域: {m.domain}\n"
            body += "\n"

        # Children (variants)
        children = [
            data["methods_by_id"].get(str(e.child_method_id))
            for e in data["method_edges"]
            if str(e.parent_method_id) == str(m.id)
        ]
        children = [c for c in children if c]
        if children:
            body += "## 变体\n\n"
            for c in children:
                body += f"- [[M__{_slug(c.name)}]] — {c.description or ''}\n"
            body += "\n"

        # Papers using this method (via method_family match)
        method_papers = [
            p for p in data["papers"]
            if p.method_family and p.method_family.lower() == m.name.lower()
        ]
        if method_papers:
            body += f"## 使用本方法的论文 ({len(method_papers)})\n\n"
            body += "| 论文 | 会议 | 年份 |\n|------|------|------|\n"
            for p in method_papers[:30]:
                pslug = _slug(p.title_sanitized or p.title)
                body += f"| [[P__{pslug}]] | {p.venue or ''} | {p.year or ''} |\n"

        (method_dir / f"M__{mslug}.md").write_text(fm + body, encoding="utf-8")
        stats["methods"] += 1


# ── Dataset Pages ────────────────────────────────────────────────

async def _export_datasets(vault: Path, data: dict, stats: dict):
    """Export dataset/D__xxx.md pages."""
    ds_dir = vault / "dataset"
    ds_dir.mkdir(exist_ok=True)

    datasets = data["tax_by_dim"].get("dataset", [])
    for ds in datasets:
        dslug = _slug(ds.name)
        profile = data["profile_map"].get(f"taxonomy_node:{ds.id}")

        fm = _fm({
            "title": ds.name,
            "type": "dataset",
            "name_zh": ds.name_zh,
        })

        body = f"# {ds.name}\n\n"
        if profile and profile.short_intro_md:
            body += profile.short_intro_md + "\n\n"
        elif ds.description:
            body += ds.description + "\n\n"

        # Papers using this dataset
        paper_ids = data["node_papers"].get(str(ds.id), [])
        if paper_ids:
            body += f"## 使用本数据集的论文 ({len(paper_ids)})\n\n"
            for pid in paper_ids[:30]:
                p = data["papers_by_id"].get(pid)
                if p:
                    pslug = _slug(p.title_sanitized or p.title)
                    body += f"- [[P__{pslug}]] ({p.venue or ''} {p.year or ''})\n"

        (ds_dir / f"D__{dslug}.md").write_text(fm + body, encoding="utf-8")
        stats["datasets"] += 1


# ── Paper Pages ──────────────────────────────────────────────────

async def _export_papers(vault: Path, session: AsyncSession, data: dict, stats: dict):
    """Export paper/venue_year/P__xxx.md pages."""
    paper_dir = vault / "paper"
    paper_dir.mkdir(exist_ok=True)

    for p in data["papers"]:
        # Determine venue_year directory
        vy = _venue_year(p.venue, p.year)
        vy_dir = paper_dir / vy
        vy_dir.mkdir(exist_ok=True)

        pslug = _slug(p.title_sanitized or p.title)

        # Build facet wikilinks
        facets = data["paper_facets"].get(str(p.id), [])
        task_links = []
        method_links = []
        dataset_links = []
        for f in facets:
            node = data["tax_by_id"].get(f["node_id"])
            if not node:
                continue
            if node.dimension == "task":
                task_links.append(f"[[T__{_slug(node.name)}]]")
            elif node.dimension == "dataset":
                dataset_links.append(f"[[D__{_slug(node.name)}]]")

        if p.method_family:
            method_links.append(f"[[M__{_slug(p.method_family)}]]")

        # Frontmatter
        fm = _fm({
            "title": p.title,
            "type": "paper",
            "venue": p.venue,
            "year": p.year,
            "acceptance": p.acceptance_type,
            "cited_by": p.cited_by_count,
            "structurality_score": p.dc_struct,
            "paper_link": p.paper_link,
            "code_url": p.code_url,
            "tasks": [l.strip("[]") for l in task_links] if task_links else None,
            "method": p.method_family,
            "datasets": [l.strip("[]") for l in dataset_links] if dataset_links else None,
        })

        body = f"# {p.title}\n\n"

        # Links section
        links = []
        if p.paper_link:
            links.append(f"[Paper]({p.paper_link})")
        if p.code_url:
            links.append(f"[Code]({p.code_url})")
        if links:
            body += " | ".join(links) + "\n\n"

        # Wikilinks
        if task_links or method_links or dataset_links:
            body += "**Tasks**: " + ", ".join(task_links) if task_links else ""
            if method_links:
                body += (" | " if task_links else "") + "**Method**: " + ", ".join(method_links)
            if dataset_links:
                body += " | **Datasets**: " + ", ".join(dataset_links)
            body += "\n\n"

        # Delta statement
        if p.delta_statement:
            body += f"> {p.delta_statement}\n\n"

        # Full report OR structured sections
        if p.full_report_md:
            body += p.full_report_md + "\n"
        else:
            if p.problem_summary:
                body += f"## 研究动机\n\n{p.problem_summary}\n\n"
            if p.method_summary:
                body += f"## 方法概述\n\n{p.method_summary}\n\n"
            if p.core_intuition:
                body += f"## 核心直觉\n\n{p.core_intuition}\n\n"
            if p.evidence_summary:
                body += f"## 证据与局限\n\n{p.evidence_summary}\n\n"

        # Key equations
        if p.key_equations and isinstance(p.key_equations, list):
            body += "## 关键公式\n\n"
            for eq in p.key_equations[:5]:
                if isinstance(eq, dict):
                    body += f"$$\n{eq.get('latex', '')}\n$$\n"
                    if eq.get("explanation_zh") or eq.get("explanation"):
                        body += f"\n{eq.get('explanation_zh') or eq.get('explanation')}\n\n"

        (vy_dir / f"P__{pslug}.md").write_text(fm + body, encoding="utf-8")
        stats["papers"] += 1


# ── Views ────────────────────────────────────────────────────────

def _export_views(vault: Path, data: dict, stats: dict):
    """Export views/ aggregated tables."""
    views_dir = vault / "views"
    views_dir.mkdir(exist_ok=True)

    # Papers by year
    body = "# Papers by Year\n\n"
    body += "| Paper | Year | Venue | Task | Method |\n|-------|------|-------|------|--------|\n"
    for p in sorted(data["papers"], key=lambda x: -(x.year or 0)):
        pslug = _slug(p.title_sanitized or p.title)
        # Get primary task from facets
        facets = data["paper_facets"].get(str(p.id), [])
        task_name = ""
        for f in facets:
            node = data["tax_by_id"].get(f["node_id"])
            if node and node.dimension == "task":
                task_name = node.name
                break
        body += (
            f"| [[P__{pslug}]] | {p.year or ''} | {p.venue or ''} "
            f"| {task_name} | {p.method_family or ''} |\n"
        )
    (views_dir / "papers_by_year.md").write_text(body, encoding="utf-8")

    # Method evolution
    body = "# Method Evolution\n\n"
    body += "| Method | Type | Maturity | Downstream | Domain |\n|--------|------|----------|------------|--------|\n"
    for m in data["methods"]:
        mslug = _slug(m.name)
        body += (
            f"| [[M__{mslug}]] | {m.type} | {m.maturity} "
            f"| {m.downstream_count or 0} | {m.domain or ''} |\n"
        )
    (views_dir / "method_evolution.md").write_text(body, encoding="utf-8")


# ── Home ─────────────────────────────────────────────────────────

def _export_home(vault: Path, data: dict, stats: dict):
    """Export 00_Home/00_方向总览.md overview page."""
    home_dir = vault / "00_Home"
    home_dir.mkdir(exist_ok=True)

    body = "# 研究方向总览\n\n"
    body += f"- 论文: {stats['papers']} 篇\n"
    body += f"- 方法: {stats['methods']} 个\n"
    body += f"- 任务: {stats['tasks']} 个\n"
    body += f"- 数据集: {stats['datasets']} 个\n"
    body += f"- 领域: {stats['domains']} 个\n\n"

    # Top domains
    body += "## 领域\n\n"
    domains = data["tax_by_dim"].get("domain", [])
    if domains:
        for d in domains:
            dslug = _slug(d.name)
            paper_count = len(data["node_papers"].get(str(d.id), []))
            body += f"- [[domain/{dslug}/_overview|{d.name}]] ({paper_count} 篇)\n"
    else:
        # Fallback: list unique categories
        categories = {}
        for p in data["papers"]:
            if p.category:
                categories[p.category] = categories.get(p.category, 0) + 1
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            body += f"- {cat} ({count} 篇)\n"

    body += "\n## 最近论文\n\n"
    recent = sorted(data["papers"], key=lambda x: -(x.year or 0))[:10]
    for p in recent:
        pslug = _slug(p.title_sanitized or p.title)
        body += f"- [[P__{pslug}]] ({p.venue or ''} {p.year or ''})\n"

    (home_dir / "00_方向总览.md").write_text(body, encoding="utf-8")
