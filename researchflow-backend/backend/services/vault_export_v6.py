"""Obsidian Vault Export — knowledge graph → Obsidian wiki.

Design principles:
  1. NO big nodes — no page links to all papers
  2. D-level peripheral papers NOT exported
  3. Short aliases in frontmatter for readable graph view
  4. Figures embedded via OSS public URLs
  5. Venue names normalized (arXiv, not "arXiv (Cornell University)")
  6. No views/ pages — paper/ grouped by venue_year/ is sufficient
"""

import logging
import re
import shutil
from pathlib import Path

import yaml
from sqlalchemy import text, select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings

logger = logging.getLogger(__name__)

# ── Venue Normalization ────────────────────────────────────────

VENUE_NORMALIZE: dict[str, str] = {
    "arXiv (Cornell University)": "arXiv",
    "ArXiv.org": "arXiv",
    "arXiv.org": "arXiv",
    "Advances in Neural Information Processing Systems": "NeurIPS",
    "Advances in Neural Information Processing Systems 36": "NeurIPS",
    "Neural Information Processing Systems": "NeurIPS",
    "NIPS": "NeurIPS",
    "Computer Vision and Pattern Recognition": "CVPR",
    "International Conference on Learning Representations": "ICLR",
    "International Conference on Machine Learning": "ICML",
    "Association for Computational Linguistics": "ACL",
    "Cambridge University Press eBooks": "CUP",
    "Medical Image Anal.": "MedIA",
    "International Journal of Compu": "IJCV",
    "Trans. Mach. Learn. Res.": "TMLR",
    "The European Respiratory Society eBooks": "ERS",
    "Proceedings of the International": "AAAI",
    "Proceedings of the 2024 Conference": "EMNLP",
    "Open MIND": "OpenMIND",
}


def _normalize_venue(venue: str | None) -> str:
    if not venue:
        return "Unknown"
    if venue in VENUE_NORMALIZE:
        return VENUE_NORMALIZE[venue]
    for key, val in VENUE_NORMALIZE.items():
        if venue.startswith(key) or key.startswith(venue):
            return val
    # Strip trailing year
    clean = re.sub(r'\s*\d{4}\s*$', '', venue).strip()
    if clean in VENUE_NORMALIZE:
        return VENUE_NORMALIZE[clean]
    return venue


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


def _paper_acronym(p) -> str:
    """Best-effort English acronym for a paper.

    Priority: method_family → first 16 chars of method_family →
    initialism from the first capital-led tokens of the title.
    """
    method = (getattr(p, "method_family", None) or "").strip()
    if 2 < len(method) <= 16:
        return method
    if method:
        return method[:16]
    title = (getattr(p, "title", "") or "").strip()
    # Acronym: first letter of each capital-led token (e.g. "Diffusion Transformer" → "DT")
    caps = re.findall(r'\b([A-Z][A-Za-z0-9\-]{0,5})', title)
    if caps and len("".join(c[0] for c in caps[:6])) >= 2:
        return "".join(c[0] for c in caps[:6])
    # Fall back to first 16 chars of sanitized title
    return _slug(getattr(p, "title_sanitized", "") or title or "Untitled", 16)


def _paper_slug(p) -> str:
    """Short, readable slug for paper pages.

    Priority: title_zh + acronym → method + title context → title.
    Chinese characters are preserved (modern filesystems and Obsidian handle them).
    """
    title_zh = (getattr(p, "title_zh", None) or "").strip()
    if title_zh:
        zh = _slug(title_zh, 16)
        acr = _paper_acronym(p)
        return f"{zh}_{_slug(acr, 16)}" if acr else zh
    method = getattr(p, "method_family", None)
    if method and len(method) > 2:
        title_ctx = _slug(p.title_sanitized or p.title or "", 25)
        return f"{_slug(method, 30)}_{title_ctx}"
    return _slug(p.title_sanitized or p.title or "Untitled", 55)


def _paper_aliases(p) -> list[str]:
    """Build short aliases for the Obsidian graph view.

    Order: title_zh first (display priority), then acronym, then method_family.
    Filters duplicates and items shorter than 2 chars.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _push(v):
        if not v:
            return
        v = v.strip()
        if len(v) < 2 or v in seen:
            return
        seen.add(v)
        out.append(v)

    _push(getattr(p, "title_zh", None))
    _push(_paper_acronym(p))
    _push(getattr(p, "method_family", None))
    delta = getattr(p, "delta_statement", None)
    if delta:
        short = delta.split("，")[0].split(",")[0][:35]
        _push(short)
    return out[:4]


def _venue_year(venue: str | None, year: int | None) -> str:
    """Build venue_year directory name with normalized venue."""
    v = _slug(_normalize_venue(venue), 30)
    y = str(year) if year else "Unknown"
    return f"{v}_{y}"


def _is_arxiv_id_title(title: str | None) -> bool:
    """Check if title is just an arXiv ID placeholder."""
    return bool(title and re.match(r'^\d{4}\.\d{4,5}', title))


def _is_paper_exported(p) -> bool:
    """A paper renders as P__xxx.md only if it survives both export filters."""
    if p is None:
        return False
    if _paper_level(p) == "D":
        return False
    if _is_arxiv_id_title(getattr(p, "title", None)):
        return False
    return True


def _method_canonical(name: str) -> str:
    """Strip parenthetical expansion: 'AFL (Analytic Federated Learning)' → 'AFL'.

    Single source of truth — both `_export_methods` (which writes the file)
    and `_export_papers` (which links to it) MUST use this same fn, otherwise
    paper-link slugs and method-file slugs diverge and methods become orphans.
    """
    s = (name or "").strip()
    s = re.sub(r"\s*\(.*\)\s*$", "", s)
    return s.strip() or name


def _summarize_year_distribution(years) -> str:
    """Render a compact year histogram like '2024 (12) · 2025 (8)'."""
    from collections import Counter
    valid = [int(y) for y in years if y]
    if not valid:
        return ""
    c = Counter(valid)
    return " · ".join(f"{y} ({n})" for y, n in sorted(c.items()))


def _summarize_venue_distribution(venues, top: int = 5) -> str:
    """Render a compact venue histogram, normalized."""
    from collections import Counter
    norm = [_normalize_venue(v) for v in venues if v]
    if not norm:
        return ""
    c = Counter(norm)
    items = c.most_common(top)
    rest = sum(n for _, n in c.most_common()[top:])
    out = " · ".join(f"{v} ({n})" for v, n in items)
    if rest:
        out += f" · 其他 ({rest})"
    return out


# ── Main Export ──────────────────────────────────────────────────

async def export_vault(session: AsyncSession, vault_dir: str | None = None) -> dict:
    """Export the full Obsidian vault from DB state.

    Returns stats dict with counts per section.
    """
    vault_dir = vault_dir or getattr(settings, "obsidian_vault_dir", None) or "/obsidian-vault"
    vault = Path(vault_dir)
    # Clear CONTENTS instead of removing the directory itself, because the
    # production vault is a bind-mount and rmtree fails with EBUSY.
    if vault.exists():
        for child in vault.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
    else:
        vault.mkdir(parents=True, exist_ok=True)

    stats = {"domains": 0, "tasks": 0, "methods": 0, "datasets": 0, "papers": 0}

    # Load all data first
    data = await _load_all_data(session)

    # Generate pages (no views — paper/ by venue_year/ is sufficient)
    await _export_domains_and_tasks(vault, data, stats)
    await _export_methods(vault, data, stats)
    await _export_datasets(vault, data, stats)
    await _export_papers(vault, session, data, stats)
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

    # Papers with DeltaCard + Analysis + latest PaperReport (for title_zh)
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
               a.core_intuition, a.full_report_md, a.changed_slots,
               pr.title_zh
        FROM papers p
        LEFT JOIN delta_cards dc ON dc.id = p.current_delta_card_id
        LEFT JOIN paper_analyses a ON a.paper_id = p.id
            AND a.is_current = true AND a.level = 'l4_deep'
        LEFT JOIN LATERAL (
            SELECT title_zh FROM paper_reports
            WHERE paper_id = p.id
            ORDER BY created_at DESC LIMIT 1
        ) pr ON true
        WHERE p.state NOT IN ('wait', 'skip', 'archived_or_expired')
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

    # Figure images from L2 parse
    figure_rows = (await session.execute(text("""
        SELECT paper_id, extracted_figure_images
        FROM paper_analyses
        WHERE level = 'l2_parse' AND is_current = true
          AND extracted_figure_images IS NOT NULL
    """))).fetchall()

    # Paper-paper relations (baseline / cite DAG). Optional table — may not
    # exist on pre-024 deployments, so swallow the error.
    relation_rows = []
    try:
        relation_rows = (await session.execute(text("""
            SELECT source_paper_id, target_paper_id, relation_type,
                   evidence, confidence, ref_title_raw
            FROM paper_relations
        """))).fetchall()
    except Exception:
        pass

    # Build lookup dicts
    figure_map: dict[str, list] = {}
    for fr in figure_rows:
        figs = fr.extracted_figure_images
        if isinstance(figs, list):
            figure_map[str(fr.paper_id)] = figs
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

    # Paper relations: source → list of (target_pid, relation_type, evidence, raw_title)
    # and reverse: target → list of (source_pid, relation_type, evidence)
    relations_out: dict[str, list[dict]] = {}
    relations_in: dict[str, list[dict]] = {}
    for r in relation_rows:
        src, tgt = str(r.source_paper_id), str(r.target_paper_id)
        relations_out.setdefault(src, []).append({
            "target": tgt,
            "type": r.relation_type,
            "evidence": r.evidence or "",
            "ref_title_raw": r.ref_title_raw or "",
        })
        relations_in.setdefault(tgt, []).append({
            "source": src,
            "type": r.relation_type,
            "evidence": r.evidence or "",
        })

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

    # Identify parent task IDs (tasks that have children in taxonomy_edges)
    # Papers should link to leaf tasks only to avoid big parent nodes
    parent_task_ids: set[str] = set()
    for e in tax_edges:
        parent = tax_by_id.get(str(e.parent_id))
        child = tax_by_id.get(str(e.child_id))
        if parent and child and parent.dimension == "task" and child.dimension == "task":
            parent_task_ids.add(str(e.parent_id))

    # Task children: parent_task_id → [child_task_ids]
    task_children: dict[str, list[str]] = {}
    for e in tax_edges:
        parent = tax_by_id.get(str(e.parent_id))
        child = tax_by_id.get(str(e.child_id))
        if parent and child and parent.dimension == "task" and child.dimension == "task":
            task_children.setdefault(str(e.parent_id), []).append(str(e.child_id))

    return {
        "papers": papers_raw,
        "papers_by_id": papers_by_id,
        "tax_nodes": tax_nodes,
        "tax_by_id": tax_by_id,
        "tax_by_dim": tax_by_dim,
        "paper_facets": paper_facets,
        "node_papers": node_papers,
        "relations_out": relations_out,
        "relations_in": relations_in,
        "methods": methods,
        "methods_by_id": methods_by_id,
        "method_edges": method_edges,
        "profile_map": profile_map,
        "domain_tasks": domain_tasks,
        "parent_task_ids": parent_task_ids,
        "task_children": task_children,
        "figure_map": figure_map,
    }


# ── Domain & Task Pages ─────────────────────────────────────────

async def _export_domains_and_tasks(vault: Path, data: dict, stats: dict):
    """Export taxonomy nodes as browsable pages.

    Groups all taxonomy nodes that have papers into categories.
    Creates T__ pages for any node with paper links (task, modality, paradigm, etc).
    """
    domain_dir = vault / "domain"
    domain_dir.mkdir(exist_ok=True)

    # Find taxonomy nodes with ≥1 non-peripheral paper, then bucket by
    # canonical name. Tasks use `canonicalize_task_name` (strips dataset
    # prefix), datasets use `canonicalize_facet_name`.
    from backend.services.facet_normalizer import (
        canonicalize_facet_name as _canon_ds,
        canonicalize_task_name as _canon_task,
    )
    # Names that are data-source / pipeline-stage tags rather than real
    # research topics — these should never appear in the research overview.
    _TAG_BLACKLIST = {
        "hf_daily", "hf-daily", "huggingface daily", "papers with code",
        "arxiv daily", "openreview", "data source", "to_review", "pending",
        "uncategorized", "general", "misc",
    }
    raw_nodes = []
    for node in data["tax_nodes"]:
        name = (node.name or "").strip()
        if not name or name.startswith("__") or name.lower() in _TAG_BLACKLIST:
            continue
        paper_ids = data["node_papers"].get(str(node.id), [])
        visible = [pid for pid in paper_ids
                    if pid in data["papers_by_id"] and _paper_level(data["papers_by_id"][pid]) != "D"]
        if visible:
            raw_nodes.append((node, name, visible))

    # Bucket by (dimension, canonical_name) — merge all variants into one node
    bucket: dict[tuple[str, str], dict] = {}
    for node, name, visible in raw_nodes:
        if node.dimension in ("task", "domain", "modality", "learning_paradigm", "mechanism"):
            cname = _canon_task(name)
        else:
            cname = _canon_ds(name, node.dimension)
        key = (node.dimension, cname)
        b = bucket.setdefault(key, {
            "canonical": cname,
            "dimension": node.dimension,
            "name_zh": node.name_zh,
            "description": node.description,
            "node_ids": [],
            "papers": set(),
            "primary_node": node,
        })
        b["node_ids"].append(str(node.id))
        b["papers"].update(visible)

    active_nodes = []
    for (dim, cname), b in bucket.items():
        # Synthesize a node-like object the rest of the code consumes
        class _N: pass
        _node = _N()
        _node.id = b["primary_node"].id
        _node.name = cname
        _node.name_zh = b["name_zh"]
        _node.dimension = dim
        _node.description = b["description"]
        _node.status = b["primary_node"].status
        data["node_papers"][str(_node.id)] = list(b["papers"])
        active_nodes.append((_node, len(b["papers"])))

    # Group by dimension
    dim_nodes: dict[str, list] = {}
    for node, count in active_nodes:
        dim_nodes.setdefault(node.dimension, []).append((node, count))

    # Sort each group by paper count descending
    for dim in dim_nodes:
        dim_nodes[dim].sort(key=lambda x: -x[1])

    # Write overview
    # NOTE: this page is a stats dashboard, not a graph hub. We render names
    # as plain text (no [[wikilinks]]) so it does NOT pull every T__/D__/M__
    # node into one giant hub in Obsidian's graph view. Use the folder
    # browser (domain/, dataset/, method/) or tag search for navigation.
    # `cssclasses: hub-page` lets the user CSS-hide it in graph if desired.
    overview = _fm({
        "title": "研究方向总览", "type": "domain",
        "cssclasses": ["hub-page", "no-graph"],
    })
    overview += "# 研究方向总览\n\n"
    overview += "> 这是一份**统计页**——研究节点 ([[T__]] / [[D__]] / [[M__]]) 通过 paper 页和文件夹浏览器互联，本页不参与图谱拓扑。\n\n"

    dim_label = {
        "domain": "领域", "task": "任务", "modality": "模态",
        "learning_paradigm": "学习范式", "dataset": "数据集",
        "mechanism": "机制", "subtask": "子任务",
        "constraint": "约束", "scenario": "场景",
        "model_family": "模型族", "method_baseline": "方法基线",
    }
    dim_order = ["task", "domain", "learning_paradigm", "modality",
                 "mechanism", "scenario", "constraint", "model_family"]
    for dim in dim_order:
        if dim not in dim_nodes:
            continue
        nodes = dim_nodes[dim]
        shown = [(n, c) for n, c in nodes if c >= 2]
        label = dim_label.get(dim, dim)
        overview += f"## {label} ({len(shown)})\n\n"
        if not shown:
            overview += "_无跨论文节点_\n\n"
            continue
        for node, count in shown[:60]:
            overview += f"- {node.name} ({count} 篇)\n"
        if len(shown) > 60:
            overview += f"- _… 另 {len(shown) - 60} 个_\n"
        overview += "\n"

    if "dataset" in dim_nodes:
        ds_nodes = [(n, c) for n, c in dim_nodes["dataset"] if c >= 2]
        overview += f"## 数据集 ({len(ds_nodes)})\n\n"
        if not ds_nodes:
            overview += "_无跨论文节点_\n\n"
        else:
            for node, count in ds_nodes[:60]:
                overview += f"- {node.name} ({count} 篇)\n"
            if len(ds_nodes) > 60:
                overview += f"- _… 另 {len(ds_nodes) - 60} 个_\n"
            overview += "\n"

    # ── Methods grouped by primary research task ─────────────────────
    # Avoids dumping 67 single-paper methods flat. Each method page already
    # has a "研究领域" link back to its parent task.
    from backend.services.facet_normalizer import canonicalize_task_name as _ct
    from collections import defaultdict, Counter as _Counter
    method_by_task: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for m in data["methods"]:
        cname_m = re.sub(r"\s*\(.*\)\s*$", "", (m.name or "").strip()) or m.name
        # Find this method's papers via method_family match
        m_papers = [
            p for p in data["papers"]
            if p.method_family and re.sub(r"\s*\(.*\)\s*$", "", p.method_family.strip()) == cname_m
        ]
        if not m_papers:
            continue
        # Vote primary research bucket across this method's papers. Try task
        # first (most specific), then domain, then learning_paradigm, then
        # modality, then paper.category (e.g. "CVPR" venue) as last fallback.
        task_votes_m = _Counter()
        for p in m_papers:
            for f in data["paper_facets"].get(str(p.id), []):
                tn = data["tax_by_id"].get(f["node_id"])
                if not tn:
                    continue
                nm = (tn.name or "").strip()
                if not nm or nm.startswith("__") or nm.lower() in _TAG_BLACKLIST:
                    continue
                if tn.dimension == "task":
                    task_votes_m[_ct(nm)] += 5
                elif tn.dimension == "domain":
                    task_votes_m[_ct(nm)] += 4
                elif tn.dimension == "learning_paradigm":
                    task_votes_m[_ct(nm)] += 3
                elif tn.dimension == "modality":
                    task_votes_m[_ct(nm)] += 2
            # If still nothing, fall back to paper's category field
            if not task_votes_m and p.category and not p.category.startswith("__"):
                task_votes_m[p.category] += 1
        primary = (task_votes_m.most_common(1) or [("Other", 0)])[0][0]
        method_by_task[primary].append((cname_m, len(m_papers)))

    if method_by_task:
        overview += f"## 方法（按研究任务分组）({sum(len(v) for v in method_by_task.values())})\n\n"
        for task, methods in sorted(method_by_task.items(), key=lambda kv: -len(kv[1])):
            overview += f"### {task} ({len(methods)} methods)\n\n"
            for mn, n in sorted(methods, key=lambda x: -x[1])[:15]:
                overview += f"- {mn} ({n} 篇)\n"
            if len(methods) > 15:
                overview += f"- _… 另 {len(methods) - 15} 个_\n"
            overview += "\n"

    (domain_dir / "_overview.md").write_text(overview, encoding="utf-8")
    stats["domains"] += 1

    # Write individual T__ pages — skip 1-paper noise AND skip dataset
    # dimension (datasets get their own D__ pages in dataset/).
    # Different DIMENSIONS (task / domain / modality / learning_paradigm) can
    # share a canonical name (e.g. "Agent" as both task and domain). Both
    # write to the same T__Agent.md slug → second write overwrites first.
    # Dedupe by slug so stats counter matches on-disk file count.
    exported_task_names: set[str] = data.setdefault("exported_task_names", set())
    written_task_slugs: set[str] = set()
    # Only task and domain dimensions become T__ pages. Modality / paradigm
    # are paper-level metadata (rendered in paper frontmatter) — they should
    # not become first-class hub nodes (e.g. T__Image with 45 papers).
    EXPORT_DIMS = {"task", "domain"}
    for node, count in active_nodes:
        if count < 2:
            continue
        if node.dimension not in EXPORT_DIMS:
            continue
        slug = _slug(node.name)
        if slug in written_task_slugs:
            continue
        _write_task_page(domain_dir, node, node.dimension, data)
        written_task_slugs.add(slug)
        exported_task_names.add(node.name)
        stats["tasks"] += 1


def _write_task_page(parent_dir: Path, task, domain_name: str, data: dict):
    """Write a T__xxx.md task page.

    Parent tasks link to child sub-tasks (not papers) to keep link count low.
    Leaf tasks link directly to their papers.
    """
    tslug = _slug(task.name)
    profile = data["profile_map"].get(f"taxonomy_node:{task.id}")
    task_children = data.get("task_children", {})
    parent_task_ids = data.get("parent_task_ids", set())
    is_parent = str(task.id) in parent_task_ids

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

    # Resolve papers first (we need the count for the auto-generated intro).
    # Use `_is_paper_exported` so the count matches what actually appears in
    # the vault — D-level + arxiv-id-only titles are excluded uniformly.
    paper_ids = data["node_papers"].get(str(task.id), [])
    visible_papers = [
        p for pid in paper_ids
        if _is_paper_exported(p := data["papers_by_id"].get(pid))
    ]

    # Intro: profile > taxonomy description > auto-generated skeleton.
    # The skeleton is a one-liner so the page is never naked.
    dim_label = {
        "domain": "研究领域", "task": "任务", "modality": "模态",
        "learning_paradigm": "学习范式", "dataset": "数据集",
        "mechanism": "机制",
    }.get(domain_name, domain_name)
    if profile and profile.short_intro_md:
        body += profile.short_intro_md.strip() + "\n\n"
    elif task.description:
        body += task.description.strip() + "\n\n"
    else:
        intro = f"_{dim_label}节点_。本节点目前由 {len(visible_papers)} 篇知识库论文支撑"
        if visible_papers:
            top_venues = sorted({(p.venue or "Unknown") for p in visible_papers})[:3]
            top_venues = [_normalize_venue(v) for v in top_venues]
            intro += "，主要来源：" + "、".join(top_venues)
        intro += "。"
        if not is_parent and len(visible_papers) <= 1:
            intro += "（待 kb_profiler agent 补充正式介绍）"
        body += intro + "\n\n"

    # If parent task: show sub-tasks (not papers directly)
    if is_parent:
        child_ids = task_children.get(str(task.id), [])
        if child_ids:
            body += "## 子任务\n\n"
            for cid in child_ids:
                child = data["tax_by_id"].get(cid)
                if child:
                    cslug = _slug(child.name)
                    child_paper_count = len(data["node_papers"].get(cid, []))
                    body += f"- [[T__{cslug}]] ({child_paper_count} 篇)\n"
            body += "\n"

    # ── Distribution stats ───────────────────────────────────────
    if visible_papers and not is_parent:
        from collections import Counter
        from backend.services.facet_normalizer import (
            canonicalize_facet_name as _canon_ds,
        )
        # Top methods used in this task
        method_counter: Counter = Counter()
        for p in visible_papers:
            mf = (p.method_family or "").strip()
            if mf:
                cname = re.sub(r"\s*\(.*\)\s*$", "", mf).strip() or mf
                method_counter[cname] += 1
        # Top datasets used in this task (via paper_facets)
        ds_counter: Counter = Counter()
        for p in visible_papers:
            for f in data.get("paper_facets", {}).get(str(p.id), []):
                node = data["tax_by_id"].get(f["node_id"])
                if not node or node.dimension != "dataset":
                    continue
                nm = (node.name or "").strip()
                if not nm or nm.startswith("__"):
                    continue
                ds_counter[_canon_ds(nm, "dataset")] += 1

        # Cross-dimension stats are PLAIN TEXT — graph edges should only be
        # paper→{task,method,dataset}, never task↔method or task↔dataset.
        # The relationship is implied by paper co-occurrence, not direct link.
        if method_counter:
            top_methods = method_counter.most_common(8)
            body += "## 代表方法\n\n"
            for mn, n in top_methods:
                body += f"- {mn} ({n} 篇)\n"
            body += "\n"
        if ds_counter:
            top_ds = ds_counter.most_common(8)
            body += "## 常用数据集\n\n"
            for dn, n in top_ds:
                body += f"- {dn} ({n} 篇)\n"
            body += "\n"
        # Year + venue lines (compact, no wikilinks → no hub edges)
        year_line = _summarize_year_distribution(p.year for p in visible_papers)
        venue_line = _summarize_venue_distribution(p.venue for p in visible_papers)
        if year_line or venue_line:
            body += "## 分布\n\n"
            if year_line:
                body += f"- 年份: {year_line}\n"
            if venue_line:
                body += f"- 会议: {venue_line}\n"
            body += "\n"

    if visible_papers:
        body += f"## 相关论文 ({len(visible_papers)})\n\n"
        body += "| 论文 | 会议 | 年份 | 核心贡献 |\n|------|------|------|----------|\n"
        for p in visible_papers[:30]:
            pslug = _paper_slug(p)
            venue = _normalize_venue(p.venue)
            year = p.year or ""
            # Drop placeholder "Analysis of paper <uuid>" delta_statements —
            # they leak the internal id and add no signal. Show "—" instead.
            ds = (p.delta_statement or "").strip()
            if ds.lower().startswith("analysis of paper "):
                ds = ""
            contribution = ds[:60] if ds else "—"
            body += f"| [[P__{pslug}]] | {venue} | {year} | {contribution} |\n"

    (parent_dir / f"T__{tslug}.md").write_text(fm + body, encoding="utf-8")


# ── Method Pages ─────────────────────────────────────────────────

async def _export_methods(vault: Path, data: dict, stats: dict):
    """Export method/M__xxx.md pages.

    Improvements over the original:
      * Bucket variants by acronym (e.g. drop "(Bootstrapped Preference Optimization)"
        suffix) so PPO/GRPO/DPO etc. each have a single canonical page.
      * Skip single-paper methods (idiosyncratic naming noise).
      * Auto-generated fallback intro when neither profile nor description exists.
      * Lineage: surface parent/child via existing method_edges OR via
        paper_relations bridge (paper-baseline → method-of-baseline).
    """
    if not data["methods"]:
        logger.info("No method_nodes — skipping method/ directory")
        return
    method_dir = vault / "method"
    method_dir.mkdir(exist_ok=True)

    # ── 1. Canonicalize + bucket method names ─────────────────────────
    # Uses module-level _method_canonical so paper pages (which also call
    # this fn for their [[M__]] wikilinks) stay in lock-step.
    bucket: dict[str, dict] = {}
    for m in data["methods"]:
        cname = _method_canonical(m.name)
        b = bucket.setdefault(cname, {
            "canonical": cname, "name_zh": m.name_zh, "description": m.description,
            "method_type": m.type, "maturity": m.maturity, "domain": m.domain,
            "parent_method_id": m.parent_method_id, "ids": [], "primary": m,
            "papers": set(),
        })
        b["ids"].append(str(m.id))

    # Attach papers: a paper "uses" a method if its method_family matches by
    # canon name. Skip non-exported papers (D-level / placeholder) — otherwise
    # the method node ends up orphan in the vault graph (it has a paper-count
    # of 1 but the paper file doesn't exist).
    for p in data["papers"]:
        if not _is_paper_exported(p):
            continue
        mf = (p.method_family or "").strip()
        if not mf:
            continue
        cname = _method_canonical(mf)
        if cname in bucket:
            bucket[cname]["papers"].add(str(p.id))
        elif mf in bucket:
            bucket[mf]["papers"].add(str(p.id))

    # ── 2. Render every method bucket (incl. single-paper). Each unique
    # method name is a paper-specific instantiation of a broader family
    # (AnyAttack, AptGCD, DVDet are all variants of larger research areas).
    # The overview groups them by primary research task so a 67-method long
    # list isn't dumped flat. Paper page already wikilinks to method, so
    # navigation works regardless of overview presence.
    relations_out = data.get("relations_out", {})
    relations_in = data.get("relations_in", {})
    papers_by_id = data.get("papers_by_id", {})
    written = 0
    for cname, b in bucket.items():
        if not b["papers"]:
            continue
        m = b["primary"]
        mslug = _slug(cname)
        profile = data["profile_map"].get(f"method_node:{m.id}")

        # Lineage: parent
        parent_name = None
        if b["parent_method_id"]:
            parent = data["methods_by_id"].get(str(b["parent_method_id"]))
            if parent:
                parent_name = _method_canonical(parent.name)

        fm_data = {
            "title": cname, "type": "method",
            "method_type": b["method_type"], "maturity": b["maturity"],
            "parent_method": parent_name,
            "n_papers": len(b["papers"]),
        }
        # Drop placeholder domain like "__pending__"
        if b["domain"] and not b["domain"].startswith("__"):
            fm_data["domain"] = b["domain"]
        fm = _fm(fm_data)

        body = f"# {cname}"
        if b["name_zh"]:
            body += f" ({b['name_zh']})"
        body += "\n\n"

        # Derive primary research task from the papers that use this method
        # — used both in intro skeleton AND as a "research domain" link so
        # the overview can group methods by their task family.
        from backend.services.facet_normalizer import canonicalize_task_name as _ct
        from collections import Counter as _Counter
        task_votes = _Counter()
        for pid in b["papers"]:
            for f in data.get("paper_facets", {}).get(pid, []):
                node = data["tax_by_id"].get(f["node_id"])
                if not node or node.dimension not in ("task", "domain"):
                    continue
                nm = (node.name or "").strip()
                if not nm or nm.startswith("__"):
                    continue
                task_votes[_ct(nm)] += 1
        primary_tasks = [t for t, _ in task_votes.most_common(3)]

        # Intro: profile > description > auto skeleton (mirrors task page logic)
        if profile and profile.short_intro_md:
            body += profile.short_intro_md.strip() + "\n\n"
        elif b["description"]:
            body += b["description"].strip() + "\n\n"
        else:
            top_venues = sorted({
                _normalize_venue(papers_by_id[pid].venue)
                for pid in b["papers"] if pid in papers_by_id
            })[:3]
            intro = (
                f"_方法节点_。目前由知识库中 **{len(b['papers'])} 篇论文**使用或提出"
            )
            if top_venues:
                intro += "，主要发表于：" + "、".join(top_venues)
            intro += "。（待 kb_profiler agent 补充正式介绍）"
            body += intro + "\n\n"

        # Research-area: PLAIN TEXT, not wikilink. method↔task edges should
        # only exist via paper, never directly.
        if primary_tasks:
            body += "**研究领域**: " + ", ".join(primary_tasks) + "\n\n"
        # Carry into frontmatter so dataview/overview can group on this field.
        if primary_tasks and "primary_tasks" not in fm_data:
            fm_data["primary_tasks"] = primary_tasks
            fm = _fm(fm_data)

        # Lineage: parent → this → children. Method-method edges via lineage
        # ARE legitimate (this is structural method evolution, not stat
        # co-occurrence), so we keep these wikilinks. The user's "no
        # cross-dimension edges" rule only forbids edges that should be
        # implied by paper co-occurrence (task↔method, task↔dataset, etc).
        children = [
            data["methods_by_id"].get(str(e.child_method_id))
            for e in data["method_edges"]
            if str(e.parent_method_id) in b["ids"]
        ]
        children = [c for c in children if c]
        if parent_name or children:
            body += "## 方法谱系\n\n"
            if parent_name:
                body += f"- **父方法**: [[M__{_slug(parent_name)}]]\n"
            if children:
                body += f"- **子方法/变体** ({len(children)}):\n"
                for c in children:
                    body += f"  - [[M__{_slug(_method_canonical(c.name))}]] — {(c.description or '').strip()[:80]}\n"
            body += "\n"

        # Papers using this method
        if b["papers"]:
            body += f"## 使用本方法的论文 ({len(b['papers'])})\n\n"
            body += "| 论文 | 会议 | 年份 |\n|------|------|------|\n"
            for pid in sorted(b["papers"])[:30]:
                p = papers_by_id.get(pid)
                if p:
                    pslug = _paper_slug(p)
                    body += f"| [[P__{pslug}]] | {_normalize_venue(p.venue)} | {p.year or ''} |\n"
            body += "\n"

        # Cross-paper baseline DAG: which method-papers cite which others as baselines?
        baseline_methods_seen: set[str] = set()
        cross_baselines = []
        for pid in b["papers"]:
            for rel in relations_out.get(pid, []):
                if rel.get("type") not in ("direct_baseline", "method_source"):
                    continue
                tgt = papers_by_id.get(rel["target"])
                if not tgt or not tgt.method_family:
                    continue
                tgt_cname = _method_canonical(tgt.method_family)
                if tgt_cname == cname or tgt_cname in baseline_methods_seen:
                    continue
                baseline_methods_seen.add(tgt_cname)
                cross_baselines.append((tgt_cname, rel.get("evidence", "")))
        # Plain text — method↔method baselines are derived from paper-paper
        # citations, so the edge already exists at the paper level. Adding a
        # direct method↔method link here would double-count.
        if cross_baselines:
            body += "## 本方法的方法学基线\n\n"
            for bcname, ev in cross_baselines[:8]:
                line = f"- {bcname}"
                if ev:
                    line += f" — _{(ev or '').strip()[:80]}_"
                body += line + "\n"
            body += "\n"

        (method_dir / f"M__{mslug}.md").write_text(fm + body, encoding="utf-8")
        written += 1
    stats["methods"] += written


# ── Dataset Pages ────────────────────────────────────────────────

async def _export_datasets(vault: Path, data: dict, stats: dict):
    """Export dataset/D__xxx.md pages."""
    ds_dir = vault / "dataset"
    ds_dir.mkdir(exist_ok=True)

    # Bucket dataset nodes by canonical name (same logic as task pages above)
    from backend.services.facet_normalizer import canonicalize_facet_name as _canon
    raw_ds = data["tax_by_dim"].get("dataset", [])
    bucket: dict[str, dict] = {}
    for ds in raw_ds:
        name = (ds.name or "").strip()
        if not name or name.startswith("__"):
            continue
        cname = _canon(name, "dataset")
        b = bucket.setdefault(cname, {
            "canonical": cname,
            "name_zh": ds.name_zh,
            "description": ds.description,
            "node_ids": [],
            "papers": set(),
            "primary": ds,
        })
        b["node_ids"].append(str(ds.id))
        for pid in data["node_papers"].get(str(ds.id), []):
            b["papers"].add(pid)

    papers_by_id = data.get("papers_by_id", {})
    # Track canonical dataset names that DO get a page so paper pages can
    # avoid emitting wikilinks to non-existent D__ files.
    exported_ds_names: set[str] = data.setdefault("exported_dataset_names", set())
    for cname, b in bucket.items():
        # Filter to only papers that actually appear in the vault. Dataset
        # pages built from raw node_papers can list non-exported papers,
        # creating broken [[P__]] links in the dataset's "使用本数据集的论文"
        # section.
        b["papers"] = {pid for pid in b["papers"]
                       if _is_paper_exported(papers_by_id.get(pid))}
        # Only render dataset pages with ≥2 EXPORTED papers — single-paper
        # nodes are extractor noise (paper-private benchmark variants).
        if len(b["papers"]) < 2:
            continue
        exported_ds_names.add(cname)
        dslug = _slug(cname)
        fm = _fm({"title": cname, "type": "dataset",
                  "name_zh": b["name_zh"], "n_papers": len(b["papers"])})
        body = f"# {cname}\n\n"

        # Intro: profile > description > auto skeleton (mirror task page logic)
        # Try profile first by checking any of the bucketed taxonomy_node ids.
        profile_text = None
        for nid in b.get("node_ids", []):
            prof = data["profile_map"].get(f"taxonomy_node:{nid}")
            if prof and prof.short_intro_md:
                profile_text = prof.short_intro_md.strip()
                break
        if profile_text:
            body += profile_text + "\n\n"
        elif b["description"]:
            body += (b["description"] or "").strip() + "\n\n"
        else:
            top_venues = sorted({
                _normalize_venue(papers_by_id[pid].venue)
                for pid in b["papers"] if pid in papers_by_id
            })[:3]
            intro = f"_数据集节点_。被知识库中 **{len(b['papers'])} 篇论文**用于评测或训练"
            if top_venues:
                intro += "，主要见于：" + "、".join(top_venues)
            intro += "。（待 kb_profiler agent 补充正式介绍）"
            body += intro + "\n\n"

        # ── Distribution stats: which tasks / methods use this dataset ──
        from collections import Counter
        from backend.services.facet_normalizer import (
            canonicalize_task_name as _ct,
        )
        ds_papers = [papers_by_id.get(pid) for pid in b["papers"]]
        ds_papers = [p for p in ds_papers if p is not None]

        task_counter: Counter = Counter()
        for p in ds_papers:
            for f in data.get("paper_facets", {}).get(str(p.id), []):
                node = data["tax_by_id"].get(f["node_id"])
                if not node or node.dimension not in ("task", "domain", "learning_paradigm"):
                    continue
                nm = (node.name or "").strip()
                if not nm or nm.startswith("__"):
                    continue
                task_counter[_ct(nm)] += 1

        method_counter: Counter = Counter()
        for p in ds_papers:
            mf = (p.method_family or "").strip()
            if mf:
                cname2 = re.sub(r"\s*\(.*\)\s*$", "", mf).strip() or mf
                method_counter[cname2] += 1

        # Plain text — see same comment in _write_task_page.
        if task_counter:
            top_tasks = task_counter.most_common(8)
            body += "## 服务的研究任务\n\n"
            for tn, n in top_tasks:
                body += f"- {tn} ({n} 次使用)\n"
            body += "\n"
        if method_counter:
            top_methods = method_counter.most_common(8)
            body += "## 在本数据集上评测的方法\n\n"
            for mn, n in top_methods:
                body += f"- {mn} ({n} 篇)\n"
            body += "\n"
        year_line = _summarize_year_distribution(p.year for p in ds_papers)
        venue_line = _summarize_venue_distribution(p.venue for p in ds_papers)
        if year_line or venue_line:
            body += "## 分布\n\n"
            if year_line:
                body += f"- 年份: {year_line}\n"
            if venue_line:
                body += f"- 会议: {venue_line}\n"
            body += "\n"

        body += f"## 使用本数据集的论文 ({len(b['papers'])})\n\n"
        for pid in sorted(b["papers"])[:30]:
            p = papers_by_id.get(pid)
            if p:
                pslug = _paper_slug(p)
                body += f"- [[P__{pslug}]] ({_normalize_venue(p.venue)} {p.year or ''})\n"
        (ds_dir / f"D__{dslug}.md").write_text(fm + body, encoding="utf-8")
        stats["datasets"] += 1


# ── Paper Level Classification ──────────────────────────────────

def _paper_level(p) -> str:
    """Classify paper into A/B/C/D level.

    A = baseline (structurality >= 0.7 or ring=baseline)
    B = structural (structurality 0.5-0.7 or ring=structural)
    C = plugin (structurality 0.3-0.5 or ring=plugin)
    D = peripheral (default — NOT exported to vault)
    """
    ring = getattr(p, "ring", None) or ""
    if ring == "baseline":
        return "A"
    if ring == "structural":
        return "B"
    if ring == "plugin":
        return "C"

    s = float(p.dc_struct) if p.dc_struct is not None else 0.0
    if s >= 0.7:
        return "A"
    if s >= 0.5:
        return "B"
    if s >= 0.3:
        return "C"

    # Papers with any analysis content are at least C
    if p.full_report_md or p.problem_summary or p.delta_statement:
        return "C"

    return "D"


# ── Paper Pages ──────────────────────────────────────────────────

async def _export_papers(vault: Path, session: AsyncSession, data: dict, stats: dict):
    """Export paper/venue_year/P__xxx.md pages.

    D-level (peripheral) papers are SKIPPED — they have no analysis content
    and would pollute the Obsidian graph with empty nodes.
    """
    paper_dir = vault / "paper"
    paper_dir.mkdir(exist_ok=True)

    parent_task_ids = data.get("parent_task_ids", set())
    figure_map = data.get("figure_map", {})
    relations_out = data.get("relations_out", {})
    relations_in = data.get("relations_in", {})
    papers_by_id = data.get("papers_by_id", {})
    skipped = 0

    # Roles whose targets we expose at the top of the page as "本文 baseline"
    BASELINE_ROLES = {"direct_baseline", "comparison_baseline",
                      "method_source", "formula_source"}
    # Reverse — papers that cite THIS paper as a baseline
    FOLLOWUP_ROLES = BASELINE_ROLES

    for p in data["papers"]:
        level = _paper_level(p)

        # Skip D-level peripheral papers
        if level == "D":
            skipped += 1
            continue

        # Skip arxiv-ID-only titles (enrichment failed)
        if _is_arxiv_id_title(p.title):
            skipped += 1
            continue

        vy = _venue_year(p.venue, p.year)
        vy_dir = paper_dir / vy
        vy_dir.mkdir(exist_ok=True)

        pslug = _paper_slug(p)

        # Build facet wikilinks. Skip placeholder nodes; canonicalize names.
        # Tasks use `canonicalize_task_name` (strips dataset prefix + maps to
        # canonical activity, e.g. "CIFAR-10 OOD detection" → "OOD Detection").
        # Datasets use `canonicalize_facet_name` (e.g. "CIFAR-10 OOD detection"
        # → "CIFAR-10"). Without this, 89% of nodes had only one paper.
        from backend.services.facet_normalizer import (
            canonicalize_facet_name as _canon_ds,
            canonicalize_task_name as _canon_task,
        )
        # Sets of canonical names that DID get an exported page. Anything not
        # in here would render as a broken [[T__]] / [[D__]] wikilink, so we
        # fall back to plain text mention in `task_text_only` / `ds_text_only`.
        exported_task_names: set[str] = data.get("exported_task_names", set())
        exported_ds_names: set[str] = data.get("exported_dataset_names", set())

        facets = data["paper_facets"].get(str(p.id), [])
        task_links_seen: set[str] = set()
        dataset_links_seen: set[str] = set()
        task_links = []
        method_links = []
        dataset_links = []
        task_text_only: list[str] = []
        ds_text_only: list[str] = []
        # Track modality / paradigm separately — render as paper metadata,
        # not as graph nodes. Otherwise "Image" (modality) and "Reinforcement
        # Learning" (paradigm) become 30-50-paper hubs that aren't real
        # research tasks.
        modality_seen: set[str] = set()
        paradigm_seen: set[str] = set()
        modalities: list[str] = []
        paradigms: list[str] = []
        for f in facets:
            node = data["tax_by_id"].get(f["node_id"])
            if not node:
                continue
            name = (node.name or "").strip()
            if not name or name.startswith("__") or name == "__pending__":
                continue
            if node.dimension == "task":
                if str(node.id) in parent_task_ids:
                    continue
                cname = _canon_task(name)
                if cname in task_links_seen:
                    continue
                task_links_seen.add(cname)
                if cname in exported_task_names:
                    task_links.append(f"[[T__{_slug(cname)}]]")
                else:
                    task_text_only.append(cname)
            elif node.dimension == "domain":
                # Domain is a coarser grouping. If a task-dim tag with the
                # same canonical name already linked, skip — otherwise the
                # paper would emit two [[T__Agent]] wikilinks (one for
                # domain, one for task), but they collapse to the same slug
                # so it's a duplicate edge in graph.
                if str(node.id) in parent_task_ids:
                    continue
                cname = _canon_task(name)
                if cname not in task_links_seen and cname in exported_task_names:
                    task_links_seen.add(cname)
                    task_links.append(f"[[T__{_slug(cname)}]]")
            elif node.dimension == "modality":
                if name not in modality_seen:
                    modality_seen.add(name)
                    modalities.append(name)
            elif node.dimension == "learning_paradigm":
                if name not in paradigm_seen:
                    paradigm_seen.add(name)
                    paradigms.append(name)
            elif node.dimension == "dataset":
                cname = _canon_ds(name, "dataset")
                if cname in dataset_links_seen:
                    continue
                dataset_links_seen.add(cname)
                if cname in exported_ds_names:
                    dataset_links.append(f"[[D__{_slug(cname)}]]")
                else:
                    ds_text_only.append(cname)

        if p.method_family:
            method_links.append(f"[[M__{_slug(_method_canonical(p.method_family))}]]")

        # Build aliases for graph view: title_zh first, then acronym, then method
        aliases = _paper_aliases(p)

        # Resolve baseline / follow-up edges from paper_relations
        baseline_targets = []   # papers THIS paper builds on
        for rel in relations_out.get(str(p.id), []):
            if rel["type"] not in BASELINE_ROLES:
                continue
            tgt_p = papers_by_id.get(rel["target"])
            if not tgt_p:
                continue
            baseline_targets.append((tgt_p, rel))
        # Dedup by target paper
        seen_t = set()
        baseline_targets_dedup = []
        for tp, rel in baseline_targets:
            if tp.id in seen_t:
                continue
            seen_t.add(tp.id)
            baseline_targets_dedup.append((tp, rel))

        followup_sources = []   # papers that build on THIS one
        for rel in relations_in.get(str(p.id), []):
            if rel["type"] not in FOLLOWUP_ROLES:
                continue
            src_p = papers_by_id.get(rel["source"])
            if not src_p:
                continue
            followup_sources.append((src_p, rel))
        seen_s = set()
        followup_sources_dedup = []
        for sp, rel in followup_sources:
            if sp.id in seen_s:
                continue
            seen_s.add(sp.id)
            followup_sources_dedup.append((sp, rel))

        # Frontmatter
        venue_display = _normalize_venue(p.venue)
        fm_data = {
            "title": p.title,
            "type": "paper",
            "paper_level": level,
            "venue": venue_display,
            "year": p.year,
            "paper_link": p.paper_link,
        }
        if aliases:
            fm_data["aliases"] = aliases
        if p.acceptance_type:
            fm_data["acceptance"] = p.acceptance_type
        if p.cited_by_count:
            fm_data["cited_by"] = p.cited_by_count
        if p.code_url:
            fm_data["code_url"] = p.code_url
        if p.method_family:
            fm_data["method"] = p.method_family
        # Modality + paradigm as frontmatter only (no graph node) — these are
        # paper attributes, not research hubs. Avoids "Image" (45 papers) and
        # "Reinforcement Learning" (35 papers) from becoming pseudo-task hubs
        # at scale.
        if modalities:
            fm_data["modalities"] = modalities[:5]
        if paradigms:
            fm_data["paradigm"] = paradigms[0] if len(paradigms) == 1 else paradigms[:3]
        # Expose baselines/followups in frontmatter so Obsidian's graph view
        # picks up the directed edges. Only include EXPORTED targets — listing
        # a non-exported slug here creates a "phantom" node in graph view.
        exported_baselines = [_paper_slug(tp) for tp, _ in baseline_targets_dedup[:10]
                              if _is_paper_exported(tp)]
        exported_followups = [_paper_slug(sp) for sp, _ in followup_sources_dedup[:10]
                              if _is_paper_exported(sp)]
        if exported_baselines:
            fm_data["baselines"] = exported_baselines
        if exported_followups:
            fm_data["followups"] = exported_followups
        fm = _fm(fm_data)

        body = f"# {p.title}\n\n"

        # Links
        links = []
        if p.paper_link:
            links.append(f"[Paper]({p.paper_link})")
        if p.code_url:
            links.append(f"[Code]({p.code_url})")
        if links:
            body += " | ".join(links) + "\n\n"

        # Wikilinks. Single-paper task / dataset nodes (which weren't
        # exported) appear as plain-text mentions instead of broken [[]].
        wiki_parts = []
        topic_part = ", ".join(task_links[:5])
        if task_text_only:
            extra = ", ".join(task_text_only[:3])
            topic_part = (topic_part + " (其他: " + extra + ")") if topic_part else extra
        if topic_part:
            wiki_parts.append("**Topics**: " + topic_part)
        if method_links:
            wiki_parts.append("**Method**: " + ", ".join(method_links))
        ds_part = ", ".join(dataset_links[:5])
        if ds_text_only:
            extra = ", ".join(ds_text_only[:5])
            ds_part = (ds_part + " (其他: " + extra + ")") if ds_part else extra
        if ds_part:
            wiki_parts.append("**Datasets**: " + ds_part)
        if wiki_parts:
            body += " | ".join(wiki_parts) + "\n\n"

        # Delta statement as callout
        if p.delta_statement:
            body += f"> [!tip] 核心洞察\n> {p.delta_statement}\n\n"

        # Full report with figure resolution
        figures = figure_map.get(str(p.id), [])
        if p.full_report_md:
            # Strip embedded figure_placements comment and use it to resolve markers
            raw = p.full_report_md
            placements, raw = _extract_figure_placements(raw)
            if "{{FIG:" in raw:
                report_body = _resolve_figure_markers(raw, figures, placements)
            else:
                # Report has no markers — auto-inject by semantic_role at section ends
                report_body = _autoinject_figures_by_role(raw, figures)
            body += report_body + "\n"
        else:
            if p.problem_summary:
                body += f"## 背景与动机\n\n{p.problem_summary}\n\n"
            if p.method_summary:
                body += f"## 方法概述\n\n{p.method_summary}\n\n"
            if p.core_intuition:
                body += f"## 核心直觉\n\n{p.core_intuition}\n\n"
            if p.evidence_summary:
                body += f"## 实验证据\n\n{p.evidence_summary}\n\n"
            # Auto-inject figures into the legacy summary block as well
            if figures:
                body = _autoinject_figures_by_role(body, figures)

        # Key equations (if not already in full_report)
        if p.key_equations and isinstance(p.key_equations, list) and not p.full_report_md:
            body += "## 关键公式\n\n"
            for eq in p.key_equations[:5]:
                if isinstance(eq, dict):
                    body += f"$$\n{eq.get('latex', '')}\n$$\n"
                    expl = eq.get("explanation_zh") or eq.get("explanation")
                    if expl:
                        body += f"\n{expl}\n\n"

        # ── Baseline / follow-up DAG ──────────────────────────────
        # Soft-degrade dead links: a baseline target that won't be exported
        # (D-level, or arxiv-id-only title) renders as plain text rather than
        # a [[wikilink]], so the vault has no broken link arrows.
        if baseline_targets_dedup or followup_sources_dedup:
            body += "## 引用网络\n\n"
        if baseline_targets_dedup:
            body += "### 直接 baseline（本文基于）\n\n"
            for tp, rel in baseline_targets_dedup[:8]:
                evidence = (rel.get("evidence") or "").strip()
                ev_clip = (": " + evidence[:80]) if evidence else ""
                role_tag = {
                    "direct_baseline": "直接 baseline",
                    "comparison_baseline": "实验对比",
                    "method_source": "方法来源",
                    "formula_source": "公式来源",
                }.get(rel["type"], rel["type"])
                if _is_paper_exported(tp):
                    target_slug = _paper_slug(tp)
                    body += f"- [[P__{target_slug}]] _({role_tag})_{ev_clip}\n"
                else:
                    venue_y = f"{_normalize_venue(tp.venue)} {tp.year or ''}".strip()
                    body += f"- {tp.title} _({venue_y}, {role_tag}, 未深度分析)_{ev_clip}\n"
            body += "\n"
        if followup_sources_dedup:
            body += "### 后续工作（建立在本文之上）\n\n"
            for sp, rel in followup_sources_dedup[:8]:
                evidence = (rel.get("evidence") or "").strip()
                ev_clip = (": " + evidence[:80]) if evidence else ""
                if _is_paper_exported(sp):
                    src_slug = _paper_slug(sp)
                    body += f"- [[P__{src_slug}]]{ev_clip}\n"
                else:
                    venue_y = f"{_normalize_venue(sp.venue)} {sp.year or ''}".strip()
                    body += f"- {sp.title} _({venue_y}, 未深度分析)_{ev_clip}\n"
            body += "\n"

        (vy_dir / f"P__{pslug}.md").write_text(fm + body, encoding="utf-8")
        stats["papers"] += 1

    if skipped:
        logger.info("Skipped %d peripheral/placeholder papers", skipped)


def _extract_figure_placements(body: str) -> tuple[list[dict], str]:
    """Pull out the `<!-- figure_placements: [...] -->` comment if present.

    Returns (placements_list, body_without_comment). The comment is appended by
    _persist_paper_report so the LLM-decided marker→label mapping survives.
    """
    import json as _json
    m = re.search(r'<!--\s*figure_placements:\s*(\[.*?\])\s*-->', body, re.DOTALL)
    if not m:
        return [], body
    try:
        placements = _json.loads(m.group(1))
        if not isinstance(placements, list):
            placements = []
    except Exception:
        placements = []
    cleaned = (body[:m.start()] + body[m.end():]).rstrip() + "\n"
    return placements, cleaned


def _render_figure(fig: dict) -> str:
    """Render a single figure as an inline markdown image + caption line."""
    url = fig.get("public_url", "")
    label = fig.get("label", "Figure")
    caption = (fig.get("caption") or "").strip()
    role = fig.get("semantic_role", "")
    if not url:
        return ""
    out = f"\n![{label}]({url})\n"
    if caption:
        suffix = f" ({role})" if role and role != "other" else ""
        out += f"*{label}{suffix}: {caption}*\n"
    return out + "\n"


def _resolve_figure_markers(body: str, figures: list[dict], placements: list[dict]) -> str:
    """Replace {{FIG:xxx}} and {{TBL:xxx}} markers with embedded OSS images.

    Resolution priority for each marker:
      1. Match by `placements[i].marker == "{{...:xxx}}"` and look up its
         preferred_labels in the figures list (exact label match).
      2. Fall back to semantic_role match against the marker hint.
      3. For {{TBL:...}} markers, restrict the candidate pool to type='table'
         (and only fall back to figures if no table candidates remain).
      4. Drop the marker if nothing fits.
    """
    if not figures:
        return re.sub(r'\{\{(?:FIG|TBL):\w+\}\}', '', body)

    by_label: dict[str, dict] = {f.get("label", ""): f for f in figures if f.get("label")}
    by_role: dict[str, list[dict]] = {}
    for f in figures:
        by_role.setdefault(f.get("semantic_role", "other"), []).append(f)

    tables = [f for f in figures if f.get("type") == "table"]
    figs_only = [f for f in figures if f.get("type") != "table"]

    placement_map: dict[str, dict] = {}
    for pl in placements:
        marker = (pl.get("marker") or "").strip()
        if marker:
            placement_map[marker] = pl

    used: set[str] = set()

    def _pick(marker: str, kind: str, hint: str):
        """kind ∈ {'fig', 'tbl'}. Determines which subset to prefer."""
        candidate_pool = tables if kind == "tbl" else figs_only
        # 1. preferred_labels from LLM placements, restricted to right type if possible
        pl = placement_map.get(marker)
        if pl:
            for lbl in pl.get("preferred_labels", []) or []:
                fig = by_label.get(lbl)
                if fig and fig.get("object_key") not in used:
                    if kind == "tbl" and fig.get("type") != "table":
                        continue  # don't accept non-tables for TBL markers
                    used.add(fig.get("object_key", lbl))
                    return fig
            role = (pl.get("semantic_role") or hint).lower()
        else:
            role = hint.lower()
        # 2. Fall back to semantic_role match WITHIN the right pool
        for fig in candidate_pool:
            r = fig.get("semantic_role", "")
            if role and (role == r or role in r or r in role) and fig.get("object_key") not in used:
                used.add(fig.get("object_key", fig.get("label", "")))
                return fig
        # 3. For TBL markers with no role match: pick any unused table — better
        #    to show some result table than none. FIG markers drop silently
        #    (don't insert random figures the LLM didn't ask for).
        if kind == "tbl":
            for fig in tables:
                if fig.get("object_key") not in used:
                    used.add(fig.get("object_key", fig.get("label", "")))
                    return fig
        return None

    def _replace(match):
        marker = match.group(0)
        kind_token = match.group(1).lower()  # "fig" or "tbl"
        hint = match.group(2)
        kind = "tbl" if kind_token == "tbl" else "fig"
        fig = _pick(marker, kind, hint)
        return _render_figure(fig) if fig else ""

    return re.sub(r'\{\{(FIG|TBL):(\w+)\}\}', _replace, body)


# Map figure semantic_role → which section_type heading it belongs after.
_ROLE_TO_SECTION_TITLE = {
    "motivation": "背景与动机",
    "example": "背景与动机",
    "pipeline": "整体框架",
    "architecture": "整体框架",
    "result": "实验与分析",
    "ablation": "实验与分析",
    "comparison": "实验与分析",
    "qualitative": "实验与分析",
    "quantitative": "实验与分析",
}


def _autoinject_figures_by_role(body: str, figures: list[dict]) -> str:
    """Insert figures inline at the END of the matching section heading.

    Used when the report body has no {{FIG:xxx}} markers (legacy reports
    or LLM omissions). Figures are grouped by semantic_role and appended
    just before the next "## " heading. Anything still unplaced is appended
    inline at the end of its closest-match section, NEVER at document end.
    """
    if not figures:
        return body

    # Bucket figures by target section title
    buckets: dict[str, list[dict]] = {}
    leftovers: list[dict] = []
    for fig in figures:
        role = (fig.get("semantic_role") or "").lower()
        sec_title = _ROLE_TO_SECTION_TITLE.get(role)
        if sec_title:
            buckets.setdefault(sec_title, []).append(fig)
        else:
            leftovers.append(fig)

    if not buckets and not leftovers:
        return body

    # Split body by H2 headings, keep heading attached to its block
    parts = re.split(r'(?m)(^##\s+.+$)', body)
    # parts = [pre, h1, body1, h2, body2, ...]
    if len(parts) < 3:
        # No H2 headings — append all figures at end as a fallback
        tail = "".join(_render_figure(f) for f in figures)
        return body.rstrip() + "\n\n" + tail

    out = [parts[0]]
    i = 1
    while i < len(parts):
        heading = parts[i]
        block = parts[i + 1] if i + 1 < len(parts) else ""
        out.append(heading)
        # Find which bucket matches this heading
        matched_key = None
        for key in buckets:
            if key in heading:
                matched_key = key
                break
        if matched_key:
            inj = "".join(_render_figure(f) for f in buckets.pop(matched_key))
            block = block.rstrip() + "\n\n" + inj
        out.append(block)
        i += 2

    result = "".join(out)
    # Anything left (no matching section) → append to last block
    remaining = []
    for figs in buckets.values():
        remaining.extend(figs)
    remaining.extend(leftovers)
    if remaining:
        # Inject at end of LAST H2 block instead of after a "论文图表" heading
        result = result.rstrip() + "\n\n" + "".join(_render_figure(f) for f in remaining)
    return result


# _export_views removed — paper/ grouped by venue_year/ is sufficient


# ── Home ─────────────────────────────────────────────────────────

def _export_home(vault: Path, data: dict, stats: dict):
    """Export 00_Home/00_方向总览.md overview page.

    Links to domain overviews and venue views — NOT directly to papers.
    """
    home_dir = vault / "00_Home"
    home_dir.mkdir(exist_ok=True)

    # Frontmatter marks this as a hub-page so the user can graph-filter on it.
    body = "---\ntitle: 方向总览\ntype: home\ncssclasses:\n  - hub-page\n  - no-graph\n---\n\n"
    body += "# 研究方向总览\n\n"
    body += "> **统计页**——研究节点通过 paper 页互联，本页不参与图谱拓扑。\n\n"
    body += f"- 论文: {stats['papers']} 篇 (已过滤外围参考)\n"
    body += f"- 方法: {stats['methods']} 个\n"
    body += f"- 任务: {stats['tasks']} 个\n"
    body += f"- 数据集: {stats['datasets']} 个\n"
    body += f"- 领域: {stats['domains']} 个\n\n"

    # Top domains — plain text (no wikilinks → no big hub).
    body += "## 领域\n\n"
    domains = data["tax_by_dim"].get("domain", [])
    if domains:
        for d in domains:
            paper_count = len(data["node_papers"].get(str(d.id), []))
            body += f"- {d.name} ({paper_count} 篇)\n"
    else:
        categories = {}
        for p in data["papers"]:
            if p.category and _paper_level(p) != "D":
                categories[p.category] = categories.get(p.category, 0) + 1
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            body += f"- {cat} ({count} 篇)\n"

    # Summary stats by venue (no links to avoid big nodes)
    venue_counts: dict[str, int] = {}
    for p in data["papers"]:
        if _paper_level(p) != "D" and not _is_arxiv_id_title(p.title):
            v = _normalize_venue(p.venue)
            venue_counts[v] = venue_counts.get(v, 0) + 1

    if venue_counts:
        body += "\n## 论文来源\n\n"
        for v, count in sorted(venue_counts.items(), key=lambda x: -x[1])[:15]:
            body += f"- {v}: {count} 篇\n"

    body += f"\n---\n*共 {stats['papers']} 篇论文 (已过滤外围参考)*\n"

    (home_dir / "00_方向总览.md").write_text(body, encoding="utf-8")
