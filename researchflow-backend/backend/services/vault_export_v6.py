"""Obsidian Vault Export v6 — enhances v5 with profiles + edge one-liners.

Additions over v5:
- Node profiles: each T/M/C/D page gets rich introductions from kb_node_profiles
- Edge one-liners: wikilinks followed by contextual descriptions from kb_edge_profiles
- Lab__ pages: research team/lab nodes
- Lineage story pages: narrative method evolution descriptions
"""

import logging
import re
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.vault_export_v5 import export_vault_v5, _fm, _slug

logger = logging.getLogger(__name__)

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


async def export_vault_v6(session: AsyncSession, out_dir: str | None = None) -> dict:
    """Export knowledge base as Obsidian vault v6 (v5 + profiles + labs)."""

    # ── Step 1: Generate base v5 vault ──
    stats = await export_vault_v5(session, out_dir)
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
