#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import ingest_pdfs
import semantic_scholar_api


REPO_ROOT = ingest_pdfs.REPO_ROOT
PAPER_ANALYSIS_DIR = ingest_pdfs.PAPER_ANALYSIS_DIR
PAPER_PDFS_DIR = ingest_pdfs.PAPER_PDFS_DIR
INCREMENTAL_ROOT = PAPER_ANALYSIS_DIR / "processing" / "incremental"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return deepcopy(default)


def merge_unique(existing: Iterable[str], incoming: Iterable[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for value in list(existing) + list(incoming):
        s = (value or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        merged.append(s)
    return merged


def detect_tool(explicit_cmd: str, candidates: List[str], env_var: str) -> Dict[str, str]:
    configured = explicit_cmd.strip() or os.environ.get(env_var, "").strip()
    executable = ""
    if configured:
        executable = shutil.which(configured.split()[0]) or ""
    else:
        for candidate in candidates:
            executable = shutil.which(candidate) or ""
            if executable:
                configured = candidate
                break
    status = "detected" if executable else ("configured_not_found" if configured else "missing")
    return {
        "configured_command": configured,
        "detected_executable": executable,
        "status": status,
    }


def artifact_status(path: Path, existing_status: str = "") -> str:
    if path.is_file():
        return "available"
    if path.is_dir():
        try:
            next(path.iterdir())
            return "available"
        except StopIteration:
            return "initialized"
    if existing_status:
        return existing_status
    return "pending"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Initialize or update the incremental analysis bundle for one PDF or a directory of PDFs. "
            "This command preserves sidecar state for sources, graph edges, and extraction artifacts."
        )
    )
    p.add_argument("input", help="Path to a PDF file or a directory containing PDFs")
    p.add_argument("--category", default="", help="Category when the PDF is outside paperPDFs/")
    p.add_argument("--venue", default="", help="Venue name when the PDF is outside paperPDFs/")
    p.add_argument("--year", default="", help="Publication year when the PDF is outside paperPDFs/")
    p.add_argument("--title", default="", help="Paper title override when the PDF is outside paperPDFs/")
    p.add_argument("--paper-link", action="append", default=[], help="Paper landing page or arXiv URL")
    p.add_argument("--venue-link", action="append", default=[], help="Conference or journal page URL")
    p.add_argument("--project-link", action="append", default=[], help="Project page URL")
    p.add_argument("--repo-link", action="append", default=[], help="Repository URL")
    p.add_argument("--hf-link", action="append", default=[], help="Hugging Face model or dataset URL")
    p.add_argument("--author", action="append", default=[], help="Author name; repeat for multiple authors")
    p.add_argument("--team", action="append", default=[], help="Team/lab/org name; repeat for multiple teams")
    p.add_argument("--arxiv-id", default="", help="arXiv identifier if known")
    p.add_argument("--semantic-scholar-id", default="", help="Semantic Scholar paper identifier if known")
    p.add_argument("--refresh-semantic-scholar", action="store_true", help="Refresh metadata and citation graph from Semantic Scholar Graph API")
    p.add_argument("--semantic-scholar-key", default="", help="Semantic Scholar API key override")
    p.add_argument("--semantic-scholar-limit", type=int, default=20, help="Max references/citations to fetch from Semantic Scholar")
    p.add_argument("--base-paper", action="append", default=[], help="Base paper node/id; repeatable")
    p.add_argument("--cite", action="append", default=[], help="Reference paper node/id; repeatable")
    p.add_argument("--cited-by", action="append", default=[], help="Cited-by paper node/id; repeatable")
    p.add_argument("--semantic-scholar-cmd", default="", help="Semantic Scholar CLI/wrapper command")
    p.add_argument("--marker-cmd", default="", help="Marker command override")
    p.add_argument("--pdffigures-cmd", default="", help="pdffigures2 command override")
    p.add_argument("--dry-run", action="store_true", help="Print the merged bundle state without writing files")
    return p.parse_args(argv)


def resolve_info(pdf: Path, args: argparse.Namespace) -> Dict[str, str]:
    existing = ingest_pdfs.derive_from_paperpdfs_path(pdf)
    if existing:
        category, venue, year, title = existing
        return ingest_pdfs.compute_paths(category=category, venue=venue, year=year, title=title)

    year_from_name, title_from_name = ingest_pdfs.infer_year_title_from_filename(pdf)
    category = args.category.strip()
    venue = args.venue.strip()
    year = args.year.strip() or (year_from_name or "")
    title = args.title.strip() or title_from_name
    missing = [name for name, value in (("category", category), ("venue", venue), ("year", year)) if not value]
    if missing:
        raise ValueError(
            f"PDF outside paperPDFs requires metadata: missing {', '.join(missing)} for {pdf.as_posix()}"
        )
    return ingest_pdfs.compute_paths(category=category, venue=venue, year=year, title=title)


def build_paths(info: Dict[str, str], input_pdf: Path) -> Dict[str, str]:
    pdf_ref_parts = info["pdf_ref"].split("/")
    category = pdf_ref_parts[1]
    venue_year = pdf_ref_parts[2]
    paper_stem = Path(pdf_ref_parts[3]).stem
    bundle_dir = INCREMENTAL_ROOT / category / venue_year / paper_stem
    return {
        "input_pdf": input_pdf.as_posix(),
        "pdf_abs": info["pdf_abs"],
        "pdf_ref": info["pdf_ref"],
        "analysis_note_abs": info["analysis_md_abs"],
        "analysis_note_ref": info["analysis_rel"],
        "bundle_dir_abs": bundle_dir.as_posix(),
        "bundle_dir_ref": bundle_dir.relative_to(REPO_ROOT).as_posix(),
        "bundle_manifest_abs": (bundle_dir / "bundle_manifest.json").as_posix(),
        "bundle_manifest_ref": (bundle_dir / "bundle_manifest.json").relative_to(REPO_ROOT).as_posix(),
        "sources_abs": (bundle_dir / "sources.json").as_posix(),
        "sources_ref": (bundle_dir / "sources.json").relative_to(REPO_ROOT).as_posix(),
        "graph_abs": (bundle_dir / "graph.json").as_posix(),
        "graph_ref": (bundle_dir / "graph.json").relative_to(REPO_ROOT).as_posix(),
        "parsed_dir_ref": (bundle_dir / "parsed").relative_to(REPO_ROOT).as_posix(),
        "figures_dir_ref": (bundle_dir / "figures").relative_to(REPO_ROOT).as_posix(),
        "tables_dir_ref": (bundle_dir / "tables").relative_to(REPO_ROOT).as_posix(),
        "formulas_dir_ref": (bundle_dir / "formulas").relative_to(REPO_ROOT).as_posix(),
        "runs_dir_ref": (bundle_dir / "runs").relative_to(REPO_ROOT).as_posix(),
    }


def merge_sources(existing: Dict[str, Any], args: argparse.Namespace, info: Dict[str, str], paths: Dict[str, str]) -> Dict[str, Any]:
    links = existing.get("links", {})
    people = existing.get("people", {})
    identifiers = existing.get("identifiers", {})
    return {
        "updated_at": now_iso(),
        "paper": {
            "title": info["title"],
            "venue": info["venue"],
            "year": info["year"],
            "category": info["category"],
            "pdf_ref": paths["pdf_ref"],
            "analysis_note_ref": paths["analysis_note_ref"],
        },
        "identifiers": {
            "arxiv_id": args.arxiv_id.strip() or identifiers.get("arxiv_id", ""),
            "semantic_scholar_id": args.semantic_scholar_id.strip() or identifiers.get("semantic_scholar_id", ""),
        },
        "links": {
            "paper": merge_unique(links.get("paper", []), args.paper_link),
            "venue": merge_unique(links.get("venue", []), args.venue_link),
            "project": merge_unique(links.get("project", []), args.project_link),
            "repo": merge_unique(links.get("repo", []), args.repo_link),
            "hf": merge_unique(links.get("hf", []), args.hf_link),
        },
        "people": {
            "authors": merge_unique(people.get("authors", []), args.author),
            "teams": merge_unique(people.get("teams", []), args.team),
        },
    }


def merge_graph(existing: Dict[str, Any], args: argparse.Namespace, info: Dict[str, str]) -> Dict[str, Any]:
    edges = existing.get("edges", {})
    node_id = existing.get("node_id", f"paper:{info['year']}:{Path(info['pdf_abs']).stem}")
    return {
        "updated_at": now_iso(),
        "shape": "dag",
        "node_id": node_id,
        "paper": {
            "title": info["title"],
            "venue": info["venue"],
            "year": info["year"],
            "category": info["category"],
        },
        "edges": {
            "base_papers": merge_unique(edges.get("base_papers", []), args.base_paper),
            "citations": merge_unique(edges.get("citations", []), args.cite),
            "cited_by": merge_unique(edges.get("cited_by", []), args.cited_by),
            "authored_by": merge_unique(edges.get("authored_by", []), args.author),
            "belongs_to_team": merge_unique(edges.get("belongs_to_team", []), args.team),
            "implemented_by": merge_unique(edges.get("implemented_by", []), args.repo_link),
            "hosted_on_hf": merge_unique(edges.get("hosted_on_hf", []), args.hf_link),
        },
    }


def semantic_edge_label(node: Dict[str, Any]) -> str:
    paper_id = (node.get("paperId") or "").strip()
    title = (node.get("title") or "").strip()
    year = str(node.get("year") or "").strip()
    descriptor = " | ".join(part for part in [title, year] if part)
    if paper_id and descriptor:
        return f"S2:{paper_id} | {descriptor}"
    if paper_id:
        return f"S2:{paper_id}"
    return descriptor


def apply_semantic_scholar_enrichment(
    sources: Dict[str, Any],
    graph: Dict[str, Any],
    info: Dict[str, str],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not args.refresh_semantic_scholar:
        return sources, graph

    result = semantic_scholar_api.enrich_paper(
        title=info["title"],
        year=info["year"],
        api_key=args.semantic_scholar_key,
        limit=max(1, args.semantic_scholar_limit),
    )
    sync = {
        "attempted_at": now_iso(),
        "status": result.status,
        "http_status": result.http_status,
        "error": result.error,
    }
    sources["semantic_scholar_sync"] = sync
    graph["semantic_scholar_sync"] = sync

    if not result.ok or not result.data:
        return sources, graph

    data = result.data
    paper = data.get("paper", {}) or data.get("search_match", {})
    match = data.get("search_match", {})
    external_ids = paper.get("externalIds", {}) or match.get("externalIds", {}) or {}
    paper_url = (paper.get("url") or match.get("url") or "").strip()
    open_access = (paper.get("openAccessPdf") or {}).get("url", "").strip()
    author_names = [author.get("name", "").strip() for author in paper.get("authors", []) if author.get("name")]
    references = data.get("references", [])
    citations = data.get("citations", [])

    sources["identifiers"]["semantic_scholar_id"] = paper.get("paperId", "") or sources["identifiers"].get("semantic_scholar_id", "")
    if external_ids.get("ArXiv"):
        sources["identifiers"]["arxiv_id"] = sources["identifiers"].get("arxiv_id") or external_ids.get("ArXiv", "")
    sources["links"]["paper"] = merge_unique(sources["links"].get("paper", []), [paper_url, open_access])
    sources["people"]["authors"] = merge_unique(sources["people"].get("authors", []), author_names)
    sources["semantic_scholar"] = {
        "paper_id": paper.get("paperId", ""),
        "paper_url": paper_url,
        "open_access_pdf": open_access,
        "citation_count": paper.get("citationCount", 0),
        "influential_citation_count": paper.get("influentialCitationCount", 0),
        "reference_count": paper.get("referenceCount", 0),
        "references_status": data.get("references_status", ""),
        "citations_status": data.get("citations_status", ""),
        "references_sample": references,
        "citations_sample": citations,
    }

    graph["edges"]["citations"] = merge_unique(graph["edges"].get("citations", []), [semantic_edge_label(item) for item in references])
    graph["edges"]["cited_by"] = merge_unique(graph["edges"].get("cited_by", []), [semantic_edge_label(item) for item in citations])
    graph["edges"]["authored_by"] = merge_unique(graph["edges"].get("authored_by", []), author_names)
    graph["semantic_scholar"] = {
        "paper_id": paper.get("paperId", ""),
        "citation_count": paper.get("citationCount", 0),
        "influential_citation_count": paper.get("influentialCitationCount", 0),
        "reference_count": paper.get("referenceCount", 0),
        "references_sample": references,
        "citations_sample": citations,
    }
    return sources, graph


def build_bundle(
    info: Dict[str, str],
    paths: Dict[str, str],
    sources: Dict[str, Any],
    graph: Dict[str, Any],
    existing_manifest: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    artifacts_existing = existing_manifest.get("artifacts", {})
    tooling_existing = existing_manifest.get("tooling", {})
    semantic = detect_tool(
        args.semantic_scholar_cmd,
        ["semantic-scholar", "semantic-scholar-cli"],
        "RF_SEMANTIC_SCHOLAR_CMD",
    )
    marker = detect_tool(args.marker_cmd, ["marker_single", "marker"], "RF_MARKER_CMD")
    pdffigures = detect_tool(args.pdffigures_cmd, ["pdffigures2"], "RF_PDFFIGURES_CMD")

    bundle_dir = Path(paths["bundle_dir_abs"])
    parsed_dir = bundle_dir / "parsed"
    figures_dir = bundle_dir / "figures"
    tables_dir = bundle_dir / "tables"
    formulas_dir = bundle_dir / "formulas"
    note_path = Path(paths["analysis_note_abs"])
    semantic_sync = sources.get("semantic_scholar_sync", {})
    semantic_key_present = bool(args.semantic_scholar_key.strip() or os.environ.get("RF_SEMANTIC_SCHOLAR_API_KEY", "").strip())

    return {
        "schema_version": 1,
        "updated_at": now_iso(),
        "analysis_mode": "incremental_full",
        "paper": {
            "title": info["title"],
            "venue": info["venue"],
            "year": info["year"],
            "category": info["category"],
        },
        "paths": paths,
        "sources_ref": paths["sources_ref"],
        "graph_ref": paths["graph_ref"],
        "tooling": {
            "semantic_scholar": {
                "preferred_transport": "api_first",
                "api_key_present": semantic_key_present,
                "configured_command": semantic["configured_command"] or tooling_existing.get("semantic_scholar", {}).get("configured_command", ""),
                "detected_executable": semantic["detected_executable"] or tooling_existing.get("semantic_scholar", {}).get("detected_executable", ""),
                "status": semantic_sync.get("status") or (semantic["status"] if semantic["status"] != "missing" else tooling_existing.get("semantic_scholar", {}).get("status", "missing")),
                "last_http_status": semantic_sync.get("http_status", 0),
                "last_error": semantic_sync.get("error", ""),
            },
            "marker": {
                "configured_command": marker["configured_command"] or tooling_existing.get("marker", {}).get("configured_command", ""),
                "detected_executable": marker["detected_executable"] or tooling_existing.get("marker", {}).get("detected_executable", ""),
                "status": marker["status"] if marker["status"] != "missing" else tooling_existing.get("marker", {}).get("status", "missing"),
            },
            "pdffigures2": {
                "configured_command": pdffigures["configured_command"] or tooling_existing.get("pdffigures2", {}).get("configured_command", ""),
                "detected_executable": pdffigures["detected_executable"] or tooling_existing.get("pdffigures2", {}).get("detected_executable", ""),
                "status": pdffigures["status"] if pdffigures["status"] != "missing" else tooling_existing.get("pdffigures2", {}).get("status", "missing"),
            },
        },
        "sources_summary": {
            "paper_links": len(sources["links"]["paper"]),
            "venue_links": len(sources["links"]["venue"]),
            "project_links": len(sources["links"]["project"]),
            "repo_links": len(sources["links"]["repo"]),
            "hf_links": len(sources["links"]["hf"]),
            "authors": len(sources["people"]["authors"]),
            "teams": len(sources["people"]["teams"]),
        },
        "graph_summary": {
            "shape": "dag",
            "base_papers": len(graph["edges"]["base_papers"]),
            "citations": len(graph["edges"]["citations"]),
            "cited_by": len(graph["edges"]["cited_by"]),
        },
        "artifacts": {
            "parsed_markdown": {
                "path": (parsed_dir / "marker.md").relative_to(REPO_ROOT).as_posix(),
                "status": artifact_status(parsed_dir / "marker.md", artifacts_existing.get("parsed_markdown", {}).get("status", "")),
            },
            "sections_json": {
                "path": (parsed_dir / "sections.json").relative_to(REPO_ROOT).as_posix(),
                "status": artifact_status(parsed_dir / "sections.json", artifacts_existing.get("sections_json", {}).get("status", "")),
            },
            "figures_dir": {
                "path": figures_dir.relative_to(REPO_ROOT).as_posix(),
                "status": artifact_status(figures_dir, artifacts_existing.get("figures_dir", {}).get("status", "")),
            },
            "tables_dir": {
                "path": tables_dir.relative_to(REPO_ROOT).as_posix(),
                "status": artifact_status(tables_dir, artifacts_existing.get("tables_dir", {}).get("status", "")),
            },
            "formulas_dir": {
                "path": formulas_dir.relative_to(REPO_ROOT).as_posix(),
                "status": artifact_status(formulas_dir, artifacts_existing.get("formulas_dir", {}).get("status", "")),
            },
            "analysis_note": {
                "path": note_path.relative_to(REPO_ROOT).as_posix(),
                "status": "available" if note_path.exists() else artifacts_existing.get("analysis_note", {}).get("status", "pending"),
            },
        },
        "tasks": [
            {
                "name": "source_registry",
                "status": "initialized",
            },
            {
                "name": "citation_graph",
                "status": semantic_sync.get("status")
                or ("ready_for_refresh" if sources["identifiers"]["semantic_scholar_id"] or semantic_key_present or semantic["configured_command"] else "pending"),
            },
            {
                "name": "figure_table_extraction",
                "status": "ready_for_refresh" if marker["configured_command"] or pdffigures["configured_command"] else "pending",
            },
            {
                "name": "formula_preservation",
                "status": "ready_for_refresh" if marker["configured_command"] else "pending",
            },
            {
                "name": "analysis_note_sync",
                "status": "ready" if note_path.exists() else "pending",
            },
        ],
    }


def write_bundle(bundle: Dict[str, Any], sources: Dict[str, Any], graph: Dict[str, Any]) -> None:
    bundle_dir = Path(bundle["paths"]["bundle_dir_abs"])
    parsed_dir = bundle_dir / "parsed"
    figures_dir = bundle_dir / "figures"
    tables_dir = bundle_dir / "tables"
    formulas_dir = bundle_dir / "formulas"
    runs_dir = bundle_dir / "runs"
    for path in (bundle_dir, parsed_dir, figures_dir, tables_dir, formulas_dir, runs_dir):
        path.mkdir(parents=True, exist_ok=True)

    Path(bundle["paths"]["bundle_manifest_abs"]).write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(bundle["paths"]["sources_abs"]).write_text(
        json.dumps(sources, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(bundle["paths"]["graph_abs"]).write_text(
        json.dumps(graph, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    run_record = {
        "timestamp": now_iso(),
        "action": "prepare_incremental_bundle",
        "analysis_mode": bundle["analysis_mode"],
        "paper": bundle["paper"],
        "bundle_manifest_ref": bundle["paths"]["bundle_manifest_ref"],
        "sources_ref": bundle["paths"]["sources_ref"],
        "graph_ref": bundle["paths"]["graph_ref"],
        "tooling": bundle["tooling"],
    }
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_prepare_bundle.json")
    (runs_dir / run_name).write_text(json.dumps(run_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_bundle_dirs(bundle_dir: Path) -> None:
    for relative in ("", "parsed", "figures", "tables", "formulas", "runs"):
        (bundle_dir / relative).mkdir(parents=True, exist_ok=True)


def process_pdf(pdf: Path, args: argparse.Namespace) -> Dict[str, Any]:
    info = resolve_info(pdf, args)
    paths = build_paths(info, pdf)
    manifest_path = Path(paths["bundle_manifest_abs"])
    sources_path = Path(paths["sources_abs"])
    graph_path = Path(paths["graph_abs"])
    bundle_dir = Path(paths["bundle_dir_abs"])
    if not args.dry_run:
        ensure_bundle_dirs(bundle_dir)
    existing_manifest = load_json(manifest_path, {})
    existing_sources = load_json(sources_path, {})
    existing_graph = load_json(graph_path, {})
    sources = merge_sources(existing_sources, args, info, paths)
    graph = merge_graph(existing_graph, args, info)
    sources, graph = apply_semantic_scholar_enrichment(sources, graph, info, args)
    bundle = build_bundle(info, paths, sources, graph, existing_manifest, args)
    if not args.dry_run:
        write_bundle(bundle, sources, graph)
    return {
        "paper": bundle["paper"],
        "bundle_manifest_ref": bundle["paths"]["bundle_manifest_ref"],
        "sources_ref": bundle["paths"]["sources_ref"],
        "graph_ref": bundle["paths"]["graph_ref"],
        "analysis_note_ref": bundle["paths"]["analysis_note_ref"],
        "tooling": bundle["tooling"],
        "graph_summary": bundle["graph_summary"],
        "sources_summary": bundle["sources_summary"],
        "artifacts": bundle["artifacts"],
        "mode": "dry_run" if args.dry_run else "written",
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"[ERR] input not found: {input_path}", file=sys.stderr)
        return 2
    if not PAPER_PDFS_DIR.exists() or not PAPER_ANALYSIS_DIR.exists():
        print(f"[ERR] expected repository folders missing under: {REPO_ROOT}", file=sys.stderr)
        return 3

    pdfs = list(ingest_pdfs.iter_pdfs(input_path))
    if not pdfs:
        print("[ERR] no PDFs found", file=sys.stderr)
        return 4

    results: List[Dict[str, Any]] = []
    for pdf in sorted(set(pdfs), key=lambda p: p.as_posix().lower()):
        try:
            results.append(process_pdf(pdf, args))
        except Exception as exc:  # pragma: no cover - operator-facing path
            print(json.dumps({"input": pdf.as_posix(), "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 5

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
