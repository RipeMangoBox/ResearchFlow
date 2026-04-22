#!/usr/bin/env python3
from __future__ import annotations

"""
batch_analyze_api_claude.py — Batch paper analysis through a Claude model
exposed by an OpenAI-compatible third-party gateway.

Reads Downloaded entries from analysis_log.csv, extracts PDF text via PyMuPDF,
sends to the configured Claude model, writes .md files, and updates log state.

Environment variables:
  export CLAUDE_API_KEY="sk-..."
  export CLAUDE_BASE_URL="https://your-gateway.example.com/v1"
  export CLAUDE_MODEL="claude-..."

Legacy env names are still accepted for compatibility:
  claude_OPENAI_API_KEY / claude_OPENAI_BASE_URL / claude_OPENAI_MODEL

Usage:
  pip install openai pymupdf
  cd <repo_root>
  python3 scripts/batch_analyze_api_claude.py [--batch-size 8] [--max-papers 100]
"""
import argparse, csv, difflib, os, pathlib, re, sys, time
from dataclasses import dataclass
import json


def _find_repo_root() -> pathlib.Path:
    """Walk upward until we find the ResearchFlow repository root."""
    current = pathlib.Path(__file__).resolve().parent
    required_entries = ("AGENTS.md", "paperAnalysis", "paperPDFs")
    for candidate in (current, *current.parents):
        if all((candidate / entry).exists() for entry in required_entries):
            return candidate
    raise RuntimeError(f"Could not locate repo root from {__file__}")


REPO_ROOT = _find_repo_root()

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ── Dependency check ──────────────────────────────────────────────
def _check_deps():
    missing = []
    if OpenAI is None:
        missing.append("openai")
    if fitz is None:
        missing.append("pymupdf")
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

# ── Config ────────────────────────────────────────────────────────
DEFAULT_LOG = "paperAnalysis/analysis_log.csv"
BATCH_SIZE = 4
MAX_PDF_TEXT_CHARS = 100_000  # ~50-60k tokens; covers most papers' core content
COOLDOWN_SECS = 5
API_TIMEOUT_SECS = 1800
API_MAX_RETRIES = 3
API_TEMPERATURE = 0.3
API_MAX_OUTPUT_TOKENS = 8000


@dataclass(frozen=True)
class ClaudeGatewayConfig:
    api_key: str
    base_url: str
    model: str


# ── Shared prompt templates ───────────────────────────────────────
PROMPT_TEMPLATE_DIR = pathlib.Path(__file__).resolve().parent / "prompt_templates"


def _load_prompt_template(filename: str) -> str:
    path = PROMPT_TEMPLATE_DIR / filename
    if not path.exists():
        raise RuntimeError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


SYSTEM_PROMPT = _load_prompt_template("batch_analyze_system_prompt.txt")

SURVEY_ROUTE_TERMS = (
    "survey",
    "systematic review",
    "literature review",
    "comprehensive review",
    "a review of",
    "review of recent",
    "review on ",
    "perspective",
    "taxonomy",
    "landscape",
    "tutorial",
    "综述",
    "调研",
    "综观",
)

TECH_REPORT_ROUTE_TERMS = (
    "technical report",
    "tech report",
    "model card",
    "system card",
    "model report",
    "system overview",
    "white paper",
    "技术报告",
    "系统概览",
    "模型卡",
    "系统卡",
    "系统报告",
)

PROMPT_ROUTE_DEFAULT = "default"
PROMPT_ROUTE_SURVEY = "survey"
PROMPT_ROUTE_TECH_REPORT = "technical_report"

QUALITATIVE_EVAL_TERMS = (
    "preliminary exploration",
    "preliminary explorations",
    "qualitative analysis",
    "qualitative",
    "capability boundary",
    "capability boundaries",
    "case study",
    "case-study",
    "exploration",
    "explorations",
    "probe",
    "probing",
    "定性",
    "能力边界",
    "探索",
    "探测",
)

SYNTHESIS_NOTE_TERMS = SURVEY_ROUTE_TERMS + (
    "overview",
    "overviews",
    "state of the art",
    "state-of-the-art",
    "research overview",
    "foundations and trends",
    "systematic overview",
    "literature landscape",
    "survey paper",
    "review paper",
    "synthesis",
    "系统综述",
    "系统梳理",
    "全景图",
    "路线图",
)

SYSTEM_DEMO_NOTE_TERMS = (
    "demo",
    "system demo",
    "prototype",
    "system prototype",
    "tool-augmented",
    "tool augmented",
    "prompt manager",
    "model composition",
    "zero training",
    "zero-shot system",
    "web service",
    "case study",
    "case-study",
    "qualitative benchmark",
    "purely qualitative",
    "no quantitative benchmark",
    "no quantitative metrics",
    "无定量",
    "纯定性",
    "系统原型",
    "工具调用",
    "多轮对话 demo",
    "web服务",
)

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
_LOCAL_PDF_INVENTORY: tuple[list[pathlib.Path], dict[str, list[pathlib.Path]]] | None = None

SURVEY_ROUTE_APPENDIX = _load_prompt_template("batch_analyze_survey_route_appendix.txt")

TECH_REPORT_ROUTE_APPENDIX = _load_prompt_template("batch_analyze_technical_report_route_appendix.txt")


# ── PDF validation & text extraction ──────────────────────────────
def validate_pdf(pdf_path: pathlib.Path) -> tuple[bool, str]:
    """Check if a file is a valid, non-corrupt PDF.
    Returns (ok, reason). Catches HTML-disguised-as-PDF, truncated downloads, etc."""
    # 1. Magic bytes check
    try:
        with open(pdf_path, "rb") as f:
            header = f.read(8)
    except OSError as e:
        return False, f"cannot read file: {e}"
    if not header.startswith(b"%PDF"):
        # Detect HTML error pages saved as .pdf
        if header[:5] in (b"<!DOC", b"<html", b"<HTML", b"<?xml"):
            return False, "file is HTML, not PDF (likely a 403/404 error page)"
        return False, f"bad magic bytes: {header[:8]!r}"

    # 2. Try opening with PyMuPDF
    try:
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        doc.close()
    except Exception as e:
        return False, f"PyMuPDF cannot open: {e}"
    if page_count == 0:
        return False, "PDF has 0 pages"

    # 3. File size sanity (< 5 KB is almost certainly incomplete)
    size_kb = pdf_path.stat().st_size / 1024
    if size_kb < 5:
        return False, f"file too small ({size_kb:.1f} KB), likely incomplete download"

    return True, "ok"


def extract_pdf_text(pdf_path: pathlib.Path, max_chars: int = MAX_PDF_TEXT_CHARS) -> str:
    """Extract text from PDF using PyMuPDF. Truncate if too long."""
    doc = fitz.open(str(pdf_path))
    texts = []
    total = 0
    for page in doc:
        t = page.get_text()
        if total + len(t) > max_chars:
            texts.append(t[:max_chars - total])
            break
        texts.append(t)
        total += len(t)
    doc.close()
    return "\n".join(texts)


# ── CSV helpers ───────────────────────────────────────────────────
def read_log(log_path: pathlib.Path):
    """Read analysis_log.csv, return (header, rows) where rows are lists."""
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def write_log(log_path: pathlib.Path, header, rows):
    """Write back analysis_log.csv."""
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def get_downloaded_indices(rows, redo=False, include_mismatch=False, include_api_failed=False):
    """Return list of row indices to process.
    --redo: process ALL entries (re-analyze everything).
    Normal: only process `Downloaded` rows. `analysis_mismatch` and `api_failed`
    retries stay opt-in so a normal run does not silently consume tokens on
    backlog cleanup."""
    if redo:
        return list(range(len(rows)))

    allowed = {"Downloaded"}
    if include_mismatch:
        allowed.add("analysis_mismatch")
    if include_api_failed:
        allowed.add("api_failed")
    return [i for i, r in enumerate(rows) if r and r[0] in allowed]


def _contains_any_term(text: str, terms: tuple[str, ...]) -> bool:
    """Case-insensitive substring match for lightweight route heuristics."""
    text = (text or "").lower()
    return any(term in text for term in terms)


def _read_first_env(*names: str) -> str:
    """Return the first non-empty environment variable value."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def load_gateway_config() -> ClaudeGatewayConfig:
    """Load gateway settings from env and ensure the selected model is Claude."""
    api_key = _read_first_env("CLAUDE_API_KEY", "claude_OPENAI_API_KEY")
    base_url = _read_first_env("CLAUDE_BASE_URL", "claude_OPENAI_BASE_URL")
    model = _read_first_env("CLAUDE_MODEL", "claude_OPENAI_MODEL")

    if not api_key:
        print("ERROR: Set CLAUDE_API_KEY (or legacy claude_OPENAI_API_KEY)")
        sys.exit(1)
    if not base_url:
        print("ERROR: Set CLAUDE_BASE_URL (or legacy claude_OPENAI_BASE_URL)")
        sys.exit(1)
    if not model:
        print("ERROR: Set CLAUDE_MODEL (or legacy claude_OPENAI_MODEL)")
        sys.exit(1)
    model_lower = model.lower()
    allowed_claude_aliases = ("claude", "op-", "sonnet", "opus", "haiku")
    if not any(alias in model_lower for alias in allowed_claude_aliases):
        print(f"ERROR: This script only supports Claude models, got model='{model}'")
        sys.exit(1)

    return ClaudeGatewayConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
    )


def build_gateway_client(config: ClaudeGatewayConfig) -> OpenAI:
    """Build an OpenAI-compatible client for the configured Claude gateway."""
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=API_TIMEOUT_SECS,
        max_retries=API_MAX_RETRIES,
    )


def _infer_prompt_route(paper_title: str, venue: str) -> str:
    """Choose a route-specific prompt for clearly marked survey/report papers.

    Default to method unless the title or venue explicitly signals survey/review
    or technical-report status. Restricting the heuristic to title + venue keeps
    routing stable and avoids PDF-body false positives.
    """
    sniff = " ".join(part for part in [paper_title, venue] if part)
    if _contains_any_term(sniff, TECH_REPORT_ROUTE_TERMS):
        return PROMPT_ROUTE_TECH_REPORT
    if _contains_any_term(sniff, SURVEY_ROUTE_TERMS):
        return PROMPT_ROUTE_SURVEY
    return PROMPT_ROUTE_DEFAULT


def _build_system_prompt(route: str) -> str:
    """Append a route-specific prompt suffix when special handling is needed."""
    if route == PROMPT_ROUTE_SURVEY:
        return SYSTEM_PROMPT + "\n\n" + SURVEY_ROUTE_APPENDIX
    if route == PROMPT_ROUTE_TECH_REPORT:
        return SYSTEM_PROMPT + "\n\n" + TECH_REPORT_ROUTE_APPENDIX
    return SYSTEM_PROMPT


def _is_survey_or_tr(row):
    """Check if a row should use the survey or technical-report route."""
    try:
        return _infer_prompt_route(row[2], row[3]) != PROMPT_ROUTE_DEFAULT
    except IndexError:
        return False


def summarize_selected_states(rows, indices) -> dict[str, int]:
    """Count selected rows by state for clearer retry diagnostics."""
    counts: dict[str, int] = {}
    for idx in indices:
        state = rows[idx][0] if rows[idx] else ""
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _normalize_title_key(text: str) -> str:
    """Normalize a title for fuzzy local-PDF matching."""
    text = (text or "").lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", text)


def _extract_arxiv_id(text: str) -> str:
    """Extract an arXiv id when present in paper links or filenames."""
    match = ARXIV_ID_RE.search(text or "")
    return match.group(1) if match else ""


def _extract_year_hint(text: str) -> str:
    """Extract a plausible publication year hint from a path/link/title snippet."""
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text or "")
    return match.group(0) if match else ""


def _get_local_pdf_inventory() -> tuple[list[pathlib.Path], dict[str, list[pathlib.Path]]]:
    """Cache a local inventory of PDFs for path repair fallbacks."""
    global _LOCAL_PDF_INVENTORY
    if _LOCAL_PDF_INVENTORY is not None:
        return _LOCAL_PDF_INVENTORY

    pdf_files = list((REPO_ROOT / "paperPDFs").rglob("*.pdf"))
    pdf_id_index: dict[str, list[pathlib.Path]] = {}
    for pdf in pdf_files:
        arxiv_id = _extract_arxiv_id(pdf.name)
        if not arxiv_id:
            continue
        pdf_id_index.setdefault(arxiv_id, []).append(pdf)

    _LOCAL_PDF_INVENTORY = (pdf_files, pdf_id_index)
    return _LOCAL_PDF_INVENTORY


def resolve_pdf_rel_from_local(
    paper_title: str,
    paper_link: str,
    current_pdf_rel: str,
    threshold: float = 0.72,
) -> tuple[str | None, str, float]:
    """Find an existing local PDF when the log path is stale or malformed."""
    title_key = _normalize_title_key(paper_title)
    if not title_key:
        return None, "", 0.0

    pdf_files, pdf_id_index = _get_local_pdf_inventory()
    current_parent = pathlib.Path(current_pdf_rel or "").parent.name.lower()
    target_year = (
        _extract_year_hint(current_pdf_rel)
        or _extract_year_hint(paper_link)
        or _extract_year_hint(paper_title)
    )
    best_path: pathlib.Path | None = None
    best_score = 0.0
    best_reason = ""

    def score_candidate(candidate: pathlib.Path, reason: str) -> None:
        nonlocal best_path, best_score, best_reason
        score = difflib.SequenceMatcher(
            None,
            title_key,
            _normalize_title_key(candidate.stem),
        ).ratio()
        candidate_year = _extract_year_hint(candidate.as_posix())
        if current_parent and candidate.parent.name.lower() == current_parent:
            score += 0.02
        # Title-only fallback should not silently repair to a clearly wrong year.
        if target_year and candidate_year:
            if target_year == candidate_year:
                score += 0.03
            elif reason == "title":
                score -= 0.20
        if score > best_score:
            best_path = candidate
            best_score = score
            best_reason = reason

    arxiv_id = _extract_arxiv_id(paper_link) or _extract_arxiv_id(current_pdf_rel)
    if arxiv_id:
        for candidate in pdf_id_index.get(arxiv_id, []):
            score_candidate(candidate, "arxiv_id")

    if best_score < threshold:
        for candidate in pdf_files:
            score_candidate(candidate, "title")

    if best_path is None or best_score < threshold:
        return None, best_reason, best_score

    return best_path.relative_to(REPO_ROOT).as_posix(), best_reason, best_score


# ── Analysis via API ──────────────────────────────────────────────
def _extract_completion_text(response) -> str:
    """Handle the common content shapes returned by OpenAI-compatible gateways."""
    if isinstance(response, str):
        text = _extract_text_from_sse_string(response)
        if text:
            return text
        if response.strip():
            return response.strip()
        raise RuntimeError("API returned an empty string response")

    try:
        content = response.choices[0].message.content
    except Exception as exc:
        raise RuntimeError("API response did not contain choices[0].message.content") from exc

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
                continue
            text = getattr(part, "text", "")
            if text:
                parts.append(text)
        if parts:
            return "".join(parts)

    raise RuntimeError("API returned empty text content")


def _extract_text_from_sse_string(raw: str) -> str:
    """Parse gateways that incorrectly return SSE frames as a plain string."""
    parts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for choice in data.get("choices", []):
            delta = choice.get("delta") or {}
            if isinstance(delta, dict):
                content = delta.get("content")
                if content:
                    parts.append(content)
                    continue
            message = choice.get("message") or {}
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
                    continue
            text = choice.get("text")
            if text:
                parts.append(text)
    return "".join(parts)


def analyze_one_paper(client: OpenAI, model: str,
                      pdf_text: str, pdf_path: str,
                      paper_title: str, venue: str,
                      prompt_route: str | None = None) -> str:
    """Send PDF text to the configured Claude gateway and return Markdown."""
    if prompt_route is None:
        prompt_route = _infer_prompt_route(paper_title, venue)
    system_prompt = _build_system_prompt(prompt_route)
    print(f"    [DEBUG] Prompt route={prompt_route}")

    user_msg = f"""Please analyze the following paper and produce the structured analysis note.

**Paper title**: {paper_title}
**Venue**: {venue}
**Analysis route**: {prompt_route}
**PDF path** (use in pdf_ref and ![[...]]): {pdf_path}

---
FULL PAPER TEXT:

{pdf_text}
    """
    payload_chars = len(system_prompt) + len(user_msg)
    print(f"    [DEBUG] API base_url={client.base_url} model={model} payload_chars={payload_chars}")
    response = client.chat.completions.create(
        model=model,
        temperature=API_TEMPERATURE,
        max_tokens=API_MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    result = _extract_completion_text(response)
    return _fix_frontmatter(result)


def _fix_frontmatter(md: str) -> str:
    """Fix common LLM output issues: strip ```yaml wrapper around frontmatter."""
    md = md.strip()
    # Remove leading ```yaml ... ``` wrapper around frontmatter
    if md.startswith("```yaml"):
        md = md[len("```yaml"):].strip()
    if md.startswith("```"):
        md = md[len("```"):].strip()
    # Ensure starts with ---
    if not md.startswith("---"):
        md = "---\n" + md
    # Fix closing ``` before second ---
    lines = md.split("\n")
    new_lines = []
    in_frontmatter = False
    for i, line in enumerate(lines):
        if line.strip() == "---" and not in_frontmatter:
            in_frontmatter = True
            new_lines.append(line)
        elif line.strip() == "```" and in_frontmatter:
            # Skip stray ``` inside frontmatter area
            continue
        elif line.strip() == "---" and in_frontmatter:
            in_frontmatter = False
            new_lines.append(line)
        else:
            new_lines.append(line)
    md = "\n".join(new_lines)
    return _normalize_frontmatter(md)


def _split_yaml_scalar_items(raw_value: str) -> list[str]:
    """Split semicolon-delimited YAML scalar content into clean items."""
    value = raw_value.strip().strip('"').strip("'")
    if not value or value.lower() == "n/a":
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def _render_yaml_scalar(original_value: str, normalized_value: str) -> str:
    """Preserve quote style when rewriting a scalar frontmatter value."""
    stripped = original_value.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        return f'"{normalized_value}"'
    if stripped.startswith("'") and stripped.endswith("'"):
        return f"'{normalized_value}'"
    return normalized_value


def _rewrite_frontmatter_scalar(frontmatter: str, key: str, transform) -> str:
    pattern = rf"(^\s*{re.escape(key)}\s*:\s*)(.+)$"

    def repl(match):
        original_value = match.group(2)
        new_value = transform(original_value)
        return match.group(1) + new_value

    return re.sub(pattern, repl, frontmatter, count=1, flags=re.MULTILINE)


def _infer_missing_task_tag(frontmatter: str, body: str) -> str:
    """Infer a conservative fallback task tag when the model omitted all task tags."""
    cat_match = re.search(r'^category:\s*(.+)$', frontmatter, re.MULTILINE)
    category = cat_match.group(1).strip().strip('"').strip("'") if cat_match else ""
    context = f"{frontmatter}\n{body}".lower()

    if any(term in context for term in ("text-to-video", "text to video", "t2v")):
        return "task/text-to-video"
    if category == "Video_Generation" or "video generation" in context:
        return "task/video-generation"
    if "video understanding" in context:
        return "task/video-understanding"
    if (
        "mllm-evaluation" in context
        or ("evaluation" in context and ("mllm" in context or "multimodal" in context))
    ):
        return "task/MLLM-evaluation"
    return ""


def _ensure_task_tag(frontmatter: str, body: str) -> str:
    """Add a conservative fallback task tag if the tags block has none."""
    tags_block_match = re.search(r'(^tags:\s*\n(?:\s+-\s+.+\n?)*)', frontmatter, re.MULTILINE)
    if not tags_block_match:
        return frontmatter

    tags_block = tags_block_match.group(1)
    if re.search(r'^\s+-\s+task/', tags_block, re.MULTILINE):
        return frontmatter

    task_tag = _infer_missing_task_tag(frontmatter, body)
    if not task_tag:
        return frontmatter

    insertion = f"  - {task_tag}\n"
    updated_tags_block = tags_block + insertion
    return frontmatter.replace(tags_block, updated_tags_block, 1)


def _normalize_frontmatter(md: str) -> str:
    """Repair common frontmatter formatting violations before validation."""
    fm_match = re.match(r"^(---\s*\n)(.*?)(\n---\s*\n?)(.*)$", md, re.DOTALL)
    if not fm_match:
        return md

    prefix, frontmatter, divider, body = fm_match.groups()

    def keep_first_item(original_value: str) -> str:
        items = _split_yaml_scalar_items(original_value)
        if len(items) <= 1:
            return original_value
        return _render_yaml_scalar(original_value, items[0])

    def cap_two_items(original_value: str) -> str:
        items = _split_yaml_scalar_items(original_value)
        if len(items) <= 2:
            return original_value
        return _render_yaml_scalar(original_value, "; ".join(items[:2]))

    frontmatter = _rewrite_frontmatter_scalar(frontmatter, "extends", keep_first_item)
    frontmatter = _rewrite_frontmatter_scalar(frontmatter, "competes_with", cap_two_items)
    frontmatter = _rewrite_frontmatter_scalar(frontmatter, "complementary_to", cap_two_items)
    frontmatter = _ensure_task_tag(frontmatter, body)

    return prefix + frontmatter + divider + body


def _is_dataset_optional_eval(md_text: str, tags: list[str], cat_val: str | None) -> bool:
    """Allow dataset-free evaluation notes for qualitative exploration papers."""
    if cat_val != "Evaluation":
        return False

    lowered = md_text.lower()
    if "qualitative-analysis" in tags:
        return True
    if _contains_any_term(lowered, QUALITATIVE_EVAL_TERMS):
        return True

    evidence_types = re.findall(r"\[evidence:\s*(\S+?)\]", md_text)
    return bool(evidence_types) and all(ev in {"case-study", "analysis"} for ev in evidence_types)


def _is_dataset_optional_synthesis_note(md_text: str) -> bool:
    """Allow dataset-free synthesis/landscape notes even if title routing missed them."""
    lowered = md_text.lower()
    evidence_types = re.findall(r"\[evidence:\s*(\S+?)\]", md_text)
    if not evidence_types or "synthesis" not in evidence_types:
        return False
    if not all(ev in {"synthesis", "analysis"} for ev in evidence_types):
        return False
    return _contains_any_term(lowered, SYNTHESIS_NOTE_TERMS)


def _is_dataset_optional_system_demo_note(
    md_text: str,
    tags: list[str],
    cat_val: str | None,
) -> bool:
    """Allow dataset-free tool/system demo notes dominated by case studies."""
    lowered = md_text.lower()
    evidence_types = re.findall(r"\[evidence:\s*(\S+?)\]", md_text)
    if not evidence_types:
        return False
    allowed_evidence = {"case-study", "analysis", "theoretical", "ablation"}
    if not all(ev in allowed_evidence for ev in evidence_types):
        return False
    if "case-study" not in evidence_types:
        return False

    demo_signals = set(tags) & {
        "tool-augmented-llm",
        "tool-augmented-LLM",
        "prompt-engineering",
        "model-composition",
        "visual-prompting",
    }
    if cat_val in {"Survey_Benchmark", "Evaluation"}:
        return False
    if demo_signals:
        return True
    return _contains_any_term(lowered, SYSTEM_DEMO_NOTE_TERMS)


def validate_structure(md_text: str, route_hint: str | None = None) -> tuple[bool, list[str]]:
    """Check that required sections and frontmatter fields exist.
    Returns (ok, list_of_issues). Empty list = all good."""
    issues = []

    # --- Section checks ---
    if "---" not in md_text:
        issues.append("missing YAML frontmatter delimiters")
    if "Part I" not in md_text:
        issues.append("missing Part I section")
    if "Part II" not in md_text:
        issues.append("missing Part II section")
    if "Agent Summary" not in md_text:
        issues.append("missing Agent Summary block")
    if "核心直觉" not in md_text and "Aha" not in md_text:
        issues.append("missing 核心直觉 / Aha subsection")
    if "Part III" not in md_text:
        issues.append("missing Part III section")

    for summary_key in [
        "task_path",
        "bottleneck",
        "mechanism_delta",
        "evidence_signal",
        "reusable_ops",
        "failure_modes",
        "open_questions",
    ]:
        if f"**{summary_key}**" not in md_text:
            issues.append(f"Agent Summary missing key {summary_key}")

    # --- Frontmatter field checks ---
    fm_match = re.search(r'^---\s*\n(.*?)\n---', md_text, re.DOTALL)
    if not fm_match:
        issues.append("cannot parse YAML frontmatter block")
        return (False, issues)

    fm = fm_match.group(1)

    # Required scalar fields
    for field in ["title:", "venue:", "year:", "tags:", "core_operator:",
                   "primary_logic:", "claims:", "pdf_ref:", "category:",
                   "evidence_strength:", "related_work_position:"]:
        if field not in fm:
            issues.append(f"frontmatter missing {field}")

    # --- Category validation ---
    ALLOWED_CATS = {
        "Multimodal_LLM", "Video_Generation", "Embodied_AI",
        "3D_Gaussian_Splatting", "Motion_Generation_Text_Speech_Music_Driven",
        "Survey_Benchmark", "Speech_Audio_Language_Models",
        "Multimodal_Hallucination", "Multimodal_Chain_of_Thought",
        "Multimodal_Instruction_Tuning_and_Latest_Works", "Evaluation",
        "Others",
    }
    cat_match = re.search(r'^category:\s*(.+)$', fm, re.MULTILINE)
    cat_val = None
    if cat_match:
        cat_val = cat_match.group(1).strip().strip('"').strip("'")
        if cat_val not in ALLOWED_CATS:
            issues.append(f"category '{cat_val}' not in allowed set")

    # --- Extract tags block only (lines between "tags:" and next top-level key) ---
    tags_block_match = re.search(r'^tags:\s*\n((?:\s+-\s+.+\n?)*)', fm, re.MULTILINE)
    if tags_block_match:
        tags_block = tags_block_match.group(1)
        tag_lines = re.findall(r'^\s+-\s+(.+)$', tags_block, re.MULTILINE)
        tags = [t.strip().strip('"').strip("'") for t in tag_lines]
    else:
        tags = []
        issues.append("tags field present but no list items found")
    if cat_val:
        cat_count = sum(1 for t in tags if t == cat_val)
        if cat_count == 0:
            issues.append(f"category '{cat_val}' not found in tags list")
        elif cat_count > 1:
            issues.append(f"category '{cat_val}' appears {cat_count}x in tags (expected 1)")
    extra_cats = [t for t in tags if t in ALLOWED_CATS and t != cat_val]
    if extra_cats:
        issues.append(f"extra category tags: {extra_cats} (only 1 allowed)")

    # --- Tag count checks (task 1-2, technique 1-3, repr 0-2) ---
    task_tags = [t for t in tags if t.startswith("task/")]
    if len(task_tags) == 0:
        issues.append("no task/ tags (need 1-2)")
    elif len(task_tags) > 2:
        issues.append(f"too many task/ tags: {len(task_tags)} (max 2)")

    technique_tags = [t for t in tags
                      if not t.startswith(("task/", "dataset/", "opensource/", "repr/"))
                      and t not in ALLOWED_CATS]
    if len(technique_tags) == 0:
        issues.append("no technique tags (need 1-3)")
    elif len(technique_tags) > 3:
        issues.append(f"too many technique tags: {len(technique_tags)} (max 3)")

    repr_tags = [t for t in tags if t.startswith("repr/")]
    if len(repr_tags) > 2:
        issues.append(f"too many repr/ tags: {len(repr_tags)} (max 2)")

    # --- opensource: exactly 1, valid enum ---
    ALLOWED_OS = {"opensource/full", "opensource/partial",
                  "opensource/pretrained-only", "opensource/promised", "opensource/no"}
    os_tags = [t for t in tags if t.startswith("opensource/")]
    if len(os_tags) == 0:
        issues.append("missing opensource/ tag")
    elif len(os_tags) > 1:
        issues.append(f"multiple opensource/ tags: {os_tags}")
    for osv in os_tags:
        if osv not in ALLOWED_OS:
            issues.append(f"invalid opensource value: '{osv}'")

    # --- Technique tags must NOT have prefix ---
    for tag in tags:
        if tag.startswith("technique-") or tag.startswith("technique/"):
            issues.append(f"technique tag has wrong prefix: '{tag}'")

    # --- evidence_strength must be valid enum ---
    ALLOWED_ES = {"weak", "moderate", "strong", "very_strong"}
    es_match = re.search(r'^evidence_strength:\s*(.+)$', fm, re.MULTILINE)
    if es_match:
        es_val = es_match.group(1).strip().strip('"').strip("'")
        if es_val not in ALLOWED_ES:
            issues.append(f"invalid evidence_strength: '{es_val}'")

    # --- related_work_position: sub-keys + format + item count ---
    if "related_work_position:" in fm:
        for subkey in ["extends:", "competes_with:", "complementary_to:"]:
            if subkey not in fm:
                issues.append(f"related_work_position missing sub-key {subkey}")

        # extends must be single item (no ; separator)
        ext_match = re.search(r'^\s*extends:\s*(.+)$', fm, re.MULTILINE)
        if ext_match:
            ext_val = ext_match.group(1).strip().strip('"').strip("'")
            if ";" in ext_val:
                issues.append(f"extends should be 1 item, got semicolons: '{ext_val[:80]}'")

        # competes_with: max 2 items
        cw_match = re.search(r'^\s*competes_with:\s*(.+)$', fm, re.MULTILINE)
        if cw_match:
            cw_val = cw_match.group(1).strip().strip('"').strip("'")
            if cw_val not in ("N/A", "n/a", ""):
                cw_items = [x.strip() for x in cw_val.split(";") if x.strip()]
                if len(cw_items) > 2:
                    issues.append(f"competes_with has {len(cw_items)} items (max 2)")

        # complementary_to: max 2 items
        ct_match = re.search(r'^\s*complementary_to:\s*(.+)$', fm, re.MULTILINE)
        if ct_match:
            ct_val = ct_match.group(1).strip().strip('"').strip("'")
            if ct_val not in ("N/A", "n/a", ""):
                ct_items = [x.strip() for x in ct_val.split(";") if x.strip()]
                if len(ct_items) > 2:
                    issues.append(f"complementary_to has {len(ct_items)} items (max 2)")

        # Reject YAML list syntax under related_work_position
        rwp_start = fm.find("related_work_position:")
        if rwp_start >= 0:
            rwp_rest = fm[rwp_start + len("related_work_position:"):]
            next_key = re.search(r'^\w+:', rwp_rest, re.MULTILINE)
            rwp_block = rwp_rest[:next_key.start()] if next_key else rwp_rest
            # Sub-key lines are fine; bare list items are not
            bare_lists = re.findall(r'^\s+-\s+(?!extends:|competes_with:|complementary_to:).+',
                                    rwp_block, re.MULTILINE)
            # Filter out lines that are sub-key values (indented under extends/competes/comp)
            if bare_lists:
                issues.append("related_work_position uses YAML list syntax (should be plain strings)")

    # --- claims: count (2-3) + evidence tags (robust matching) ---
    ALLOWED_EV = {"ablation", "comparison", "case-study", "theoretical", "analysis", "synthesis"}
    # Extract claims block: everything between "claims:" and the next top-level key
    claims_match = re.search(r'^claims:\s*\n((?:\s+-\s+.+\n?)*)', fm, re.MULTILINE)
    if claims_match:
        claims_block = claims_match.group(1)
        # Match any list item (with or without quotes, any prefix style)
        all_claims = re.findall(r'^\s+-\s+["\']?(.+?)["\']?\s*$', claims_block, re.MULTILINE)

        if len(all_claims) < 2:
            issues.append(f"only {len(all_claims)} claim(s) found (need 2-3)")
        elif len(all_claims) > 3:
            issues.append(f"{len(all_claims)} claims found (max 3)")

        for cl in all_claims:
            ev_match = re.search(r'\[evidence:\s*(\S+?)\]', cl)
            if not ev_match:
                issues.append(f"claim missing [evidence: TYPE]: '{cl[:70]}...'")
            elif ev_match.group(1) not in ALLOWED_EV:
                issues.append(f"invalid evidence type '{ev_match.group(1)}' in claim")
    else:
        if "claims:" in fm:
            issues.append("claims field present but contains no list items (need 2-3)")

    # --- dataset count: type-aware (survey + technical-report detection) ---
    dataset_tags = [t for t in tags if t.startswith("dataset/")]
    title_match = re.search(r'^title:\s*(.+)$', fm, re.MULTILINE)
    title_val = title_match.group(1).strip().strip('"').strip("'").lower() if title_match else ""
    venue_match = re.search(r'^venue:\s*(.+)$', fm, re.MULTILINE)
    venue_val = venue_match.group(1).strip().strip('"').strip("'").lower() if venue_match else ""
    co_match = re.search(r'^core_operator:\s*(.+)$', fm, re.MULTILINE)
    co_val = co_match.group(1).strip().lower() if co_match else ""
    type_context = " ".join(part for part in [title_val, venue_val, co_val] if part)

    is_survey_type = (
        route_hint == PROMPT_ROUTE_SURVEY
        or (
            cat_val == "Survey_Benchmark"
            and _contains_any_term(type_context, SURVEY_ROUTE_TERMS)
        )
    )
    is_technical_report_type = (
        route_hint == PROMPT_ROUTE_TECH_REPORT
        or _contains_any_term(type_context, TECH_REPORT_ROUTE_TERMS)
    )
    is_dataset_optional_eval = _is_dataset_optional_eval(md_text, tags, cat_val)
    is_dataset_optional_synthesis = _is_dataset_optional_synthesis_note(md_text)
    is_dataset_optional_system_demo = _is_dataset_optional_system_demo_note(md_text, tags, cat_val)
    if len(dataset_tags) == 0 and not (
        is_survey_type
        or is_technical_report_type
        or is_dataset_optional_eval
        or is_dataset_optional_synthesis
        or is_dataset_optional_system_demo
    ):
        issues.append("no dataset/ tags (required for method/benchmark papers)")

    return (len(issues) == 0, issues)


# ── Main loop ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Batch paper analysis via a Claude gateway")
    parser.add_argument("--log", default=DEFAULT_LOG, help="Path to analysis_log.csv")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Papers per session (default 4)")
    parser.add_argument("--max-papers", type=int, default=9999, help="Max total papers to process")
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_SECS, help="Seconds between papers")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without calling API")
    parser.add_argument("--start-from", type=int, default=0, help="Skip first N entries")
    parser.add_argument("--redo", action="store_true",
                        help="Re-analyze ALL entries including checked (overwrites existing .md)")
    parser.add_argument("--include-mismatch", action="store_true",
                        help="Also re-analyze rows marked analysis_mismatch without requiring --redo")
    parser.add_argument("--include-api-failed", action="store_true",
                        help="Also retry rows marked api_failed without requiring --redo")
    parser.add_argument("--max-chars", type=int, default=MAX_PDF_TEXT_CHARS,
                        help="Max chars to extract from PDF (default 100000). Use 200000 for surveys/TRs")
    parser.add_argument("--filter", choices=["all", "survey_tr", "regular"], default="all",
                        help="Filter paper type: 'survey_tr' = only survey/technical reports, "
                             "'regular' = skip survey/TR, 'all' = process everything (default)")
    args = parser.parse_args()

    _check_deps()

    config = load_gateway_config()
    client = build_gateway_client(config)

    print(f"[CONFIG] model={config.model} base_url={config.base_url}")
    print(f"[CONFIG] timeout={API_TIMEOUT_SECS}s max_retries={API_MAX_RETRIES}")

    log_path = REPO_ROOT / args.log
    if not log_path.exists():
        print(f"ERROR: {log_path} not found"); sys.exit(1)

    header, rows = read_log(log_path)
    dl_indices = get_downloaded_indices(
        rows,
        redo=args.redo,
        include_mismatch=args.include_mismatch,
        include_api_failed=args.include_api_failed,
    )
    print(f"[INFO] Entries to process: {len(dl_indices)}")
    selected_state_counts = summarize_selected_states(rows, dl_indices)
    if selected_state_counts:
        print(f"[INFO] Selected states: {selected_state_counts}")

    # Apply paper type filter
    if args.filter == "survey_tr":
        dl_indices = [i for i in dl_indices if _is_survey_or_tr(rows[i])]
        print(f"[INFO] Filter=survey_tr: {len(dl_indices)} survey/TR papers")
    elif args.filter == "regular":
        dl_indices = [i for i in dl_indices if not _is_survey_or_tr(rows[i])]
        print(f"[INFO] Filter=regular: {len(dl_indices)} regular papers")
    filtered_state_counts = summarize_selected_states(rows, dl_indices)
    if filtered_state_counts:
        print(f"[INFO] States after filter: {filtered_state_counts}")
    if (
        args.filter == "regular"
        and not args.redo
        and not args.include_mismatch
        and not args.include_api_failed
    ):
        regular_mismatch = sum(
            1 for row in rows
            if row[0] == "analysis_mismatch" and not _is_survey_or_tr(row)
        )
        if regular_mismatch:
            print(
                "[INFO] Note: regular non-checked rows also include "
                f"{regular_mismatch} analysis_mismatch entries. "
                "Add --include-mismatch to re-analyze them."
            )
        regular_api_failed = sum(
            1 for row in rows
            if row[0] == "api_failed" and not _is_survey_or_tr(row)
        )
        if regular_api_failed:
            print(
                "[INFO] Note: regular rows also include "
                f"{regular_api_failed} api_failed entries. "
                "Add --include-api-failed to retry them."
            )

    # Override max chars from CLI
    max_chars = args.max_chars
    print(f"[INFO] Max PDF chars: {max_chars:,}")

    # Apply start-from offset
    if args.start_from > 0:
        dl_indices = dl_indices[args.start_from:]
        print(f"[INFO] After --start-from {args.start_from}: {len(dl_indices)} remaining")

    # Limit
    dl_indices = dl_indices[:args.max_papers]
    total = len(dl_indices)
    print(f"[INFO] Will process: {total} papers, batch size: {args.batch_size}")
    print(f"[INFO] Model: {config.model}, Base URL: {config.base_url}")
    print()

    processed = 0
    success = 0
    skipped = 0
    failed = 0
    session_count = 0

    for batch_start in range(0, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total)
        batch_indices = dl_indices[batch_start:batch_end]
        session_count += 1

        print(f"{'='*60}")
        print(f"[SESSION {session_count}] Papers {batch_start+1}-{batch_end} of {total}")
        print(f"{'='*60}")

        for idx in batch_indices:
            row = rows[idx]
            paper_title = row[2]  # paper_title
            venue = row[3]        # venue
            pdf_rel = row[7]      # pdf_path
            pdf_abs = REPO_ROOT / pdf_rel

            def derive_md_paths(current_pdf_rel: str) -> tuple[str, pathlib.Path]:
                current_md_rel = current_pdf_rel.replace("paperPDFs", "paperAnalysis").replace(".pdf", ".md")
                return current_md_rel, REPO_ROOT / current_md_rel

            md_rel, md_abs = derive_md_paths(pdf_rel)

            processed += 1
            print(f"\n[{processed}/{total}] {paper_title}")
            print(f"  PDF: {pdf_rel}")
            print(f"  MD:  {md_rel}")

            # Check PDF exists and is valid
            if not pdf_abs.exists():
                repaired_pdf_rel, repair_reason, repair_score = resolve_pdf_rel_from_local(
                    paper_title,
                    row[5],  # paper_link
                    pdf_rel,
                )
                if repaired_pdf_rel:
                    row[7] = repaired_pdf_rel
                    pdf_rel = repaired_pdf_rel
                    pdf_abs = REPO_ROOT / pdf_rel
                    md_rel, md_abs = derive_md_paths(pdf_rel)
                    print(
                        "  ↺ Repaired missing PDF path via local "
                        f"{repair_reason or 'title'} match ({repair_score:.3f})"
                    )
                    print(f"  PDF: {pdf_rel}")
                    print(f"  MD:  {md_rel}")
                else:
                    print(f"  ⚠ PDF not found, skipping (state stays {row[0]})")
                    skipped += 1
                    continue

            pdf_ok, pdf_reason = validate_pdf(pdf_abs)
            if not pdf_ok:
                print(f"  ✗ Corrupt PDF: {pdf_reason} — marking pdf_broken")
                rows[idx][0] = "pdf_broken"
                failed += 1
                continue

            # Existing MD handling:
            #   --redo: always overwrite
            #   normal: re-analyze any selected non-terminal state and overwrite
            if md_abs.exists():
                if args.redo:
                    print(f"  ♻ Redo: overwriting existing MD")
                else:
                    print(f"  ♻ Existing MD found for state '{row[0]}': re-analyzing and overwriting")

            if args.dry_run:
                print(f"  [DRY RUN] Would analyze and write {md_rel}")
                continue

            # Extract PDF text
            try:
                pdf_text = extract_pdf_text(pdf_abs, max_chars=max_chars)
                if len(pdf_text.strip()) < 200:
                    print(f"  ⚠ PDF text too short ({len(pdf_text)} chars), skipping")
                    skipped += 1
                    continue
                print(f"  Extracted {len(pdf_text)} chars from PDF")
            except Exception as e:
                print(f"  ✗ PDF read error: {e}")
                skipped += 1
                continue

            # Call API
            try:
                print(f"  Calling API...")
                prompt_route = _infer_prompt_route(paper_title, venue)
                md_content = analyze_one_paper(client, config.model, pdf_text,
                                               pdf_rel, paper_title, venue,
                                               prompt_route=prompt_route)

                # Validate structure — if incomplete, still write but mark mismatch
                # Do NOT retry (saves tokens); user can re-process later
                struct_ok, struct_issues = validate_structure(md_content, route_hint=prompt_route)
                if not struct_ok:
                    print(f"  ⚠ Validation issues ({len(struct_issues)}):")
                    for iss in struct_issues:
                        print(f"    - {iss}")
                    rows[idx][0] = "analysis_mismatch"
                    md_abs.parent.mkdir(parents=True, exist_ok=True)
                    md_abs.write_text(md_content, encoding="utf-8")
                    failed += 1
                    continue

                # Write MD
                md_abs.parent.mkdir(parents=True, exist_ok=True)
                md_abs.write_text(md_content, encoding="utf-8")
                rows[idx][0] = "checked"
                success += 1
                print(f"  ✓ Written: {md_rel}")

            except Exception as e:
                print(f"  ✗ API error: {e} — marking api_failed, skipping")
                rows[idx][0] = "api_failed"
                failed += 1
                continue

            # Cooldown between papers
            if args.cooldown > 0:
                time.sleep(args.cooldown)

        # Save log after each batch (session boundary)
        if not args.dry_run:
            write_log(log_path, header, rows)
            print(f"\n[SESSION {session_count} DONE] Log saved. "
                  f"Success: {success}, Skipped: {skipped}, Failed: {failed}")

        # Inter-session cooldown
        if batch_end < total:
            print(f"\n[COOLDOWN] 10s before next session...")
            time.sleep(10)

    # Final summary
    print(f"\n{'='*60}")
    print(f"[FINISHED] Sessions: {session_count}, "
          f"Processed: {processed}, Success: {success}, "
          f"Skipped: {skipped}, Failed: {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
