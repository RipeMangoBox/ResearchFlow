#!/usr/bin/env python3
from __future__ import annotations

"""
batch_analyze_api.py — Batch paper analysis via a GPT gateway using the
OpenAI Responses API.

Reads Downloaded entries from analysis_log.csv, extracts PDF text via PyMuPDF,
sends to the configured GPT model, writes .md files, and updates log state.

Environment variables:
  export GPT_API_KEY="sk-..."
  export GPT_BASE_URL="https://api.openai.com/v1"
  export GPT_MODEL="gpt-5.4"

Legacy env names are still accepted for compatibility:
  gpt_OPENAI_API_KEY / gpt_OPENAI_BASE_URL / gpt_OPENAI_MODEL
  OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL

Optional:
  export GPT_REASONING_EFFORT="xhigh"   # default: xhigh

Usage:
  pip install openai pymupdf
  cd <repo_root>
  python3 scripts/batch_analyze_api.py [--batch-size 8] [--max-papers 100]
  python3 scripts/batch_analyze_api.py --api-model claude [...]
"""
import argparse, csv, json, os, pathlib, re, sys, time
from dataclasses import dataclass
from typing import NoReturn


def _find_repo_root() -> pathlib.Path:
    """Walk upward until we find the ResearchFlow repository root."""
    current = pathlib.Path(__file__).resolve().parent
    required_entries = ("AGENTS.md", "paperAnalysis", "paperPDFs")
    for candidate in (current, *current.parents):
        if all((candidate / entry).exists() for entry in required_entries):
            return candidate
    raise RuntimeError(f"Could not locate repo root from {__file__}")


REPO_ROOT = _find_repo_root()


def _dispatch_to_claude(argv: list[str]) -> NoReturn:
    """Delegate the current invocation to the Claude batch analyzer."""
    claude_script = REPO_ROOT / "scripts" / "batch_analyze_api_claude.py"
    filtered: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--api-model":
            skip_next = True
            continue
        if arg.startswith("--api-model="):
            continue
        filtered.append(arg)
    print(f"[INFO] Delegating to Claude batch analyzer: {claude_script}")
    os.execv(sys.executable, [sys.executable, str(claude_script), *filtered])


def md_has_agent_summary(md_path: pathlib.Path) -> bool:
    """Treat an existing note with the Agent Summary callout as usable."""
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return "[!info] **Agent Summary**" in text

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
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "xhigh"


@dataclass(frozen=True)
class GPTGatewayConfig:
    api_key: str
    base_url: str
    model: str
    reasoning_effort: str


# ── System prompt (condensed from SKILL.md) ───────────────────────
SYSTEM_PROMPT = r"""You are a research paper analyst. Given the full text of an academic PDF, produce a structured analysis note in Markdown.

# A. Hard Schema (MUST follow exactly)

## CRITICAL FORMAT RULES
- The YAML frontmatter MUST start with a line containing only `---` and end with a line containing only `---`. Do NOT wrap it in ```yaml code blocks.
- The very first line of your output MUST be `---` (the opening frontmatter delimiter).

## YAML frontmatter (raw, NOT in code block)
---
title: "Paper Title"
venue: VENUE
year: YYYY  # or UnknownYear if not determinable
tags:
  - CategoryTag
  - task/primary-task
  - technique-tag
  - dataset/DatasetName
  - opensource/status
core_operator: 一句话描述核心机制
primary_logic: |
  输入条件 → 关键变换步骤 → 输出
claims:
  - "Claim 1: verifiable statement [evidence: ablation|comparison|case-study|theoretical|analysis|synthesis]"
  - "Claim 2: verifiable statement [evidence: ...]"
related_work_position:
  extends: "Method/Paper it builds upon"
  competes_with: "Direct competitor method(s)"
  complementary_to: "Orthogonal method(s) that could combine"
evidence_strength: moderate
pdf_ref: PDF_PATH_PLACEHOLDER
category: CategoryTag
---

## Tag convention (STRICT)
- Category (exactly 1, no prefix, Snake_Case): `Multimodal_LLM`, `Video_Generation`, `Embodied_AI`, `3D_Gaussian_Splatting`, `Motion_Generation`, `Survey_Benchmark`, `Speech_Audio_Language_Models`, `Multimodal_Hallucination`, `Multimodal_Chain_of_Thought`, `Multimodal_Instruction_Tuning_and_Latest_Works`, `Evaluation`, `Others`
- Task (1-2, prefix `task/`, kebab-case): `task/visual-question-answering`, `task/text-to-motion`, `task/video-understanding`, `task/MLLM-evaluation`
- For generic video generation papers, prefer `task/video-generation`; use `task/text-to-video` only when text-conditioned generation is the clear primary interface.
- Technique (1-3, NO prefix, kebab-case): `diffusion`, `mixture-of-experts`, `state-token`, `reinforcement-learning`, `flow-matching`, `vq-vae`
- Dataset (≥1 for method/benchmark papers; 0 allowed for survey/review/perspective/technical-report/system-overview/theoretical papers — prefix `dataset/`, original casing): `dataset/HumanML3D`, `dataset/MMBench`, `dataset/Video-MME`
- Representation (0-2, prefix `repr/`): `repr/SMPL`, `repr/joint-rotation`
- Open source (exactly 1, prefix `opensource/`): `opensource/full`, `opensource/partial`, `opensource/pretrained-only`, `opensource/promised`, `opensource/no`

WRONG examples (DO NOT generate):
- `technique-state-token` → should be `state-token` (technique has NO prefix)
- `technique/dual-model` → should be `dual-model-architecture` (technique has NO prefix)
- `MultimodalLLM` → should be `Multimodal_LLM` (category uses Snake_Case)
- `Survey` → should be `Survey_Benchmark` (category must be descriptive)
- `opensource/yes` → should be `opensource/full` (use exact values)
- `Benchmark` → should be `Survey_Benchmark` (not a valid standalone category)

## evidence_strength values
- weak: only case studies or qualitative examples, no controlled comparison
- moderate: comparisons on standard benchmarks but limited ablation or few datasets
- strong: thorough ablation + multi-dataset comparison + clear baselines
- very_strong: above + human evaluation / reproducibility evidence / cross-domain transfer

Calibration rules (STRICT — bias toward conservative):
- Evidence on only 1 dataset or lacking ablation → cap at moderate
- Use strong only when there is BOTH meaningful comparison AND solid causal/ablation support
- Use very_strong rarely; reserve for unusually comprehensive evidence
- When uncertain between two levels, choose the lower one

## related_work_position rules
- extends: exactly 1 named method/paper if identifiable, else "N/A". Single string.
- competes_with: 1-2 most direct competitors only. Use "; " to separate if multiple. Single string.
- complementary_to: 0-2 items only. Use "; " to separate if multiple. Single string, or "N/A".
- Prefer named methods/papers (e.g. "LoRA (Hu et al. 2021)"), NOT broad areas like "existing MLLMs".
- For survey/review/technical-report papers: N/A is normal and expected for most sub-fields; do not force-fill.

## Unknown / N/A policy (STRICT)
- venue unknown → `venue: arXiv` (or `Technical Report` if clearly not a preprint)
- year unknown → `year: UnknownYear` (do NOT guess)
- no project/code link found → keep paper/arXiv link if identifiable; omit missing project/code link; set `opensource/no`
- no clear dataset for survey/review/perspective/technical report/system overview → omit dataset tags entirely (allowed for those document types)
- cannot determine a field → use the explicit unknown marker above; NEVER fabricate

## claims quality rules
Each claim must be self-contained, verifiable, and falsifiable without reading the full paper.
Append `[evidence: TYPE]` inline where TYPE is one of: ablation, comparison, case-study, theoretical, analysis, synthesis.
- ablation / comparison / case-study / theoretical: standard for method papers
- analysis: for benchmark papers and technical/system reports (e.g. "reveals capability gap X via protocol Y")
- synthesis: for survey/review papers (e.g. "identifies trend X across N papers")
Good: "Method X achieves SOTA FID on HumanML3D without task-specific fine-tuning [evidence: comparison]"
Bad: "This paper proposes a novel method" (not verifiable)

## category field
The `category` frontmatter value MUST exactly match the Category tag in `tags`.

# B. Content Objectives

## Body sections

### 1. Title + Quick Links & TL;DR
Level-1 heading = paper title.
Callout block:
> [!abstract] **Quick Links & TL;DR**
> - **Links**: arXiv/project links (omit if none found)
> - **Summary**: one-sentence core contribution (Chinese)
> - **Key Performance**: 1-2 key metrics

### 2. Agent Summary Block (MUST include immediately after TL;DR)
Use this exact callout title and key names:
> [!info] **Agent Summary**
> - **task_path**: <input modality / problem setting -> output target>
> - **bottleneck**: <the true bottleneck this paper addresses>
> - **mechanism_delta**: <the key mechanism change in one sentence>
> - **evidence_signal**: <the strongest supporting evidence signal>
> - **reusable_ops**: [op1, op2]
> - **failure_modes**: [failure1, failure2]
> - **open_questions**: [question1, question2]

Rules:
- If a field is unclear, write `N/A` instead of fabricating.
- Keep each field concise and extraction-friendly.
- `reusable_ops`, `failure_modes`, and `open_questions` should use bracketed list form even if there is only 1 item.

### 3. `## Part I：问题与挑战`
What hard problem? Where is the bottleneck? Input/output interface. Boundary conditions.

### 4. `## Part II：方法与洞察`
Design philosophy. MUST include `### 核心直觉` subsection:
- what changed → which distribution/constraint/information bottleneck changed → what capability changed
- why this design works (causal, not restatement)
- strategic trade-offs table

### 5. `## Part III：证据与局限`
Key experiment signals (signal type + conclusion, not number dumping). 1-2 key metrics.
**局限性** must be specific boundary conditions, NOT vague "future work". Use this micro-template:
- Fails when: (input distribution / scale / modality where method breaks)
- Assumes: (supervision type, data requirements, backbone dependency)
- Not designed for: (explicit non-goals or out-of-scope scenarios)
Explicitly mention major resource/dependency assumptions (closed-source API, compute scale, special hardware, annotation cost) if they materially affect reproducibility or extensibility.
Reusable components.

### 6. Local PDF reference
![[PDF_PATH_PLACEHOLDER]]

## The Three Questions (every analysis must answer)
1. What/Why: What is the true bottleneck? Why solve it now?
2. How: What key causal knob did the authors introduce? What changed → which distribution/constraint/information bottleneck changed → what capability changed?
3. So what: Where is the capability jump vs prior work? Which experiment signals best support that claim?

## Expression priority
Intuition / structured explanation > mechanism-level abstraction > minimal symbolic notation.
Prioritize system-level causality over implementation details.
If formulas must appear: at most 1-2 verbalized objective descriptions, no derivations.

## Rules
- Analysis language: Chinese (zh). YAML keys stay English.
- Do NOT derive formulas or explain loss terms.
- Keep concise: problem → method → capability change thread.
- claims: 2-3 verifiable, falsifiable statements with [evidence: TYPE] tag.
- core_operator and primary_logic in Chinese.

# C. Document-Type Adapter

## How to determine paper type (apply in order)
- If the primary contribution is a new model/method/algorithm/system for solving a task → **method** (default)
- If the primary contribution is evaluation design, benchmark construction, scoring protocol, or failure diagnosis → **benchmark/evaluation**
- If the primary contribution is taxonomy, synthesis, landscape review, perspective, or trend analysis → **survey/review**
- If the primary contribution is system disclosure, training/deployment report, model/system card, or system overview rather than taxonomy → **technical report / system overview**
- If mixed, choose based on the main claimed contribution. When uncertain, default to **method**.
- Do NOT switch away from method unless the primary contribution is clearly evaluation design or synthesis/taxonomy.

Adapt emphasis based on paper type. The section structure (Part I/II/III) stays the same, but content focus shifts:

## Method paper (default)
- Part I: true bottleneck, why now
- Part II: causal knob, what distribution/constraint changed, trade-offs table
- Part III: experiment signals (ablation + comparison), specific failure boundaries, reusable operators

## Benchmark / Evaluation paper
- core_operator → describe the evaluation design innovation (scoring mechanism, task decomposition, failure taxonomy)
- primary_logic → `评测目标 → 数据/标注设计 → 评分策略 → 揭示的能力边界`
- Part I: what existing evaluations miss, why current metrics are insufficient
- Part II: evaluation coverage dimensions, scoring/protocol design, what failure modes it exposes. `### 核心直觉` should explain what measurement bottleneck changed and what diagnostic capability was gained
- Part III: key findings about model weaknesses, ranking surprises, human-vs-model gaps. Limitations = coverage gaps, annotation biases, format constraints

## Survey / Review / Perspective
- core_operator → describe the organizational/analytical contribution (taxonomy, landscape decomposition, trend identification)
- primary_logic → `领域范围 → 分类框架/分析维度 → 关键发现/未解决张力`
- Part I: scope and motivation for the survey, what landscape confusion exists
- Part II: taxonomy structure, key dimensions of comparison, identified trends. `### 核心直觉` should explain what conceptual organization was missing and what clarity was gained
- Part III: key takeaways, unresolved tensions, frontier directions. Limitations = coverage boundaries, recency cutoff, methodology bias

## Technical Report / System Overview / Model Card / System Card
- core_operator → describe the system/report contribution (system composition, training/alignment recipe, deployment framing, safety boundary disclosure)
- primary_logic → `系统目标/产品形态 → 组件/训练与对齐配置 → 能力覆盖/部署边界`
- Part I: target usage context, why this system/report matters now, and what operational or capability bottleneck it addresses
- Part II: architecture stack, training recipe, alignment/safety mechanisms, interface design, and major integration choices. `### 核心直觉` should explain what system-level knob changed and what capability/deployment benefit it enabled
- Part III: evaluation signals or demos, resource dependencies, serving constraints, safety boundaries, and explicit non-goals. If no benchmark suite is central, keep evidence_strength conservative and discuss evidence qualitatively

## Benchmark tag exemplar (correct)
```yaml
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - nonlinear-scoring
  - group-based-evaluation
  - dataset/Video-MME-v2
  - opensource/full
category: Survey_Benchmark
```
Note: technique tags have NO prefix even for benchmark papers. `Benchmark` alone is NOT a valid category — use `Survey_Benchmark`.
"""

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

SURVEY_ROUTE_APPENDIX = r"""
# D. Routed Focus Override — Survey / Review / Perspective
The script classified this document as a **survey/review/perspective**. Treat it as synthesis/taxonomy work, NOT as a system report.

- core_operator → summarize the synthesis/taxonomy contribution, not an implementation module
- primary_logic → `领域范围 → 分类框架/证据组织 → 关键趋势/未解决张力`
- claims should usually use `synthesis`; use `analysis` only if the paper performs explicit meta-analysis or diagnostic comparison across methods
- dataset tags are optional. Include them only when named datasets/benchmarks are central recurring anchors in the survey
- related_work_position: `N/A` is normal across all three sub-fields; do not force competitors
- Part I: scope, motivation, and what confusion/gap in the literature this synthesis resolves
- Part II: taxonomy axes, comparison dimensions, and what conceptual clarity is gained
- Part III: takeaways, open tensions, recency/coverage limits, and frontier questions
"""

TECH_REPORT_ROUTE_APPENDIX = r"""
# D. Routed Focus Override — Technical Report / System Overview / Model Card / System Card
The script classified this document as a **technical report / system overview**. Do NOT analyze it as a survey unless the main contribution is taxonomy/review.

- core_operator → summarize the system/report contribution (stack design, training/alignment recipe, deployment framing, safety boundary disclosure)
- primary_logic → `系统目标/产品形态 → 组件/训练与对齐配置 → 能力覆盖/部署边界`
- claims should usually use `analysis`, `comparison`, or `case-study`; use `synthesis` only if the report explicitly surveys a broader field
- dataset tags are optional. Include them only when named evaluation datasets/benchmarks are central evidence in the report
- related_work_position: `N/A` is normal if the document is mainly a disclosure/report rather than a direct competitor paper
- Part I: target usage context, why the system/report matters now, and which operational bottleneck or deployment need it addresses
- Part II: architecture stack, training/alignment choices, interface design, and major system integration decisions
- Part III: evaluation signals or demos, resource dependencies, serving constraints, safety boundaries, and explicit non-goals. If evidence is mostly demonstrations or internal studies, keep `evidence_strength` conservative
"""


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


def _read_first_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable value."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def load_gateway_config() -> GPTGatewayConfig:
    """Resolve GPT gateway config from env."""
    api_key = _read_first_env("GPT_API_KEY", "gpt_OPENAI_API_KEY", "OPENAI_API_KEY")
    base_url = _read_first_env(
        "GPT_BASE_URL",
        "gpt_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
        default=DEFAULT_BASE_URL,
    )
    model = _read_first_env(
        "GPT_MODEL",
        "gpt_OPENAI_MODEL",
        "OPENAI_MODEL",
        default=DEFAULT_MODEL,
    )
    reasoning_effort = _read_first_env(
        "GPT_REASONING_EFFORT",
        "OPENAI_REASONING_EFFORT",
        default=DEFAULT_REASONING_EFFORT,
    ).lower()

    if not api_key:
        print("ERROR: Set GPT_API_KEY (or legacy gpt_OPENAI_API_KEY / OPENAI_API_KEY)")
        sys.exit(1)
    if not base_url:
        print("ERROR: Set GPT_BASE_URL (or legacy gpt_OPENAI_BASE_URL / OPENAI_BASE_URL)")
        sys.exit(1)
    if not model:
        print("ERROR: Set GPT_MODEL (or legacy gpt_OPENAI_MODEL / OPENAI_MODEL)")
        sys.exit(1)
    allowed_reasoning_efforts = {"none", "minimal", "low", "medium", "high", "xhigh"}
    if reasoning_effort not in allowed_reasoning_efforts:
        print(
            "ERROR: GPT_REASONING_EFFORT must be one of "
            f"{sorted(allowed_reasoning_efforts)}, got '{reasoning_effort}'"
        )
        sys.exit(1)

    return GPTGatewayConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        reasoning_effort=reasoning_effort,
    )


def build_gateway_client(config: GPTGatewayConfig) -> OpenAI:
    """Build an OpenAI client for the selected GPT gateway."""
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


# ── Analysis via API ──────────────────────────────────────────────
def _extract_response_text(response) -> str:
    """Extract text from Responses API outputs with a few compatibility fallbacks."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    if isinstance(response, str):
        text = _extract_text_from_sse_string(response)
        if text:
            return text
        if response.strip():
            return response.strip()
        raise RuntimeError("API returned an empty string response")

    try:
        output = response.output
    except Exception as exc:
        raise RuntimeError("API response did not contain output_text or output items") from exc

    parts = []
    for item in output or []:
        content_list = getattr(item, "content", None)
        if not content_list and isinstance(item, dict):
            content_list = item.get("content")
        for part in content_list or []:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
                    continue
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


def analyze_one_paper(client: OpenAI, config: GPTGatewayConfig,
                      pdf_text: str, pdf_path: str,
                      paper_title: str, venue: str,
                      prompt_route: str | None = None) -> str:
    """Send PDF text to the configured GPT gateway and return Markdown."""
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
    print(f"    [DEBUG] API base_url={client.base_url} model={config.model} payload_chars={payload_chars}")
    stream_text_parts = []
    with client.responses.stream(
        model=config.model,
        instructions=system_prompt,
        input=user_msg,
        store=False,
        reasoning={"effort": config.reasoning_effort},
        temperature=API_TEMPERATURE,
        max_output_tokens=API_MAX_OUTPUT_TOKENS,
    ) as stream:
        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    stream_text_parts.append(delta)
        response = stream.get_final_response()

    result = "".join(stream_text_parts).strip() or _extract_response_text(response)
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
        "3D_Gaussian_Splatting", "Motion_Generation",
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


# Keep prompt/routing/validation requirements exactly aligned with the Claude
# variant. Provider-specific gateway code stays local in this GPT script.
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from batch_analyze_api_claude import (
    _build_system_prompt,
    _fix_frontmatter,
    _infer_prompt_route,
    _is_survey_or_tr,
    extract_pdf_text,
    get_downloaded_indices,
    read_log,
    resolve_pdf_rel_from_local,
    summarize_selected_states,
    validate_pdf,
    validate_structure,
    write_log,
)


# ── Main loop ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Batch paper analysis via a GPT Responses API gateway")
    parser.add_argument("--api-model", choices=["gpt", "claude"], default="gpt",
                        help="Compatibility switch. Use 'claude' to delegate to batch_analyze_api_claude.py")
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

    if args.api_model == "claude":
        _dispatch_to_claude(sys.argv[1:])

    config = load_gateway_config()
    client = build_gateway_client(config)

    print(f"[CONFIG] model={config.model} base_url={config.base_url}")
    print(
        f"[CONFIG] timeout={API_TIMEOUT_SECS}s max_retries={API_MAX_RETRIES} "
        f"reasoning_effort={config.reasoning_effort} store=false"
    )

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
                if not args.redo and md_has_agent_summary(md_abs):
                    rows[idx][0] = "checked"
                    success += 1
                    print("  ✓ Existing MD already has Agent Summary: marking checked and skipping")
                    continue
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
                md_content = analyze_one_paper(client, config, pdf_text,
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
