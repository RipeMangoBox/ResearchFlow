"""LLM service — unified interface for Anthropic/OpenAI with tracking.

Every call is logged to model_runs for cost tracking and auditability.
Falls back to mock mode when no API key is configured.

v4: Garbage detection for proxy APIs, robust streaming, auto-retry.
"""

import asyncio
import json
import logging
import re
import time
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.models.system import ModelRun

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds

# Patterns that indicate a garbage/non-completion response from proxy API
_GARBAGE_PATTERNS = [
    re.compile(r"^我是\s*(Claude|ChatGPT|AI)", re.MULTILINE),
    re.compile(r"^I am (Claude|ChatGPT|an AI)", re.MULTILINE),
    re.compile(r"^I'm (Claude|ChatGPT|an AI)", re.MULTILINE),
    re.compile(r"由 Anthropic 开发"),
    re.compile(r"developed by (Anthropic|OpenAI)", re.IGNORECASE),
    re.compile(r"^(Hello|Hi)!?\s*(I'm|I am)\s*(Claude|an AI)", re.MULTILINE),
    re.compile(r"模型 ID 是"),
]


class LLMResponse:
    """Unified response from any LLM provider."""
    def __init__(self, text: str, input_tokens: int, output_tokens: int,
                 model: str, provider: str, latency_ms: int):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model
        self.provider = provider
        self.latency_ms = latency_ms


def _is_garbage_response(text: str) -> bool:
    """Detect if the response is a garbage/self-introduction from proxy API."""
    if not text:
        return True
    for pattern in _GARBAGE_PATTERNS:
        if pattern.search(text[:500]):
            return True
    return False


async def call_llm(
    prompt: str,
    system: str = "",
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    session: AsyncSession | None = None,
    paper_id: UUID | None = None,
    job_id: UUID | None = None,
    prompt_version: str = "v1",
) -> LLMResponse:
    """Call an LLM with retry and return structured response.

    Retries on:
    - Transient HTTP errors (429, 500, 502, 503, timeout)
    - Garbage responses (proxy API self-introduction)
    - Empty responses
    """
    start = time.monotonic()

    if not settings.anthropic_api_key and not settings.openai_api_key:
        resp = _mock_response(prompt, system)
    else:
        last_error = None
        resp = None
        for attempt in range(MAX_RETRIES):
            try:
                # Prefer OpenAI-compatible proxy when OPENAI_BASE_URL is set
                # (proxy handles routing; Anthropic direct often fails in CN)
                if settings.openai_api_key and getattr(settings, "openai_base_url", None):
                    default_model = settings.openai_model or "kimi-k2.6"
                    resp = await asyncio.wait_for(
                        _call_openai(prompt, system, model or default_model, max_tokens, temperature),
                        timeout=180,
                    )
                elif settings.anthropic_api_key:
                    # Use kimi-k2.6 when custom base_url is set (Kimi API)
                    default_model = "kimi-k2.6" if settings.anthropic_base_url else "claude-sonnet-4.6"
                    resp = await asyncio.wait_for(
                        _call_anthropic(prompt, system, model or default_model, max_tokens, temperature),
                        timeout=180,
                    )
                else:
                    default_model = settings.openai_model or "kimi-k2.6"
                    resp = await asyncio.wait_for(
                        _call_openai(prompt, system, model or default_model, max_tokens, temperature),
                        timeout=180,
                    )

                # ── Garbage detection ──────────────────────────
                if _is_garbage_response(resp.text):
                    last_error = f"Garbage response detected: {resp.text[:80]}"
                    logger.warning(
                        f"LLM garbage response (attempt {attempt+1}/{MAX_RETRIES}): "
                        f"{resp.text[:100]}"
                    )
                    resp = None  # Force retry
                else:
                    break  # Valid response

            except asyncio.TimeoutError:
                last_error = "LLM call timed out after 180s"
                logger.warning(f"LLM timeout (attempt {attempt+1}/{MAX_RETRIES})")
            except Exception as e:
                last_error = str(e)
                err_str = str(e).lower()
                if any(kw in err_str for kw in ("rate_limit", "429", "500", "502", "503", "timeout", "connection", "too large")):
                    logger.warning(f"LLM transient error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                else:
                    raise  # Non-transient error, fail immediately

            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.info(f"LLM retrying in {delay}s...")
                await asyncio.sleep(delay)

        if resp is None:
            raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}")

    # Log to model_runs
    if session:
        try:
            run = ModelRun(
                job_id=job_id,
                paper_id=paper_id,
                model_provider=resp.provider,
                model_name=resp.model,
                prompt_version=prompt_version[:50],  # Prevent varchar overflow
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost_usd=_estimate_cost(resp),
                latency_ms=resp.latency_ms,
            )
            session.add(run)
        except Exception as e:
            logger.debug(f"Failed to log model_run: {e}")

    return resp


async def _call_anthropic(prompt: str, system: str, model: str,
                          max_tokens: int, temperature: float) -> LLMResponse:
    import anthropic
    kwargs = {"api_key": settings.anthropic_api_key}
    # Support custom base_url for Kimi and other Anthropic-compatible APIs
    base_url = getattr(settings, "anthropic_base_url", None)
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.AsyncAnthropic(**kwargs)

    start = time.monotonic()
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages, "temperature": temperature}
    if system:
        kwargs["system"] = system

    response = await client.messages.create(**kwargs)
    latency = int((time.monotonic() - start) * 1000)

    text = response.content[0].text if response.content else ""
    return LLMResponse(
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model=model,
        provider="anthropic",
        latency_ms=latency,
    )


async def _call_openai(prompt: str, system: str, model: str,
                       max_tokens: int, temperature: float) -> LLMResponse:
    import openai
    kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    # Kimi API requires coding-agent User-Agent header
    kwargs["default_headers"] = {"User-Agent": "claude-code/1.0"}
    client = openai.AsyncOpenAI(**kwargs)

    start = time.monotonic()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Always use streaming — proxy APIs may truncate non-streaming responses
    stream = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, stream=True,
    )
    chunks = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
    latency = int((time.monotonic() - start) * 1000)
    text = "".join(chunks)
    return LLMResponse(
        text=text,
        input_tokens=len(prompt.split()),  # estimate (streaming has no usage)
        output_tokens=len(text.split()),
        model=model,
        provider="openai",
        latency_ms=latency,
    )


def _mock_response(prompt: str, system: str) -> LLMResponse:
    """Generate a structured mock response matching each agent's expected schema.

    Detects the agent type from system prompt keywords and returns
    the correct JSON structure so the pipeline can run end-to-end.
    """
    logger.info("LLM mock mode — no API key configured")

    sys_lower = system.lower()

    if "paper_essence" in sys_lower and "method_delta" in sys_lower:
        # shallow_extractor agent
        mock = {
            "paper_essence": {
                "problem_statement": "[Mock] The paper addresses a key challenge in the domain.",
                "core_claim": "[Mock] The proposed method achieves state-of-the-art performance.",
                "method_summary": "[Mock] A novel approach combining hierarchical representations with diffusion models.",
                "main_contributions": ["[Mock] Novel architecture", "[Mock] New training recipe"],
                "target_tasks": ["text-to-motion"],
                "target_modalities": ["text", "motion"],
                "training_paradigm": "supervised",
                "limitations": ["[Mock] Limited to short sequences"],
                "evidence_refs": [
                    {"claim": "[Mock] SOTA on HumanML3D", "confidence": 0.8, "basis": "experiment_backed", "reasoning": "Table 1"}
                ],
            },
            "method_delta": {
                "proposed_method_name": "[Mock] ProposedMethod",
                "baseline_methods": [{"name": "T2M-GPT", "role": "primary_baseline"}],
                "changed_slots": [
                    {"slot_name": "generation_backbone", "baseline_value": "autoregressive", "proposed_value": "diffusion", "change_type": "replace", "is_novel": True}
                ],
                "is_plugin_patch": False,
                "is_structural_change": True,
                "should_create_method_node": True,
                "creation_reason": "[Mock] Novel architecture",
                "key_equations": ["[Mock] L_diff = E[||noise - predicted||^2]"],
            },
        }
    elif "classify each reference" in sys_lower:
        # reference_role agent
        mock = {
            "classifications": [
                {"ref_index": "[1]", "ref_title": "T2M-GPT", "role": "direct_baseline", "confidence": 0.9,
                 "where_mentioned": ["method", "experiments"], "recommended_ingest_level": "deep",
                 "reason": "Primary baseline in experiment tables"},
                {"ref_index": "[2]", "ref_title": "MDM", "role": "comparison_baseline", "confidence": 0.8,
                 "where_mentioned": ["experiments"], "recommended_ingest_level": "abstract",
                 "reason": "Comparison in Table 1"},
            ],
            "anchor_baselines": ["[1]"],
            "method_sources": [],
        }
    elif "method" in sys_lower and "experiment" in sys_lower and "formulas" in sys_lower:
        # deep_analyzer agent
        mock = {
            "method": {
                "proposed_method_name": "[Mock] ProposedMethod",
                "baseline_methods": [{"name": "T2M-GPT", "role": "primary_baseline", "paper_title": None, "evidence_refs": []}],
                "changed_slots": [
                    {"slot_name": "generation_backbone", "baseline_value": "autoregressive", "proposed_value": "masked_modeling",
                     "change_type": "replaced", "is_novel": True, "evidence_refs": [{"claim": "Section 3.1", "section": "method", "confidence": 0.9}]}
                ],
                "new_components": [{"name": "[Mock] MaskedTransformer", "description": "Bidirectional masked token predictor", "role_in_pipeline": "core generator"}],
                "pipeline_modules": [{"name": "VQ-VAE", "input": "motion sequence", "output": "discrete tokens", "is_new": False, "replaces": None},
                                     {"name": "MaskedTransformer", "input": "masked tokens", "output": "predicted tokens", "is_new": True, "replaces": "GPT decoder"}],
                "should_create_method_node": True, "should_create_lineage_edge": True, "lineage_parent": "T2M-GPT",
            },
            "experiment": {
                "main_results": [{"benchmark": "HumanML3D", "metric": "FID", "proposed_score": 0.045, "baseline_scores": [{"name": "T2M-GPT", "score": 0.141}],
                                   "improvement": "-68%", "is_sota": True, "evidence_refs": [{"table_or_figure": "Table 1", "section": "experiments", "confidence": 0.95}]}],
                "ablations": [{"component_removed": "residual layers", "effect": "FID degrades to 0.228", "delta_value": 0.183, "delta_metric": "FID", "supports_core_claim": True}],
                "costs": {"training_compute": None, "inference_latency": None, "model_parameters": None, "gpu_type": None, "training_time": None},
                "fairness_assessment": {"are_comparisons_fair": True, "are_baselines_strongest": True, "missing_baselines": [], "potential_issues": [], "overall_evidence_strength": 0.85},
            },
            "formulas": {
                "key_formulas": [{"latex": "L = E_{t}[||\\epsilon - \\epsilon_\\theta(x_t, t)||^2]", "name": "Diffusion Loss",
                                  "explanation_zh": "[Mock] 标准扩散训练损失", "slot_affected": "objective", "differs_from_baseline": False, "baseline_formula_latex": None}],
                "pipeline_figure": {"description": "[Mock] VQ-VAE encoder → Masked Transformer → VQ-VAE decoder", "modules": [{"name": "VQ-VAE", "role": "tokenizer"}], "flow_description": "Input → Tokenize → Mask → Predict → Decode"},
                "figure_roles": [{"fig_ref": "Figure 1", "semantic_role": "pipeline", "description_zh": "[Mock] 方法流程图"}],
                "formula_derivation_steps": [],
            },
        }
    elif "knowledge graph" in sys_lower and "node_candidates" in sys_lower:
        # graph_candidate agent
        mock = {
            "node_candidates": [
                {"node_type": "method", "name": "[Mock] ProposedMethod", "name_zh": None, "one_liner": "[Mock] A novel masked modeling approach",
                 "evidence_refs": [{"claim": "Section 3", "section": "method", "confidence": 0.9}], "confidence": 0.85},
                {"node_type": "task", "name": "text-to-motion", "name_zh": "文本到运动生成", "one_liner": "Generate 3D motion from text",
                 "evidence_refs": [], "confidence": 0.95},
            ],
            "edge_candidates": [
                {"source_type": "paper", "source_ref": "this_paper", "target_type": "method", "target_ref": "[Mock] ProposedMethod",
                 "relation_type": "proposes_method", "slot_name": None, "one_liner": "This paper proposes the method",
                 "evidence_refs": [{"claim": "Abstract", "section": "abstract", "confidence": 0.95}], "confidence": 0.9},
                {"source_type": "paper", "source_ref": "this_paper", "target_type": "task", "target_ref": "text-to-motion",
                 "relation_type": "belongs_to_task", "slot_name": None, "one_liner": "Paper addresses text-to-motion",
                 "evidence_refs": [], "confidence": 0.9},
            ],
            "lineage_candidates": [
                {"child_method": "[Mock] ProposedMethod", "parent_method": "T2M-GPT", "relation": "builds_on",
                 "changed_slots": ["generation_backbone"], "evidence": "Section 3 describes modifications to T2M-GPT"},
            ],
        }
    elif "node_profiles" in sys_lower and "edge_profiles" in sys_lower:
        # kb_profiler agent
        mock = {
            "node_profiles": [
                {"node_name": "[Mock] ProposedMethod", "one_liner": "[Mock] 一种新的掩码建模方法",
                 "short_intro_md": "[Mock] 本方法通过掩码建模实现双向注意力。",
                 "detailed_md": "## 概述\n\n[Mock] 详细描述...\n\n## 核心特点\n\n- 双向注意力\n- 迭代解码",
                 "structured_json": {"architecture_type": "transformer", "training_paradigm": "supervised"},
                 "evidence_refs": [{"claim": "Novel architecture", "source_paper": "this_paper", "confidence": 0.8}]},
            ],
            "edge_profiles": [
                {"source_ref": "this_paper", "target_ref": "[Mock] ProposedMethod", "relation_type": "proposes_method",
                 "one_liner": "[Mock] 本文提出了 ProposedMethod", "relation_summary": "[Mock] 该方法在本文中首次提出",
                 "source_context": "[Mock] 来源论文视角", "target_context": "[Mock] 目标节点视角",
                 "evidence_refs": [{"claim": "Abstract", "source_paper": "this_paper", "confidence": 0.9}]},
            ],
        }
    elif "6 sections" in sys_lower or "paper report" in sys_lower.replace("_", " ") or "narrative report" in sys_lower:
        # paper_report agent (new 6-section narrative format)
        mock = {
            "title_zh": "[Mock] 方法深度分析报告",
            "title_en": "[Mock] Method Analysis Report",
            "sections": [
                {"section_type": "background_motivation", "title": "背景与动机", "body_md": "[Mock] 文本驱动3D运动生成面临质量与多样性的权衡。T2M-GPT 采用自回归方式���成离散 token，MDM 使用��散模型在连续空间去噪。前者速度快但缺乏全局一致性，后者质量好但推理慢。本文提出用掩码建模替代自回归，在离散 token 空间实现双向注意力。"},
                {"section_type": "core_innovation", "title": "核心创新", "body_md": "[Mock] 核心洞察：掩码建模允许每个 token 在生成时看到所有已确定的 token，打破自回归的单向约束。\n\n{{FIG:motivation}}"},
                {"section_type": "framework_overview", "title": "整体框架", "body_md": "[Mock] 整体架构分为三个阶段：\n\n{{FIG:pipeline}}\n\n1. **RVQ Tokenizer**: 输入运动序列 → 分层离散 token\n2. **Masked Transformer**: 基层 token ���掩码预测\n3. **Residual Transformer**: 残差层细化"},
                {"section_type": "module_decomposition", "title": "核心模块与公式", "body_md": "[Mock]\n### RVQ 量化器\n\n直觉：将连续运动信号离散化为多层 token。\n\n$$z = \\text{VQ}(x) + \\sum_{k=1}^{K} r_k$$\n\n### Masked Transformer\n\n$$p(m_i | m_{\\text{visible}}) = \\text{Transformer}(m_{\\text{visible}})$$"},
                {"section_type": "experiment_analysis", "title": "实验与分析", "body_md": "[Mock]\n| Benchmark | Metric | Ours | T2M-GPT | MDM |\n|-----------|--------|------|---------|-----|\n| HumanML3D | FID↓ | 0.045 | 0.141 | 0.544 |\n\n{{FIG:result}}\n\n消融：移除残差层 FID 0.045→0.228"},
                {"section_type": "lineage_positioning", "title": "方法谱系与定位", "body_md": "[Mock] 本方法属于离散运动生成方法族，继承 T2M-GPT 的 VQ 表示，将解码��从自回归替换为掩码建模。"},
            ],
            "figure_placements": [
                {"marker": "{{FIG:pipeline}}", "preferred_labels": ["Figure 1", "Figure 2"], "semantic_role": "pipeline"},
                {"marker": "{{FIG:result}}", "preferred_labels": ["Table 1"], "semantic_role": "result"},
            ],
        }
    else:
        # Fallback — return generic structure
        mock = {
            "problem_summary": "[Mock] This paper addresses a key challenge.",
            "method_summary": "[Mock] Novel approach proposed.",
            "evidence_summary": "[Mock] Experiments show improvements.",
            "core_intuition": "[Mock] Key insight is hierarchical decomposition.",
        }

    mock_text = json.dumps(mock, ensure_ascii=False, indent=2)
    return LLMResponse(
        text=mock_text, input_tokens=len(prompt.split()),
        output_tokens=len(mock_text.split()),
        model="mock", provider="mock", latency_ms=0,
    )


def _estimate_cost(resp: LLMResponse) -> float:
    """Rough cost estimate in USD."""
    costs = {
        # Kimi K2.6 (kimi.com/coding)
        "kimi-k2.6": (1.0 / 1_000_000, 4.0 / 1_000_000),
        # Legacy proxy model names (keep for old records)
        "so-4.6": (3.0 / 1_000_000, 15.0 / 1_000_000),
        "op-4.6": (15.0 / 1_000_000, 75.0 / 1_000_000),
        "mock": (0, 0),
    }
    rate = costs.get(resp.model, (1.0 / 1_000_000, 3.0 / 1_000_000))
    return resp.input_tokens * rate[0] + resp.output_tokens * rate[1]
