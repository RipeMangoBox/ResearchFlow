"""LLM service — unified interface for Anthropic/OpenAI with tracking.

Every call is logged to model_runs for cost tracking and auditability.
Falls back to mock mode when no API key is configured.
"""

import json
import logging
import time
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.models.system import ModelRun

logger = logging.getLogger(__name__)


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
    """Call an LLM and return structured response.

    Provider selection:
    1. If ANTHROPIC_API_KEY is set → use Claude
    2. If OPENAI_API_KEY is set → use OpenAI
    3. Otherwise → mock mode (returns placeholder)
    """
    start = time.monotonic()

    if settings.anthropic_api_key:
        resp = await _call_anthropic(prompt, system, model or "claude-sonnet-4-20250514", max_tokens, temperature)
    elif settings.openai_api_key:
        default_model = settings.openai_model or "gpt-4o-mini"
        resp = await _call_openai(prompt, system, model or default_model, max_tokens, temperature)
    else:
        resp = _mock_response(prompt, system)

    # Log to model_runs
    if session:
        run = ModelRun(
            job_id=job_id,
            paper_id=paper_id,
            model_provider=resp.provider,
            model_name=resp.model,
            prompt_version=prompt_version,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cost_usd=_estimate_cost(resp),
            latency_ms=resp.latency_ms,
        )
        session.add(run)
        # Don't flush here — let the caller handle transaction

    return resp


async def _call_anthropic(prompt: str, system: str, model: str,
                          max_tokens: int, temperature: float) -> LLMResponse:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

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
    client = openai.AsyncOpenAI(**kwargs)

    start = time.monotonic()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        # Try non-streaming first
        response = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            temperature=temperature, stream=False,
        )
        latency = int((time.monotonic() - start) * 1000)
        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
            provider="openai",
            latency_ms=latency,
        )
    except Exception as e:
        # Fallback: some proxies force streaming — collect chunks
        logger.info(f"Non-streaming failed ({e}), trying streaming mode")
        stream = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            temperature=temperature, stream=True,
        )
        chunks = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        latency = int((time.monotonic() - start) * 1000)
        return LLMResponse(
            text="".join(chunks),
            input_tokens=len(prompt.split()),  # estimate
            output_tokens=len("".join(chunks).split()),
            model=model,
            provider="openai",
            latency_ms=latency,
        )


def _mock_response(prompt: str, system: str) -> LLMResponse:
    """Generate a structured mock response for testing without API keys."""
    logger.info("LLM mock mode — no API key configured")

    # Parse what kind of analysis is being requested
    mock_text = json.dumps({
        "problem_summary": "[Mock] This paper addresses a key challenge in the domain.",
        "method_summary": "[Mock] The proposed method introduces a novel approach using structured components.",
        "evidence_summary": "[Mock] Experiments show improvements over baselines on standard benchmarks.",
        "core_intuition": "[Mock] The key insight is applying a hierarchical decomposition to the problem.",
        "changed_slots": ["denoiser", "conditioning"],
        "is_plugin_patch": False,
        "worth_deep_read": True,
        "confidence_notes": [
            {"claim": "Novel architecture design", "confidence": 0.5, "basis": "speculative",
             "reasoning": "Mock analysis — no actual LLM evaluation performed"},
        ],
    }, ensure_ascii=False, indent=2)

    return LLMResponse(
        text=mock_text,
        input_tokens=len(prompt.split()),
        output_tokens=len(mock_text.split()),
        model="mock",
        provider="mock",
        latency_ms=0,
    )


def _estimate_cost(resp: LLMResponse) -> float:
    """Rough cost estimate in USD."""
    costs = {
        "claude-sonnet-4-20250514": (3.0 / 1_000_000, 15.0 / 1_000_000),
        "claude-haiku-4-5-20251001": (0.80 / 1_000_000, 4.0 / 1_000_000),
        "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
        "gpt-4o": (2.50 / 1_000_000, 10.0 / 1_000_000),
        "mock": (0, 0),
    }
    rate = costs.get(resp.model, (1.0 / 1_000_000, 3.0 / 1_000_000))
    return resp.input_tokens * rate[0] + resp.output_tokens * rate[1]
