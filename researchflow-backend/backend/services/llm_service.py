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
    if not text or len(text) < 20:
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
                if settings.anthropic_api_key:
                    resp = await asyncio.wait_for(
                        _call_anthropic(prompt, system, model or "claude-sonnet-4.6", max_tokens, temperature),
                        timeout=180,
                    )
                else:
                    default_model = settings.openai_model or "gpt-4o-mini"
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
    """Generate a structured mock response for testing without API keys."""
    logger.info("LLM mock mode — no API key configured")
    mock_text = json.dumps({
        "problem_summary": "[Mock] This paper addresses a key challenge in the domain.",
        "method_summary": "[Mock] The proposed method introduces a novel approach.",
        "evidence_summary": "[Mock] Experiments show improvements over baselines.",
        "core_intuition": "[Mock] The key insight is a hierarchical decomposition.",
        "changed_slots": ["denoiser", "conditioning"],
        "is_plugin_patch": False,
        "worth_deep_read": True,
        "confidence_notes": [],
    }, ensure_ascii=False, indent=2)
    return LLMResponse(
        text=mock_text, input_tokens=len(prompt.split()),
        output_tokens=len(mock_text.split()),
        model="mock", provider="mock", latency_ms=0,
    )


def _estimate_cost(resp: LLMResponse) -> float:
    """Rough cost estimate in USD."""
    costs = {
        "claude-sonnet-4.6": (3.0 / 1_000_000, 15.0 / 1_000_000),
        "claude-haiku-4.5": (0.80 / 1_000_000, 4.0 / 1_000_000),
        "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
        "gpt-4o": (2.50 / 1_000_000, 10.0 / 1_000_000),
        "mock": (0, 0),
    }
    rate = costs.get(resp.model, (1.0 / 1_000_000, 3.0 / 1_000_000))
    return resp.input_tokens * rate[0] + resp.output_tokens * rate[1]
