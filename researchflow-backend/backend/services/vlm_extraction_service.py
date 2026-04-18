"""VLM extraction service — use Claude Vision API for figure/formula/table understanding.

No GPU needed. Uses Claude Opus 4.6 API to:
  1. Classify figure semantic role (motivation/pipeline/result/ablation)
  2. Describe figure content (modules, arrows, data flow)
  3. OCR formulas from figure images → LaTeX
  4. Structure complex tables from images → Markdown
  5. Judge ambiguous acceptance status from webpage text

Architecture: GROBID + PyMuPDF (CPU, deterministic) → extract raw content
              → Claude API (cloud, intelligent) → understand and filter
"""

import base64
import json
import logging
import time
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.models.system import ModelRun

logger = logging.getLogger(__name__)

# Model selection: use Sonnet for routine VLM tasks (cheaper), Opus for complex judgment
VLM_MODEL_ROUTINE = "claude-sonnet-4-20250514"  # figure classification, simple OCR
VLM_MODEL_COMPLEX = "claude-opus-4-20250514"    # formula derivation, acceptance judgment


async def call_vlm(
    prompt: str,
    images: list[dict],
    system: str = "",
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    session: AsyncSession | None = None,
    paper_id: UUID | None = None,
) -> dict:
    """Call Claude Vision API with text + images.

    Args:
        prompt: Text prompt
        images: List of {data: bytes, media_type: "image/png"} or {url: "https://..."}
        system: System prompt
        model: Model override
        max_tokens: Max output tokens
        session: DB session for cost tracking

    Returns: {text: str, input_tokens: int, output_tokens: int, model: str}
    """
    if not settings.anthropic_api_key:
        return {"text": "[VLM mock — no API key]", "input_tokens": 0, "output_tokens": 0, "model": "mock"}

    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    # Build multimodal content
    content = []
    for img in images:
        if "data" in img:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.get("media_type", "image/png"),
                    "data": base64.b64encode(img["data"]).decode("utf-8"),
                },
            })
        elif "url" in img:
            content.append({
                "type": "image",
                "source": {
                    "type": "url",
                    "url": img["url"],
                },
            })

    content.append({"type": "text", "text": prompt})

    model = model or VLM_MODEL_ROUTINE
    start = time.monotonic()

    try:
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)
        latency = int((time.monotonic() - start) * 1000)

        text = response.content[0].text if response.content else ""
        result = {
            "text": text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": model,
            "latency_ms": latency,
        }

        # Log cost
        if session:
            run = ModelRun(
                paper_id=paper_id,
                model_provider="anthropic",
                model_name=model,
                prompt_version="vlm_v1",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=_estimate_vlm_cost(response.usage.input_tokens, response.usage.output_tokens, model),
                latency_ms=latency,
            )
            session.add(run)

        return result

    except Exception as e:
        logger.error(f"VLM call failed: {e}")
        return {"text": "", "input_tokens": 0, "output_tokens": 0, "model": model, "error": str(e)}


# ── Figure understanding ──────────────────────────────────────

async def classify_and_describe_figures(
    session: AsyncSession,
    paper_id: UUID,
    figure_images: list[dict],
    figure_captions: list[dict],
    paper_title: str = "",
    max_figures: int = 5,
) -> list[dict]:
    """Classify figure roles and generate semantic descriptions.

    For each figure:
      - semantic_role: motivation / pipeline / architecture / result / ablation / comparison / example
      - description: what the figure shows (modules, data flow, comparisons)
      - is_key_figure: whether this is a critical figure for understanding the paper

    Only processes top N figures (sorted by size, as larger figures are more likely to be key).
    """
    if not figure_images:
        return []

    # Sort by size (larger = more likely to be important), take top N
    sorted_figs = sorted(figure_images, key=lambda f: f.get("size_bytes", 0), reverse=True)
    selected = sorted_figs[:max_figures]

    # Load image bytes from object storage
    from backend.services.object_storage import get_storage
    storage = get_storage()

    results = []
    for fig in selected:
        object_key = fig.get("object_key", "")
        if not object_key:
            continue

        img_data = await storage.get(object_key)
        if not img_data:
            continue

        # Find matching caption
        fig_num = fig.get("figure_num", 0)
        caption = ""
        for cap in figure_captions:
            if cap.get("figure_num") == fig_num or cap.get("label", "").endswith(str(fig_num)):
                caption = cap.get("caption", "")
                break

        ext = object_key.rsplit(".", 1)[-1] if "." in object_key else "png"
        media_type = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "gif", "webp") else "image/png"

        prompt = f"""Analyze this figure from the paper "{paper_title}".
Caption: {caption if caption else "(no caption available)"}

Return JSON:
{{
  "semantic_role": "motivation|pipeline|architecture|result|ablation|comparison|example|other",
  "description": "2-3 sentence description of what this figure shows, in Chinese",
  "key_elements": ["element1", "element2", ...],
  "is_key_figure": true/false,
  "contains_formula": true/false,
  "contains_table": true/false
}}

Only return the JSON, no other text."""

        vlm_result = await call_vlm(
            prompt=prompt,
            images=[{"data": img_data, "media_type": media_type}],
            model=VLM_MODEL_ROUTINE,
            max_tokens=500,
            session=session,
            paper_id=paper_id,
        )

        try:
            parsed = json.loads(vlm_result["text"].strip().strip("```json").strip("```"))
        except (json.JSONDecodeError, KeyError):
            parsed = {"semantic_role": "other", "description": vlm_result.get("text", "")[:200]}

        results.append({
            "figure_num": fig_num,
            "object_key": object_key,
            "public_url": fig.get("public_url"),
            "caption": caption,
            **parsed,
        })

    return results


# ── Formula OCR from images ───────────────────────────────────

async def extract_formulas_from_images(
    session: AsyncSession,
    paper_id: UUID,
    figure_images: list[dict],
    paper_title: str = "",
) -> list[dict]:
    """Extract LaTeX formulas from figure images that contain equations.

    This is for formulas that PyMuPDF/GROBID missed because they were
    rendered as images rather than embedded text.
    """
    # First classify figures to find ones containing formulas
    from backend.services.object_storage import get_storage
    storage = get_storage()

    formula_figs = []
    for fig in figure_images:
        # Only process smaller images (formula images are usually small)
        if fig.get("size_bytes", 0) > 500_000:  # Skip large figures (likely plots)
            continue
        # Check aspect ratio — formulas are usually wide and short
        w = fig.get("width", 0)
        h = fig.get("height", 0)
        if w > 0 and h > 0 and w / h > 2.0:  # Wide aspect ratio
            formula_figs.append(fig)

    if not formula_figs:
        return []

    results = []
    for fig in formula_figs[:10]:  # Cap at 10
        img_data = await storage.get(fig.get("object_key", ""))
        if not img_data:
            continue

        ext = fig.get("object_key", "").rsplit(".", 1)[-1] if "." in fig.get("object_key", "") else "png"

        prompt = """This image may contain a mathematical formula or equation from an academic paper.
If it contains a formula, convert it to LaTeX.
If it does NOT contain a formula (it's a diagram, chart, or text), respond with just: NOT_FORMULA

Return format (only if formula):
```latex
<the LaTeX code>
```"""

        vlm_result = await call_vlm(
            prompt=prompt,
            images=[{"data": img_data, "media_type": f"image/{ext}"}],
            model=VLM_MODEL_ROUTINE,
            max_tokens=300,
            session=session,
            paper_id=paper_id,
        )

        text = vlm_result.get("text", "")
        if "NOT_FORMULA" not in text and "```" in text:
            import re
            m = re.search(r'```(?:latex)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if m:
                latex = m.group(1).strip()
                if latex and len(latex) > 3:
                    results.append({
                        "latex": latex,
                        "source": "vlm_ocr",
                        "figure_num": fig.get("figure_num", 0),
                        "page": fig.get("page_num", -1),
                    })

    return results


# ── Acceptance judgment for ambiguous cases ───────────────────

async def judge_acceptance_status(
    session: AsyncSession,
    paper_id: UUID,
    paper_title: str,
    observations: list[dict],
) -> dict:
    """Use LLM to judge acceptance status when multiple sources conflict.

    Called when the metadata resolver finds unresolved conflicts in the
    acceptance_status field across sources.

    Args:
        observations: list of {source, value, confidence} from different sources
    """
    obs_text = "\n".join(
        f"- {o['source']}: \"{o['value']}\" (confidence: {o.get('confidence', '?')})"
        for o in observations
    )

    from backend.services.llm_service import call_llm

    prompt = f"""Paper title: "{paper_title}"

The following metadata sources report different acceptance status:
{obs_text}

Based on these observations, determine:
1. Is this paper accepted at a conference/journal? (yes/no/uncertain)
2. If yes, which venue? (e.g., ICLR, CVPR, NeurIPS)
3. If yes, what type? (oral/poster/spotlight/workshop/unknown)
4. Confidence level (0.0-1.0)

Return JSON only:
{{"accepted": true/false/null, "venue": "...", "acceptance_type": "...", "confidence": 0.X, "reasoning": "..."}}"""

    resp = await call_llm(
        prompt=prompt,
        system="You are a metadata resolver for academic papers. Be conservative — only say 'accepted' if evidence is strong.",
        model=VLM_MODEL_ROUTINE,
        max_tokens=300,
        temperature=0.1,
        session=session,
        paper_id=paper_id,
        prompt_version="acceptance_judge_v1",
    )

    try:
        return json.loads(resp.text.strip().strip("```json").strip("```"))
    except (json.JSONDecodeError, AttributeError):
        return {"accepted": None, "confidence": 0.0, "reasoning": "Failed to parse LLM response"}


def _estimate_vlm_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate cost for VLM calls (image tokens are more expensive)."""
    costs = {
        "claude-sonnet-4-20250514": (3.0 / 1_000_000, 15.0 / 1_000_000),
        "claude-opus-4-20250514": (15.0 / 1_000_000, 75.0 / 1_000_000),
    }
    rate = costs.get(model, (3.0 / 1_000_000, 15.0 / 1_000_000))
    return input_tokens * rate[0] + output_tokens * rate[1]
