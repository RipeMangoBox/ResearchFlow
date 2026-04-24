"""Formula extraction service — VLM page scan + PyMuPDF crop.

Pipeline (no GPU, no GROBID needed):
  1. PyMuPDF scans pages → rank by math symbol density
  2. Render top pages as images (batches of 3-5)
  3. ONE Claude VLM call per batch: page images → LaTeX formulas

Fallback when no API key:
  - PyMuPDF regex extraction (low quality, from pdf_extract.py)

Optional GROBID enhancement (if available):
  - GROBID coords → precise crops → VLM OCR (higher quality per-formula)
"""

import base64
import json
import logging
import re
import time
from uuid import UUID

import fitz

from backend.config import settings

logger = logging.getLogger(__name__)

# Math symbols that indicate a page likely contains formulas
_MATH_INDICATOR = re.compile(
    r'[∑∫∏∂∇∆≤≥≠≈∈∉⊂⊃∀∃αβγδεζηθλμνξπρστφψωΩΓΔΘΛΣΦΨ]|'
    r'\\(?:frac|sum|int|prod|mathbb|mathcal|text|left|right)\b|'
    r'\(\d+\)\s*$'
)


async def extract_formulas(
    pdf_path: str,
    paper_id: UUID,
    grobid_formulas: list[dict] | None = None,
    session=None,
) -> list[dict]:
    """Extract formulas from PDF as LaTeX.

    Primary: VLM page scan (sends page images, VLM extracts all formulas).
    Enhanced: If GROBID coords provided, also do precise crop + OCR.
    Fallback: Return GROBID raw text if no API key.

    Returns: [{latex, label, page, context, source}]
    """
    if not (settings.anthropic_api_key or settings.openai_api_key):
        # No API key — can only return GROBID text if available
        if grobid_formulas:
            return [
                {"latex": f.get("text", ""), "label": f.get("label", ""),
                 "page": f.get("page", -1), "source": "grobid_text"}
                for f in grobid_formulas if f.get("text")
            ]
        return []

    doc = fitz.open(pdf_path)

    # ── Primary: VLM page scan ──────────────────────────────────
    vlm_results = await _vlm_page_scan(doc, paper_id, session)

    # ── Enhanced: GROBID crop + OCR (if coords available) ───────
    if grobid_formulas:
        crop_results = await _grobid_crop_ocr(doc, grobid_formulas, paper_id, session)
        # Merge: deduplicate by label, prefer VLM scan (has context)
        vlm_results = _merge_formula_results(vlm_results, crop_results)

    doc.close()

    logger.info(f"Formula extraction for {paper_id}: {len(vlm_results)} formulas")
    return vlm_results


# ── VLM Page Scan ────────────────────────────────────────────────────

def _rank_pages_by_math_density(doc, max_pages: int = 15) -> list[int]:
    """Rank pages by math symbol density. Returns page indices sorted by density."""
    page_scores = []
    for i in range(min(len(doc), max_pages)):
        text = doc[i].get_text()
        hits = len(_MATH_INDICATOR.findall(text))
        page_scores.append((i, hits))

    # Filter pages with any math content, sort by density
    math_pages = [(i, h) for i, h in page_scores if h >= 2]
    math_pages.sort(key=lambda x: x[1], reverse=True)

    if not math_pages:
        # No math detected — try pages 1-5 (method section usually there)
        return list(range(min(5, len(doc))))

    return [i for i, _ in math_pages]


async def _vlm_page_scan(
    doc,
    paper_id: UUID,
    session=None,
    pages_per_batch: int = 3,
    max_batches: int = 3,
) -> list[dict]:
    """Send page images to VLM, extract all formulas directly.

    Batches pages to stay within token limits (~3 pages per call).
    """
    ranked_pages = _rank_pages_by_math_density(doc)
    if not ranked_pages:
        return []

    # Cap total pages
    max_pages = pages_per_batch * max_batches
    selected_pages = ranked_pages[:max_pages]

    all_results = []
    for batch_start in range(0, len(selected_pages), pages_per_batch):
        batch_indices = selected_pages[batch_start:batch_start + pages_per_batch]

        # Render pages
        page_images = []
        for idx in sorted(batch_indices):
            page = doc[idx]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            page_images.append({
                "page": idx,
                "png_bytes": pix.tobytes("png"),
            })

        batch_results = await _call_vlm_page_scan(page_images, paper_id, session)
        all_results.extend(batch_results)

        # Stop if this batch found nothing (unlikely to find more)
        if not batch_results and batch_start > 0:
            break

    return all_results


async def _call_vlm_page_scan(
    page_images: list[dict],
    paper_id: UUID,
    session=None,
) -> list[dict]:
    """One VLM call: page images → all formulas as LaTeX."""
    import openai
    client = openai.AsyncOpenAI(
        api_key=settings.openai_api_key or settings.anthropic_api_key,
        base_url=settings.openai_base_url or None,
        default_headers={"User-Agent": "claude-code/1.0"},
    )
    actual_model = settings.openai_model or "kimi-k2.6"

    content = []
    for p in page_images:
        content.append({"type": "text", "text": f"--- Page {p['page'] + 1} ---"})
        b64 = base64.b64encode(p["png_bytes"]).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    content.append({"type": "text", "text": """Extract ALL mathematical formulas/equations from these pages.

Return a JSON array:
[
  {"page": 3, "label": "(1)", "latex": "\\mathcal{L} = ...", "context": "loss function for..."},
  ...
]

Rules:
- Include BOTH display equations (numbered) and important multi-symbol equations
- Skip trivial inline math (single variables like x, y)
- Use standard LaTeX notation (not Unicode)
- "label" = equation number if visible, null otherwise
- "context" = brief description of what the formula represents
- "page" is 1-indexed
- Return ONLY the JSON array, no explanation"""})

    start = time.monotonic()
    try:
        stream = await client.chat.completions.create(
            model=actual_model,
            max_tokens=settings.vlm_max_tokens_heavy,
            temperature=0.1,
            messages=[{"role": "user", "content": content}],
            stream=True,
        )
        chunks = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        elapsed = time.monotonic() - start
        text = "".join(chunks).strip() if chunks else "[]"
        in_tokens = len(page_images) * 1500  # ~1500 tokens per page image
        out_tokens = len(text.split())

        logger.info(
            f"VLM page scan: {len(page_images)} pages, "
            f"~{in_tokens}→{out_tokens} tokens, {elapsed:.1f}s"
        )

        # Log cost
        if session:
            try:
                from backend.models.system import ModelRun
                run = ModelRun(
                    paper_id=paper_id,
                    model_provider="anthropic",
                    model_name=actual_model,
                    prompt_version="formula_page_scan_v1",
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    cost_usd=(in_tokens * 3.0 + out_tokens * 15.0) / 1_000_000,
                    latency_ms=int(elapsed * 1000),
                )
                session.add(run)
            except Exception:
                pass

        # Parse JSON
        parsed = _parse_vlm_json(text)
        if not isinstance(parsed, list):
            return []

        return [
            {
                "latex": r.get("latex", ""),
                "label": r.get("label") or "",
                "page": (r.get("page", 1) - 1),  # Convert to 0-indexed
                "context": r.get("context", ""),
                "source": "vlm_page_scan",
            }
            for r in parsed
            if r.get("latex")
        ]

    except Exception as e:
        logger.error(f"VLM page scan failed: {e}")
        return []


# ── GROBID Crop + OCR (optional enhancement) ────────────────────────

async def _grobid_crop_ocr(
    doc,
    grobid_formulas: list[dict],
    paper_id: UUID,
    session=None,
) -> list[dict]:
    """Crop formula regions using GROBID coords, OCR via VLM.

    This is the original pipeline, kept as optional enhancement when
    GROBID is available.
    """
    formula_images = []
    for f in grobid_formulas:
        if not f.get("bbox") or f.get("page", -1) < 0:
            continue
        page_num = f["page"]
        if page_num >= len(doc):
            continue

        page = doc[page_num]
        rect = fitz.Rect(*f["bbox"])

        # Expand horizontally to full text width
        text_margin = 50
        page_mid = page.rect.width / 2
        if rect.x0 < page_mid and rect.x1 > page_mid:
            rect = fitz.Rect(text_margin, rect.y0,
                             page.rect.width - text_margin, rect.y1)
        elif rect.x0 < page_mid:
            rect = fitz.Rect(text_margin, rect.y0,
                             max(rect.x1 + 30, page_mid - 5), rect.y1)
        else:
            rect = fitz.Rect(min(rect.x0 - 30, page_mid + 5), rect.y0,
                             page.rect.width - text_margin, rect.y1)

        rect = fitz.Rect(
            max(rect.x0, page.rect.x0), max(rect.y0 - 3, page.rect.y0),
            min(rect.x1, page.rect.x1), min(rect.y1 + 3, page.rect.y1),
        )

        if rect.width < 20 or rect.height < 8:
            continue

        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=rect)
        formula_images.append({
            "png_bytes": pix.tobytes("png"),
            "page": page_num,
            "bbox": [round(rect.x0, 1), round(rect.y0, 1),
                     round(rect.x1, 1), round(rect.y1, 1)],
            "grobid_text": f.get("text", ""),
            "label": f.get("label", ""),
        })

    if not formula_images:
        return []

    return await _ocr_formula_crops(formula_images, paper_id, session)


async def _ocr_formula_crops(
    formula_images: list[dict],
    paper_id: UUID,
    session=None,
) -> list[dict]:
    """ONE VLM call: batch formula crop images → LaTeX."""
    import openai
    client = openai.AsyncOpenAI(
        api_key=settings.openai_api_key or settings.anthropic_api_key,
        base_url=settings.openai_base_url or None,
        default_headers={"User-Agent": "claude-code/1.0"},
    )
    actual_model = settings.openai_model or "kimi-k2.6"

    images_to_send = formula_images[:20]
    content = []
    for i, f in enumerate(images_to_send):
        content.append({
            "type": "text",
            "text": f"--- Formula {i} (page {f['page']}, label: {f.get('label', '?')}) ---",
        })
        b64 = base64.b64encode(f["png_bytes"]).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    content.append({"type": "text", "text": """Convert each formula image to LaTeX.
Return a JSON array: [{"index": 0, "latex": "...", "label": "(1)", "is_formula": true}, ...]
Rules:
- Standard LaTeX math notation (not Unicode)
- Include equation numbers as "label"
- If NOT a formula, set is_formula=false
- Return ONLY the JSON array"""})

    start = time.monotonic()
    try:
        stream = await client.chat.completions.create(
            model=actual_model, max_tokens=settings.vlm_max_tokens_heavy, temperature=0.1,
            messages=[{"role": "user", "content": content}], stream=True,
        )
        chunks = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        elapsed = time.monotonic() - start
        text = "".join(chunks).strip() if chunks else "[]"
        in_tokens = len(images_to_send) * 500
        out_tokens = len(text.split())

        logger.info(
            f"VLM crop OCR: {len(images_to_send)} formulas, "
            f"~{in_tokens}→{out_tokens} tokens, {elapsed:.1f}s"
        )

        if session:
            try:
                from backend.models.system import ModelRun
                run = ModelRun(
                    paper_id=paper_id,
                    model_provider="anthropic",
                    model_name=actual_model,
                    prompt_version="formula_crop_ocr_v1",
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    cost_usd=(in_tokens * 3.0 + out_tokens * 15.0) / 1_000_000,
                    latency_ms=int(elapsed * 1000),
                )
                session.add(run)
            except Exception:
                pass

        parsed = _parse_vlm_json(text)
        if not isinstance(parsed, list):
            return []

        results = []
        for ocr in parsed:
            idx = ocr.get("index", -1)
            if idx < 0 or idx >= len(images_to_send) or not ocr.get("is_formula", False):
                continue
            orig = images_to_send[idx]
            results.append({
                "latex": ocr.get("latex", ""),
                "label": ocr.get("label", orig.get("label", "")),
                "page": orig.get("page", -1),
                "bbox": orig.get("bbox"),
                "source": "grobid_crop_vlm_ocr",
            })
        return results

    except Exception as e:
        logger.error(f"VLM crop OCR failed: {e}")
        return [
            {"latex": f.get("grobid_text", ""), "label": f.get("label", ""),
             "page": f.get("page", -1), "source": "grobid_text_fallback"}
            for f in formula_images if f.get("grobid_text")
        ]


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_vlm_json(text: str):
    """Parse JSON from VLM response, handling markdown code blocks."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    logger.warning(f"Could not parse VLM JSON response: {text[:200]}")
    return []


def _merge_formula_results(
    vlm_results: list[dict],
    crop_results: list[dict],
) -> list[dict]:
    """Merge VLM page scan results with GROBID crop OCR results.

    Dedup by label — if both found the same equation, prefer VLM scan
    (has context field). Add any crop-only formulas.
    """
    if not crop_results:
        return vlm_results
    if not vlm_results:
        return crop_results

    # Index VLM results by label
    vlm_by_label = {}
    for r in vlm_results:
        if r.get("label"):
            vlm_by_label[r["label"]] = r

    # Add crop results that VLM missed
    for cr in crop_results:
        label = cr.get("label", "")
        if label and label in vlm_by_label:
            continue  # Already have from VLM scan
        # Check if any VLM result on same page has similar LaTeX
        found_dup = False
        for vr in vlm_results:
            if vr.get("page") == cr.get("page"):
                # Simple similarity: first 30 chars of LaTeX match
                v_prefix = vr.get("latex", "")[:30].replace(" ", "")
                c_prefix = cr.get("latex", "")[:30].replace(" ", "")
                if v_prefix and c_prefix and v_prefix == c_prefix:
                    found_dup = True
                    break
        if not found_dup:
            vlm_results.append(cr)

    return vlm_results
