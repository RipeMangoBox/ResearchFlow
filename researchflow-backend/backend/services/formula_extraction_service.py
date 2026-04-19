"""Formula extraction service — GROBID coords + PyMuPDF crop + VLM OCR.

Pipeline (no GPU needed):
  1. GROBID detects formula regions → coordinates in TEI XML
  2. PyMuPDF crops each formula region from the PDF page → PNG
  3. ONE Claude VLM call: all formula images → LaTeX source code

Fallback when GROBID unavailable:
  - PyMuPDF regex extraction (existing, low quality)
  - VLM scan of pages with detected equation regions

Total API calls: 1 (batch all formula images in single request)
"""

import base64
import json
import logging
import time
from uuid import UUID

import fitz

from backend.config import settings

logger = logging.getLogger(__name__)


async def extract_formulas(
    pdf_path: str,
    paper_id: UUID,
    grobid_formulas: list[dict] | None = None,
    session=None,
) -> list[dict]:
    """Extract formulas from PDF as LaTeX.

    Args:
        pdf_path: Path to PDF file
        paper_id: For cost tracking
        grobid_formulas: Pre-parsed GROBID formula data with coords
            [{text, label, page, bbox: [x0,y0,x1,y1]}]
        session: DB session for cost logging

    Returns: [{latex, label, page, bbox, source, context}]
    """
    doc = fitz.open(pdf_path)
    formula_images = []

    # ── Step 1: Get formula regions ──────────────────────────
    if grobid_formulas:
        # GROBID provided coordinates — crop each formula
        for f in grobid_formulas:
            if not f.get("bbox") or f.get("page", -1) < 0:
                continue
            page_num = f["page"]
            if page_num >= len(doc):
                continue

            page = doc[page_num]
            bbox = f["bbox"]
            rect = fitz.Rect(*bbox)

            # Expand horizontally to full text width (formulas often wider than GROBID bbox)
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

            # Vertical margin
            rect = fitz.Rect(
                max(rect.x0, page.rect.x0),
                max(rect.y0 - 3, page.rect.y0),
                min(rect.x1, page.rect.x1),
                min(rect.y1 + 3, page.rect.y1),
            )

            if rect.width < 20 or rect.height < 8:
                continue

            # Crop at 3x zoom for OCR clarity
            pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=rect)
            formula_images.append({
                "png_bytes": pix.tobytes("png"),
                "page": page_num,
                "bbox": [round(rect.x0, 1), round(rect.y0, 1),
                         round(rect.x1, 1), round(rect.y1, 1)],
                "grobid_text": f.get("text", ""),
                "label": f.get("label", ""),
                "width": pix.width,
                "height": pix.height,
            })
    else:
        # No GROBID — detect formula regions heuristically
        formula_images = _detect_formula_regions_heuristic(doc)

    doc.close()

    if not formula_images:
        logger.info(f"No formula regions found for {paper_id}")
        return []

    logger.info(f"Found {len(formula_images)} formula regions for {paper_id}")

    # ── Step 2: VLM OCR — convert images to LaTeX ────────────
    if settings.anthropic_api_key or settings.openai_api_key:
        return await _ocr_formulas_vlm(formula_images, session, paper_id)
    else:
        # No API key — return GROBID text as fallback
        return [
            {
                "latex": f.get("grobid_text", ""),
                "label": f.get("label", ""),
                "page": f.get("page", -1),
                "bbox": f.get("bbox"),
                "source": "grobid_text",
            }
            for f in formula_images
            if f.get("grobid_text")
        ]


def _detect_formula_regions_heuristic(doc, pdf_path: str = "") -> list[dict]:
    """Fallback: detect formula regions without GROBID.

    Strategy: look for pages with text blocks containing math Unicode
    characters (∑, ∫, ∏, ≤, ≥, θ, α, β, etc.) that form isolated
    short lines (likely equations, not inline text).
    """
    import re
    math_chars = re.compile(r'[∑∫∏∂∇∆≤≥≠≈∈∉⊂⊃∀∃αβγδεζηθλμνξπρστφψωΩΓΔΘΛΣΦΨ]')
    formula_images = []

    for page_num in range(min(len(doc), 15)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] != 0:
                continue

            bbox = block["bbox"]
            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")

            block_text = block_text.strip()

            # Heuristic: short block with math symbols = likely a formula
            if (len(block_text) < 200
                    and len(math_chars.findall(block_text)) >= 2
                    and bbox[3] - bbox[1] < 80):  # Not too tall

                rect = fitz.Rect(*bbox)

                # Expand horizontally to full text column width
                # Academic papers: single-column formulas span ~90% page width
                # Double-column: formulas span column width
                page_mid = page.rect.width / 2
                text_margin = 50  # Typical margin in academic papers

                if rect.x0 < page_mid and rect.x1 > page_mid:
                    # Full-width formula (single column or spanning)
                    rect = fitz.Rect(text_margin, rect.y0,
                                     page.rect.width - text_margin, rect.y1)
                elif rect.x0 < page_mid:
                    # Left column formula
                    rect = fitz.Rect(text_margin, rect.y0,
                                     page_mid - 5, rect.y1)
                else:
                    # Right column formula
                    rect = fitz.Rect(page_mid + 5, rect.y0,
                                     page.rect.width - text_margin, rect.y1)

                # Vertical margin
                rect = fitz.Rect(
                    max(rect.x0, page.rect.x0),
                    max(rect.y0 - 5, page.rect.y0),
                    min(rect.x1, page.rect.x1),
                    min(rect.y1 + 5, page.rect.y1),
                )

                pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=rect)
                formula_images.append({
                    "png_bytes": pix.tobytes("png"),
                    "page": page_num,
                    "bbox": [round(rect.x0, 1), round(rect.y0, 1),
                             round(rect.x1, 1), round(rect.y1, 1)],
                    "grobid_text": block_text,
                    "label": "",
                    "width": pix.width,
                    "height": pix.height,
                })

                if len(formula_images) >= 30:
                    break

    # Merge vertically adjacent formula rects on same page BEFORE cropping
    # (multi-line equations get split into separate text blocks)
    merged_rects = _merge_adjacent_rects(
        [(f["page"], f["bbox"], f["grobid_text"]) for f in formula_images]
    )

    # Re-crop merged regions
    result = []
    for page_num, bbox, text in merged_rects:
        if page_num >= len(doc):
            continue
        page = doc[page_num]
        rect = fitz.Rect(*bbox)
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=rect)
        result.append({
            "png_bytes": pix.tobytes("png"),
            "page": page_num,
            "bbox": bbox,
            "grobid_text": text,
            "label": "",
            "width": pix.width,
            "height": pix.height,
        })

    return result


def _merge_adjacent_rects(
    items: list[tuple[int, list[float], str]],
    gap: float = 15.0,
) -> list[tuple[int, list[float], str]]:
    """Merge vertically adjacent formula rects on the same page."""
    if len(items) <= 1:
        return items

    # Sort by page then y
    items = sorted(items, key=lambda x: (x[0], x[1][1]))
    merged = []
    cur_page, cur_bbox, cur_text = items[0]
    cur_bbox = list(cur_bbox)

    for page, bbox, text in items[1:]:
        if page == cur_page and bbox[1] - cur_bbox[3] <= gap:
            # Adjacent — merge
            cur_bbox[0] = min(cur_bbox[0], bbox[0])
            cur_bbox[1] = min(cur_bbox[1], bbox[1])
            cur_bbox[2] = max(cur_bbox[2], bbox[2])
            cur_bbox[3] = max(cur_bbox[3], bbox[3])
            cur_text += " " + text
        else:
            merged.append((cur_page, cur_bbox, cur_text))
            cur_page, cur_bbox, cur_text = page, list(bbox), text

    merged.append((cur_page, cur_bbox, cur_text))
    return merged


async def _ocr_formulas_vlm(
    formula_images: list[dict],
    session,
    paper_id: UUID,
) -> list[dict]:
    """ONE Claude VLM call: batch all formula images → LaTeX.

    Sends all formula crops in a single request for cost efficiency.
    """
    import openai
    client = openai.AsyncOpenAI(
        api_key=settings.openai_api_key or settings.anthropic_api_key,
        base_url=settings.openai_base_url or None,
    )
    actual_model = settings.openai_model or "claude-sonnet-4-20250514"

    # Cap at 20 formulas per call (token limit)
    images_to_send = formula_images[:20]

    content = []
    for i, f in enumerate(images_to_send):
        content.append({
            "type": "text",
            "text": f"--- Formula {i} (page {f['page']}, label: {f.get('label', '?')}) ---",
        })
        b64 = base64.b64encode(f["png_bytes"]).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    prompt = f"""Convert each formula image to LaTeX.

Return a JSON array with one entry per formula:
[
  {{"index": 0, "latex": "\\\\mathcal{{J}}_{{DAPO}}(\\\\theta) = \\\\mathbb{{E}}_{{...}}[...]", "label": "(5)", "is_formula": true}},
  {{"index": 1, "latex": "", "is_formula": false}}
]

Rules:
- Use standard LaTeX math notation (not Unicode)
- Include equation numbers if visible (as "label" field)
- If the image is NOT a formula (just text or a diagram), set is_formula=false
- Wrap display equations in $$ ... $$ notation
- Return ONLY the JSON array"""

    content.append({"type": "text", "text": prompt})

    start = time.monotonic()
    try:
        stream = await client.chat.completions.create(
            model=actual_model,
            max_tokens=4000,
            temperature=0.1,
            messages=[{"role": "user", "content": content}],
            stream=True,
        )
        chunks = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        latency = int((time.monotonic() - start) * 1000)
        text = "".join(chunks) if chunks else "[]"
        in_tokens = len(images_to_send) * 500
        out_tokens = len(text.split())

        logger.info(
            f"VLM formula OCR: {len(images_to_send)} formulas, "
            f"~{in_tokens}→{out_tokens} tokens, {latency}ms"
        )

        if session:
            from backend.models.system import ModelRun
            run = ModelRun(
                paper_id=paper_id,
                model_provider="anthropic",
                model_name=actual_model,
                prompt_version="formula_ocr_v1",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=(response.usage.input_tokens * 3.0 + response.usage.output_tokens * 15.0) / 1_000_000,
                latency_ms=latency,
            )
            session.add(run)

        # Parse
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        ocr_results = json.loads(text)
        if not isinstance(ocr_results, list):
            ocr_results = []

        # Merge OCR results with original metadata
        results = []
        for ocr in ocr_results:
            idx = ocr.get("index", -1)
            if idx < 0 or idx >= len(images_to_send) or not ocr.get("is_formula", False):
                continue

            orig = images_to_send[idx]
            results.append({
                "latex": ocr.get("latex", ""),
                "label": ocr.get("label", orig.get("label", "")),
                "page": orig.get("page", -1),
                "bbox": orig.get("bbox"),
                "source": "vlm_ocr",
                "grobid_text": orig.get("grobid_text", ""),
            })

        return results

    except Exception as e:
        logger.error(f"VLM formula OCR failed: {e}")
        # Fallback: return GROBID text
        return [
            {
                "latex": f.get("grobid_text", ""),
                "label": f.get("label", ""),
                "page": f.get("page", -1),
                "bbox": f.get("bbox"),
                "source": "grobid_text_fallback",
            }
            for f in formula_images
            if f.get("grobid_text")
        ]
