"""Compare formula detection: GROBID coords vs Heuristic-only.

Usage:
  # Step 1: Local (no API needed) — compare detection regions
  python tests/compare_formula_extraction.py --pdf storage/papers/raw-pdf/Motion_Generation/MoMask_Test.pdf

  # Step 2: With VLM OCR (needs API key)
  OPENAI_API_KEY=xxx OPENAI_BASE_URL=xxx python tests/compare_formula_extraction.py \
    --pdf storage/papers/raw-pdf/Motion_Generation/MoMask_Test.pdf --ocr

  # Step 3: Full-page VLM scan (no GROBID, no heuristic — pure VLM)
  OPENAI_API_KEY=xxx python tests/compare_formula_extraction.py \
    --pdf storage/papers/raw-pdf/Motion_Generation/MoMask_Test.pdf --vlm-scan
"""

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time

import fitz

# ── Heuristic detection (same as formula_extraction_service.py) ──────

MATH_CHARS = re.compile(
    r'[∑∫∏∂∇∆≤≥≠≈∈∉⊂⊃∀∃αβγδεζηθλμνξπρστφψωΩΓΔΘΛΣΦΨ]'
)


def detect_heuristic(doc) -> list[dict]:
    """PyMuPDF heuristic: find text blocks with math Unicode chars."""
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
            if (len(block_text) < 200
                    and len(MATH_CHARS.findall(block_text)) >= 2
                    and bbox[3] - bbox[1] < 80):
                rect = _expand_rect(page, fitz.Rect(*bbox))
                if rect.width < 20 or rect.height < 8:
                    continue
                pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=rect)
                formula_images.append({
                    "png_bytes": pix.tobytes("png"),
                    "page": page_num,
                    "bbox": [round(rect.x0, 1), round(rect.y0, 1),
                             round(rect.x1, 1), round(rect.y1, 1)],
                    "text": block_text,
                    "label": "",
                    "source": "heuristic",
                })
                if len(formula_images) >= 30:
                    break
    # Merge adjacent
    return _merge_adjacent(formula_images, doc)


def _expand_rect(page, rect):
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
    return fitz.Rect(
        max(rect.x0, page.rect.x0),
        max(rect.y0 - 5, page.rect.y0),
        min(rect.x1, page.rect.x1),
        min(rect.y1 + 5, page.rect.y1),
    )


def _merge_adjacent(items, doc):
    if len(items) <= 1:
        return items
    items = sorted(items, key=lambda x: (x["page"], x["bbox"][1]))
    merged = []
    cur = items[0].copy()
    cur["bbox"] = list(cur["bbox"])
    for it in items[1:]:
        if it["page"] == cur["page"] and it["bbox"][1] - cur["bbox"][3] <= 15:
            cur["bbox"][0] = min(cur["bbox"][0], it["bbox"][0])
            cur["bbox"][1] = min(cur["bbox"][1], it["bbox"][1])
            cur["bbox"][2] = max(cur["bbox"][2], it["bbox"][2])
            cur["bbox"][3] = max(cur["bbox"][3], it["bbox"][3])
            cur["text"] += " " + it["text"]
        else:
            merged.append(cur)
            cur = it.copy()
            cur["bbox"] = list(cur["bbox"])
    merged.append(cur)
    # Re-crop merged
    result = []
    for m in merged:
        page = doc[m["page"]]
        rect = fitz.Rect(*m["bbox"])
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=rect)
        m["png_bytes"] = pix.tobytes("png")
        result.append(m)
    return result


# ── Full-page VLM scan (方案 B: no GROBID, no heuristic) ────────────

def render_pages_with_formulas(doc, max_pages=15) -> list[dict]:
    """Render pages likely to contain formulas as images for VLM.

    Strategy: scan text for math indicators, only render those pages.
    Cap at 5 pages per batch to stay within token limits.
    """
    import re
    math_indicators = re.compile(
        r'[∑∫∏∂∇∆≤≥≠≈∈∉⊂⊃∀∃αβγδεζηθλμνξπρστφψωΩΓΔΘΛΣΦΨ]|'
        r'\\(?:frac|sum|int|prod|mathbb|mathcal|text|left|right)\b|'
        r'\(\d+\)\s*$'  # equation numbers like "(1)"
    )

    candidate_pages = []
    for i in range(min(len(doc), max_pages)):
        page = doc[i]
        text = page.get_text()
        # Count math indicators
        hits = len(math_indicators.findall(text))
        if hits >= 2 or i < 5:  # Always include first 5 pages (intro/method)
            candidate_pages.append((i, hits))

    # Sort by math density, take top 5
    candidate_pages.sort(key=lambda x: x[1], reverse=True)
    selected = sorted([p[0] for p in candidate_pages[:5]])

    pages = []
    for i in selected:
        page = doc[i]
        # 1.5x zoom — enough for VLM to read formulas, keeps token cost low
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        pages.append({
            "page": i,
            "png_bytes": pix.tobytes("png"),
            "width": pix.width,
            "height": pix.height,
        })
    print(f"  Selected pages (by math density): {selected}")
    return pages


# ── VLM calls ────────────────────────────────────────────────────────

async def vlm_ocr_crops(formula_images: list[dict], model: str) -> list[dict]:
    """OCR formula crops → LaTeX (same as production code)."""
    import openai
    client = openai.AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    content = []
    for i, f in enumerate(formula_images[:20]):
        content.append({"type": "text",
                        "text": f"--- Formula {i} (page {f['page']}, label: {f.get('label', '?')}) ---"})
        b64 = base64.b64encode(f["png_bytes"]).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})

    content.append({"type": "text", "text": """Convert each formula image to LaTeX.
Return a JSON array: [{"index": 0, "latex": "...", "label": "(1)", "is_formula": true}, ...]
Rules:
- Standard LaTeX math notation
- Include equation numbers as "label"
- If NOT a formula, set is_formula=false
- Return ONLY the JSON array"""})

    start = time.monotonic()
    stream = await client.chat.completions.create(
        model=model, max_tokens=4000, temperature=0.1,
        messages=[{"role": "user", "content": content}], stream=True,
    )
    chunks = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)

    elapsed = time.monotonic() - start
    text = "".join(chunks).strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    results = json.loads(text)
    print(f"  VLM OCR: {len(formula_images)} images → {len([r for r in results if r.get('is_formula')])} formulas, {elapsed:.1f}s")
    return results


async def vlm_scan_pages(pages: list[dict], model: str) -> list[dict]:
    """Send full-page images → VLM extracts all formulas directly.

    This is the "pure VLM" approach: no GROBID, no heuristic.
    One API call with page images → all formulas as LaTeX.
    """
    import openai
    client = openai.AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    content = []
    for p in pages:
        content.append({"type": "text", "text": f"--- Page {p['page'] + 1} ---"})
        b64 = base64.b64encode(p["png_bytes"]).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})

    content.append({"type": "text", "text": """Extract ALL mathematical formulas/equations from these pages.

Return a JSON array:
[
  {"page": 1, "label": "(1)", "latex": "\\mathcal{L} = ...", "context": "loss function for..."},
  {"page": 3, "label": null, "latex": "p(x|z) = ...", "context": "inline but important"},
  ...
]

Rules:
- Include BOTH display equations (numbered) and important inline equations
- Skip trivial inline math like single variables
- Use standard LaTeX notation
- "context" = brief description of what the formula represents
- "page" is 1-indexed
- Return ONLY the JSON array"""})

    start = time.monotonic()
    stream = await client.chat.completions.create(
        model=model, max_tokens=8000, temperature=0.1,
        messages=[{"role": "user", "content": content}], stream=True,
    )
    chunks = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)

    elapsed = time.monotonic() - start
    text = "".join(chunks).strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        print(f"  VLM raw response (first 500 chars): {text[:500]}")
        # Try to extract JSON from markdown code block
        if "```" in text:
            parts = text.split("```")
            for part in parts[1:]:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                try:
                    results = json.loads(cleaned)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                print("  ERROR: Could not parse VLM response as JSON")
                return []
        else:
            print("  ERROR: Could not parse VLM response as JSON")
            return []
    print(f"  VLM page scan: {len(pages)} pages → {len(results)} formulas, {elapsed:.1f}s")
    return results


# ── Output helpers ───────────────────────────────────────────────────

def save_crops(formula_images, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    for i, f in enumerate(formula_images):
        path = os.path.join(output_dir, f"{prefix}_{i}_p{f['page']}.png")
        with open(path, "wb") as fp:
            fp.write(f["png_bytes"])
    print(f"  Saved {len(formula_images)} crops to {output_dir}/")


def print_comparison(heuristic_results, vlm_scan_results=None):
    print("\n" + "=" * 70)
    print("COMPARISON: Formula Detection")
    print("=" * 70)

    print(f"\n[Heuristic] Detected {len(heuristic_results)} formula regions:")
    for i, f in enumerate(heuristic_results):
        text_preview = f.get("text", "")[:60].replace("\n", " ")
        print(f"  {i}: page {f['page']}, bbox={f['bbox']}, text='{text_preview}'")

    if vlm_scan_results:
        print(f"\n[VLM Scan] Detected {len(vlm_scan_results)} formulas:")
        for i, f in enumerate(vlm_scan_results):
            latex = f.get("latex", "")[:60]
            ctx = f.get("context", "")[:40]
            print(f"  {i}: page {f.get('page', '?')}, label={f.get('label', '-')}, "
                  f"latex='{latex}', ctx='{ctx}'")


# ── Main ─────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Compare formula extraction approaches")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--ocr", action="store_true", help="Run VLM OCR on detected crops")
    parser.add_argument("--vlm-scan", action="store_true", help="Run full-page VLM scan (no GROBID)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="VLM model name")
    parser.add_argument("--output-dir", default="tests/formula_compare_output", help="Output directory for crops")
    args = parser.parse_args()

    doc = fitz.open(args.pdf)
    print(f"PDF: {args.pdf} ({len(doc)} pages)")

    # ── Step 1: Heuristic detection (always runs, no API needed) ─────
    print("\n── Heuristic Detection ──")
    h_start = time.monotonic()
    heuristic = detect_heuristic(doc)
    h_time = time.monotonic() - h_start
    print(f"  Found {len(heuristic)} formula regions in {h_time:.2f}s")
    save_crops(heuristic, args.output_dir, "heuristic")

    # ── Step 2: VLM OCR on heuristic crops (optional) ────────────────
    ocr_results = None
    if args.ocr and heuristic:
        print("\n── VLM OCR (heuristic crops) ──")
        ocr_results = await vlm_ocr_crops(heuristic, args.model)

    # ── Step 3: Full-page VLM scan (optional) ────────────────────────
    vlm_results = None
    if args.vlm_scan:
        print("\n── Full-Page VLM Scan (方案 B: pure VLM, no GROBID) ──")
        pages = render_pages_with_formulas(doc)
        # Save page images too
        save_crops(
            [{"png_bytes": p["png_bytes"], "page": p["page"]} for p in pages],
            args.output_dir, "page"
        )
        vlm_results = await vlm_scan_pages(pages, args.model)

    doc.close()

    # ── Results ──────────────────────────────────────────────────────
    print_comparison(heuristic, vlm_results)

    if ocr_results:
        print(f"\n── OCR Results (from heuristic crops) ──")
        for r in ocr_results:
            if r.get("is_formula"):
                print(f"  [{r['index']}] label={r.get('label', '-')}: {r.get('latex', '')[:80]}")

    if vlm_results:
        print(f"\n── VLM Scan Results (方案 B) ──")
        for r in vlm_results:
            print(f"  page {r.get('page')}, label={r.get('label', '-')}: {r.get('latex', '')[:80]}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Heuristic regions:    {len(heuristic)}")
    if ocr_results:
        n_formula = len([r for r in ocr_results if r.get("is_formula")])
        print(f"  OCR valid formulas:   {n_formula}")
    if vlm_results:
        print(f"  VLM scan formulas:    {len(vlm_results)}")
    print(f"  API calls used:       {'1 (OCR)' if ocr_results and not vlm_results else ''}"
          f"{'1 (scan)' if vlm_results and not ocr_results else ''}"
          f"{'2 (OCR + scan)' if ocr_results and vlm_results else ''}"
          f"{'0' if not ocr_results and not vlm_results else ''}")


if __name__ == "__main__":
    asyncio.run(main())
