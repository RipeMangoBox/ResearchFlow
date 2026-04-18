"""Figure extraction service — deterministic detection + VLM understanding.

Architecture (no GPU needed):
  1. PyMuPDF detects figure regions deterministically (CPU, free)
     - Image bbox clustering on each page
     - Full-page fallback for pages with large images
  2. Crop each candidate region → low-res thumbnail
  3. ONE Claude API call with all thumbnails → classify and describe each
     (Claude does NOT return bbox — it only judges what each crop is)
  4. Confirmed figures → high-res crop → OSS upload

Total API calls: 1 per paper
Claude's role: understanding and filtering, NOT coordinate detection
"""

import base64
import json
import logging
import time
from pathlib import Path
from uuid import UUID

import fitz  # pymupdf

from backend.config import settings
from backend.services.object_storage import get_storage

logger = logging.getLogger(__name__)


async def extract_figures_precise(
    pdf_path: str,
    paper_id: UUID,
    paper_title: str = "",
    session=None,
) -> list[dict]:
    """Extract all figures and tables from a PDF.

    Pipeline:
      1. PyMuPDF detects candidate figure regions (deterministic, CPU)
      2. Crop candidates as low-res thumbnails
      3. ONE Claude API call classifies all candidates
      4. Confirmed figures → high-res crop → OSS

    Returns list of figure records for PaperAnalysis.extracted_figure_images.
    """
    doc = fitz.open(pdf_path)

    # Step 1: Detect candidate figure regions
    candidates = _detect_candidate_regions(doc)
    if not candidates:
        doc.close()
        return []

    logger.info(f"Found {len(candidates)} candidate regions for {paper_id}")

    # Step 2: Crop low-res thumbnails for Claude
    thumbnails = []
    for c in candidates:
        page = doc[c["page_num"]]
        # 1x zoom thumbnail for classification (saves API tokens)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), clip=fitz.Rect(*c["bbox"]))
        thumbnails.append({
            **c,
            "thumb_bytes": pix.tobytes("png"),
            "thumb_w": pix.width,
            "thumb_h": pix.height,
        })

    # Step 3: ONE Claude API call to classify all candidates
    if settings.anthropic_api_key:
        classifications = await _classify_candidates_vlm(
            thumbnails, paper_title, session, paper_id
        )
    else:
        # No API key → treat all candidates as figures
        classifications = [
            {"index": i, "is_figure_or_table": True, "label": f"Figure {i+1}",
             "type": "figure", "semantic_role": "other", "description": "", "caption": ""}
            for i in range(len(thumbnails))
        ]

    # Step 4: Confirmed candidates → high-res crop → OSS
    storage = get_storage()
    records = []

    for cls in classifications:
        if not cls.get("is_figure_or_table", False):
            continue

        idx = cls.get("index", -1)
        if idx < 0 or idx >= len(candidates):
            continue

        c = candidates[idx]
        page = doc[c["page_num"]]
        rect = fitz.Rect(*c["bbox"])

        # High-res crop (2.5x zoom)
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=rect)
            image_bytes = pix.tobytes("png")
        except Exception as e:
            logger.warning(f"High-res crop failed for candidate {idx}: {e}")
            continue

        if len(image_bytes) < 2000:
            continue

        # Build object key
        label = cls.get("label", f"fig_{idx+1}").replace(" ", "_").replace(":", "")[:30]
        fig_type = cls.get("type", "figure")
        object_key = f"papers/{paper_id}/figures/{label}.png"

        try:
            await storage.put(object_key, image_bytes)
        except Exception as e:
            logger.warning(f"OSS upload failed for {object_key}: {e}")
            continue

        public_url = storage.get_public_url(object_key)

        records.append({
            "figure_num": idx + 1,
            "label": cls.get("label", f"Figure {idx+1}"),
            "type": fig_type,
            "object_key": object_key,
            "public_url": public_url,
            "page_num": c["page_num"],
            "bbox": c["bbox"],
            "width": pix.width,
            "height": pix.height,
            "size_bytes": len(image_bytes),
            "caption": cls.get("caption", ""),
            "semantic_role": cls.get("semantic_role", ""),
            "description": cls.get("description", ""),
            "extraction_method": "vlm_precise",
        })

    doc.close()
    logger.info(f"Extracted {len(records)} figures/tables for {paper_id}")
    return records


# ── Step 1: Deterministic candidate detection ────────────────

def _detect_candidate_regions(doc) -> list[dict]:
    """Detect figure/table regions using caption-anchored scanning.

    Strategy (top-down):
    1. Find all "Figure X" / "Table X" caption text positions on each page
    2. For each caption: scan upward to find the figure content boundary
       (images, drawings, or non-text region above the caption)
    3. The bbox = content region + caption text
    4. Fallback: if no captions found, cluster image bboxes (old method)

    This is far more reliable than bottom-up image clustering because
    the caption text position is deterministic and unambiguous.
    """
    import re
    candidates = []

    for page_num in range(min(len(doc), 15)):
        page = doc[page_num]
        page_rect = page.rect

        # ── Strategy A: Caption-anchored detection ────────────
        # Find caption text blocks: "Figure 1.", "Fig. 2:", "Table 3."
        # Must START with the label (not "...see Figure 1..." in body text)
        caption_pattern = re.compile(
            r'^(Figure|Fig\.|Table)\s*(\d+)\s*[.:\s]',
            re.IGNORECASE
        )

        text_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        text_block_list = []  # [(y0, y1, x0, x1, text)]
        for block in text_blocks:
            if block["type"] != 0:  # 0 = text block
                continue
            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
            bbox = block["bbox"]  # (x0, y0, x1, y1)
            text_block_list.append((bbox[1], bbox[3], bbox[0], bbox[2], block_text.strip()))

        # Sort text blocks by y-position
        text_block_list.sort(key=lambda b: b[0])

        # Find caption blocks — must START with "Figure X" / "Table X"
        # This avoids matching "...in Table 1..." in body text
        caption_blocks = []
        seen_labels = set()
        for y0, y1, x0, x1, text in text_block_list:
            m = caption_pattern.match(text)
            if m:
                fig_type = "table" if m.group(1).lower() == "table" else "figure"
                fig_num = int(m.group(2))
                label_key = f"{fig_type}_{fig_num}"
                if label_key in seen_labels:
                    continue  # Skip duplicate captions on same page
                seen_labels.add(label_key)
                caption_blocks.append({
                    "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                    "text": text, "type": fig_type, "num": fig_num,
                })

        # For each caption, find figure content above it
        for cap in caption_blocks:
            figure_bbox = _find_content_above_caption(
                page, cap, text_block_list
            )
            if figure_bbox:
                candidates.append({
                    "page_num": page_num,
                    "bbox": [round(figure_bbox.x0), round(figure_bbox.y0),
                             round(figure_bbox.x1), round(figure_bbox.y1)],
                    "source": "caption_anchored",
                    "label": f"{'Figure' if cap['type'] == 'figure' else 'Table'} {cap['num']}",
                    "caption_text": cap["text"][:200],
                })

        # ── Strategy B: Fallback — image/drawing cluster ──────
        # Only for pages where no captions were found
        if not any(c["page_num"] == page_num for c in candidates):
            fallback = _fallback_cluster_detection(page, page_num)
            candidates.extend(fallback)

    # Deduplicate overlapping candidates on same page
    candidates = _deduplicate_candidates(candidates)
    candidates.sort(key=lambda c: (c["page_num"], c["bbox"][1]))
    return candidates[:25]


def _find_content_above_caption(page, caption: dict, text_blocks: list) -> "fitz.Rect | None":
    """Given a caption position, find the figure/table content region above it.

    Key insight: the figure region is the FULL-WIDTH space between
    the previous text block (or page top) and the caption.

    For academic papers, figures occupy either:
    - Full page width (single-column figures)
    - Half page width (in-column figures)

    We detect which case by checking if there's a text gap spanning
    more than half the page width above the caption.
    """
    import re as _re
    page_rect = page.rect
    cap_y = caption["y0"]  # Top of caption text
    cap_x0 = caption["x0"]
    cap_x1 = caption["x1"]

    # Find the nearest BODY TEXT block ABOVE the caption
    # Skip: other captions, very short blocks (labels), blocks very close to caption
    above_text_y1 = page_rect.y0  # Default: top of page

    for y0, y1, x0, x1, text in text_blocks:
        if y1 > cap_y - 5:
            continue  # Not above
        if len(text) < 20:
            continue  # Too short (likely label, equation number)
        if _re.match(r'^(Figure|Fig\.|Table)\s*\d+', text, _re.IGNORECASE):
            continue  # Another caption
        # Must be a real paragraph — at least 2 lines or 50 chars
        if len(text) < 50 and y1 - y0 < 15:
            continue
        if y1 > above_text_y1:
            above_text_y1 = y1

    # The figure region is between above_text_y1 and caption bottom
    gap = cap_y - above_text_y1

    if gap < 20:
        # Very small gap — no figure content above this caption
        # Maybe the caption is below a table rendered as text
        # Try expanding downward instead (for tables)
        if caption["type"] == "table":
            # Look for table content below caption
            below_blocks = [(y0, y1, x0, x1) for y0, y1, x0, x1, t in text_blocks
                            if y0 > caption["y1"] + 2]
            if below_blocks:
                table_bottom = min(b[1] for b in below_blocks[:8])
                return fitz.Rect(
                    max(page_rect.x0 + 5, cap_x0 - 20),
                    cap_y - 5,
                    min(page_rect.x1 - 5, cap_x1 + 20),
                    table_bottom + 5,
                )
        return None

    # Determine width: if caption spans >55% of page width, assume full-width figure
    caption_width = cap_x1 - cap_x0
    if caption_width > page_rect.width * 0.55:
        # Full-width figure (common in academic papers)
        fig_x0 = page_rect.x0
        fig_x1 = page_rect.x1
    else:
        # In-column figure — try to detect column boundaries
        # Use the caption x-range as a guide, expand slightly
        mid_x = page_rect.width / 2
        if cap_x0 < mid_x:
            # Left column figure
            fig_x0 = page_rect.x0
            fig_x1 = mid_x + 10
        else:
            # Right column figure
            fig_x0 = mid_x - 10
            fig_x1 = page_rect.x1

    result = fitz.Rect(
        max(fig_x0, page_rect.x0),
        max(above_text_y1, page_rect.y0),
        min(fig_x1, page_rect.x1),
        min(caption["y1"] + 5, page_rect.y1),
    )

    # Sanity check: figure region should be reasonably sized
    if result.height < 40 or result.width < 60:
        return None

    return result


def _fallback_cluster_detection(page, page_num: int) -> list[dict]:
    """Fallback: cluster images + drawings when no captions found."""
    page_rect = page.rect
    candidates = []

    # Cluster images
    img_rects = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            for r in page.get_image_rects(xref):
                if r.width > 40 and r.height > 40:
                    img_rects.append(r)
        except Exception:
            continue

    if img_rects:
        clusters = _cluster_rects(img_rects, merge_dist=80)
        for cluster in clusters:
            expanded = fitz.Rect(
                max(cluster.x0 - 10, page_rect.x0),
                max(cluster.y0 - 10, page_rect.y0),
                min(cluster.x1 + 10, page_rect.x1),
                min(cluster.y1 + 45, page_rect.y1),
            )
            candidates.append({
                "page_num": page_num,
                "bbox": [round(expanded.x0), round(expanded.y0),
                         round(expanded.x1), round(expanded.y1)],
                "source": "image_cluster_fallback",
            })

    # Cluster drawings
    drawings = page.get_drawings()
    if len(drawings) > 20:
        draw_rects = [fitz.Rect(d["rect"]) for d in drawings if d.get("rect")]
        draw_rects = [r for r in draw_rects if r.width > 50 and r.height > 50]
        if draw_rects:
            clusters = _cluster_rects(draw_rects, merge_dist=40)
            for cluster in clusters:
                if not any(fitz.Rect(*c["bbox"]).intersects(cluster)
                           for c in candidates if c["page_num"] == page_num):
                    expanded = fitz.Rect(
                        max(cluster.x0 - 10, page_rect.x0),
                        max(cluster.y0 - 10, page_rect.y0),
                        min(cluster.x1 + 10, page_rect.x1),
                        min(cluster.y1 + 45, page_rect.y1),
                    )
                    candidates.append({
                        "page_num": page_num,
                        "bbox": [round(expanded.x0), round(expanded.y0),
                                 round(expanded.x1), round(expanded.y1)],
                        "source": "drawing_cluster_fallback",
                    })

    return candidates


def _cluster_rects(rects: list, merge_dist: int = 80) -> list:
    """Merge overlapping or nearby rectangles. merge_dist controls gap tolerance."""
    if not rects:
        return []

    sorted_rects = sorted(rects, key=lambda r: (r.y0, r.x0))
    clusters = [fitz.Rect(sorted_rects[0])]

    for rect in sorted_rects[1:]:
        merged = False
        for i, cluster in enumerate(clusters):
            expanded = fitz.Rect(
                cluster.x0 - merge_dist, cluster.y0 - merge_dist,
                cluster.x1 + merge_dist, cluster.y1 + merge_dist,
            )
            if expanded.intersects(rect):
                clusters[i] = cluster | rect
                merged = True
                break
        if not merged:
            clusters.append(fitz.Rect(rect))

    return [c for c in clusters if c.width > 60 and c.height > 60]


def _deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    """Remove heavily overlapping candidates on same page."""
    if len(candidates) <= 1:
        return candidates

    result = []
    for c in candidates:
        c_rect = fitz.Rect(*c["bbox"])
        overlap = False
        for existing in result:
            if existing["page_num"] != c["page_num"]:
                continue
            e_rect = fitz.Rect(*existing["bbox"])
            if c_rect.intersects(e_rect):
                intersection = c_rect & e_rect
                smaller_area = min(c_rect.get_area(), e_rect.get_area())
                if smaller_area > 0 and intersection.get_area() / smaller_area > 0.5:
                    # Keep the larger or the caption-anchored one
                    if c.get("source") == "caption_anchored":
                        existing["bbox"] = c["bbox"]
                        existing["source"] = c["source"]
                        existing["label"] = c.get("label", existing.get("label", ""))
                    elif c_rect.get_area() > e_rect.get_area():
                        existing["bbox"] = c["bbox"]
                    overlap = True
                    break
        if not overlap:
            result.append(c)

    return result


# ── Step 3: VLM classification (1 API call) ──────────────────

async def _classify_candidates_vlm(
    thumbnails: list[dict],
    paper_title: str,
    session,
    paper_id: UUID,
) -> list[dict]:
    """ONE Claude API call to classify all candidate crops.

    Claude sees each thumbnail and judges:
    - Is this a figure or table? (or just text/noise)
    - What label does it have? (Figure 1, Table 2, etc.)
    - What semantic role? (motivation, pipeline, result, etc.)
    - Brief description (Chinese)

    Claude does NOT return coordinates — those come from PyMuPDF.
    """
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    content = []
    for i, thumb in enumerate(thumbnails):
        content.append({
            "type": "text",
            "text": f"--- Candidate {i} (page {thumb['page_num']}) ---",
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(thumb["thumb_bytes"]).decode("utf-8"),
            },
        })

    prompt = f"""You are analyzing {len(thumbnails)} cropped regions from the paper "{paper_title}".

For each candidate, determine if it is a figure, table, or neither (just text/equation/noise).

Return a JSON array with one entry per candidate:
[
  {{
    "index": 0,
    "is_figure_or_table": true,
    "label": "Figure 1",
    "type": "figure",
    "caption": "first sentence of visible caption",
    "semantic_role": "pipeline",
    "description": "整体框架图"
  }},
  {{
    "index": 1,
    "is_figure_or_table": false,
    "label": "",
    "type": "",
    "caption": "",
    "semantic_role": "",
    "description": "纯文本段落，不是图表"
  }}
]

semantic_role must be one of: motivation, pipeline, architecture, result, ablation, comparison, qualitative, quantitative, example, other

Rules:
- description in Chinese, 1 sentence
- If you can see a label like "Figure 1" or "Table 2" in the image, use it
- Return ONLY the JSON array"""

    content.append({"type": "text", "text": prompt})

    start = time.monotonic()
    try:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            temperature=0.1,
            messages=[{"role": "user", "content": content}],
        )
        latency = int((time.monotonic() - start) * 1000)

        text = response.content[0].text if response.content else "[]"
        logger.info(
            f"VLM figure classification: {len(thumbnails)} candidates, "
            f"{response.usage.input_tokens}→{response.usage.output_tokens} tokens, "
            f"{latency}ms"
        )

        # Track cost
        if session:
            from backend.models.system import ModelRun
            run = ModelRun(
                paper_id=paper_id,
                model_provider="anthropic",
                model_name="claude-sonnet-4-20250514",
                prompt_version="figure_classify_v1",
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

        results = json.loads(text)
        return results if isinstance(results, list) else []

    except Exception as e:
        logger.error(f"VLM figure classification failed: {e}")
        # Fallback: treat all candidates as figures
        return [
            {"index": i, "is_figure_or_table": True, "label": f"Figure {i+1}",
             "type": "figure", "semantic_role": "other", "description": "", "caption": ""}
            for i in range(len(thumbnails))
        ]
