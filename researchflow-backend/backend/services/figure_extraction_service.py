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
    """Detect candidate figure/table regions on each page using PyMuPDF.

    Strategy:
    - Find all image bboxes on each page
    - Cluster nearby images into figure regions
    - Expand regions to include captions
    - Also detect large blank/graphic areas (for vector-drawn figures)

    Returns: [{page_num, bbox: [x0,y0,x1,y1], source: "image_cluster"|"large_block"}]
    """
    candidates = []

    for page_num in range(min(len(doc), 15)):  # First 15 pages
        page = doc[page_num]
        page_rect = page.rect

        # Strategy A: cluster embedded image bboxes
        img_rects = []
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                for inst_rect in page.get_image_rects(xref):
                    if inst_rect.width > 40 and inst_rect.height > 40:
                        img_rects.append(inst_rect)
            except Exception:
                continue

        if img_rects:
            clusters = _cluster_rects(img_rects)
            for cluster in clusters:
                expanded = _expand_for_caption(cluster, page_rect)
                candidates.append({
                    "page_num": page_num,
                    "bbox": [round(expanded.x0), round(expanded.y0),
                             round(expanded.x1), round(expanded.y1)],
                    "source": "image_cluster",
                })

        # Strategy B: detect drawing regions (vector graphics)
        # Pages with few text blocks but drawings often have figures
        drawings = page.get_drawings()
        if len(drawings) > 20:  # Lots of drawing commands = likely a vector figure
            # Find bounding box of all drawings
            draw_rects = [fitz.Rect(d["rect"]) for d in drawings if d.get("rect")]
            draw_rects = [r for r in draw_rects
                          if r.width > 50 and r.height > 50
                          and r.width < page_rect.width * 0.95]  # Not full-page
            if draw_rects:
                drawing_clusters = _cluster_rects(draw_rects)
                for cluster in drawing_clusters:
                    # Check it doesn't overlap with already-found image clusters
                    overlap = False
                    for existing in candidates:
                        if existing["page_num"] == page_num:
                            er = fitz.Rect(*existing["bbox"])
                            if er.intersects(cluster) and er.get_area() > 0:
                                overlap_area = (er & cluster).get_area()
                                if overlap_area / min(er.get_area(), cluster.get_area()) > 0.5:
                                    overlap = True
                                    break
                    if not overlap:
                        expanded = _expand_for_caption(cluster, page_rect)
                        candidates.append({
                            "page_num": page_num,
                            "bbox": [round(expanded.x0), round(expanded.y0),
                                     round(expanded.x1), round(expanded.y1)],
                            "source": "drawing_cluster",
                        })

    # Deduplicate and sort by page then y-position
    candidates = _deduplicate_candidates(candidates)
    candidates.sort(key=lambda c: (c["page_num"], c["bbox"][1]))

    return candidates[:25]  # Cap at 25 candidates


def _cluster_rects(rects: list) -> list:
    """Merge overlapping or nearby rectangles into clusters."""
    if not rects:
        return []

    sorted_rects = sorted(rects, key=lambda r: (r.y0, r.x0))
    clusters = [fitz.Rect(sorted_rects[0])]

    for rect in sorted_rects[1:]:
        merged = False
        for i, cluster in enumerate(clusters):
            # Check if rect is within 25pt of cluster
            expanded = fitz.Rect(
                cluster.x0 - 25, cluster.y0 - 25,
                cluster.x1 + 25, cluster.y1 + 25,
            )
            if expanded.intersects(rect):
                clusters[i] = cluster | rect  # Union
                merged = True
                break
        if not merged:
            clusters.append(fitz.Rect(rect))

    # Filter out tiny clusters
    return [c for c in clusters if c.width > 60 and c.height > 60]


def _expand_for_caption(rect, page_rect, margin: int = 10) -> fitz.Rect:
    """Expand rect with margin and extra space below for caption."""
    return fitz.Rect(
        max(rect.x0 - margin, page_rect.x0),
        max(rect.y0 - margin, page_rect.y0),
        min(rect.x1 + margin, page_rect.x1),
        min(rect.y1 + margin + 35, page_rect.y1),  # +35 for caption line
    )


def _deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    """Remove candidates that heavily overlap with each other."""
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
                smaller = min(c_rect.get_area(), e_rect.get_area())
                if smaller > 0 and intersection.get_area() / smaller > 0.6:
                    # Keep the larger one
                    if c_rect.get_area() > e_rect.get_area():
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
