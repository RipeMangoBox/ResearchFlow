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

MAX_PAGES_TO_SCAN = 15


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

    # Step 2: Crop low-res thumbnails for VLM (0.5x to reduce request size)
    thumbnails = []
    for c in candidates:
        page = doc[c["page_num"]]
        pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), clip=fitz.Rect(*c["bbox"]))
        thumbnails.append({
            **c,
            "thumb_bytes": pix.tobytes("png"),
            "thumb_w": pix.width,
            "thumb_h": pix.height,
        })

    # Step 3: ONE Claude API call — classify candidates + detect missed figures
    if settings.anthropic_api_key or settings.openai_api_key:
        # Also render full-page thumbnails for missed figure detection
        page_thumbs = []
        page_count = min(len(doc), MAX_PAGES_TO_SCAN)
        for i in range(page_count):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))  # Half-res for cost
            page_thumbs.append({
                "page_num": i,
                "thumb_bytes": pix.tobytes("png"),
                "page_width": page.rect.width,
                "page_height": page.rect.height,
            })

        classifications, missed_figures = await _classify_and_detect_missed(
            thumbnails, page_thumbs, paper_title, session, paper_id
        )

        # Step 3.5: For missed figures, add FULL PAGE as candidate
        # Better to crop too much than to miss content
        for missed in missed_figures:
            page_num = missed.get("page", -1)
            if page_num < 0 or page_num >= len(doc):
                continue
            page = doc[page_num]
            # Use full page — better to include extra than miss part of figure
            rect = page.rect

            candidates.append({
                "page_num": page_num,
                "bbox": [round(rect.x0), round(rect.y0), round(rect.x1), round(rect.y1)],
                "source": "vlm_missed_recovery",
                "label": missed.get("label", ""),
            })
            # Add classification for this new candidate
            classifications.append({
                "index": len(candidates) - 1,
                "is_figure_or_table": True,
                "label": missed.get("label", ""),
                "type": missed.get("type", "figure"),
                "caption": missed.get("caption", ""),
                "semantic_role": missed.get("semantic_role", "other"),
                "description": missed.get("description", ""),
            })
    else:
        # No VLM — use label from detection, keep BOTH figures and tables
        classifications = []
        for i, c in enumerate(candidates):
            label = c.get("label", f"Figure {i+1}")
            is_table = label.lower().startswith("table")
            classifications.append({
                "index": i, "is_figure_or_table": True,
                "label": label, "type": "table" if is_table else "figure",
                "semantic_role": "other", "description": "",
                "caption": c.get("caption_text", ""),
            })

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

    # Persist into the dedicated paper_figures table when a session is available.
    # Idempotent via uq_paper_figures_paper_label (paper_id + label).
    if session is not None and records:
        await _persist_paper_figures(session, paper_id, records)

    return records


async def _persist_paper_figures(session, paper_id: UUID, records: list[dict]) -> None:
    """Upsert figure records into the paper_figures table.

    Falls back silently if the table doesn't exist yet (pre-migration 024).
    """
    from sqlalchemy import text as _sql_text
    import json as _json
    insert_sql = _sql_text("""
        INSERT INTO paper_figures (
            paper_id, label, type, semantic_role, page_num, bbox,
            object_key, public_url, caption, description,
            width, height, size_bytes, extraction_method
        )
        VALUES (
            :paper_id, :label, :type, :semantic_role, :page_num, CAST(:bbox AS jsonb),
            :object_key, :public_url, :caption, :description,
            :width, :height, :size_bytes, :extraction_method
        )
        ON CONFLICT (paper_id, label) DO UPDATE SET
            type=EXCLUDED.type,
            semantic_role=EXCLUDED.semantic_role,
            page_num=EXCLUDED.page_num,
            bbox=EXCLUDED.bbox,
            object_key=EXCLUDED.object_key,
            public_url=EXCLUDED.public_url,
            caption=EXCLUDED.caption,
            description=EXCLUDED.description,
            width=EXCLUDED.width,
            height=EXCLUDED.height,
            size_bytes=EXCLUDED.size_bytes,
            extraction_method=EXCLUDED.extraction_method
    """)
    try:
        for r in records:
            label = (r.get("label") or "").strip()[:64]
            if not label or not r.get("object_key"):
                continue
            await session.execute(insert_sql, {
                "paper_id": str(paper_id),
                "label": label,
                "type": (r.get("type") or "figure")[:16],
                "semantic_role": (r.get("semantic_role") or "other")[:32],
                "page_num": r.get("page_num"),
                "bbox": _json.dumps(r.get("bbox")) if r.get("bbox") is not None else None,
                "object_key": r.get("object_key", "")[:500],
                "public_url": r.get("public_url"),
                "caption": (r.get("caption") or "")[:8000],
                "description": (r.get("description") or "")[:8000],
                "width": r.get("width"),
                "height": r.get("height"),
                "size_bytes": r.get("size_bytes"),
                "extraction_method": (r.get("extraction_method") or "vlm_precise")[:32],
            })
        await session.flush()
    except Exception as e:
        # Pre-migration deployments don't have the table yet — keep ingest going.
        logger.warning("paper_figures upsert skipped (table missing?): %s", e)


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

        # For each caption, find content region:
        #   - figure: scan UP (caption is below the figure)
        #   - table:  scan DOWN first (CVPR/NeurIPS convention has caption
        #             ABOVE the table); fall back to UP for ICLR-style layouts
        for cap in caption_blocks:
            if cap["type"] == "table":
                figure_bbox = (
                    _find_content_below_caption(page, cap, text_block_list)
                    or _find_content_above_caption(page, cap, text_block_list)
                )
            else:
                figure_bbox = _find_content_above_caption(page, cap, text_block_list)
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


def _find_content_below_caption(page, caption: dict, text_blocks: list) -> "fitz.Rect | None":
    """Locate table content directly BELOW its caption.

    Tables in CVPR / NeurIPS / ICML usually place the caption above the body,
    so the natural anchor scans downward. Strategy:
      1. Take the caption's x-range, expand a bit horizontally.
      2. Walk text blocks below the caption in y-order. Accept rows whose
         x-range overlaps the caption — these are table rows.
      3. Stop at the first big vertical gap (>30 px), the next caption, or
         the bottom of the page.
      4. Return the union rect (caption + body rows).
    """
    import re as _re
    page_rect = page.rect
    cap_y_top = caption["y0"]
    cap_y_bot = caption["y1"]
    cap_x0 = caption["x0"]
    cap_x1 = caption["x1"]
    cap_width = cap_x1 - cap_x0

    # Captions wider than 55% of page = full-width table
    full_width = cap_width > page_rect.width * 0.55
    if full_width:
        x_lo = page_rect.x0 + 5
        x_hi = page_rect.x1 - 5
    else:
        x_lo = max(page_rect.x0 + 5, cap_x0 - 25)
        x_hi = min(page_rect.x1 - 5, cap_x1 + 25)

    body_blocks = []
    last_y = cap_y_bot
    for y0, y1, x0, x1, text in text_blocks:
        if y0 < cap_y_bot + 2:
            continue
        if _re.match(r'^(Figure|Fig\.|Table)\s*\d+', text, _re.IGNORECASE):
            break  # next caption — stop
        # Accept row if its x-range overlaps the caption width band
        if x1 < x_lo - 10 or x0 > x_hi + 10:
            continue
        gap = y0 - last_y
        if body_blocks and gap > 30:
            break  # large vertical gap — table body has ended
        body_blocks.append((y0, y1, x0, x1))
        last_y = max(last_y, y1)
        if len(body_blocks) > 30:
            break  # don't grab whole-page

    if len(body_blocks) < 2:
        # Not enough rows to be a table
        return None

    bottom = max(b[1] for b in body_blocks)
    rect = fitz.Rect(x_lo, cap_y_top - 3, x_hi, min(bottom + 6, page_rect.y1))
    if rect.height < 30 or rect.width < 60:
        return None
    return rect


def _find_content_above_caption(page, caption: dict, text_blocks: list) -> "fitz.Rect | None":
    """Given a caption position, find the figure/table content region above it.

    Strategy:
    1. Try to find embedded images (page.get_image_info) near the caption
    2. If found, use the image bbox directly (most reliable)
    3. Otherwise, scan upward for the gap between body text and caption
    """
    import re as _re
    page_rect = page.rect
    cap_y = caption["y0"]  # Top of caption text
    cap_x0 = caption["x0"]
    cap_x1 = caption["x1"]

    # ── Strategy 1: Find embedded images OR dense drawing regions above caption ──
    try:
        nearby_rects = []

        # Check embedded raster images
        img_infos = page.get_image_info(xrefs=True)
        for img in img_infos:
            img_rect = fitz.Rect(img["bbox"])
            if img_rect.y1 > cap_y - 300 and img_rect.y0 < cap_y + 10:
                if img_rect.width > 50 and img_rect.height > 30:
                    nearby_rects.append(img_rect)

        # Check vector drawings (lines, curves, rects) — figures often use these
        drawings = page.get_drawings()
        if drawings:
            # Cluster drawing paths above caption into a bounding rect
            draw_rects = []
            for d in drawings:
                r = fitz.Rect(d["rect"])
                if r.y1 > cap_y - 300 and r.y0 < cap_y and r.width > 10 and r.height > 10:
                    draw_rects.append(r)
            if len(draw_rects) >= 3:  # Need multiple drawing elements to be a figure
                union = draw_rects[0]
                for r in draw_rects[1:]:
                    union = union | r
                if union.width > 50 and union.height > 30:
                    nearby_rects.append(union)

        if nearby_rects:
            union = nearby_rects[0]
            for r in nearby_rects[1:]:
                union = union | r
            result = fitz.Rect(
                min(union.x0, cap_x0) - 5,
                union.y0 - 5,
                max(union.x1, cap_x1) + 5,
                caption["y1"] + 5,
            )
            result = result & page_rect
            if result.width > 30 and result.height > 50:
                return result
    except Exception:
        pass

    # ── Strategy 2: Text gap scanning (original logic, relaxed) ──
    above_text_y1 = page_rect.y0  # Default: top of page

    for y0, y1, x0, x1, text in text_blocks:
        if y1 > cap_y - 5:
            continue  # Not above
        if len(text) < 30:
            continue  # Too short (likely label, axis tick)
        if _re.match(r'^(Figure|Fig\.|Table)\s*\d+', text, _re.IGNORECASE):
            continue  # Another caption
        # Must be a real paragraph — at least 50 chars
        if len(text) < 50 and y1 - y0 < 15:
            continue
        if y1 > above_text_y1:
            above_text_y1 = y1

    gap = cap_y - above_text_y1

    if gap < 15:
        # Very small gap — try table content below caption
        if caption["type"] == "table":
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


# ── Step 3: VLM classify + detect missed (1 API call) ────────

async def _classify_and_detect_missed(
    thumbnails: list[dict],
    page_thumbs: list[dict],
    paper_title: str,
    session,
    paper_id: UUID,
) -> tuple[list[dict], list[dict]]:
    """ONE Claude API call to:
    1. Classify each candidate crop (figure/table/noise)
    2. Scan all pages for figures that the deterministic method MISSED

    Returns: (classifications, missed_figures)
    - classifications: [{index, is_figure_or_table, label, type, ...}]
    - missed_figures: [{page, label, type, semantic_role, description}]
    """
    import openai
    # Endpoint resolution. We invoke through the openai client because the VLM
    # call uses the OpenAI chat/completions JSON schema, but the live server
    # only sets ANTHROPIC_* (pointing at Kimi's Anthropic-protocol gateway).
    # Kimi also exposes an OpenAI-compatible endpoint at the same host with a
    # `/v1` suffix — derive it when openai_base_url is empty.
    api_key = settings.openai_api_key or settings.anthropic_api_key
    base_url = settings.openai_base_url or None
    if not base_url and settings.anthropic_base_url:
        a = settings.anthropic_base_url.rstrip("/")
        base_url = a if a.endswith("/v1") else a + "/v1"
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"User-Agent": "claude-code/1.0"},
    )
    actual_model = settings.openai_model or "kimi-k2.6"

    content = []

    # Part A: candidate crops (OpenAI image_url format)
    # Cap at 10 candidates + 3 pages to stay within VLM token limits
    detected_labels = set()
    capped_thumbnails = thumbnails[:10] if thumbnails else []
    if capped_thumbnails:
        content.append({"type": "text", "text": f"=== PART A: {len(capped_thumbnails)} candidate crops ==="})
        for i, thumb in enumerate(capped_thumbnails):
            content.append({
                "type": "text",
                "text": f"--- Candidate {i} (page {thumb['page_num']}) ---",
            })
            b64 = base64.b64encode(thumb["thumb_bytes"]).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
            if thumb.get("label"):
                detected_labels.add(thumb["label"])

    # Part B: full page thumbnails for missed detection
    # Cap at 5 pages to avoid token limit (each page ~1500 tokens)
    capped_pages = page_thumbs[:3]
    content.append({"type": "text", "text": f"\n=== PART B: {len(capped_pages)} full page thumbnails ==="})
    for pt in capped_pages:
        content.append({
            "type": "text",
            "text": f"--- Page {pt['page_num']} ---",
        })
        b64 = base64.b64encode(pt["thumb_bytes"]).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    detected_str = ", ".join(sorted(detected_labels)) if detected_labels else "none yet"

    prompt = f"""You are analyzing the paper "{paper_title}".

TASK 1: For each candidate crop in PART A, classify it:
- Is it a figure or table? Or just text/equation/noise?
- What label? (Figure 1, Table 2, etc.)
- Semantic role and Chinese description

TASK 2: Look at ALL pages in PART B. The deterministic method already detected: [{detected_str}].
Find any figures or tables that were MISSED (not in the detected list above).

Return JSON with two keys:
{{
  "classifications": [
    {{"index": 0, "is_figure_or_table": true, "label": "Figure 1", "type": "figure", "caption": "...", "semantic_role": "pipeline", "description": "整体框架图"}},
    {{"index": 1, "is_figure_or_table": false, "label": "", "type": "", "caption": "", "semantic_role": "", "description": "纯文本"}}
  ],
  "missed": [
    {{"page": 5, "label": "Figure 5", "type": "figure", "caption": "first sentence of caption", "semantic_role": "result", "description": "推理时间对比散点图和用户偏好柱状图"}}
  ]
}}

semantic_role: motivation / pipeline / architecture / result / ablation / comparison / qualitative / quantitative / example / other
description: Chinese, 1 sentence

Rules:
- For TASK 1: return one entry per candidate, same order
- For TASK 2: only report GENUINELY MISSED figures/tables — not ones already in PART A
- Return ONLY the JSON object"""

    content.append({"type": "text", "text": prompt})

    start = time.monotonic()
    try:
        # Kimi's OpenAI-compatible endpoint at /v1/chat/completions silently
        # truncates streaming responses for multi-image inputs (observed:
        # 8702→1 tokens). Use a single-shot (stream=False) request instead;
        # the Anthropic-protocol gateway handles this correctly.
        resp = await client.chat.completions.create(
            model=actual_model,
            max_tokens=settings.vlm_max_tokens_medium,
            temperature=0.1,
            messages=[{"role": "user", "content": content}],
            stream=False,
        )
        latency = int((time.monotonic() - start) * 1000)
        text = (resp.choices[0].message.content or "") if resp.choices else "{}"
        in_tokens = len(prompt.split()) + (len(thumbnails) + len(page_thumbs)) * 500
        out_tokens = len(text.split())

        logger.info(
            f"VLM classify+detect: {len(thumbnails)} candidates + {len(page_thumbs)} pages, "
            f"~{in_tokens}→{out_tokens} tokens, {latency}ms"
        )

        if session:
            from backend.models.system import ModelRun
            run = ModelRun(
                paper_id=paper_id,
                model_provider="openai",
                model_name=actual_model,
                prompt_version="figure_classify_detect_v1",
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=(in_tokens * 3.0 + out_tokens * 15.0) / 1_000_000,
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

        result = json.loads(text)
        classifications = result.get("classifications", [])
        missed = result.get("missed", [])

        if not isinstance(classifications, list):
            classifications = []
        if not isinstance(missed, list):
            missed = []

        logger.info(f"VLM result: {len(classifications)} classified, {len(missed)} missed recovered")
        return classifications, missed

    except Exception as e:
        logger.error(f"VLM classify+detect failed: {e}")
        # Fallback: use original labels, keep BOTH figures and tables
        fallback = []
        for i, t in enumerate(thumbnails):
            label = t.get("label", f"Figure {i+1}")
            is_table = label.lower().startswith("table")
            fallback.append({
                "index": i, "is_figure_or_table": True,
                "label": label, "type": "table" if is_table else "figure",
                "semantic_role": "other", "description": "",
                "caption": t.get("caption_text", ""),
            })
        return fallback, []
