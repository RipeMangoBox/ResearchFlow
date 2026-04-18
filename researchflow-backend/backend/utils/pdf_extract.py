"""PDF text extraction and section parsing using pymupdf.

L2 parse: extract sections, formulas, tables, figure captions.
No LLM calls — pure local processing.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf


@dataclass
class ParsedPDF:
    """Result of L2 PDF parsing."""
    full_text: str = ""
    page_count: int = 0
    sections: dict[str, str] = field(default_factory=dict)
    formulas: list[str] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    figure_captions: list[dict] = field(default_factory=list)
    figure_images: list[dict] = field(default_factory=list)
    references_text: str = ""


# Common section header patterns
SECTION_PATTERNS = [
    # Numbered: "1. Introduction", "2 Method", "3.1 Architecture"
    (r"^(\d+\.?\d*\.?)\s+(Introduction|Related Work|Background|Preliminary|Preliminaries)",
     "introduction"),
    (r"^(\d+\.?\d*\.?)\s+(Method|Approach|Proposed Method|Our Method|Framework|Model|Architecture)",
     "method"),
    (r"^(\d+\.?\d*\.?)\s+(Experiment|Evaluation|Results|Empirical)",
     "experiments"),
    (r"^(\d+\.?\d*\.?)\s+(Conclusion|Discussion|Summary|Limitation|Future Work)",
     "conclusion"),
    (r"^(\d+\.?\d*\.?)\s+(Abstract|ABSTRACT)", "abstract"),
    (r"^(\d+\.?\d*\.?)\s+(Related Work|Prior Work|Literature Review)", "related_work"),
    (r"^(\d+\.?\d*\.?)\s+(Ablation|Analysis)", "ablation"),
    (r"^(\d+\.?\d*\.?)\s+(Appendix|Supplementary)", "appendix"),
    # Un-numbered: "Introduction", "Method"
    (r"^(Introduction|INTRODUCTION)$", "introduction"),
    (r"^(Abstract|ABSTRACT)$", "abstract"),
    (r"^(Method|Methods|METHODS?|Approach|APPROACH|Methodology)$", "method"),
    (r"^(Experiments?|EXPERIMENTS?|Evaluation|Results|RESULTS)$", "experiments"),
    (r"^(Conclusions?|CONCLUSIONS?|Discussion|DISCUSSION)$", "conclusion"),
    (r"^(Related Work|RELATED WORK)$", "related_work"),
    (r"^(References|REFERENCES|Bibliography)$", "references"),
]

# Formula detection patterns
FORMULA_PATTERNS = [
    r"\\begin\{equation\}",    # LaTeX
    r"\\begin\{align",         # LaTeX align
    r"\$\$.+?\$\$",            # Display math
    r"\\frac\{",               # Fraction
    r"\\sum_",                 # Summation
    r"\\int_",                 # Integral
    r"\\mathcal\{",            # Calligraphic
    r"L\s*=\s*",               # Loss function
    r"\\nabla",                # Gradient
    r"\\arg\s*min",            # Argmin
]

# Figure caption pattern
FIGURE_CAPTION_RE = re.compile(
    r"(Figure|Fig\.?)\s*(\d+)[.:]\s*(.+?)(?=\n\n|\n[A-Z]|\n\d+\.|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Table caption pattern
TABLE_CAPTION_RE = re.compile(
    r"(Table)\s*(\d+)[.:]\s*(.+?)(?=\n\n|\n[A-Z]|\n\d+\.|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def parse_pdf(pdf_path: str | Path) -> ParsedPDF:
    """Parse a PDF file and extract structured content.

    Returns a ParsedPDF with sections, formulas, tables, and figure captions.
    """
    doc = fitz.open(str(pdf_path))
    result = ParsedPDF(page_count=len(doc))

    # Extract full text
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    result.full_text = "\n\n".join(pages_text)

    # Extract figure images
    result.figure_images = _extract_figure_images(doc)

    doc.close()

    # Parse sections
    result.sections = _extract_sections(result.full_text)

    # Extract formulas (lines containing formula patterns)
    result.formulas = _extract_formulas(result.full_text)

    # Extract figure captions
    result.figure_captions = _extract_figure_captions(result.full_text)

    # Extract table captions
    result.tables = _extract_table_captions(result.full_text)

    # Extract references section
    if "references" in result.sections:
        result.references_text = result.sections["references"]

    return result


def _extract_sections(text: str) -> dict[str, str]:
    """Split text into named sections based on header patterns."""
    lines = text.split("\n")
    sections: dict[str, str] = {}
    current_section = "preamble"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue

        matched_section = None
        for pattern, section_name in SECTION_PATTERNS:
            if re.match(pattern, stripped, re.IGNORECASE):
                matched_section = section_name
                break

        if matched_section:
            # Save previous section
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content and len(content) > 20:
                    sections[current_section] = content
            current_section = matched_section
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content and len(content) > 20:
            sections[current_section] = content

    return sections


def _extract_formulas(text: str) -> list[str]:
    """Extract lines that likely contain mathematical formulas."""
    formulas = []
    lines = text.split("\n")

    for i, line in enumerate(lines):
        for pattern in FORMULA_PATTERNS:
            if re.search(pattern, line):
                # Get context: line before + formula line + line after
                context_lines = []
                if i > 0:
                    context_lines.append(lines[i-1].strip())
                context_lines.append(line.strip())
                if i < len(lines) - 1:
                    context_lines.append(lines[i+1].strip())
                formula_text = " ".join(l for l in context_lines if l)
                if formula_text and formula_text not in formulas:
                    formulas.append(formula_text)
                break

    return formulas[:30]  # Cap at 30 formulas


def _extract_figure_captions(text: str) -> list[dict]:
    """Extract figure captions with figure number."""
    captions = []
    for m in FIGURE_CAPTION_RE.finditer(text):
        caption_text = m.group(3).strip().replace("\n", " ")
        if len(caption_text) > 10:
            captions.append({
                "figure_num": int(m.group(2)),
                "caption": caption_text[:500],
            })
    return captions


def _extract_table_captions(text: str) -> list[dict]:
    """Extract table captions with table number."""
    captions = []
    for m in TABLE_CAPTION_RE.finditer(text):
        caption_text = m.group(3).strip().replace("\n", " ")
        if len(caption_text) > 10:
            captions.append({
                "table_num": int(m.group(2)),
                "caption": caption_text[:500],
            })
    return captions


def _extract_figure_images(doc) -> list[dict]:
    """Extract figure images from a pymupdf Document.

    Two-strategy approach:
    1. Page-region screenshot: detect figure areas by clustering image
       bounding boxes on each page → render that page region as PNG.
       Captures the full figure including labels, subfigures, axes.
    2. Fallback to embedded xref images for pages without detectable regions.

    Returns list of dicts with image_bytes + metadata.
    """
    images = []

    for page_num, page in enumerate(doc):
        page_images = list(page.get_images(full=True))
        if not page_images:
            continue

        # Collect bounding boxes of all images on this page
        img_rects = []
        for img_info in page_images:
            xref = img_info[0]
            try:
                img_instances = page.get_image_rects(xref)
                for rect in img_instances:
                    if rect.width > 50 and rect.height > 50:  # Skip tiny icons
                        img_rects.append(rect)
            except Exception:
                continue

        if not img_rects:
            continue

        # Cluster nearby image rects into figure regions
        figure_regions = _cluster_image_rects(img_rects, page.rect)

        for region in figure_regions:
            # Expand region slightly to include captions/labels
            expanded = _expand_rect(region, page.rect, margin=15)

            try:
                # Render the page region as high-res PNG (2x zoom for clarity)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat, clip=expanded)
                image_bytes = pix.tobytes("png")

                if len(image_bytes) < 3000:  # Skip tiny results
                    continue

                images.append({
                    "page_num": page_num,
                    "width": pix.width,
                    "height": pix.height,
                    "ext": "png",
                    "size_bytes": len(image_bytes),
                    "image_bytes": image_bytes,
                    "bbox": [round(expanded.x0, 1), round(expanded.y0, 1),
                             round(expanded.x1, 1), round(expanded.y1, 1)],
                    "extraction_method": "page_region",
                })

                if len(images) >= 20:
                    return images
            except Exception:
                continue

    # Fallback: if page-region extraction found nothing, use embedded xref images
    if not images:
        images = _extract_xref_images(doc)

    return images


def _cluster_image_rects(rects: list, page_rect) -> list:
    """Cluster nearby image rectangles into figure regions.

    Images within the same figure (e.g., subfigures a/b/c) are close together.
    Merges rects that overlap or are within 30pt of each other.
    """
    import fitz

    if not rects:
        return []

    # Sort by y-position (top to bottom)
    sorted_rects = sorted(rects, key=lambda r: (r.y0, r.x0))

    clusters = []
    current_cluster = fitz.Rect(sorted_rects[0])

    for rect in sorted_rects[1:]:
        # Check if this rect is close to the current cluster
        expanded = fitz.Rect(
            current_cluster.x0 - 30, current_cluster.y0 - 30,
            current_cluster.x1 + 30, current_cluster.y1 + 30,
        )
        if expanded.intersects(rect):
            # Merge into current cluster
            current_cluster = current_cluster | rect  # Union
        else:
            # Save current cluster, start new one
            if current_cluster.width > 80 and current_cluster.height > 80:
                clusters.append(current_cluster)
            current_cluster = fitz.Rect(rect)

    # Don't forget the last cluster
    if current_cluster.width > 80 and current_cluster.height > 80:
        clusters.append(current_cluster)

    return clusters


def _expand_rect(rect, page_rect, margin: int = 15):
    """Expand a rect by margin but clamp within page bounds.

    Also extends downward to capture figure captions below the image.
    """
    import fitz
    return fitz.Rect(
        max(rect.x0 - margin, page_rect.x0),
        max(rect.y0 - margin, page_rect.y0),
        min(rect.x1 + margin, page_rect.x1),
        min(rect.y1 + margin + 40, page_rect.y1),  # +40 for caption text below
    )


def _extract_xref_images(doc) -> list[dict]:
    """Fallback: extract embedded image objects by xref.

    Used when page-region extraction finds nothing (e.g., scanned PDFs).
    """
    images = []
    seen_xrefs = set()

    for page_num, page in enumerate(doc):
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                if len(image_bytes) < 5000:
                    continue

                ext = base_image.get("ext", "png")
                images.append({
                    "page_num": page_num,
                    "width": base_image.get("width", 0),
                    "height": base_image.get("height", 0),
                    "ext": ext,
                    "size_bytes": len(image_bytes),
                    "image_bytes": image_bytes,
                    "extraction_method": "xref_embedded",
                })

                if len(images) >= 20:
                    return images
            except Exception:
                continue

    return images


def get_pdf_size_mb(pdf_path: str | Path) -> float:
    """Get PDF file size in MB."""
    return Path(pdf_path).stat().st_size / (1024 * 1024)
