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
    sections_hierarchy: list[dict] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    figure_captions: list[dict] = field(default_factory=list)
    figure_images: list[dict] = field(default_factory=list)
    references_text: str = ""
    citation_contexts: list[dict] = field(default_factory=list)
    dataset_mentions: list[dict] = field(default_factory=list)


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
    result.sections_hierarchy = _extract_sections_hierarchical(result.full_text)

    # Extract formulas (lines containing formula patterns)
    result.formulas = _extract_formulas(result.full_text)

    # Extract figure captions
    result.figure_captions = _extract_figure_captions(result.full_text)

    # Extract table captions
    result.tables = _extract_table_captions(result.full_text)

    # Extract references section
    if "references" in result.sections:
        result.references_text = result.sections["references"]

    # Citation contexts & dataset mentions
    result.citation_contexts = _extract_citation_contexts(result.full_text, result.sections)
    result.dataset_mentions = _extract_dataset_mentions(result.full_text, result.sections)

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


# ── Hierarchical section heading regex ────────────────────────────
# Matches: "1 Introduction", "2. Method", "3.1 Encoder", "3.1.2 Details", "A.1 Appendix"
_HEADING_RE = re.compile(
    r"^([A-Z]?\d+(?:\.\d+)*)\s*\.?\s+([A-Z][^\n]{2,80})$", re.MULTILINE
)


def _extract_sections_hierarchical(text: str) -> list[dict]:
    """Extract sections preserving heading hierarchy (1 → 1.1 → 1.1.1).

    Returns a flat list of sections ordered by appearance, each with:
      {number, title, level, text, start_char}
    Level is determined by dot-count: "3" → 1, "3.1" → 2, "3.1.2" → 3.
    """
    headings: list[dict] = []
    for m in _HEADING_RE.finditer(text):
        number = m.group(1)
        title = m.group(2).strip()
        # Level = number of numeric components
        level = number.count(".") + 1
        headings.append({
            "number": number,
            "title": title,
            "level": level,
            "start_char": m.start(),
        })

    if not headings:
        return []

    # Fill text between headings
    sections: list[dict] = []
    for i, h in enumerate(headings):
        text_start = h["start_char"] + len(f"{h['number']} {h['title']}")
        text_end = headings[i + 1]["start_char"] if i + 1 < len(headings) else len(text)
        body = text[text_start:text_end].strip()
        sections.append({
            "number": h["number"],
            "title": h["title"],
            "level": h["level"],
            "text": body[:5000],  # cap per section
            "start_char": h["start_char"],
        })

    return sections


# ── Citation context extraction ───────────────────────────────────
# Numeric: [1], [1,2], [1-3], [1, 2, 3]
_CITE_NUMERIC_RE = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")
# Author-year: (Smith et al., 2023), Smith et al. (2023)
_CITE_AUTHOR_RE = re.compile(r"\(([A-Z][a-z]+\s+et\s+al\.?,?\s*\d{4})\)")


def _extract_citation_contexts(
    full_text: str, sections: dict[str, str], max_contexts: int = 150,
) -> list[dict]:
    """Extract in-text citation mentions with ±120 chars context.

    Returns: [{ref_marker, context, section, char_offset}]
    """
    # Build section boundary map: char_offset → section_name
    section_ranges: list[tuple[int, int, str]] = []
    for sec_name, sec_text in sections.items():
        idx = full_text.find(sec_text[:100])
        if idx >= 0:
            section_ranges.append((idx, idx + len(sec_text), sec_name))
    section_ranges.sort()

    def _find_section(pos: int) -> str:
        for start, end, name in section_ranges:
            if start <= pos < end:
                return name
        return "unknown"

    contexts: list[dict] = []
    seen: dict[str, list[str]] = {}  # ref_marker → list of sections already recorded

    for pattern in (_CITE_NUMERIC_RE, _CITE_AUTHOR_RE):
        for m in pattern.finditer(full_text):
            if len(contexts) >= max_contexts:
                break
            raw_marker = m.group(0)
            # Expand numeric ranges: "[1-3]" → "1", "2", "3"
            markers = _expand_citation_markers(m.group(1))
            ctx_start = max(0, m.start() - 120)
            ctx_end = min(len(full_text), m.end() + 120)
            context = full_text[ctx_start:ctx_end].replace("\n", " ").strip()
            section = _find_section(m.start())

            # Skip very short / non-prose contexts
            if len(context) < 30:
                continue

            for marker in markers:
                key = marker
                if key not in seen:
                    seen[key] = []
                # Max 3 contexts per marker, prefer different sections
                if len(seen[key]) >= 3 and section in seen[key]:
                    continue
                seen[key].append(section)
                contexts.append({
                    "ref_marker": marker,
                    "context": context,
                    "section": section,
                    "char_offset": m.start(),
                })

    return contexts


def _expand_citation_markers(raw: str) -> list[str]:
    """Expand "[1, 3-5]" into ["1", "3", "4", "5"]."""
    markers: list[str] = []
    for part in re.split(r"[,\s]+", raw):
        part = part.strip()
        range_m = re.match(r"(\d+)\s*[-–]\s*(\d+)", part)
        if range_m:
            lo, hi = int(range_m.group(1)), int(range_m.group(2))
            for n in range(lo, min(hi + 1, lo + 20)):  # safety cap
                markers.append(str(n))
        elif part.isdigit():
            markers.append(part)
        else:
            markers.append(part)  # author-year passthrough
    return markers


# ── Dataset mention detection ─────────────────────────────────────
# Well-known ML dataset names (case-insensitive matching)
_KNOWN_DATASETS = {
    # Vision
    "imagenet", "coco", "cifar-10", "cifar-100", "mnist", "fashion-mnist",
    "lsun", "celeba", "voc", "ade20k", "cityscapes", "kitti",
    "laion", "cc3m", "cc12m", "webvid", "howto100m", "ego4d",
    # NLP
    "glue", "superglue", "squad", "wmt", "xsum", "cnn/dailymail",
    "mmlu", "hellaswag", "winogrande", "arc", "truthfulqa",
    # Multimodal
    "vqa", "gqa", "visual genome", "nocaps", "flickr30k",
    "textvqa", "okvqa", "a-okvqa",
    # Motion / 3D
    "humanml3d", "amass", "babel", "kit-ml", "h3.6m", "human3.6m",
    "aist++", "motionx", "motion-x", "grab", "ntu rgb+d",
    # Audio / speech
    "librispeech", "commonvoice", "voxceleb", "audioset",
    # Agent / tool
    "webshop", "alfworld", "scienceworld", "toolbench", "agentbench",
    # Video
    "kinetics-400", "kinetics-700", "something-something",
    "activitynet", "ucf101", "hmdb51", "charades",
    "msrvtt", "msvd", "didemo", "youcook2", "vatex",
    # Generation benchmarks
    "fid", "is", "clip-score", "fvd",
}

# URL patterns for datasets
_DATASET_URL_RE = re.compile(
    r"(https?://(?:"
    r"huggingface\.co/datasets/[\w\-\.\/]+"
    r"|kaggle\.com/[\w\-\/]+"
    r"|zenodo\.org/record/\d+"
    r"|figshare\.com/[\w\-\/]+"
    r"|drive\.google\.com/[\w\-\/\?=&]+"
    r"|github\.com/[\w\-]+/[\w\-]+/(?:tree|blob)/[\w\-]+/data"
    r"))",
    re.IGNORECASE,
)

# Detect paper-released dataset: "we release/publish/provide/introduce ... dataset"
_RELEASE_VERB_RE = re.compile(
    r"\b(?:we|our)\s+(?:release|publish|provide|introduce|present|contribute|propose)\b"
    r"[^.]{0,80}\b(?:dataset|benchmark|corpus|data)\b",
    re.IGNORECASE,
)

# Generic dataset mention in experiments: "train/evaluate on the X dataset"
_DATASET_USAGE_RE = re.compile(
    r"(?:train|evaluat|test|benchmark|fine-?tun)\w*\s+"
    r"(?:on|with|using)\s+(?:the\s+)?([A-Z][\w\-\.]+(?:\s+[\w\-\.]+){0,2})"
    r"\s+(?:dataset|benchmark|corpus)",
    re.IGNORECASE,
)


def _extract_dataset_mentions(
    full_text: str, sections: dict[str, str],
) -> list[dict]:
    """Detect dataset references from PDF text.

    Three-layer detection:
    1. Data Availability / Datasets section parsing
    2. Known dataset name matching in experiments section
    3. URL pattern matching across full text

    Returns: [{name, url, source_section, context, is_released_by_paper}]
    """
    mentions: list[dict] = []
    seen_names: set[str] = set()

    # Check if paper releases its own dataset
    is_releasing = bool(_RELEASE_VERB_RE.search(full_text))

    # 1. Scan dedicated sections: "Data Availability", "Datasets", "Experimental Setup"
    data_sections = {}
    for sec_name, sec_text in sections.items():
        lower = sec_name.lower()
        if any(kw in lower for kw in ("data", "dataset", "experiment", "setup", "benchmark")):
            data_sections[sec_name] = sec_text

    # 2. Known dataset name matching
    search_text = ""
    for sec_name in ("experiments", "method", "ablation"):
        if sec_name in sections:
            search_text += sections[sec_name] + "\n"
    if not search_text:
        search_text = full_text

    for name in _KNOWN_DATASETS:
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        m = pattern.search(search_text)
        if m:
            norm = name.lower()
            if norm not in seen_names:
                seen_names.add(norm)
                ctx_start = max(0, m.start() - 60)
                ctx_end = min(len(search_text), m.end() + 60)
                mentions.append({
                    "name": name,
                    "url": None,
                    "source_section": "experiments",
                    "context": search_text[ctx_start:ctx_end].replace("\n", " ").strip(),
                    "is_released_by_paper": False,
                })

    # Also check generic pattern: "evaluate on the FooBar dataset"
    for m in _DATASET_USAGE_RE.finditer(search_text):
        candidate = m.group(1).strip()
        norm = candidate.lower()
        if norm not in seen_names and len(candidate) > 2:
            seen_names.add(norm)
            ctx_start = max(0, m.start() - 40)
            ctx_end = min(len(search_text), m.end() + 40)
            mentions.append({
                "name": candidate,
                "url": None,
                "source_section": "experiments",
                "context": search_text[ctx_start:ctx_end].replace("\n", " ").strip(),
                "is_released_by_paper": False,
            })

    # 3. URL pattern matching
    for m in _DATASET_URL_RE.finditer(full_text):
        url = m.group(1)
        # Infer name from URL
        parts = url.rstrip("/").split("/")
        name = parts[-1] if parts else url
        ctx_start = max(0, m.start() - 60)
        ctx_end = min(len(full_text), m.end() + 60)
        mentions.append({
            "name": name,
            "url": url,
            "source_section": "body",
            "context": full_text[ctx_start:ctx_end].replace("\n", " ").strip(),
            "is_released_by_paper": is_releasing,
        })

    return mentions[:50]  # cap


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
