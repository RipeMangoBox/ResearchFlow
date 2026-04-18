"""PDF text extraction and section parsing using pymupdf.

L2 parse: extract sections, formulas, tables, figure captions.
No LLM calls — pure local processing.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


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
    import fitz  # pymupdf

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

    Filters out tiny images (<5KB, likely icons/logos) and returns
    metadata + raw bytes for each significant figure.
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
                if len(image_bytes) < 5000:  # Skip tiny images (<5KB)
                    continue

                ext = base_image.get("ext", "png")
                images.append({
                    "xref": xref,
                    "page_num": page_num,
                    "width": base_image.get("width", 0),
                    "height": base_image.get("height", 0),
                    "ext": ext,
                    "size_bytes": len(image_bytes),
                    "image_bytes": image_bytes,
                })

                if len(images) >= 20:  # Cap at 20 images per PDF
                    return images
            except Exception:
                continue

    return images


def get_pdf_size_mb(pdf_path: str | Path) -> float:
    """Get PDF file size in MB."""
    return Path(pdf_path).stat().st_size / (1024 * 1024)
