"""MinerU adapter for high-fidelity PDF content extraction.

MinerU (magic-pdf) provides:
- Full text with reading order preservation
- LaTeX formula extraction (inline + display)
- Cross-page table merging → Markdown tables
- Figure extraction as separate PNG/JPG files
- CJK language support

Requires: pip install magic-pdf
GPU recommended for speed (0.21s/page on Nvidia L4).
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MinerUResult:
    """Result of MinerU PDF parsing."""
    markdown_text: str = ""
    formulas: list[dict] = field(default_factory=list)
    # Schema: [{latex, type: "inline"|"display", page, context}]
    tables: list[dict] = field(default_factory=list)
    # Schema: [{markdown, caption, page}]
    figures: list[dict] = field(default_factory=list)
    # Schema: [{path, caption, page, bbox}]
    metadata: dict = field(default_factory=dict)
    page_count: int = 0
    success: bool = False
    error: str = ""


def is_available() -> bool:
    """Check if MinerU (magic-pdf) is installed."""
    try:
        import magic_pdf  # noqa: F401
        return True
    except ImportError:
        return False


def parse_pdf(pdf_path: str | Path) -> MinerUResult:
    """Parse PDF with MinerU and return structured content.

    Falls back gracefully if magic-pdf is not installed.
    """
    if not is_available():
        return MinerUResult(error="magic-pdf not installed", success=False)

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return MinerUResult(error=f"PDF not found: {pdf_path}", success=False)

    try:
        return _run_mineru(pdf_path)
    except Exception as e:
        logger.error(f"MinerU parse failed for {pdf_path}: {e}")
        return MinerUResult(error=str(e), success=False)


def _run_mineru(pdf_path: Path) -> MinerUResult:
    """Internal: run MinerU extraction pipeline."""
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

    result = MinerUResult()

    # Create temp output directory
    with tempfile.TemporaryDirectory(prefix="mineru_") as tmpdir:
        output_dir = Path(tmpdir)
        image_dir = output_dir / "images"
        image_dir.mkdir()

        # Read PDF bytes
        pdf_bytes = pdf_path.read_bytes()

        # Create dataset
        ds = PymuDocDataset(pdf_bytes)
        result.page_count = len(ds)

        # Analyze document layout
        infer_result = doc_analyze(ds)

        # Create writers
        image_writer = FileBasedDataWriter(str(image_dir))
        md_writer = FileBasedDataWriter(str(output_dir))

        # Run pipeline (auto-detect scan vs text PDF)
        if ds.classify() == "ocr":
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        # Get markdown content
        md_content = pipe_result.dump_md(md_writer, f"{pdf_path.stem}")
        result.markdown_text = md_content if isinstance(md_content, str) else ""

        # If markdown was written to file, read it
        if not result.markdown_text:
            md_file = output_dir / f"{pdf_path.stem}.md"
            if md_file.exists():
                result.markdown_text = md_file.read_text(encoding="utf-8")

        # Extract formulas from content_list
        try:
            content_list = pipe_result.dump_content_list(md_writer, f"{pdf_path.stem}")
            if isinstance(content_list, list):
                for item in content_list:
                    item_type = item.get("type", "")
                    if item_type == "equation":
                        result.formulas.append({
                            "latex": item.get("latex", item.get("text", "")),
                            "type": "display",
                            "page": item.get("page_idx", -1),
                        })
                    elif item_type == "table":
                        result.tables.append({
                            "markdown": item.get("text", ""),
                            "caption": item.get("caption", ""),
                            "page": item.get("page_idx", -1),
                        })
                    elif item_type == "image":
                        img_path = item.get("img_path", "")
                        result.figures.append({
                            "path": str(image_dir / img_path) if img_path else "",
                            "caption": item.get("caption", ""),
                            "page": item.get("page_idx", -1),
                        })
        except Exception as e:
            logger.warning(f"MinerU content_list extraction failed: {e}")

        # Collect any extracted images from the directory
        if not result.figures:
            for img_file in image_dir.glob("*"):
                if img_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    result.figures.append({
                        "path": str(img_file),
                        "caption": "",
                        "page": -1,
                    })

        result.success = True
        result.metadata = {
            "pdf_type": ds.classify(),
            "page_count": result.page_count,
            "formula_count": len(result.formulas),
            "table_count": len(result.tables),
            "figure_count": len(result.figures),
        }

    return result


def extract_formulas_from_markdown(markdown_text: str) -> list[dict]:
    """Extract LaTeX formulas from MinerU markdown output.

    Supplements the content_list extraction for inline formulas.
    """
    import re

    formulas = []

    # Display math: $$...$$
    for m in re.finditer(r'\$\$(.+?)\$\$', markdown_text, re.DOTALL):
        latex = m.group(1).strip()
        if len(latex) > 5:  # Skip trivial
            formulas.append({
                "latex": latex,
                "type": "display",
                "context": markdown_text[max(0, m.start()-50):m.end()+50],
            })

    # Inline math: $...$  (not $$)
    for m in re.finditer(r'(?<!\$)\$([^$\n]+?)\$(?!\$)', markdown_text):
        latex = m.group(1).strip()
        if len(latex) > 3 and any(c in latex for c in ['\\', '_', '^', '{', '}']):
            formulas.append({
                "latex": latex,
                "type": "inline",
                "context": markdown_text[max(0, m.start()-30):m.end()+30],
            })

    return formulas
