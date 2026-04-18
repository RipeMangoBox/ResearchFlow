"""End-to-end extraction test — verify all metadata extraction capabilities.

Usage: python -m pytest tests/test_e2e_extraction.py -v
Or:    python tests/test_e2e_extraction.py  (standalone)

Tests extraction quality WITHOUT database or API keys.
Uses local PDFs only.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def find_test_pdfs() -> list[tuple[str, str]]:
    """Find available test PDFs."""
    candidates = [
        ("MoMask", "storage/papers/raw-pdf/Motion_Generation/MoMask_Test.pdf"),
        ("MemAgent", os.path.expanduser("~/Desktop/简历/streamvideo-project/MemAgent/paper/paper.pdf")),
        ("RankGRPO", os.path.expanduser("~/Desktop/简历/Rank-GRPO/RANK-GRPO- TRAINING LLM-BASED CONVERSATIONAL RECOMMENDER SYSTEMS WITH REINFORCEMENT LEARNING.pdf")),
    ]
    return [(name, path) for name, path in candidates if os.path.exists(path)]


class TestPyMuPDFExtraction:
    """Test PyMuPDF basic extraction."""

    def test_sections(self):
        from backend.utils.pdf_extract import parse_pdf
        pdfs = find_test_pdfs()
        assert pdfs, "No test PDFs found"

        for name, path in pdfs:
            result = parse_pdf(path)
            assert result.page_count > 0, f"{name}: no pages"
            assert len(result.sections) >= 3, f"{name}: only {len(result.sections)} sections"
            print(f"  {name}: {result.page_count} pages, sections={list(result.sections.keys())}")

    def test_figure_images(self):
        from backend.utils.pdf_extract import parse_pdf
        pdfs = find_test_pdfs()
        total_with_figs = 0
        for name, path in pdfs:
            result = parse_pdf(path)
            if result.figure_images:
                total_with_figs += 1
                for fig in result.figure_images[:3]:
                    assert "image_bytes" in fig, f"{name}: figure missing bytes"
                    assert fig["size_bytes"] > 1000, f"{name}: figure too small"
            print(f"  {name}: {len(result.figure_images)} figures")
        # At least some papers should have extractable figures
        assert total_with_figs > 0, "No papers had extractable figures"


class TestFigureDetection:
    """Test figure region detection."""

    def test_candidate_detection(self):
        import fitz
        from backend.services.figure_extraction_service import _detect_candidate_regions

        pdfs = find_test_pdfs()
        for name, path in pdfs:
            doc = fitz.open(path)
            candidates = _detect_candidate_regions(doc)
            doc.close()

            assert len(candidates) > 0, f"{name}: no figure candidates detected"
            # Check at least some are caption-anchored
            caption_anchored = [c for c in candidates if c["source"] == "caption_anchored"]
            print(f"  {name}: {len(candidates)} candidates ({len(caption_anchored)} caption-anchored)")


class TestFormulaDetection:
    """Test formula region detection."""

    def test_heuristic_detection(self):
        import fitz
        from backend.services.formula_extraction_service import _detect_formula_regions_heuristic

        pdfs = find_test_pdfs()
        for name, path in pdfs:
            doc = fitz.open(path)
            formulas = _detect_formula_regions_heuristic(doc)
            doc.close()

            # At least some papers should have formulas
            print(f"  {name}: {len(formulas)} formulas detected")
            for f in formulas[:3]:
                assert "png_bytes" in f, f"{name}: formula missing image"
                assert len(f["grobid_text"]) > 3, f"{name}: formula text too short"


class TestPDFMetadata:
    """Test PDF first-page metadata extraction."""

    def test_title_extraction(self):
        from backend.utils.pdf_metadata import extract_metadata_from_pdf

        pdfs = find_test_pdfs()
        for name, path in pdfs:
            result = extract_metadata_from_pdf(path)
            print(f"  {name}: title={result.get('title', '?')[:60]}")
            # Title should exist and not be an arXiv header
            if result.get("title"):
                assert "arXiv:" not in result["title"], f"{name}: title is arXiv header"

    def test_acceptance_from_pdf(self):
        from backend.utils.pdf_metadata import extract_metadata_from_pdf

        pdfs = find_test_pdfs()
        for name, path in pdfs:
            result = extract_metadata_from_pdf(path)
            venue = result.get("venue", "")
            status = result.get("acceptance_status", "")
            print(f"  {name}: venue={venue}, status={status}")


class TestAcceptanceParsing:
    """Test acceptance status parsing from text."""

    def test_positive_patterns(self):
        from backend.services.enrich_service import _parse_acceptance_from_comment

        positives = [
            ("Accepted at ICLR 2025", "ICLR", "accepted"),
            ("Published as a conference paper at ICLR 2026", "ICLR", "accepted"),
            ("To appear at CVPR 2025", "CVPR", "accepted"),
            ("Accepted to NeurIPS 2025 as a spotlight paper", "NEURIPS", "accepted"),
            ("Accepted at ECCV 2024 (Oral)", "ECCV", "accepted"),
        ]
        for text, expected_venue, expected_status in positives:
            result = _parse_acceptance_from_comment(text)
            assert result is not None, f"Failed to parse: {text}"
            assert result["venue"] == expected_venue, f"{text}: got venue={result['venue']}"
            assert result["acceptance_status"] == expected_status

    def test_negative_patterns(self):
        from backend.services.enrich_service import _parse_acceptance_from_comment

        negatives = [
            "Under review at ICML 2025",
            "Submitted to NeurIPS 2025",
            "Work in progress",
            "Rejected from ICLR 2025",
            "Project Page: https://example.com",
            "12 pages, 5 figures",
        ]
        for text in negatives:
            result = _parse_acceptance_from_comment(text)
            assert result is None, f"False positive: {text} → {result}"


class TestDatasetExtraction:
    """Test dataset link extraction."""

    def test_extract_dataset_links(self):
        from backend.services.enrich_service import _extract_dataset_links

        text = """
        Download from https://huggingface.co/datasets/my-org/benchmark
        Google Drive: https://drive.google.com/drive/folders/abc123
        Zenodo: https://zenodo.org/records/12345
        Dataset: https://example.com/data.zip
        """
        urls = _extract_dataset_links(text)
        assert len(urls) >= 3, f"Only found {len(urls)} URLs"
        assert any("huggingface.co/datasets" in u for u in urls)
        assert any("drive.google.com" in u for u in urls)
        assert any("zenodo.org" in u for u in urls)


# ── Standalone runner ────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ResearchFlow E2E Extraction Tests")
    print("=" * 60)

    test_classes = [
        TestPyMuPDFExtraction,
        TestFigureDetection,
        TestFormulaDetection,
        TestPDFMetadata,
        TestAcceptanceParsing,
        TestDatasetExtraction,
    ]

    passed = 0
    failed = 0

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        obj = cls()
        for method_name in dir(obj):
            if not method_name.startswith("test_"):
                continue
            try:
                getattr(obj, method_name)()
                print(f"  ✅ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if failed else 0)
