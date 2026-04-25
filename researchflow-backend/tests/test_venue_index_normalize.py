"""Unit tests for venue_index normalization functions."""

from backend.services.venue_index.normalize import (
    extract_arxiv_id,
    extract_doi,
    extract_openreview_forum_id,
    extract_aaai_article_id,
    normalize_authors,
    normalize_link,
    normalize_title,
    normalize_whitespace,
    slugify_short_title,
    strip_latex,
)


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("  hello   world  ") == "hello world"

    def test_collapses_tabs_newlines(self):
        assert normalize_whitespace("hello\t\n  world") == "hello world"

    def test_empty(self):
        assert normalize_whitespace("") == ""

    def test_none(self):
        assert normalize_whitespace(None) == ""


class TestStripLatex:
    def test_inline_math(self):
        assert strip_latex("$O(n)$ Attention") == "O(n) Attention"

    def test_mathcal(self):
        assert "F" in strip_latex("\\mathcal{F}-divergence")

    def test_greek_letters(self):
        result = strip_latex("\\alpha-blending with \\beta")
        assert "alpha" not in result.lower()
        assert "blending" in result

    def test_braces(self):
        result = strip_latex("{x}^{2}")
        assert "x" in result and "2" in result

    def test_no_latex(self):
        assert strip_latex("Plain Title") == "Plain Title"


class TestNormalizeTitle:
    def test_lowercase_alphanum_only(self):
        assert normalize_title("Hello, World! 2025") == "hello world 2025"

    def test_strips_articles(self):
        result = normalize_title("The Transformer Architecture")
        assert "transformer" in result

    def test_latex_in_title(self):
        # \log is stripped, both n's remain
        assert normalize_title("$O(n\\log n)$ Attention") == "o n n attention"

    def test_latex_mathcal(self):
        assert normalize_title("Learning $\\mathcal{F}$-Divergence") == "learning f divergence"

    def test_latex_and_plain_match(self):
        # A latex title and its plain equivalent should normalize the same
        latex = normalize_title("$O(n)$ Complexity for Transformers")
        plain = normalize_title("O(n) Complexity for Transformers")
        assert latex == plain


class TestNormalizeAuthors:
    def test_semicolon_delimiter(self):
        assert normalize_authors("Alice; Bob; Carol") == "Alice; Bob; Carol"

    def test_comma_delimiter(self):
        assert normalize_authors("Alice, Bob, Carol") == "Alice; Bob; Carol"

    def test_and_delimiter(self):
        assert normalize_authors("Alice and Bob") == "Alice; Bob"

    def test_empty(self):
        assert normalize_authors("") == ""


class TestNormalizeLink:
    def test_strips_fragment(self):
        result = normalize_link("https://example.com/page#section")
        assert "#" not in result

    def test_strips_wrapping_chars(self):
        assert normalize_link("<https://example.com>") == "https://example.com"

    def test_relative_to_base(self):
        result = normalize_link("/paper.pdf", "https://example.com")
        assert result == "https://example.com/paper.pdf"

    def test_empty(self):
        assert normalize_link("") == ""


class TestExtractArxivId:
    def test_abs_url(self):
        assert extract_arxiv_id("https://arxiv.org/abs/2503.12345") == "2503.12345"

    def test_pdf_url(self):
        assert extract_arxiv_id("https://arxiv.org/pdf/2503.12345") == "2503.12345"

    def test_no_match(self):
        assert extract_arxiv_id("https://google.com") == ""

    def test_five_digit(self):
        assert extract_arxiv_id("https://arxiv.org/abs/2503.12345") == "2503.12345"

    def test_none(self):
        assert extract_arxiv_id(None) == ""


class TestExtractOpenReviewForumId:
    def test_basic(self):
        assert extract_openreview_forum_id("https://openreview.net/forum?id=abc123") == "abc123"

    def test_with_extra_params(self):
        assert extract_openreview_forum_id("https://openreview.net/forum?noteId=x&id=def456") == "def456"

    def test_no_match(self):
        assert extract_openreview_forum_id("https://google.com") == ""

    def test_none(self):
        assert extract_openreview_forum_id(None) == ""


class TestExtractDoi:
    def test_doi_org(self):
        assert extract_doi("https://doi.org/10.1145/12345") == "10.1145/12345"

    def test_not_doi(self):
        assert extract_doi("https://arxiv.org/abs/2503.12345") == ""


class TestExtractAAAIArticleId:
    def test_basic(self):
        assert extract_aaai_article_id("https://ojs.aaai.org/index.php/AAAI/article/view/28815") == "28815"

    def test_with_pdf_suffix(self):
        assert extract_aaai_article_id("https://ojs.aaai.org/index.php/AAAI/article/view/28815/29555") == "28815"

    def test_no_match(self):
        assert extract_aaai_article_id("https://google.com") == ""


class TestSlugifyShortTitle:
    def test_basic(self):
        result = slugify_short_title("Vision Transformer for Dense Prediction")
        assert result == "VisionTransformerFor"

    def test_default_fallback(self):
        assert slugify_short_title("") == "Paper"
