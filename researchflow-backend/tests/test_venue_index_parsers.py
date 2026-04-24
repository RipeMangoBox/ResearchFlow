"""Unit tests for venue_index parsers — verify each parser extracts correct fields.

Each test uses a minimal HTML/JSON fixture inline to avoid external dependencies.
Tests run without DB or network access.
"""

from __future__ import annotations

import pytest

from backend.services.venue_index.models import AcceptedPaperRecord, ConferenceYearConfig, SourceConfig
from backend.services.venue_index.parsers import (
    PARSERS,
    parse_acl_anthology_html,
    parse_acmmm_html,
    parse_acmmm_vue_accepted,
    parse_anthropic_research,
    parse_cvpr_openaccess_html,
    parse_hf_daily_papers,
    parse_iclr_proceedings_html,
    parse_iclr_virtual_html,
    parse_jmlr_html,
    parse_kdd_html,
    parse_kesen_siggraph_html,
    parse_neurips_virtual_html,
    parse_openalex_works,
    parse_openreview_api_v2,
    parse_openreview_notes_json,
    parse_s2_bulk_papers,
    parse_simple_html_paper_list,
    parse_virtual_conference_json,
    parse_aaai_ojs_html,
)


# ── Helpers ──────────────────────────────────────────────────────

def _conf(venue: str = "TEST", year: int = 2025) -> ConferenceYearConfig:
    return ConferenceYearConfig(
        venue=venue, year=year, conf_year=f"{venue}_{year}",
        status="active", skip_reason="",
        primary_source=SourceConfig(kind="test", url="http://test", parser="test"),
    )


def _src(kind: str = "test", url: str = "http://test", parser: str = "test") -> SourceConfig:
    return SourceConfig(kind=kind, url=url, parser=parser)


def _assert_basic(records: list[AcceptedPaperRecord], min_count: int = 1):
    assert len(records) >= min_count, f"Expected >= {min_count} records, got {len(records)}"
    for r in records:
        assert r.title, "title must be non-empty"
        assert r.venue, "venue must be set"
        assert r.year > 0, "year must be positive"


# ── parse_virtual_conference_json ────────────────────────────────

class TestVirtualConferenceJSON:
    """Tests for EventHosts unified JSON (ICLR/NeurIPS/ICML/CVPR/ECCV)."""

    FIXTURE = {
        "results": [
            {
                "name": "Attention Is All You Need: Revisited",
                "authors": [
                    {"fullname": "Alice Zhang", "institution": "MIT"},
                    {"fullname": "Bob Chen", "institution": "Stanford"},
                ],
                "abstract": "We revisit the transformer architecture.",
                "keywords": ["transformers", "attention", "NLP"],
                "decision": "Accept (Oral)",
                "eventtype": "oral",
                "event_type": "Oral Presentation",
                "paper_pdf_url": "https://openreview.net/pdf?id=abc123",
                "sourceurl": "https://openreview.net/forum?id=abc123",
                "virtualsite_url": "/virtual/2025/poster/12345",
                "topic": "Deep Learning",
                "id": "12345",
                "uid": "uid_12345",
                "session": "Morning Session A",
                "room_name": "Hall B",
            },
            {
                "name": "  Sparse   Attention  for  Long  Documents ",
                "authors": [{"fullname": "Carol Li"}],
                "abstract": "",
                "keywords": [],
                "decision": "poster",
                "eventtype": "poster",
                "event_type": "",
                "paper_pdf_url": "",
                "sourceurl": "",
                "virtualsite_url": "/virtual/2025/poster/12346",
                "topic": "",
                "id": "12346",
            },
        ]
    }

    def test_basic_extraction(self):
        records = parse_virtual_conference_json(self.FIXTURE, _conf("ICLR"), _src())
        _assert_basic(records, 2)

    def test_title_whitespace_normalization(self):
        records = parse_virtual_conference_json(self.FIXTURE, _conf("ICLR"), _src())
        assert records[1].title == "Sparse Attention for Long Documents"

    def test_authors_semicolon_separated(self):
        records = parse_virtual_conference_json(self.FIXTURE, _conf("ICLR"), _src())
        assert records[0].authors == "Alice Zhang; Bob Chen"

    def test_keywords_extracted(self):
        records = parse_virtual_conference_json(self.FIXTURE, _conf("ICLR"), _src())
        assert "transformers" in records[0].keywords_raw

    def test_decision_normalized(self):
        records = parse_virtual_conference_json(self.FIXTURE, _conf("ICLR"), _src())
        assert records[0].decision == "Accept (Oral)"
        assert records[0].acceptance_type == "Oral"

    def test_poster_decision(self):
        records = parse_virtual_conference_json(self.FIXTURE, _conf("ICLR"), _src())
        assert records[1].acceptance_type == "Poster"

    def test_openreview_forum_id_extracted(self):
        records = parse_virtual_conference_json(self.FIXTURE, _conf("ICLR"), _src())
        assert records[0].openreview_forum_id == "abc123"

    def test_pdf_url_as_paper_link(self):
        records = parse_virtual_conference_json(self.FIXTURE, _conf("ICLR"), _src())
        assert "pdf" in records[0].paper_link

    def test_empty_results(self):
        records = parse_virtual_conference_json({"results": []}, _conf("ICLR"), _src())
        assert records == []

    def test_skip_empty_title(self):
        payload = {"results": [{"name": "", "authors": []}]}
        records = parse_virtual_conference_json(payload, _conf("ICLR"), _src())
        assert records == []


# ── parse_openreview_api_v2 ──────────────────────────────────────

class TestOpenReviewAPIv2:

    FIXTURE = {
        "notes": [
            {
                "id": "xYz789",
                "forum": "xYz789",
                "content": {
                    "title": {"value": "Graph Neural Networks for Code"},
                    "authors": {"value": ["David Wu", "Eve Park"]},
                    "keywords": {"value": ["GNN", "code understanding"]},
                    "abstract": {"value": "We propose GNN4Code."},
                    "pdf": {"value": "/pdf/xYz789.pdf"},
                },
            }
        ]
    }

    def test_basic_extraction(self):
        records = parse_openreview_api_v2(self.FIXTURE, _conf("NeurIPS"), _src())
        _assert_basic(records)
        assert records[0].title == "Graph Neural Networks for Code"
        assert records[0].openreview_forum_id == "xYz789"

    def test_authors(self):
        records = parse_openreview_api_v2(self.FIXTURE, _conf("NeurIPS"), _src())
        assert records[0].authors == "David Wu; Eve Park"

    def test_keywords(self):
        records = parse_openreview_api_v2(self.FIXTURE, _conf("NeurIPS"), _src())
        assert "GNN" in records[0].keywords_raw

    def test_pdf_url_constructed(self):
        records = parse_openreview_api_v2(self.FIXTURE, _conf("NeurIPS"), _src())
        assert records[0].has_pdf_camera_ready == "true"

    def test_empty_notes(self):
        records = parse_openreview_api_v2({"notes": []}, _conf("NeurIPS"), _src())
        assert records == []


# ── parse_openreview_notes_json ──────────────────────────────────

class TestOpenReviewNotesJSON:

    FIXTURE = {
        "notes": [
            {
                "id": "abc123",
                "content": {
                    "title": "Learning to Align",
                    "authors": ["Frank Lee", "Grace Kim"],
                    "keywords": ["alignment", "LLM"],
                    "abstract": "A method for alignment.",
                    "pdf": "/pdf/abc123.pdf",
                },
            }
        ]
    }

    def test_basic(self):
        records = parse_openreview_notes_json(self.FIXTURE, _conf("ICLR"), _src())
        _assert_basic(records)
        assert records[0].title == "Learning to Align"
        assert records[0].openreview_forum_id == "abc123"
        assert records[0].authors == "Frank Lee; Grace Kim"


# ── parse_cvpr_openaccess_html ───────────────────────────────────

class TestCVPROpenAccessHTML:

    FIXTURE = """
    <dt class="ptitle"><br><a href="/content/CVPR2025/html/Zhang_Vision_Transformer_CVPR_2025_paper.html">Vision Transformer for Dense Prediction</a></dt>
    <dd><a href="/CVPR2025/search?q=Zhang">Hao Zhang</a>, <a href="/CVPR2025/search?q=Li">Wei Li</a></dd>
    <dd>[<a href="/content/CVPR2025/papers/Zhang_Vision_Transformer_CVPR_2025_paper.pdf">pdf</a>] [<a href="http://arxiv.org/abs/2503.12345">arXiv</a>]</dd>
    """

    def test_basic(self):
        records = parse_cvpr_openaccess_html(self.FIXTURE, _conf("CVPR"), _src())
        _assert_basic(records)
        assert records[0].title == "Vision Transformer for Dense Prediction"

    def test_authors_no_links(self):
        records = parse_cvpr_openaccess_html(self.FIXTURE, _conf("CVPR"), _src())
        assert "Hao Zhang" in records[0].authors
        assert "Wei Li" in records[0].authors
        assert "pdf" not in records[0].authors.lower()

    def test_arxiv_id_extracted(self):
        records = parse_cvpr_openaccess_html(self.FIXTURE, _conf("CVPR"), _src())
        assert records[0].arxiv_id == "2503.12345"


# ── parse_iclr_virtual_html ─────────────────────────────────────

class TestICLRVirtualHTML:

    FIXTURE = """
    <li><a href="/virtual/2025/poster/27719">Point-SAM: Efficient 3D Segmentation</a></li>
    <li><a href="/virtual/2025/poster/27720">Scaling Laws for Language Models</a></li>
    """

    def test_basic(self):
        records = parse_iclr_virtual_html(self.FIXTURE, _conf("ICLR"), _src())
        _assert_basic(records, 2)
        assert records[0].title == "Point-SAM: Efficient 3D Segmentation"

    def test_paper_link(self):
        records = parse_iclr_virtual_html(self.FIXTURE, _conf("ICLR"), _src())
        assert "iclr.cc/virtual/2025/poster/27719" in records[0].paper_link


# ── parse_iclr_proceedings_html ──────────────────────────────────

class TestICLRProceedingsHTML:

    FIXTURE = """
    <a title="paper title" href="/paper_files/paper/2025/hash/abc123-Paper-Conference.html">
    Transformers Meet Graphs</a>
    <span class="paper-authors">John Doe, Jane Smith</span>
    """

    def test_basic(self):
        records = parse_iclr_proceedings_html(self.FIXTURE, _conf("ICLR"), _src())
        _assert_basic(records)
        assert records[0].title == "Transformers Meet Graphs"
        assert "proceedings.iclr.cc" in records[0].paper_link


# ── parse_aaai_ojs_html ─────────────────────────────────────────

class TestAAAIOJSHTML:

    FIXTURE = """
    <div class="obj_article_summary">
    <h3 class="title">
      <a id="article-28815" href="https://ojs.aaai.org/index.php/AAAI/article/view/28815">Efficient Pruning for LLMs</a>
    </h3>
    <div class="authors">Alice Wang, Bob Johnson</div>
    <a class="obj_galley_link pdf" href="https://ojs.aaai.org/index.php/AAAI/article/view/28815/29555">PDF</a>
    </div>
    <div class="obj_article_summary">
    """

    def test_basic(self):
        records = parse_aaai_ojs_html(self.FIXTURE, _conf("AAAI", 2024), _src())
        _assert_basic(records)
        assert records[0].title == "Efficient Pruning for LLMs"
        assert "Alice Wang" in records[0].authors


# ── parse_acl_anthology_html ────────────────────────────────────

class TestACLAnthologyHTML:

    FIXTURE = """
    <span class="d-block">
    <strong><a href="/2024.acl-long.123/">Multimodal Learning with Transformers</a></strong>
    <a href="/people/a/alice-zhang/">Alice Zhang</a>, <a href="/people/b/bob-li/">Bob Li</a>
    </span>
    <div class="card-body p-3 small">We study multimodal transformers.</div>
    <span class="d-block">
    <strong><a href="/2024.acl-workshop.5/">Workshop Paper Title</a></strong>
    <a href="/people/c/carol/">Carol</a>
    </span>
    """

    def test_basic(self):
        records = parse_acl_anthology_html(self.FIXTURE, _conf("ACL", 2024), _src())
        _assert_basic(records)
        assert records[0].title == "Multimodal Learning with Transformers"

    def test_filters_workshops(self):
        records = parse_acl_anthology_html(self.FIXTURE, _conf("ACL", 2024), _src())
        titles = [r.title for r in records]
        assert "Workshop Paper Title" not in titles

    def test_abstract_extracted(self):
        records = parse_acl_anthology_html(self.FIXTURE, _conf("ACL", 2024), _src())
        assert "multimodal" in records[0].abstract_raw.lower()


# ── parse_kdd_html ───────────────────────────────────────────────

class TestKDDHTML:

    FIXTURE = """
    <tr><td><strong>Graph Mining at Scale</strong><br>DOI: https://doi.org/10.1145/3637528.1234</td></tr>
    <tr><td>David Kim (MIT); Eve Park (Stanford)</td></tr>
    """

    def test_basic(self):
        records = parse_kdd_html(self.FIXTURE, _conf("KDD", 2024), _src())
        _assert_basic(records)
        assert records[0].title == "Graph Mining at Scale"
        assert "David Kim" in records[0].authors


# ── parse_kesen_siggraph_html ────────────────────────────────────

class TestKesenSiggraphHTML:

    FIXTURE = """
    <dt><B>Neural Radiance Caching</B>
    (<B>SIG</B>)
    <a href="https://doi.org/10.1145/12345"><img alt="ACM DOI"></a>
    <a href="https://arxiv.org/abs/2503.99999"><img alt="Author Preprint"></a>
    </dt>
    <dd>Tom Anderson, Lisa Chen* (Equal Contribution), Bob Smith</dd>
    """

    def test_basic(self):
        records = parse_kesen_siggraph_html(self.FIXTURE, _conf("SIGGRAPH"), _src())
        _assert_basic(records)
        assert records[0].title == "Neural Radiance Caching"

    def test_arxiv_id(self):
        records = parse_kesen_siggraph_html(self.FIXTURE, _conf("SIGGRAPH"), _src())
        assert records[0].arxiv_id == "2503.99999"

    def test_authors_cleaned(self):
        records = parse_kesen_siggraph_html(self.FIXTURE, _conf("SIGGRAPH"), _src())
        # Asterisks and "(Equal Contribution)" should be removed
        assert "*" not in records[0].authors


# ── parse_jmlr_html ─────────────────────────────────────────────

class TestJMLRHTML:

    FIXTURE = """
    <dl>
    <dt>Stochastic Optimization Under Constraints</dt>
    <dd><b><i>Frank Lee, Grace Wang</i></b>; 26(1):1-30, 2025.
    <br>[<a href='/papers/v26/lee25a.html'>abs</a>][<a href='/papers/volume26/lee25a/lee25a.pdf'>pdf</a>][<a href='https://github.com/franklee/soc'>code</a>]
    </dl>
    """

    def test_basic(self):
        records = parse_jmlr_html(self.FIXTURE, _conf("JMLR"), _src())
        _assert_basic(records)
        assert records[0].title == "Stochastic Optimization Under Constraints"

    def test_code_url(self):
        records = parse_jmlr_html(self.FIXTURE, _conf("JMLR"), _src())
        assert "github.com" in records[0].code_url

    def test_filters_editorials(self):
        editorial = """
        <dl><dt>Editorial Board</dt><dd><b><i>Editors</i></b></dd></dl>
        """
        records = parse_jmlr_html(editorial, _conf("JMLR"), _src())
        assert records == []


# ── parse_openalex_works ─────────────────────────────────────────

class TestOpenAlexWorks:

    FIXTURE = [
        {
            "title": "Diffusion Models: A Survey",
            "authorships": [
                {"author": {"display_name": "Alice Zhang"}},
                {"author": {"display_name": "Bob Li"}},
            ],
            "doi": "https://doi.org/10.1109/TPAMI.2025.12345",
            "abstract_inverted_index": {"We": [0], "survey": [1], "diffusion": [2], "models": [3]},
            "biblio": {},
        }
    ]

    def test_basic(self):
        records = parse_openalex_works(self.FIXTURE, _conf("TPAMI"), _src())
        _assert_basic(records)
        assert records[0].title == "Diffusion Models: A Survey"

    def test_doi_cleaned(self):
        records = parse_openalex_works(self.FIXTURE, _conf("TPAMI"), _src())
        assert records[0].doi == "10.1109/TPAMI.2025.12345"

    def test_abstract_reconstructed(self):
        records = parse_openalex_works(self.FIXTURE, _conf("TPAMI"), _src())
        assert records[0].abstract_raw == "We survey diffusion models"


# ── parse_s2_bulk_papers ─────────────────────────────────────────

class TestS2BulkPapers:

    FIXTURE = [
        {
            "title": "Self-Supervised Vision Transformers",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "abstract": "We present SSVT.",
            "externalIds": {"ArXiv": "2503.11111", "DOI": "10.1234/ssvt"},
            "citationCount": 150,
            "openAccessPdf": {"url": "https://arxiv.org/pdf/2503.11111.pdf"},
            "url": "https://api.semanticscholar.org/...",
        },
        {
            "title": "No ArXiv Paper",
            "authors": [],
            "externalIds": {"DOI": "10.5555/nope"},
            "citationCount": 5,
        },
    ]

    def test_filters_non_arxiv(self):
        records = parse_s2_bulk_papers(self.FIXTURE, _conf("ArXiv_HiCite"), _src())
        assert len(records) == 1  # only arxiv paper

    def test_fields(self):
        records = parse_s2_bulk_papers(self.FIXTURE, _conf("ArXiv_HiCite"), _src())
        r = records[0]
        assert r.arxiv_id == "2503.11111"
        assert r.doi == "10.1234/ssvt"
        assert r.extras["citation_count"] == "150"


# ── parse_hf_daily_papers ────────────────────────────────────────

class TestHFDailyPapers:

    FIXTURE = [
        {
            "paper": {
                "id": "2503.22222",
                "title": "Fast Inference for LLMs",
                "authors": [{"name": "Dave"}],
                "summary": "We speed up LLM inference.",
            },
            "_upvotes": 42,
        }
    ]

    def test_basic(self):
        records = parse_hf_daily_papers(self.FIXTURE, _conf("HF_DailyPapers"), _src())
        _assert_basic(records)
        assert records[0].arxiv_id == "2503.22222"
        assert records[0].extras["hf_upvotes"] == "42"


# ── parse_anthropic_research ─────────────────────────────────────

class TestAnthropicResearch:

    FIXTURE = [
        {
            "title": "Constitutional AI",
            "arxiv_url": "https://arxiv.org/abs/2212.08073",
            "full_paper_url": "https://arxiv.org/pdf/2212.08073.pdf",
            "page_url": "https://www.anthropic.com/research/constitutional-ai",
            "description": "We explore constitutional AI approaches.",
            "date": "2022-12-15",
        }
    ]

    def test_basic(self):
        records = parse_anthropic_research(self.FIXTURE, _conf("Anthropic_Research", 2024), _src())
        _assert_basic(records)
        assert records[0].arxiv_id == "2212.08073"


# ── parse_acmmm_vue_accepted ────────────────────────────────────

class TestACMMMVueAccepted:

    FIXTURE = """
    some_prefix,contents:[{type:"paperTitle",text:"1234 <b>Video Understanding via Attention</b>"},{type:"paperAuthor",text:"Alice, Bob, Carol"}]some_suffix
    """

    def test_basic(self):
        records = parse_acmmm_vue_accepted(self.FIXTURE, _conf("ACMMM", 2024), _src())
        _assert_basic(records)
        assert records[0].title == "Video Understanding via Attention"


# ── parse_acmmm_html ────────────────────────────────────────────

class TestACMMMHTML:

    FIXTURE = """
    <p>5678\xa0<b>Multi-Modal Retrieval System</b><br/>Tom Lee, Jane Park</p>
    """

    def test_basic(self):
        records = parse_acmmm_html(self.FIXTURE, _conf("ACMMM", 2024), _src())
        _assert_basic(records)
        assert records[0].title == "Multi-Modal Retrieval System"


# ── PARSERS registry completeness ────────────────────────────────

class TestParsersRegistry:

    def test_all_parsers_registered(self):
        expected = {
            "openreview_notes_json", "openreview_api_v2",
            "simple_html_paper_list", "curated_markdown_links",
            "cvpr_openaccess_html", "iclr_virtual_html", "iclr_proceedings_html",
            "virtual_conference_json", "acl_anthology_html", "aaai_ojs_html",
            "kdd_html", "kesen_siggraph_html", "acmmm_html", "acmmm_vue_accepted",
            "openalex_works", "jmlr_html", "s2_bulk_papers", "hf_daily_papers",
            "anthropic_research",
        }
        assert expected.issubset(set(PARSERS.keys())), f"Missing parsers: {expected - set(PARSERS.keys())}"

    def test_parsers_are_callable(self):
        for name, fn in PARSERS.items():
            assert callable(fn), f"Parser {name} is not callable"
