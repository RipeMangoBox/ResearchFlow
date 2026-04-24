from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourceConfig:
    kind: str
    url: str
    parser: str
    expected_count: Optional[int] = None
    parser_args: Optional[str] = None


@dataclass
class ConferenceYearConfig:
    venue: str
    year: int
    conf_year: str
    status: str
    skip_reason: str
    primary_source: SourceConfig
    auxiliary_sources: list[SourceConfig] = field(default_factory=list)
    notes: str = ""


@dataclass
class AcceptedPaperRecord:
    paper_id: str = ""
    short_id: str = ""
    venue: str = ""
    year: int = 0
    conf_year: str = ""
    title: str = ""
    authors: str = ""
    source_type: str = ""
    source_url: str = ""
    paper_link: str = ""
    landing_url: str = ""
    pdf_url: str = ""
    pdf_status: str = ""
    pdf_source: str = ""
    source_text_status: str = ""
    source_text_url: str = ""
    source_text_source: str = ""
    source_text_evidence: str = ""
    source_text_search_query: str = ""
    source_text_checked_at: str = ""
    arxiv_id: str = ""
    arxiv_url: str = ""
    keywords_raw: str = ""
    abstract_raw: str = ""
    doi: str = ""
    openreview_forum_id: str = ""
    has_pdf_camera_ready: str = ""
    decision: str = ""
    acceptance_type: str = ""
    topic: str = ""
    code_url: str = ""
    review_score_status: str = ""
    paper_url: str = ""
    virtual_id: str = ""
    virtual_uid: str = ""
    virtualsite_url: str = ""
    sourceid: str = ""
    sourceurl: str = ""
    session: str = ""
    eventtype: str = ""
    event_type: str = ""
    room_name: str = ""
    starttime: str = ""
    endtime: str = ""
    poster_position: str = ""
    # Extras hold any CSV columns not defined above (e.g. review_*, code_is_real,
    # code_stars, has_pretrained_weights, has_dataset, and future enrich fields).
    # They are preserved across build/merge cycles so downstream enrich output
    # is never lost by a rebuild.
    extras: dict = field(default_factory=dict)


@dataclass
class SourceRunResult:
    source: SourceConfig
    records: list[AcceptedPaperRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
