"""Shared enum definitions for database models."""

import enum


class PaperState(str, enum.Enum):
    # ── Ephemeral / ingestion states (library-external input) ──
    EPHEMERAL_RECEIVED = "ephemeral_received"  # just arrived, temporary object
    CANONICALIZED = "canonicalized"            # identity resolved, deduped
    ENRICHED = "enriched"                      # related assets auto-completed

    # ── Main pipeline states (library-internal) ──
    WAIT = "wait"
    DOWNLOADED = "downloaded"
    L1_METADATA = "l1_metadata"
    L2_PARSED = "l2_parsed"
    L3_SKIMMED = "l3_skimmed"
    L4_DEEP = "l4_deep"
    CHECKED = "checked"

    # ── Out-of-band states ──
    SKIP = "skip"
    MISSING = "missing"
    TOO_LARGE = "too_large"
    ANALYSIS_MISMATCH = "analysis_mismatch"
    ARCHIVED_OR_EXPIRED = "archived_or_expired"  # ephemeral object expired or archived


class Importance(str, enum.Enum):
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class AnalysisLevel(str, enum.Enum):
    L1_METADATA = "l1_metadata"
    L2_PARSE = "l2_parse"
    L3_SKIM = "l3_skim"
    L4_DEEP = "l4_deep"


class AssetType(str, enum.Enum):
    RAW_PDF = "raw_pdf"
    RAW_HTML = "raw_html"
    EXTRACTED_TEXT = "extracted_text"
    FIGURE = "figure"
    CODE_SNAPSHOT = "code_snapshot"
    SKIM_REPORT = "skim_report"
    DEEP_REPORT = "deep_report"
    EXPORTED_MD = "exported_md"


class PeriodType(str, enum.Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FeedbackType(str, enum.Enum):
    CORRECTION = "correction"
    CONFIRMATION = "confirmation"
    REJECTION = "rejection"
    TAG_EDIT = "tag_edit"


class Tier(str, enum.Enum):
    A_OPEN_DATA = "A_open_data"
    B_OPEN_CODE = "B_open_code"
    C_ACCEPTED_NO_CODE = "C_accepted_no_code"
    D_PREPRINT = "D_preprint"


class EvidenceBasis(str, enum.Enum):
    """How well a claim is supported by source material."""
    CODE_VERIFIED = "code_verified"        # confirmed in source code
    EXPERIMENT_BACKED = "experiment_backed" # supported by ablation/table/figure
    TEXT_STATED = "text_stated"             # author explicitly states it
    INFERRED = "inferred"                   # logically inferred by system
    SPECULATIVE = "speculative"             # system's best guess, no direct evidence
