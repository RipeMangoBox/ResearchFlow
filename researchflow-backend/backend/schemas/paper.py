"""Pydantic schemas for Paper CRUD and filtering."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from backend.models.enums import (
    AnalysisLevel,
    EvidenceBasis,
    Importance,
    PaperState,
    Tier,
)


# ── Base / shared fields ───────────────────────────────────────

class PaperBase(BaseModel):
    title: str
    venue: str | None = None
    year: int | None = None
    category: str
    paper_link: str | None = None
    project_link: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    abstract: str | None = None
    tags: list[str] = Field(default_factory=list)
    mechanism_family: str | None = None
    core_operator: str | None = None
    primary_logic: str | None = None


# ── Create / Update ────────────────────────────────────────────

class PaperCreate(PaperBase):
    importance: Importance | None = Importance.C
    tier: Tier | None = None
    is_ephemeral: bool = False
    retention_days: int | None = 30


class PaperUpdate(BaseModel):
    title: str | None = None
    venue: str | None = None
    year: int | None = None
    category: str | None = None
    state: PaperState | None = None
    importance: Importance | None = None
    tier: Tier | None = None
    paper_link: str | None = None
    project_link: str | None = None
    abstract: str | None = None
    tags: list[str] | None = None
    mechanism_family: str | None = None
    core_operator: str | None = None
    primary_logic: str | None = None
    is_ephemeral: bool | None = None


# ── Response ───────────────────────────────────────────────────

class PaperResponse(PaperBase):
    id: UUID
    title_sanitized: str
    state: PaperState
    importance: Importance | None = None
    tier: Tier | None = None

    # Scores
    keep_score: float | None = None
    analysis_priority: float | None = None
    structurality_score: float | None = None
    extensionability_score: float | None = None

    # Metadata
    authors: dict | None = None
    keywords: list[str] | None = None
    claims: list[str] | None = None
    open_code: bool = False
    open_data: bool = False
    code_url: str | None = None

    # Storage
    pdf_path_local: str | None = None
    pdf_object_key: str | None = None

    # Ephemeral
    is_ephemeral: bool = False
    expires_at: datetime | None = None

    # Source
    source: str | None = None

    # Timestamps
    collected_at: datetime | None = None
    analyzed_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class PaperBrief(BaseModel):
    """Lightweight paper card for list views."""
    id: UUID
    title: str
    venue: str | None = None
    year: int | None = None
    category: str
    state: PaperState
    importance: Importance | None = None
    tier: Tier | None = None
    tags: list[str] = Field(default_factory=list)
    core_operator: str | None = None
    keep_score: float | None = None
    structurality_score: float | None = None
    is_ephemeral: bool = False

    model_config = {"from_attributes": True}


# ── Analysis response ──────────────────────────────────────────

class AnalysisBrief(BaseModel):
    id: UUID
    level: AnalysisLevel
    model_provider: str | None = None
    model_name: str | None = None
    confidence: float | None = None
    problem_summary: str | None = None
    method_summary: str | None = None
    core_intuition: str | None = None
    changed_slots: list[str] | None = None
    is_plugin_patch: bool | None = None
    worth_deep_read: bool | None = None
    generated_at: datetime | None = None

    model_config = {"from_attributes": True}


class PaperDetail(PaperResponse):
    """Full paper with latest analysis."""
    latest_analysis: AnalysisBrief | None = None


# ── Filter / query ─────────────────────────────────────────────

class PaperFilter(BaseModel):
    """Query parameters for paper listing."""
    q: str | None = None  # full-text search
    state: PaperState | None = None
    category: str | None = None
    venue: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    importance: Importance | None = None
    tier: Tier | None = None
    tags: list[str] | None = None
    is_ephemeral: bool | None = None
    min_keep_score: float | None = None
    min_structurality: float | None = None
    min_extensionability: float | None = None

    # Pagination
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)

    # Sort
    sort_by: str = "updated_at"  # updated_at, keep_score, analysis_priority, year, title
    sort_order: str = "desc"     # asc, desc


# ── Paginated response ─────────────────────────────────────────

class PaperListResponse(BaseModel):
    items: list[PaperBrief]
    total: int
    page: int
    size: int
    pages: int
