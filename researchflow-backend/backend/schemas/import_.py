"""Pydantic schemas for import endpoints."""

from uuid import UUID

from pydantic import BaseModel, Field


class LinkImportItem(BaseModel):
    """A single paper link to import."""
    url: str
    title: str | None = None
    venue: str | None = None
    year: int | None = None
    category: str | None = None
    project_link: str | None = None
    tags: list[str] = Field(default_factory=list)


class LinkImportRequest(BaseModel):
    """Import a batch of paper links."""
    items: list[LinkImportItem] = Field(..., min_length=1, max_length=200)
    default_category: str = "Uncategorized"
    is_ephemeral: bool = False
    retention_days: int = 30


class ImportResultItem(BaseModel):
    paper_id: UUID
    title: str
    status: str  # "created", "duplicate", "error"
    message: str | None = None


class ImportResponse(BaseModel):
    total: int
    created: int
    duplicates: int
    errors: int
    items: list[ImportResultItem]
