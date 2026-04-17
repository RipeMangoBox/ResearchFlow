"""Import API router — ingest papers from various sources."""

from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.models.enums import AssetType, PaperState
from backend.models.paper import Paper, PaperAsset
from backend.schemas.import_ import (
    ImportResponse,
    LinkImportRequest,
)
from backend.schemas.paper import PaperResponse
from backend.services import ingestion_service, parse_service
from backend.services.object_storage import compute_checksum, get_storage
from backend.utils.sanitize import sanitize_filename

router = APIRouter(prefix="/import", tags=["import"])


@router.post("/links", response_model=ImportResponse)
async def import_links(
    data: LinkImportRequest,
    session: AsyncSession = Depends(get_session),
):
    """Import paper links in batch.

    Each link is normalized, deduped, and ingested. Duplicates are
    reported but not created again.
    """
    try:
        results = await ingestion_service.ingest_links(
            session,
            data.items,
            default_category=data.default_category,
            is_ephemeral=data.is_ephemeral,
            retention_days=data.retention_days,
        )
        await session.commit()

        created = sum(1 for r in results if r.status == "created")
        duplicates = sum(1 for r in results if r.status == "duplicate")
        errors = sum(1 for r in results if r.status == "error")

        return ImportResponse(
            total=len(results),
            created=created,
            duplicates=duplicates,
            errors=errors,
            items=results,
        )
    except Exception:
        await session.rollback()
        raise


@router.post("/{paper_id}/accept", response_model=PaperResponse)
async def accept_to_kb(
    paper_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Promote an ephemeral paper to the main knowledge base."""
    try:
        paper = await ingestion_service.accept_to_kb(session, paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        await session.commit()
        return PaperResponse.model_validate(paper)
    except HTTPException:
        raise
    except Exception:
        await session.rollback()
        raise


@router.post("/pdf", response_model=PaperResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    title: str | None = Query(default=None),
    category: str = Query(default="Uncategorized"),
    is_ephemeral: bool = Query(default=False),
    session: AsyncSession = Depends(get_session),
):
    """Upload a PDF file, store it, and create a paper record.

    Optionally triggers L2 parsing immediately.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Read file
    data = await file.read()
    if len(data) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="PDF exceeds 50MB limit")

    # Determine title
    paper_title = title or file.filename.replace(".pdf", "").replace("_", " ")
    title_san = sanitize_filename(paper_title)

    try:
        # Store in object storage
        storage = get_storage()
        object_key = f"papers/raw-pdf/{category}/{title_san}.pdf"
        await storage.put(object_key, data)
        checksum = compute_checksum(data)

        # Create paper record
        paper = Paper(
            title=paper_title,
            title_sanitized=title_san,
            category=category,
            state=PaperState.DOWNLOADED,
            pdf_object_key=object_key,
            is_ephemeral=is_ephemeral,
            tags=[category],
            source="pdf_upload",
        )
        session.add(paper)
        await session.flush()

        # Create asset record
        asset = PaperAsset(
            paper_id=paper.id,
            asset_type=AssetType.RAW_PDF,
            object_key=object_key,
            mime_type="application/pdf",
            size_bytes=len(data),
            checksum=checksum,
        )
        session.add(asset)
        await session.flush()
        await session.refresh(paper)
        await session.commit()

        return PaperResponse.model_validate(paper)
    except Exception:
        await session.rollback()
        raise


@router.post("/parse", summary="L2 parse PDFs")
async def parse_pdfs(
    limit: int = Query(default=5, ge=1, le=20),
    session: AsyncSession = Depends(get_session),
):
    """Run L2 parse (pymupdf section extraction) on unprocessed PDFs."""
    try:
        results = await parse_service.parse_all_unprocessed(session, limit=limit)
        await session.commit()
        return {"processed": len(results), "results": results}
    except Exception:
        await session.rollback()
        raise


@router.post("/{paper_id}/parse")
async def parse_single_pdf(
    paper_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Run L2 parse on a single paper's PDF."""
    try:
        analysis = await parse_service.parse_paper_pdf(session, paper_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Paper or PDF not found")
        await session.commit()

        sections = list(analysis.extracted_sections.keys()) if analysis.extracted_sections else []
        return {
            "paper_id": str(paper_id),
            "analysis_id": str(analysis.id),
            "level": analysis.level.value,
            "sections": sections,
            "formulas": len(analysis.extracted_formulas or []),
            "figures": len(analysis.figure_captions or []),
            "tables": len(analysis.extracted_tables or []),
        }
    except HTTPException:
        raise
    except Exception:
        await session.rollback()
        raise


@router.post("/cleanup-expired")
async def cleanup_expired(
    session: AsyncSession = Depends(get_session),
):
    """Archive expired ephemeral papers."""
    try:
        count = await ingestion_service.cleanup_expired(session)
        await session.commit()
        return {"archived": count}
    except Exception:
        await session.rollback()
        raise
