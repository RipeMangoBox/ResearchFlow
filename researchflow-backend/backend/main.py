from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.analyses import router as analyses_router
from backend.api.import_ import router as import_router
from backend.api.papers import router as papers_router
from backend.api.reports import router as reports_router
from backend.api.search import router as search_router
from backend.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="ResearchFlow",
    description="Research operating system backend",
    version="0.1.0",
    lifespan=lifespan,
)

# Routers
app.include_router(papers_router, prefix=settings.api_prefix)
app.include_router(import_router, prefix=settings.api_prefix)
app.include_router(analyses_router, prefix=settings.api_prefix)
app.include_router(reports_router, prefix=settings.api_prefix)
app.include_router(search_router, prefix=settings.api_prefix)


@app.get(f"{settings.api_prefix}/health")
async def health():
    return {"status": "ok"}
