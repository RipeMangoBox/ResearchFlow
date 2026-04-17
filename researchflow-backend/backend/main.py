from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.analyses import router as analyses_router
from backend.api.import_ import router as import_router
from backend.api.papers import router as papers_router
from backend.api.digests import router as digests_router
from backend.api.directions import router as directions_router
from backend.api.feedback import router as feedback_router
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

# CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(papers_router, prefix=settings.api_prefix)
app.include_router(import_router, prefix=settings.api_prefix)
app.include_router(analyses_router, prefix=settings.api_prefix)
app.include_router(reports_router, prefix=settings.api_prefix)
app.include_router(search_router, prefix=settings.api_prefix)
app.include_router(digests_router, prefix=settings.api_prefix)
app.include_router(directions_router, prefix=settings.api_prefix)
app.include_router(feedback_router, prefix=settings.api_prefix)


@app.get(f"{settings.api_prefix}/health")
async def health():
    return {"status": "ok"}
