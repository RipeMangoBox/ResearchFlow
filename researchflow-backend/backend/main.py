import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.analyses import router as analyses_router
from backend.api.import_ import router as import_router
from backend.api.papers import router as papers_router
from backend.api.digests import router as digests_router
from backend.api.directions import router as directions_router
from backend.api.feedback import router as feedback_router
from backend.api.assertions import router as assertions_router
from backend.api.exploration import router as exploration_router
from backend.api.graph import router as graph_router
from backend.api.pipeline import router as pipeline_router
from backend.api.reports import router as reports_router
from backend.api.reviews import router as reviews_router
from backend.api.bottlenecks import router as bottlenecks_router
from backend.api.search import router as search_router
from backend.api.taxonomy import router as taxonomy_router
from backend.api.methods import router as methods_router
from backend.api.candidates import router as candidates_router
from backend.config import settings

logger = logging.getLogger(__name__)


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
app.include_router(graph_router, prefix=settings.api_prefix)
app.include_router(assertions_router, prefix=settings.api_prefix)
app.include_router(pipeline_router, prefix=settings.api_prefix)
app.include_router(exploration_router, prefix=settings.api_prefix)
app.include_router(reviews_router, prefix=settings.api_prefix)
app.include_router(bottlenecks_router, prefix=settings.api_prefix)
app.include_router(taxonomy_router, prefix=settings.api_prefix)
app.include_router(methods_router, prefix=settings.api_prefix)
app.include_router(candidates_router, prefix=settings.api_prefix)


# Global exception handler
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception on %s %s:\n%s",
        request.method,
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get(f"{settings.api_prefix}/health")
async def health():
    return {"status": "ok"}
