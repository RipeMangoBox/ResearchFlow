from fastapi import FastAPI

from backend.config import settings

app = FastAPI(
    title="ResearchFlow",
    description="Research operating system backend",
    version="0.1.0",
)


@app.get(f"{settings.api_prefix}/health")
async def health():
    return {"status": "ok"}
