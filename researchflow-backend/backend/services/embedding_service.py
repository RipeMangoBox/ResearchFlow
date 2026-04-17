"""Embedding service — generate vector embeddings for papers.

Uses OpenAI text-embedding-3-small when API key is available,
falls back to deterministic mock embeddings for testing.
"""

import hashlib
import logging
import struct
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.models.paper import Paper

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536
EMBEDDING_MODEL = "text-embedding-3-small"


async def embed_text(content: str) -> list[float]:
    """Generate embedding for a text string."""
    if settings.openai_api_key:
        return await _embed_openai(content)
    return _mock_embedding(content)


async def _embed_openai(content: str) -> list[float]:
    import openai
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=content[:8000],  # API limit
    )
    return resp.data[0].embedding


def _mock_embedding(content: str) -> list[float]:
    """Deterministic mock embedding from content hash.

    Same input always produces same vector, so dedup and similarity
    tests work correctly in mock mode.
    """
    h = hashlib.sha256(content.encode()).digest()
    # Expand hash to fill 1536 dims deterministically
    vectors = []
    for i in range(EMBEDDING_DIM // 8):
        seed = hashlib.sha256(h + struct.pack("i", i)).digest()
        for j in range(0, 32, 4):
            val = struct.unpack("f", seed[j:j+4])[0]
            # Normalize to [-1, 1]
            val = max(-1.0, min(1.0, val / 1e30))
            vectors.append(val)
    return vectors[:EMBEDDING_DIM]


async def embed_paper(session: AsyncSession, paper_id: UUID) -> bool:
    """Generate and store embedding for a single paper."""
    paper = await session.get(Paper, paper_id)
    if not paper:
        return False

    content = _build_paper_text(paper)
    embedding = await embed_text(content)

    # Update using raw SQL since pgvector column needs special handling
    await session.execute(
        text("UPDATE papers SET embedding = :emb WHERE id = :id"),
        {"emb": str(embedding), "id": paper_id},
    )
    await session.flush()
    return True


async def embed_batch(session: AsyncSession, limit: int = 50) -> int:
    """Generate embeddings for papers that don't have one yet."""
    result = await session.execute(
        text("""
            SELECT id, title, abstract, core_operator, category, venue
            FROM papers
            WHERE embedding IS NULL
            ORDER BY analysis_priority DESC NULLS LAST
            LIMIT :limit
        """),
        {"limit": limit},
    )
    rows = result.fetchall()

    count = 0
    for row in rows:
        pid, title, abstract, core_op, category, venue = row
        content = f"{title}. {abstract or ''} {core_op or ''} {category} {venue or ''}"
        embedding = await embed_text(content.strip())

        await session.execute(
            text("UPDATE papers SET embedding = :emb WHERE id = :id"),
            {"emb": str(embedding), "id": pid},
        )
        count += 1

    await session.flush()
    return count


def _build_paper_text(paper: Paper) -> str:
    """Build text representation for embedding."""
    parts = [paper.title]
    if paper.abstract:
        parts.append(paper.abstract)
    if paper.core_operator:
        parts.append(paper.core_operator)
    if paper.category:
        parts.append(paper.category)
    if paper.venue:
        parts.append(paper.venue)
    if paper.tags:
        parts.append(" ".join(paper.tags[:10]))
    return ". ".join(parts)
