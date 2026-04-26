"""Qdrant retriever — dense similarity search with citation metadata."""

from __future__ import annotations

from typing import TypedDict

import structlog
from qdrant_client import AsyncQdrantClient

from fraudlens.core.config import get_settings
from fraudlens.rag.embedder import embed_query
from fraudlens.rag.store import COLLECTION_NAME

logger = structlog.get_logger(__name__)

_TOP_K = 5


class RetrievedChunk(TypedDict):
    """A chunk returned from Qdrant with its similarity score and citation info."""

    text: str
    source: str
    page: int
    score: float


async def retrieve(query: str, top_k: int = _TOP_K) -> list[RetrievedChunk]:
    """Retrieve the most relevant regulation chunks for a query.

    Embeds the query with text-embedding-3-small, searches the Qdrant
    fraudlens_regulations collection using cosine similarity, and returns
    up to top_k results with source citation metadata.

    Args:
        query: Free-text question or regulatory topic to look up.
        top_k: Maximum number of chunks to return (default 5).

    Returns:
        List of RetrievedChunk dicts sorted by score descending.
        Returns an empty list if the collection is empty or unavailable.
    """
    settings = get_settings()
    api_key = settings.qdrant_api_key.get_secret_value() or None
    client = AsyncQdrantClient(url=settings.qdrant_url, api_key=api_key)

    try:
        query_vector = await embed_query(query)
        response = await client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        results = response.points
    except Exception:
        logger.exception("qdrant_retrieve_failed", query=query[:80])
        return []
    finally:
        await client.close()

    chunks: list[RetrievedChunk] = []
    for hit in results:
        payload = hit.payload or {}
        chunks.append(
            {
                "text": str(payload.get("text", "")),
                "source": str(payload.get("source", "unknown")),
                "page": int(payload.get("page", 0)),
                "score": float(hit.score),
            }
        )

    logger.info("retrieve_complete", query=query[:80], results=len(chunks))
    return chunks
