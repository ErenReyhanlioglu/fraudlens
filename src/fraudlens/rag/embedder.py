"""OpenAI embedding wrapper — produces vectors for RAG index and retrieval queries."""

from __future__ import annotations

import asyncio

import structlog
from openai import AsyncOpenAI

from fraudlens.core.config import get_settings

logger = structlog.get_logger(__name__)

_MODEL = "text-embedding-3-small"
_DIMENSIONS = 1536
_BATCH_SIZE = 100


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings with text-embedding-3-small.

    Sends requests in batches of 100 to stay within the OpenAI API limits.

    Args:
        texts: Non-empty list of strings to embed.

    Returns:
        List of float vectors in the same order as the input texts.
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = await client.embeddings.create(model=_MODEL, input=batch)
        all_vectors.extend([item.embedding for item in response.data])
        logger.debug("embedding_batch_done", batch_start=i, batch_size=len(batch))

    logger.info("embedding_complete", total_texts=len(texts))
    return all_vectors


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    """Synchronous wrapper around embed_texts for use in one-shot scripts.

    Args:
        texts: Non-empty list of strings to embed.

    Returns:
        List of float vectors in the same order as the input texts.
    """
    return asyncio.run(embed_texts(texts))


async def embed_query(query: str) -> list[float]:
    """Embed a single query string for similarity search.

    Args:
        query: The search query text.

    Returns:
        A single float vector of length 1536.
    """
    vectors = await embed_texts([query])
    return vectors[0]


EMBEDDING_DIMENSIONS: int = _DIMENSIONS
