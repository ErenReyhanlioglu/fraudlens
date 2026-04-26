"""Qdrant vector store — upserts and manages the fraudlens_regulations collection."""

from __future__ import annotations

import uuid

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from fraudlens.core.config import get_settings
from fraudlens.rag.chunker import Chunk
from fraudlens.rag.embedder import EMBEDDING_DIMENSIONS

logger = structlog.get_logger(__name__)

COLLECTION_NAME = "fraudlens_regulations"
_BATCH_SIZE = 50


def _get_client() -> QdrantClient:
    settings = get_settings()
    api_key = settings.qdrant_api_key.get_secret_value() or None
    return QdrantClient(url=settings.qdrant_url, api_key=api_key)


def ensure_collection(client: QdrantClient | None = None) -> None:
    """Create the Qdrant collection if it does not already exist.

    Args:
        client: Optional pre-built QdrantClient; a new one is created if omitted.
    """
    client = client or _get_client()
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSIONS, distance=Distance.COSINE),
        )
        logger.info("collection_created", collection=COLLECTION_NAME)
    else:
        logger.info("collection_exists", collection=COLLECTION_NAME)


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]], client: QdrantClient | None = None) -> None:
    """Upsert chunk vectors into the Qdrant collection.

    Args:
        chunks: Chunk metadata list (same order as vectors).
        vectors: Pre-computed embedding vectors.
        client: Optional pre-built QdrantClient.
    """
    client = client or _get_client()
    ensure_collection(client)

    points: list[PointStruct] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                },
            )
        )

    for i in range(0, len(points), _BATCH_SIZE):
        batch = points[i : i + _BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.debug("upsert_batch_done", batch_start=i, batch_size=len(batch))

    logger.info("upsert_complete", total_points=len(points), collection=COLLECTION_NAME)


def drop_collection(client: QdrantClient | None = None) -> None:
    """Delete the collection entirely (used for re-indexing).

    Args:
        client: Optional pre-built QdrantClient.
    """
    client = client or _get_client()
    client.delete_collection(COLLECTION_NAME)
    logger.info("collection_dropped", collection=COLLECTION_NAME)
