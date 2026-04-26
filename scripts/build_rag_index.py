"""One-shot script to build the RAG index from regulation PDFs.

Usage:
    uv run python scripts/build_rag_index.py [--rebuild]

Loads PDFs from data/docs/, chunks them, embeds with text-embedding-3-small,
and upserts into the Qdrant fraudlens_regulations collection.

Pass --rebuild to drop and recreate the collection before indexing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

# Ensure src/ is on the path when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudlens.rag.chunker import chunk_pages
from fraudlens.rag.embedder import embed_texts_sync
from fraudlens.rag.loader import load_pdfs
from fraudlens.rag.store import drop_collection, ensure_collection, upsert_chunks

logger = structlog.get_logger(__name__)

_DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"
_PDF_NAMES = [
    "fatf_40_recommendations_2012.pdf",
    "masak_finansal_kuruluslar_aml_riskleri_5549.pdf",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG index from regulation PDFs.")
    parser.add_argument("--rebuild", action="store_true", help="Drop existing collection before indexing.")
    args = parser.parse_args()

    pdf_paths = [_DOCS_DIR / name for name in _PDF_NAMES]
    missing = [p for p in pdf_paths if not p.exists()]
    if missing:
        logger.error("missing_pdfs", paths=[str(p) for p in missing])
        sys.exit(1)

    if args.rebuild:
        logger.info("dropping_collection")
        drop_collection()

    logger.info("loading_pdfs", count=len(pdf_paths))
    pages = load_pdfs(pdf_paths)
    logger.info("pages_loaded", count=len(pages))

    logger.info("chunking")
    chunks = chunk_pages(pages)
    logger.info("chunks_created", count=len(chunks))

    logger.info("embedding", model="text-embedding-3-small")
    texts = [c["text"] for c in chunks]
    vectors = embed_texts_sync(texts)
    logger.info("embeddings_created", count=len(vectors))

    logger.info("upserting_to_qdrant")
    ensure_collection()
    upsert_chunks(chunks, vectors)

    logger.info("rag_index_build_complete", chunks=len(chunks))


if __name__ == "__main__":
    main()
