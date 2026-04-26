"""Text chunker — splits page records into token-bounded chunks with metadata."""

from __future__ import annotations

from typing import TypedDict

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fraudlens.rag.loader import PageRecord

logger = structlog.get_logger(__name__)

_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 128


class Chunk(TypedDict):
    """A single text chunk with provenance metadata."""

    source: str
    page: int
    chunk_index: int
    text: str


def chunk_pages(pages: list[PageRecord]) -> list[Chunk]:
    """Split a list of page records into overlapping token-bounded chunks.

    Uses tiktoken cl100k_base tokenizer so chunk_size is measured in tokens,
    not characters. Metadata (source file name + page number) is preserved on
    every chunk for citation.

    Args:
        pages: Output of loader.load_pdfs / loader.load_pdf.

    Returns:
        Flat list of Chunk dicts ordered by source → page → chunk_index.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )

    chunks: list[Chunk] = []
    for page in pages:
        texts = splitter.split_text(page["text"])
        for idx, text in enumerate(texts):
            chunks.append(
                {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": idx,
                    "text": text,
                }
            )

    logger.info("chunking_complete", input_pages=len(pages), output_chunks=len(chunks))
    return chunks
