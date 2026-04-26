"""PDF loader — reads each page and returns structured page records."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import structlog
from pypdf import PdfReader

logger = structlog.get_logger(__name__)


class PageRecord(TypedDict):
    """Single page extracted from a PDF document."""

    source: str
    page: int
    text: str


def load_pdf(path: str | Path) -> list[PageRecord]:
    """Load a PDF file and return one record per page.

    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        List of PageRecord dicts sorted by page number. Pages with no
        extractable text are silently skipped.
    """
    path = Path(path)
    reader = PdfReader(str(path))
    records: list[PageRecord] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        records.append({"source": path.name, "page": page_num, "text": text})

    logger.info("pdf_loaded", source=path.name, pages_extracted=len(records))
    return records


def load_pdfs(paths: list[str | Path]) -> list[PageRecord]:
    """Load multiple PDF files and concatenate their page records.

    Args:
        paths: List of PDF file paths.

    Returns:
        Combined list of PageRecord dicts across all files.
    """
    all_records: list[PageRecord] = []
    for p in paths:
        all_records.extend(load_pdf(p))
    return all_records
