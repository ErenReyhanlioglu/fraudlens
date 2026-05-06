"""PDF loader — reads each page and returns structured page records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

import structlog
from pypdf import PdfReader

logger = structlog.get_logger(__name__)

_HEADER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^FATF\s+40\s+Recommendations.*$", re.IGNORECASE),
    re.compile(r"^FINANCIAL ACTION TASK FORCE.*$", re.IGNORECASE),
    re.compile(r"^\d+\s*[-–]\s*©\s*\d{4}\s+FATF.*$", re.IGNORECASE),
    re.compile(r"^-\s*\d+\s*-$"),
    re.compile(r"^\d{1,3}$"),  # standalone page numbers
]


def _clean_page_text(text: str) -> str:
    """Strip repetitive PDF header/footer lines that pollute RAG chunks."""
    lines = text.splitlines()
    cleaned = [l for l in lines if not any(p.match(l.strip()) for p in _HEADER_PATTERNS)]
    return "\n".join(cleaned).strip()


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
        text = _clean_page_text(text)
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
