"""
PDF parser for corpus ingestion.

Uses pypdf (primary) with pdfplumber as fallback for tables/complex layouts.
Returns clean text per page, suitable for chunking.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF file, one dict per page.

    Returns:
      [{"page": int, "text": str, "source": str}, ...]

    Tries pypdf first; falls back to pdfplumber for pages where pypdf
    extracts less than 50 characters (likely a scanned/complex page).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    source_tag = f"pdf:{path.name}"
    results: list[dict] = []

    try:
        import pypdf

        reader = pypdf.PdfReader(str(path))
        pypdf_pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pypdf_pages.append((i + 1, text.strip()))
    except Exception as exc:
        logger.warning("pypdf failed for %s: %s — falling back to pdfplumber.", path.name, exc)
        pypdf_pages = []

    # Identify pages that need pdfplumber fallback
    needs_fallback = {
        page_num for page_num, text in pypdf_pages if len(text) < 50
    }

    plumber_pages: dict[int, str] = {}
    if needs_fallback:
        try:
            import pdfplumber

            with pdfplumber.open(str(path)) as pdf:
                for page_num in needs_fallback:
                    idx = page_num - 1
                    if idx < len(pdf.pages):
                        text = pdf.pages[idx].extract_text() or ""
                        plumber_pages[page_num] = text.strip()
        except Exception as exc:
            logger.warning("pdfplumber fallback failed for %s: %s", path.name, exc)

    # Merge
    if pypdf_pages:
        for page_num, text in pypdf_pages:
            final_text = plumber_pages.get(page_num, text)
            if len(final_text) > 20:
                results.append({"page": page_num, "text": final_text, "source": source_tag})
    elif plumber_pages:
        for page_num, text in sorted(plumber_pages.items()):
            if len(text) > 20:
                results.append({"page": page_num, "text": text, "source": source_tag})
    else:
        # Last resort: try pdfplumber for everything
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = (page.extract_text() or "").strip()
                    if len(text) > 20:
                        results.append({"page": i + 1, "text": text, "source": source_tag})
        except Exception as exc:
            logger.error("All PDF parsing strategies failed for %s: %s", path.name, exc)

    logger.info("Parsed PDF '%s': %d pages with content.", path.name, len(results))
    return results


def parse_pdf_directory(directory: str | Path) -> list[dict]:
    """Parse all PDFs in a directory.  Returns combined list of page dicts."""
    directory = Path(directory)
    all_pages: list[dict] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        try:
            pages = parse_pdf(pdf_path)
            all_pages.extend(pages)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", pdf_path, exc)
    return all_pages
