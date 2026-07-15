"""
Text chunking for corpus ingestion.

Uses LangChain's RecursiveCharacterTextSplitter (which doesn't require
an LLM — it's pure text splitting) to produce overlapping chunks suitable
for BGE-M3's 512-token sweet spot.
"""

from __future__ import annotations

import hashlib
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """
    Split `text` into overlapping chunks and return a list of dicts:
      [{"chunk_id": str, "text": str, "source": str}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)

    results = []
    for i, chunk_text_item in enumerate(raw_chunks):
        chunk_id = _make_id(source, i, chunk_text_item)
        results.append({
            "chunk_id": chunk_id,
            "text": chunk_text_item.strip(),
            "source": source,
        })

    logger.debug("Chunked '%s' → %d chunks.", source, len(results))
    return [c for c in results if len(c["text"]) > 20]  # drop near-empty chunks


def _make_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{source}|{index}|{text[:64]}".encode()).hexdigest()[:12]
    safe_source = source.replace("/", "_").replace("\\", "_")[:30]
    return f"{safe_source}_{index}_{digest}"
