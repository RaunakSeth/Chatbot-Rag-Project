"""
Supabase client — wraps the Supabase Python SDK for:
  - Storing / fetching client configs
  - pgvector similarity search (replaces LanceDB)
  - File storage for uploaded PDFs
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ── Lazy singleton ────────────────────────────────────────────────────────────

_client = None


def get_client():
    """Return a cached Supabase client instance."""
    global _client
    if _client is None:
        try:
            from supabase import create_client
        except ImportError:
            raise RuntimeError(
                "supabase package not installed. Run: pip install supabase"
            )
        url  = os.getenv("SUPABASE_URL", "")
        key  = os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set. "
                "Get them from your Supabase project dashboard."
            )
        _client = create_client(url, key)
    return _client


# ── Vector search ─────────────────────────────────────────────────────────────

def similarity_search(
    client_id: str,
    query_embedding: list[float],
    top_k: int = 5,
    score_threshold: float = 0.35,
) -> list[dict[str, Any]]:
    """
    Run pgvector cosine similarity search for the given client's chunks.

    Parameters
    ----------
    client_id        : The client partition to search within.
    query_embedding  : The embedding vector of the user query.
    top_k            : Number of top chunks to return.
    score_threshold  : Minimum similarity score (0–1).

    Returns
    -------
    List of chunk dicts with keys: chunk_id, text, source, score.
    """
    sb = get_client()
    try:
        result = sb.rpc(
            "match_chunks",
            {
                "p_client_id": client_id,
                "query_embedding": query_embedding,
                "match_count": top_k,
                "match_threshold": score_threshold,
            },
        ).execute()
        return result.data or []
    except Exception as exc:
        logger.error("Supabase similarity search failed: %s", exc)
        return []


# ── Chunk upsert (used during onboarding) ────────────────────────────────────

def upsert_chunks(client_id: str, chunks: list[dict[str, Any]]) -> int:
    """
    Upsert document chunks + embeddings into the `chunks` table.

    Each chunk dict must have:
      chunk_id  : str  (unique ID)
      text      : str
      source    : str
      embedding : list[float]

    Returns the number of rows upserted.
    """
    sb = get_client()
    rows = [
        {
            "client_id":  client_id,
            "chunk_id":   c["chunk_id"],
            "text":       c["text"],
            "source":     c.get("source", ""),
            "embedding":  c["embedding"],
        }
        for c in chunks
    ]
    try:
        sb.table("chunks").upsert(rows, on_conflict="chunk_id").execute()
        return len(rows)
    except Exception as exc:
        logger.error("Supabase upsert_chunks failed: %s", exc)
        raise


# ── Delete client chunks ──────────────────────────────────────────────────────

def delete_client_chunks(client_id: str) -> None:
    """Remove all chunks belonging to a client (for re-indexing or deletion)."""
    sb = get_client()
    sb.table("chunks").delete().eq("client_id", client_id).execute()


# ── Client config CRUD ────────────────────────────────────────────────────────

def save_client_config(config_dict: dict[str, Any]) -> None:
    """Upsert a client config row into the `clients` table."""
    sb = get_client()
    sb.table("clients").upsert(config_dict, on_conflict="client_id").execute()


def load_client_config(client_id: str) -> dict[str, Any] | None:
    """Fetch a single client config by ID. Returns None if not found."""
    sb = get_client()
    result = sb.table("clients").select("*").eq("client_id", client_id).maybe_single().execute()
    return result.data


def list_client_configs() -> list[dict[str, Any]]:
    """Return all client configs."""
    sb = get_client()
    result = sb.table("clients").select("*").order("client_id").execute()
    return result.data or []


def count_client_chunks(client_id: str) -> int:
    """Return the number of indexed chunks for a client."""
    sb = get_client()
    result = (
        sb.table("chunks")
        .select("chunk_id", count="exact")
        .eq("client_id", client_id)
        .execute()
    )
    return result.count or 0
