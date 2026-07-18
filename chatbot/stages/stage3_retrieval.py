"""
Stage 3 — Retrieval.

Supports two backends (auto-selected via environment):
  - Supabase pgvector  : when SUPABASE_URL is set (cloud/production)
  - LanceDB            : when SUPABASE_URL is NOT set (local dev)

Embedding model: BAAI/bge-small-en-v1.5 (MIT) — dense 384-dim vectors.
  Runs locally via fastembed (ONNX Runtime) for ultra-low RAM usage.
  Downloaded once at build time via build.sh. Loaded from cache at runtime.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
    chunk_id: str = ""


# ── Embedder singleton ────────────────────────────────────────────────────────

_embedder = None
_embedder_model: str = ""


def _get_embedder(model_id: str = "BAAI/bge-small-en-v1.5"):
    global _embedder, _embedder_model
    if _embedder is None or _embedder_model != model_id:
        from fastembed import TextEmbedding
        logger.info("Loading fastembed model: %s", model_id)
        # TextEmbedding automatically manages downloads and ONNX initialization
        _embedder = TextEmbedding(model_name=model_id)
        _embedder_model = model_id
        logger.info("Embedding model loaded.")
    return _embedder


def _embed(texts: list[str], model_id: str = "BAAI/bge-small-en-v1.5") -> list[list[float]]:
    """Return dense embeddings for a list of texts."""
    embedder = _get_embedder(model_id)
    # fastembed returns a generator of numpy arrays
    embeddings_gen = embedder.embed(texts, batch_size=32)
    return [vec.tolist() for vec in embeddings_gen]


# ── Backend selector ──────────────────────────────────────────────────────────

def _use_supabase() -> bool:
    """True when SUPABASE_URL is set — use pgvector; otherwise fall back to LanceDB."""
    return bool(os.getenv("SUPABASE_URL", "").strip())


# ── Indexing (used during onboarding) ────────────────────────────────────────

def index_chunks(
    chunks: list[dict],
    lancedb_path: str = ".lancedb",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
    client_id: str | None = None,
) -> int:
    """
    Embed `chunks` and upsert into the active vector store.

    For Supabase (cloud): requires `client_id`.
    For LanceDB (local):  requires `lancedb_path`.
    """
    if not chunks:
        logger.warning("No chunks provided to index_chunks.")
        return 0

    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d chunks …", len(texts))
    all_vectors = _embed(texts, embedding_model)

    if _use_supabase():
        # ── Supabase path ─────────────────────────────────────────────────────
        if not client_id:
            raise ValueError("client_id is required when SUPABASE_URL is set.")
        from chatbot.db import upsert_chunks
        enriched = [
            {
                **chunks[i],
                "embedding": all_vectors[i],
            }
            for i in range(len(chunks))
        ]
        count = upsert_chunks(client_id, enriched)
        logger.info("Supabase: upserted %d chunks for client '%s'.", count, client_id)
        return count
    else:
        # ── LanceDB path ──────────────────────────────────────────────────────
        import lancedb
        import pyarrow as pa

        TABLE_NAME = "chunks"
        schema = pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), len(all_vectors[0]))),
        ])
        rows = [
            {
                "chunk_id": c.get("chunk_id", f"chunk_{i}"),
                "text": c["text"],
                "source": c.get("source", "unknown"),
                "vector": all_vectors[i],
            }
            for i, c in enumerate(chunks)
        ]
        db = lancedb.connect(lancedb_path)
        if TABLE_NAME in db.table_names():
            table = db.open_table(TABLE_NAME)
            table.add(rows)
        else:
            table = db.create_table(TABLE_NAME, data=rows, schema=schema)
        count = table.count_rows()
        logger.info("LanceDB '%s' now has %d rows.", lancedb_path, count)
        return count


# ── Retrieval (used at query time) ───────────────────────────────────────────

def retrieve(
    query: str,
    lancedb_path: str,
    top_k: int = 5,
    score_threshold: float = 0.35,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    client_id: str | None = None,
) -> list[RetrievedChunk]:
    """
    Embed `query` and return top-k nearest chunks from the active vector store.
    """
    query_vec = _embed([query], embedding_model)[0]

    if _use_supabase():
        # ── Supabase path ─────────────────────────────────────────────────────
        if not client_id:
            raise ValueError("client_id is required when SUPABASE_URL is set.")
        from chatbot.db import similarity_search
        rows = similarity_search(client_id, query_vec, top_k, score_threshold)
        return [
            RetrievedChunk(
                text=r["text"],
                source=r.get("source", "unknown"),
                score=float(r.get("score", 0.0)),
                chunk_id=r.get("chunk_id", ""),
            )
            for r in rows
        ]
    else:
        # ── LanceDB path ──────────────────────────────────────────────────────
        import lancedb

        TABLE_NAME = "chunks"
        db = lancedb.connect(lancedb_path)
        if TABLE_NAME not in db.table_names():
            logger.warning("No LanceDB table at %s — returning empty context.", lancedb_path)
            return []

        table = db.open_table(TABLE_NAME)
        rows = (
            table.search(query_vec)
            .limit(top_k)
            .select(["chunk_id", "text", "source", "_distance"])
            .to_list()
        )
        chunks = []
        for row in rows:
            distance = row.get("_distance", 1.0)
            score = max(0.0, 1.0 - distance)
            if score >= score_threshold:
                chunks.append(RetrievedChunk(
                    text=row["text"],
                    source=row.get("source", "unknown"),
                    score=score,
                    chunk_id=row.get("chunk_id", ""),
                ))
        logger.debug("LanceDB: retrieved %d chunks.", len(chunks))
        return chunks
