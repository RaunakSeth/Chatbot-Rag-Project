"""
Tests for the retrieval layer (chunking + LanceDB index/search).
Uses a temporary directory so no persistent state is created.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chatbot.retrieval.chunker import chunk_text, _make_id

# Skip LanceDB tests if pyarrow's native dataset extension is blocked by OS policy
def _lancedb_available() -> bool:
    try:
        import pyarrow.dataset  # noqa: F401
        import lancedb  # noqa: F401
        return True
    except (ImportError, OSError):
        return False

lancedb_available = pytest.mark.skipif(
    not _lancedb_available(),
    reason="pyarrow dataset DLL blocked by OS Application Control policy — LanceDB unavailable",
)


# ── Chunker ───────────────────────────────────────────────────────────────────

class TestChunker:
    def test_basic_chunking(self):
        text = "Hello world. " * 200  # long enough to produce multiple chunks
        chunks = chunk_text(text, source="test://example", chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1
        for c in chunks:
            assert "chunk_id" in c
            assert "text" in c
            assert "source" in c
            assert c["source"] == "test://example"

    def test_short_text_single_chunk(self):
        text = "This is a short FAQ answer about our return policy."
        chunks = chunk_text(text, source="test://faq")
        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_empty_text(self):
        chunks = chunk_text("", source="test://empty")
        assert chunks == []

    def test_chunk_ids_are_unique(self):
        text = "paragraph one\n\nparagraph two\n\nparagraph three\n\n" * 20
        chunks = chunk_text(text, source="test://doc", chunk_size=50, chunk_overlap=5)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_make_id_deterministic(self):
        id1 = _make_id("source", 0, "hello world")
        id2 = _make_id("source", 0, "hello world")
        assert id1 == id2

    def test_make_id_different_for_different_inputs(self):
        id1 = _make_id("source", 0, "hello world")
        id2 = _make_id("source", 1, "hello world")
        assert id1 != id2


# ── Retrieval with mocked embedder ───────────────────────────────────────────

class TestRetrieval:
    @pytest.fixture
    def tmp_lancedb(self, tmp_path):
        return str(tmp_path / "lancedb")

    @lancedb_available
    @patch("chatbot.stages.stage3_retrieval._get_embedder")
    def test_index_and_retrieve(self, mock_get_embedder, tmp_lancedb):
        # Mock the fastembed embedder
        mock_embedder = MagicMock()
        import numpy as np
        fake_vector = np.random.rand(384).astype("float32")
        mock_embedder.embed.return_value = iter([fake_vector] * 3)
        mock_get_embedder.return_value = mock_embedder

        from chatbot.stages.stage3_retrieval import index_chunks, retrieve

        # Index 3 chunks
        chunks = [
            {"chunk_id": f"c{i}", "text": f"Sample text about topic {i}.", "source": "test"}
            for i in range(3)
        ]
        total = index_chunks(chunks, tmp_lancedb)
        assert total == 3

        # Retrieve (mock embedder returns same vector → all distances ~0 → high score)
        mock_embedder.embed.return_value = iter([fake_vector])
        results = retrieve(
            "test query",
            tmp_lancedb,
            top_k=3,
            score_threshold=0.0,  # low threshold so we get results with random vectors
        )
        # At least the table should be queryable
        assert isinstance(results, list)

    @lancedb_available
    @patch("chatbot.stages.stage3_retrieval._get_embedder")
    def test_retrieve_empty_db(self, mock_get_embedder, tmp_lancedb):
        mock_embedder = MagicMock()
        import numpy as np
        mock_embedder.embed.return_value = iter([np.random.rand(384).astype("float32")])
        mock_get_embedder.return_value = mock_embedder
        
        from chatbot.stages.stage3_retrieval import retrieve
        results = retrieve("test", tmp_lancedb)
        assert results == []
