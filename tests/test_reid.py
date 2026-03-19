"""Tests for re-ID module."""

import numpy as np
import pytest


def test_embedding_database_add_and_search() -> None:
    """Database should return added embeddings on search."""
    faiss = pytest.importorskip("faiss")
    from src.reid.database import EmbeddingDatabase

    db = EmbeddingDatabase(embedding_dim=128, index_type="flatip")

    # Add a normalized embedding
    emb = np.random.randn(128).astype(np.float32)
    emb /= np.linalg.norm(emb)
    db.add(emb, individual_id="TOAD_001", photo_path="test.jpg")

    assert db.size == 1
    assert "TOAD_001" in db.get_individuals()

    # Search should return the added embedding
    results = db.search(emb, top_k=1)
    assert len(results) == 1
    assert results[0]["individual_id"] == "TOAD_001"
    assert results[0]["score"] > 0.99


def test_embedding_database_empty_search() -> None:
    """Searching an empty database should return empty list."""
    faiss = pytest.importorskip("faiss")
    from src.reid.database import EmbeddingDatabase

    db = EmbeddingDatabase(embedding_dim=128)
    results = db.search(np.random.randn(128).astype(np.float32))
    assert results == []
