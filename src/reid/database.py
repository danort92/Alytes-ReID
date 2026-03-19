"""Embedding database management for toad re-identification.

Stores and retrieves individual toad embeddings using FAISS for fast
similarity search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingDatabase:
    """Manages a database of toad embeddings for re-identification.

    Stores embeddings with associated metadata (individual ID, photo paths,
    capture dates) and supports fast nearest-neighbor search via FAISS.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "flatip",
    ) -> None:
        """Initialize embedding database.

        Args:
            embedding_dim: Dimensionality of embeddings.
            index_type: FAISS index type ("flatip" for inner product, "flatl2" for L2).
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss required. Install with: pip install faiss-cpu")

        self.embedding_dim = embedding_dim
        self.index_type = index_type

        if index_type == "flatip":
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "flatl2":
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.metadata: list[dict[str, Any]] = []
        logger.info(
            "EmbeddingDatabase initialized (dim=%d, index=%s)",
            embedding_dim,
            index_type,
        )

    @property
    def size(self) -> int:
        """Number of embeddings in the database."""
        return self.index.ntotal

    def add(
        self,
        embedding: np.ndarray,
        individual_id: str,
        photo_path: str = "",
        capture_date: str = "",
        extra: dict[str, Any] | None = None,
    ) -> int:
        """Add an embedding to the database.

        Args:
            embedding: Embedding vector (embedding_dim,).
            individual_id: Unique identifier for the individual.
            photo_path: Path to the source photo.
            capture_date: Date of capture.
            extra: Additional metadata.

        Returns:
            Index of the added embedding.
        """
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        self.index.add(embedding)

        meta = {
            "individual_id": individual_id,
            "photo_path": photo_path,
            "capture_date": capture_date,
            **(extra or {}),
        }
        self.metadata.append(meta)

        idx = self.size - 1
        logger.debug("Added embedding for %s at index %d", individual_id, idx)
        return idx

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for the most similar embeddings.

        Args:
            query: Query embedding vector (embedding_dim,).
            top_k: Number of top results to return.

        Returns:
            List of dicts with keys: "individual_id", "score", "metadata".
        """
        if self.size == 0:
            return []

        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self.size)
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "individual_id": self.metadata[idx]["individual_id"],
                "score": float(score),
                "metadata": self.metadata[idx],
            })

        return results

    def get_individuals(self) -> list[str]:
        """Get list of unique individual IDs in the database.

        Returns:
            Sorted list of unique individual IDs.
        """
        ids = {m["individual_id"] for m in self.metadata}
        return sorted(ids)

    def save(self, path: Path) -> None:
        """Save database to disk.

        Args:
            path: Directory to save index and metadata.
        """
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info("Database saved to %s (%d embeddings)", path, self.size)

    def load(self, path: Path) -> None:
        """Load database from disk.

        Args:
            path: Directory containing saved index and metadata.
        """
        import faiss

        path = Path(path)
        self.index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "metadata.json") as f:
            self.metadata = json.load(f)

        logger.info("Database loaded from %s (%d embeddings)", path, self.size)
