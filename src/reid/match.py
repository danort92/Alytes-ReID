"""Query matching for toad re-identification.

Usage:
    python -m src.reid.match --model data/models/reid/reid_model.pt --db data/models/reid --image query.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from src.reid.database import EmbeddingDatabase
from src.reid.model import ToadReIDModel, build_model

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_embedding(
    model: ToadReIDModel,
    image: np.ndarray,
    device: str = "cpu",
    image_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Extract embedding from a preprocessed toad image.

    Args:
        model: Loaded re-ID model.
        image: Preprocessed image (H, W, 3) in RGB.
        device: Torch device.
        image_size: Expected input size (H, W).

    Returns:
        Embedding vector as numpy array.
    """
    # Resize and normalize
    img = cv2.resize(image, (image_size[1], image_size[0]))
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    # To tensor (B, C, H, W)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        embedding = model(tensor)

    return embedding.cpu().numpy().flatten()


def match_toad(
    model: ToadReIDModel,
    database: EmbeddingDatabase,
    image: np.ndarray,
    top_k: int = 5,
    threshold: float = 0.6,
    device: str = "cpu",
) -> dict[str, Any]:
    """Match a toad image against the database.

    Args:
        model: Loaded re-ID model.
        database: Embedding database.
        image: Preprocessed toad image (H, W, 3) in RGB.
        top_k: Number of top matches to return.
        threshold: Similarity threshold below which toad is flagged as new.
        device: Torch device.

    Returns:
        Dictionary with keys:
            - "matches": list of top-K matches with scores
            - "is_new": True if best match is below threshold
            - "embedding": the query embedding
    """
    embedding = extract_embedding(model, image, device=device)
    matches = database.search(embedding, top_k=top_k)

    is_new = True
    if matches and matches[0]["score"] >= threshold:
        is_new = False

    return {
        "matches": matches,
        "is_new": is_new,
        "embedding": embedding,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Match toad against database")
    parser.add_argument("--config", type=Path, default=Path("config/reid.yaml"))
    parser.add_argument("--model", type=Path, required=True, help="Model weights")
    parser.add_argument("--db", type=Path, required=True, help="Database directory")
    parser.add_argument("--image", type=Path, required=True, help="Query image")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(str(args.model), map_location=device))

    db = EmbeddingDatabase(
        embedding_dim=config["model"]["embedding_dim"],
        index_type=config["database"]["index_type"],
    )
    db.load(args.db)

    image = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2RGB)
    result = match_toad(model, db, image, device=device, **{
        k: config["inference"][k]
        for k in ("top_k", "similarity_threshold")
        if k in config["inference"]
    })

    if result["is_new"]:
        print("NEW INDIVIDUAL (no confident match found)")
    else:
        print(f"Best match: {result['matches'][0]['individual_id']} "
              f"(score: {result['matches'][0]['score']:.3f})")

    print("\nTop matches:")
    for m in result["matches"]:
        print(f"  {m['individual_id']}: {m['score']:.3f}")


if __name__ == "__main__":
    main()
