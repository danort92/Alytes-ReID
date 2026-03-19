"""Re-ID model architecture with metric learning head."""

from __future__ import annotations

import logging
from typing import Any

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ToadReIDModel(nn.Module):
    """Toad re-identification model with embedding backbone and optional ArcFace head.

    Architecture:
        Backbone (EfficientNet/ResNet) → Global Average Pool → FC → Embedding
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        embedding_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        """Initialize re-ID model.

        Args:
            backbone: timm model name for the backbone.
            embedding_dim: Dimensionality of the output embedding.
            pretrained: Use ImageNet-pretrained weights.
            dropout: Dropout rate before embedding layer.
        """
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # remove classification head
        )
        backbone_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        logger.info(
            "ToadReIDModel: backbone=%s, backbone_dim=%d, embedding_dim=%d",
            backbone,
            backbone_dim,
            embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding from input image.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            L2-normalized embedding tensor (B, embedding_dim).
        """
        features = self.backbone(x)
        embedding = self.head(features)
        # L2 normalize
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


def build_model(config: dict[str, Any]) -> ToadReIDModel:
    """Build re-ID model from configuration.

    Args:
        config: Model configuration dictionary.

    Returns:
        Initialized ToadReIDModel.
    """
    model_cfg = config["model"]
    return ToadReIDModel(
        backbone=model_cfg["backbone"],
        embedding_dim=model_cfg["embedding_dim"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
    )
