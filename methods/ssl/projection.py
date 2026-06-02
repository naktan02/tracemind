"""Query SSL feature projection primitives."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class SslProjectionHead(nn.Module):
    """CoMatch/SimMatch류 feature similarity objective용 MLP projection head."""

    def __init__(self, *, input_dim: int, proj_size: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if proj_size <= 0:
            raise ValueError("proj_size must be positive.")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=False),
            nn.Linear(input_dim, proj_size),
        )

    def forward(self, features: Tensor) -> Tensor:
        return F.normalize(self.layers(features), dim=1)
