"""Stock identity embedding helpers."""

from __future__ import annotations

import torch
from torch import nn


class StockIdentityEmbedding(nn.Module):
    """Build stock identity tensors for learning and fixed Gaussian ID codes."""

    def __init__(
        self,
        *,
        num_stocks: int,
        representation_type: str,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if representation_type not in {"learning", "gaussian"}:
            raise ValueError(
                "Unsupported stock_id_representation_type: "
                f"{representation_type!r}."
            )

        self.representation_type = representation_type
        self.embedding_dim = embedding_dim
        if self.representation_type == "learning":
            self.stock_id_embedding = nn.Embedding(num_stocks, embedding_dim)
            self.register_buffer("stock_id_gaussian_code", None, persistent=False)
        else:
            self.stock_id_embedding = None
            self.register_buffer(
                "stock_id_gaussian_code",
                self._build_stock_id_gaussian_code(
                    num_stocks=num_stocks,
                    embedding_dim=embedding_dim,
                ),
                persistent=True,
            )

    @staticmethod
    def _build_stock_id_gaussian_code(
        *,
        num_stocks: int,
        embedding_dim: int,
    ) -> torch.Tensor:
        # Fixed Gaussian ID codes keep stock identities dense without materializing one-hot vectors.
        gaussian_code = torch.randn(num_stocks, embedding_dim)
        gaussian_code = gaussian_code / gaussian_code.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return gaussian_code

    def forward(
        self,
        *,
        stock_indices: torch.Tensor,
        time_steps: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.representation_type == "learning":
            if self.stock_id_embedding is None:
                raise RuntimeError("stock_id_embedding must be initialized in learning mode.")
            return (
                self.stock_id_embedding(stock_indices)
                .unsqueeze(1)
                .expand(-1, time_steps, -1, -1)
            )

        if self.representation_type != "gaussian":
            raise ValueError(
                "Unsupported stock_id_representation_type: "
                f"{self.representation_type!r}."
            )

        if self.stock_id_gaussian_code is None:
            raise RuntimeError("stock_id_gaussian_code must be initialized in gaussian mode.")
        identity = self.stock_id_gaussian_code[stock_indices.to(dtype=torch.long)]
        return identity.to(dtype=dtype).unsqueeze(1).expand(-1, time_steps, -1, -1)
