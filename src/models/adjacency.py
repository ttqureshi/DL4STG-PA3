"""Fixed, adaptive, and hybrid adjacency for STGNN (Q5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetric_normalized_adjacency(adj: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    A_hat = D^{-1/2} (A + I) D^{-1/2} for undirected nonnegative graphs (Q5a).
    Self-loops are added before degree normalization.
    """
    if not isinstance(adj, torch.Tensor):
        adj = torch.from_numpy(adj.astype(np.float32))
    a = adj.clone()
    a.fill_diagonal_(0.0)
    a = a + torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
    deg = a.sum(dim=1)
    d_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
    d_left = torch.diag(d_inv_sqrt)
    return d_left @ a @ d_left


class AdaptiveAdjacency(nn.Module):
    """
    Learnable sensor–sensor weights (Q5b):
      A_adaptive = softmax(row-wise, ReLU(E1 @ E2^T)).
    Softmax runs along columns so each row is a distribution over neighbors.
    """

    def __init__(
        self,
        num_nodes: int,
        embed_dim: int = 10,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.top_k = top_k
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)

    def forward(self) -> torch.Tensor:
        logits = F.relu(self.E1 @ self.E2.T)
        a = F.softmax(logits, dim=-1)
        if self.top_k is not None and self.top_k < self.num_nodes:
            # Sparsify after softmax: keep top-k per row, renormalize rows so they sum to 1.
            # Gradients flow only through retained entries (straight-through mask).
            vals, idx = torch.topk(a, k=self.top_k, dim=-1)
            mask = torch.zeros_like(a)
            mask.scatter_(-1, idx, 1.0)
            a = a * mask
            a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return a

    def dense_matrix(self) -> torch.Tensor:
        return self.forward()


@dataclass
class AdjacencyConfig:
    mode: str = "fixed"  # fixed | adaptive | hybrid | none
    hybrid_alpha: float = 0.5
    embed_dim: int = 10
    top_k: Optional[int] = None


class GraphAdjacency(nn.Module):
    """Produces batch-shared normalized adjacency used in the GCN layer."""

    def __init__(
        self,
        fixed_adj: np.ndarray,
        config: AdjacencyConfig,
    ) -> None:
        super().__init__()
        self.mode = config.mode
        self.hybrid_alpha = config.hybrid_alpha
        n = fixed_adj.shape[0]

        fixed = symmetric_normalized_adjacency(fixed_adj)
        self.register_buffer("fixed_norm", fixed)

        if config.mode in ("adaptive", "hybrid"):
            self.adaptive = AdaptiveAdjacency(n, config.embed_dim, config.top_k)
        else:
            self.adaptive = None

    def forward(self) -> torch.Tensor:
        if self.mode == "none":
            return None
        if self.mode == "fixed":
            return self.fixed_norm
        if self.mode == "adaptive":
            assert self.adaptive is not None
            return self.adaptive()
        # hybrid: combine normalized fixed graph with adaptive (both row-stochastic-ish).
        assert self.adaptive is not None
        a_ad = self.adaptive()
        alpha = self.hybrid_alpha
        a = alpha * self.fixed_norm + (1.0 - alpha) * a_ad
        # Re-scale rows for stable message passing (not necessarily doubly stochastic).
        row_sum = a.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return a / row_sum


def build_adjacency(fixed_adj: np.ndarray, config: AdjacencyConfig) -> GraphAdjacency:
    return GraphAdjacency(fixed_adj, config)