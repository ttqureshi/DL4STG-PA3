"""STGNN and temporal-only baseline (Q4b–c)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adjacency import AdjacencyConfig, GraphAdjacency


class TemporalConvEncoder(nn.Module):
    """
    Lightweight temporal encoder: 1D convolutions along history length per sensor.
    Input x: (B, T_in, N, F) -> Z: (B, N, d_model).
    """

    def __init__(
        self,
        input_len: int = 8,
        in_channels: int = 1,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        mid = max(hidden_dim // 2, 8)
        self.conv1 = nn.Conv1d(in_channels, mid, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(mid, hidden_dim, kernel_size=3, padding=1)
        self.input_len = input_len
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x.shape
        x_bn = x.permute(0, 2, 3, 1).reshape(b * n, f, t)
        h = F.relu(self.conv1(x_bn))
        h = F.relu(self.conv2(h))
        # Global average pool over time
        z = h.mean(dim=-1).view(b, n, self.hidden_dim)
        return z


class ForecastHead(nn.Module):
    """Maps node features to horizon-length predictions."""

    def __init__(self, hidden_dim: int, horizon: int = 4, out_channels: int = 1) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, horizon * out_channels)
        self.horizon = horizon
        self.out_channels = out_channels

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, N, d)
        out = self.fc(h)
        b, n, _ = out.shape
        out = out.view(b, n, self.horizon, self.out_channels)
        return out.permute(0, 2, 1, 3)  # (B, horizon, N, F)


class GraphConvLayer(nn.Module):
    """Single layer: H = ReLU(A_hat Z W_g)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w_g = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, z: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # z: (B, N, in_dim), a_hat: (N, N)
        agg = torch.einsum("ij,bjd->bid", a_hat, z)
        return F.relu(self.w_g(agg))


class ForecastingModel(nn.Module):
    """STGNN: temporal encoder -> GCN -> forecast head (Q4b, Q5)."""

    def __init__(
        self,
        num_nodes: int,
        fixed_adj,
        adjacency_config: AdjacencyConfig,
        hidden_dim: int = 32,
        input_len: int = 8,
        horizon: int = 4,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = TemporalConvEncoder(input_len, in_channels, hidden_dim)
        self.graph = GraphAdjacency(fixed_adj, adjacency_config)
        self.gcn = GraphConvLayer(hidden_dim, hidden_dim)
        self.head = ForecastHead(hidden_dim, horizon, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        a_hat = self.graph()
        if a_hat is None:
            h = z
        else:
            h = self.gcn(z, a_hat)
        return self.head(h)


class TemporalOnlyModel(nn.Module):
    """Same encoder and head as STGNN, without graph convolution (Q4c)."""

    def __init__(
        self,
        hidden_dim: int = 32,
        input_len: int = 8,
        horizon: int = 4,
        in_channels: int = 1,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = TemporalConvEncoder(input_len, in_channels, hidden_dim)
        self.head = ForecastHead(hidden_dim, horizon, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)