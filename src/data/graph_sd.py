"""
Build San Diego road-network adjacency submatrix A_SD (716×716) aligned with flow tensor columns.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from src.data.paths import CA_RN_ADJ, DATA_PROCESSED, SD_NUM_SENSORS, ensure_dirs
from src.data.preprocess_sd import load_sd_metadata


@dataclass
class AdjacencyStats:
    num_nodes: int
    num_nonzero_edges: int
    num_directed_nonzero_including_diag: int
    is_symmetric: bool
    max_asymmetry: float
    mean_degree: float
    sparsity_percent: float
    diagonal_all_zero_after_processing: bool


def extract_sd_adjacency(sd_meta=None) -> np.ndarray:
    if sd_meta is None:
        sd_meta = load_sd_metadata()
    id2 = sd_meta["ID2"].values.astype(int).tolist()
    ca_adj = np.load(CA_RN_ADJ)
    sd_adj = ca_adj[np.ix_(id2, id2)].astype(np.float64)
    if sd_adj.shape != (SD_NUM_SENSORS, SD_NUM_SENSORS):
        raise ValueError(f"Expected adjacency {SD_NUM_SENSORS}×{SD_NUM_SENSORS}, got {sd_adj.shape}")
    return sd_adj


def compute_stats(adj: np.ndarray, zero_diagonal_for_stats: bool = True) -> AdjacencyStats:
    a = adj.copy()
    if zero_diagonal_for_stats:
        np.fill_diagonal(a, 0.0)
    directed_nnz = int(np.count_nonzero(a))
    is_sym = bool(np.allclose(a, a.T, rtol=0, atol=1e-8))
    asym = float(np.max(np.abs(a - a.T))) if a.size else 0.0
    degrees = np.count_nonzero(a, axis=1)
    nn = a.shape[0]
    return AdjacencyStats(
        num_nodes=nn,
        num_nonzero_edges=directed_nnz,
        num_directed_nonzero_including_diag=int(np.count_nonzero(adj)),
        is_symmetric=is_sym,
        max_asymmetry=asym,
        mean_degree=float(np.mean(degrees)),
        sparsity_percent=float(directed_nnz / (nn * nn) * 100),
        diagonal_all_zero_after_processing=bool(np.all(np.diag(a) == 0)),
    )


def degree_per_node(adj: np.ndarray) -> np.ndarray:
    a = adj.copy()
    np.fill_diagonal(a, 0)
    return np.count_nonzero(a, axis=1)


def one_hop_neighbors(adj: np.ndarray, node_idx: int) -> np.ndarray:
    row = adj[node_idx].copy()
    row[node_idx] = 0
    return np.flatnonzero(row != 0)


def two_hop_neighbors(adj: np.ndarray, node_idx: int) -> np.ndarray:
    one_hop = set(one_hop_neighbors(adj, node_idx).tolist())
    two_hop: set[int] = set()
    for j in one_hop:
        two_hop.update(one_hop_neighbors(adj, j).tolist())
    two_hop.discard(node_idx)
    two_hop -= one_hop
    return np.array(sorted(two_hop), dtype=int)


def save_adjacency(adj: np.ndarray, stats: AdjacencyStats) -> None:
    ensure_dirs()
    np.save(DATA_PROCESSED / "sd_rn_adj.npy", adj)
    with open(DATA_PROCESSED / "sd_adjacency_stats.json", "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)
    np.save(DATA_PROCESSED / "sd_degrees.npy", degree_per_node(adj))


def run_graph_construction() -> tuple[np.ndarray, AdjacencyStats]:
    sd_meta = load_sd_metadata()
    adj = extract_sd_adjacency(sd_meta)
    stats = compute_stats(adj)
    save_adjacency(adj, stats)
    return adj, stats