"""Sliding-window dataset for spatio-temporal traffic forecasting (Q4a)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Q3 pipeline convention (see src/data/preprocess.py when present).
FLOW_CANDIDATES = (
    "sd_flow_2020_2021_15min.npy",
    "flow_15min.npy",
    "traffic_15min.npy",
    "flow.npy",
)
ADJ_CANDIDATES = ("adjacency.npy", "sd_rn_adj.npy", "adj_sd.npy")
MANIFEST_NAME = "manifest.json"


def _first_existing(processed_dir: Path, names: Tuple[str, ...]) -> Path:
    for name in names:
        path = processed_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of {names} found in {processed_dir}. Run Q3 preprocessing first."
    )


def load_processed_arrays(processed_dir: Path | str) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load flow tensor (T, N[, 1]), adjacency (N, N), and timestamps from Q3 artifacts."""
    processed_dir = Path(processed_dir)
    flow_path = _first_existing(processed_dir, FLOW_CANDIDATES)
    adj_path = _first_existing(processed_dir, ADJ_CANDIDATES)
    manifest_path = processed_dir / MANIFEST_NAME

    flow = np.load(flow_path).astype(np.float32)
    if flow.ndim == 3:
        flow = flow[..., 0]
    if flow.ndim != 2:
        raise ValueError(f"Expected flow shape (T, N), got {flow.shape}")

    adj = np.load(adj_path).astype(np.float32)
    np.fill_diagonal(adj, 0.0)

    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        start = meta.get("start", "2020-01-01 00:00:00")
        freq = meta.get("freq", "15min")
        timestamps = pd.date_range(start=start, periods=flow.shape[0], freq=freq)
    else:
        timestamps_path = processed_dir / "timestamps.npy"
        if timestamps_path.exists():
            timestamps = pd.to_datetime(np.load(timestamps_path, allow_pickle=True))
        else:
            timestamps = pd.date_range(start="2020-01-01", periods=flow.shape[0], freq="15min")

    if len(timestamps) != flow.shape[0]:
        raise ValueError(
            f"Timestamp length {len(timestamps)} != flow length {flow.shape[0]}"
        )
    return flow, adj, timestamps


def _split_masks(timestamps: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    """Chronological splits: 2020 train, Jan–Mar 2021 val, Apr–Dec 2021 test."""
    year = timestamps.year.to_numpy()
    month = timestamps.month.to_numpy()
    return {
        "train": year == 2020,
        "val": (year == 2021) & (month <= 3),
        "test": (year == 2021) & (month >= 4),
    }


def _contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive start, exclusive end indices for True runs."""
    segments: List[Tuple[int, int]] = []
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def build_sample_indices(
    mask: np.ndarray,
    input_len: int = 8,
    horizon: int = 4,
) -> np.ndarray:
    """
    Window at end index t uses inputs [t-7..t] and targets [t+1..t+horizon].
    Samples never cross split boundaries.
    """
    indices: List[int] = []
    min_span = input_len + horizon
    for seg_start, seg_end in _contiguous_segments(mask):
        if seg_end - seg_start < min_span:
            continue
        t_min = seg_start + input_len - 1
        t_max = seg_end - horizon - 1
        for t in range(t_min, t_max + 1):
            indices.append(t)
    return np.asarray(indices, dtype=np.int64)


@dataclass
class NormStats:
    mean: float
    std: float

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x * self.std + self.mean
        return x * self.std + self.mean


class TrafficForecastDataset(Dataset):
    def __init__(
        self,
        flow: np.ndarray,
        sample_ends: np.ndarray,
        stats: NormStats,
        input_len: int = 8,
        horizon: int = 4,
    ) -> None:
        self.flow = stats.normalize(flow)
        self.sample_ends = sample_ends
        self.input_len = input_len
        self.horizon = horizon
        self.stats = stats

    def __len__(self) -> int:
        return int(self.sample_ends.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = int(self.sample_ends[idx])
        x = self.flow[t - self.input_len + 1 : t + 1]  # (8, N)
        y = self.flow[t + 1 : t + 1 + self.horizon]  # (4, N)
        # (T, N, F) with F=1
        x_t = torch.from_numpy(x).unsqueeze(-1)
        y_t = torch.from_numpy(y).unsqueeze(-1)
        return x_t, y_t


class ForecastDataModule:
    """Builds train/val/test loaders with train-only normalization."""

    def __init__(
        self,
        processed_dir: Path | str,
        batch_size: int = 32,
        input_len: int = 8,
        horizon: int = 4,
        num_workers: int = 0,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.input_len = input_len
        self.horizon = horizon
        self.num_workers = num_workers

        flow, adj, timestamps = load_processed_arrays(self.processed_dir)
        self.flow_raw = flow
        self.adj_raw = adj
        self.timestamps = timestamps
        self.num_nodes = flow.shape[1]

        masks = _split_masks(timestamps)
        train_vals = flow[masks["train"]]
        self.stats = NormStats(mean=float(train_vals.mean()), std=float(train_vals.std() + 1e-6))

        self.indices = {
            split: build_sample_indices(masks[split], input_len, horizon)
            for split in ("train", "val", "test")
        }

    def dataloaders(self) -> Dict[str, DataLoader]:
        loaders: Dict[str, DataLoader] = {}
        for split in ("train", "val", "test"):
            ds = TrafficForecastDataset(
                self.flow_raw,
                self.indices[split],
                self.stats,
                self.input_len,
                self.horizon,
            )
            shuffle = split == "train"
            loaders[split] = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
            )
        return loaders