"""
Generate Q3 figures: one-week flow plot, adjacency heatmap, degree distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.graph_sd import (
    degree_per_node,
    extract_sd_adjacency,
    one_hop_neighbors,
    two_hop_neighbors,
)
from src.data.paths import DATA_PROCESSED, FIGURES_DIR, ensure_dirs
from src.data.preprocess_sd import load_sd_metadata


def _load_flow() -> tuple[pd.DataFrame, list[str]]:
    ts = pd.read_csv(DATA_PROCESSED / "sd_flow_timestamps.csv", parse_dates=["timestamp"])
    flow = np.load(DATA_PROCESSED / "sd_flow_2020_2021_15min.npy")
    meta = load_sd_metadata()
    sensor_ids = meta["ID"].astype(str).tolist()
    index = pd.DatetimeIndex(ts["timestamp"].values)
    df = pd.DataFrame(flow, index=index, columns=sensor_ids)
    return df, sensor_ids


def plot_one_week_three_sensors(sensor_indices: tuple[int, int, int] = (0, 100, 200)) -> Path:
    ensure_dirs()
    flow, sensor_ids = _load_flow()
    meta = load_sd_metadata()
    # First full week in 2020
    start = pd.Timestamp("2020-01-06")
    end = start + pd.Timedelta(days=7)
    window = flow.loc[start:end]

    fig, ax = plt.subplots(figsize=(12, 4))
    for idx in sensor_indices:
        sid = sensor_ids[idx]
        label = f"Sensor {sid} (idx {idx})"
        ax.plot(window.index, window[sid], label=label, linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic flow (15-min, resampled)")
    ax.set_title("San Diego — three sensors, one week (2020-01-06 … 2020-01-13)")
    ax.legend()
    fig.autofmt_xdate()
    out = FIGURES_DIR / "q3a_three_sensors_one_week.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_adjacency_heatmap(max_display: int = 716) -> Path:
    ensure_dirs()
    adj = extract_sd_adjacency()
    if adj.shape[0] > max_display:
        adj = adj[:max_display, :max_display]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(adj, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_title(f"San Diego adjacency heatmap ({adj.shape[0]}×{adj.shape[1]})")
    ax.set_xlabel("Sensor index (SD order)")
    ax.set_ylabel("Sensor index (SD order)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Road distance / weight")
    out = FIGURES_DIR / "q3b_adjacency_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_degree_distribution() -> Path:
    ensure_dirs()
    adj = extract_sd_adjacency()
    degrees = degree_per_node(adj)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(degrees, bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("Number of neighbors (nonzero off-diagonal edges)")
    ax.set_ylabel("Count of sensors")
    ax.set_title("Degree distribution — San Diego traffic graph")
    out = FIGURES_DIR / "q3b_degree_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def export_neighborhood_example(node_idx: int = 0) -> dict:
    ensure_dirs()
    meta = load_sd_metadata()
    adj = extract_sd_adjacency()
    one = one_hop_neighbors(adj, node_idx)
    two = two_hop_neighbors(adj, node_idx)
    row = meta.iloc[node_idx]
    payload = {
        "selected_index": node_idx,
        "selected_sensor_id": str(row["ID"]),
        "lat": float(row["Lat"]),
        "lng": float(row["Lng"]),
        "one_hop_count": int(len(one)),
        "two_hop_count": int(len(two)),
        "one_hop_neighbor_indices": one.tolist(),
        "two_hop_neighbor_indices": two.tolist(),
        "one_hop_neighbor_ids": meta.iloc[one]["ID"].astype(str).tolist(),
        "two_hop_neighbor_ids": meta.iloc[two]["ID"].astype(str).tolist(),
    }
    out = DATA_PROCESSED / "q3c_neighborhood_example.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def run_all_plots() -> list[Path]:
    paths = [
        plot_one_week_three_sensors(),
        plot_adjacency_heatmap(),
        plot_degree_distribution(),
    ]
    export_neighborhood_example(node_idx=0)
    return paths