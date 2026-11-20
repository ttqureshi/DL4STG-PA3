"""Synthetic SD flow when full LargeST HDF5 cannot be extracted locally."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.paths import DATA_PROCESSED, SD_NUM_SENSORS, YEARS, ensure_dirs
from src.data.preprocess_sd import MissingValueReport, save_processed, load_sd_metadata


def _build_index() -> pd.DatetimeIndex:
    chunks = []
    for year in YEARS:
        start = f"{year}-01-01"
        end = f"{year}-12-31 23:45:00"
        chunks.append(pd.date_range(start, end, freq="15min"))
    idx = pd.DatetimeIndex(np.concatenate(chunks))
    return idx[~idx.duplicated(keep="first")]


def generate_synthetic_flow(seed: int = 42) -> tuple[pd.DataFrame, MissingValueReport]:
    """
    Physics-inspired toy flows: daily rush-hour curves + mild spatial smoothing.
    Uses real SD sensor metadata for column order only.
    """
    rng = np.random.default_rng(seed)
    sd_meta = load_sd_metadata()
    n = len(sd_meta)
    if n != SD_NUM_SENSORS:
        raise ValueError(f"Expected {SD_NUM_SENSORS} sensors, got {n}")

    index = _build_index()
    t = np.arange(len(index)) / 96.0
    daily = 50 + 40 * np.sin(2 * np.pi * (t - 0.35))  # morning/evening structure
    weekly = 1.0 + 0.15 * np.sin(2 * np.pi * t / 7)
    sensor_scale = rng.uniform(0.7, 1.3, size=(n, 1))
    noise = rng.normal(0, 5, size=(len(index), n))
    base = (daily[:, None] * weekly[:, None] * sensor_scale.T) + noise
    base = np.clip(base, 0, None)
    flow = pd.DataFrame(base.astype(np.float32), index=index, columns=sd_meta["ID"].astype(str))

    report = MissingValueReport(
        sensors_with_any_nan=0,
        total_nan_cells=0,
        nan_fraction=0.0,
        imputation="synthetic generator (no NaNs)",
    )
    return flow, report


def run_synthetic_preprocessing(seed: int = 42) -> dict:
    ensure_dirs()
    sd_meta = load_sd_metadata()
    flow, report = generate_synthetic_flow(seed=seed)
    meta = save_processed(sd_meta, flow, report)
    note = {
        "synthetic_flow": True,
        "seed": seed,
        "reason": "Local disk quota prevented extracting full LargeST ca_his_raw_*.h5 files.",
        "replace_with": "Re-run python -m src.data.run_q3 without --synthetic-flow after raw HDF5 is available.",
    }
    with open(DATA_PROCESSED / "synthetic_note.json", "w", encoding="utf-8") as f:
        json.dump(note, f, indent=2)
    return meta