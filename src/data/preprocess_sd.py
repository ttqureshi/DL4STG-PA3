"""
Extract San Diego (District 11) sensors, resample 5-min → 15-min, handle missing values.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.paths import (
    DATA_PROCESSED,
    RESAMPLE_RULE,
    SD_DISTRICT,
    SD_NUM_SENSORS,
    YEARS,
    ensure_dirs,
    raw_h5_path,
)

H5_KEY = "t"
H5_ROW_CHUNK = 400


@dataclass
class MissingValueReport:
    """Summary of NaNs before imputation."""

    sensors_with_any_nan: int
    total_nan_cells: int
    nan_fraction: float
    imputation: str


def load_sd_metadata() -> pd.DataFrame:
    from src.data.paths import CA_META

    ca_meta = pd.read_csv(CA_META)
    sd_meta = ca_meta[ca_meta["District"] == SD_DISTRICT].copy()
    sd_meta = sd_meta.reset_index(drop=True)
    if len(sd_meta) != SD_NUM_SENSORS:
        raise ValueError(
            f"Expected {SD_NUM_SENSORS} San Diego sensors, found {len(sd_meta)}"
        )
    return sd_meta


def resample_policy() -> tuple[str, str]:
    agg = "mean"
    rationale = (
        "5-min values are interval vehicle counts. Official LargeST uses "
        "resample('15min').mean(); we match that (equivalent to average count "
        "over three 5-min bins). Alternative: .sum() for total vehicles per 15 min."
    )
    return agg, rationale


def missing_report(before_fill: pd.DataFrame) -> MissingValueReport:
    nan_mask = before_fill.isna()
    return MissingValueReport(
        sensors_with_any_nan=int(nan_mask.any(axis=0).sum()),
        total_nan_cells=int(nan_mask.sum().sum()),
        nan_fraction=float(nan_mask.mean().mean()),
        imputation="NaN after resample → fillna(0), consistent with LargeST process_ca_his",
    )


def _read_sd_h5_chunked(path: Path, sd_sensor_ids: list[str]) -> pd.DataFrame:
    """LargeST HDF5 is fixed-format; slice rows to keep RAM under ~200 MB per chunk."""
    parts: list[pd.DataFrame] = []
    with pd.HDFStore(path, mode="r") as store:
        storer = store.get_storer(H5_KEY)
        nrows = int(storer.nrows) if storer.nrows is not None else int(storer.shape[0])
        for start in range(0, nrows, H5_ROW_CHUNK):
            stop = min(start + H5_ROW_CHUNK, nrows)
            chunk = store.select(H5_KEY, start=start, stop=stop)
            chunk.columns = chunk.columns.astype(str)
            parts.append(chunk[sd_sensor_ids])
    return pd.concat(parts, axis=0).sort_index()


def _process_year_h5(sd_meta: pd.DataFrame, year: int) -> tuple[pd.DataFrame, MissingValueReport]:
    sd_sensor_ids = sd_meta["ID"].astype(str).tolist()
    subset = _read_sd_h5_chunked(raw_h5_path(year), sd_sensor_ids)
    agg, _ = resample_policy()
    resampled = subset.resample(RESAMPLE_RULE).agg(agg)
    report = missing_report(resampled)
    resampled = resampled.fillna(0).round(0)
    return resampled, report


def build_flow_tensor(sd_meta: pd.DataFrame) -> tuple[pd.DataFrame, MissingValueReport]:
    frames = []
    reports = []
    for year in YEARS:
        resampled, report = _process_year_h5(sd_meta, year)
        frames.append(resampled)
        reports.append(report)

    combined = pd.concat(frames, axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    merged_report = MissingValueReport(
        sensors_with_any_nan=max(r.sensors_with_any_nan for r in reports),
        total_nan_cells=sum(r.total_nan_cells for r in reports),
        nan_fraction=float(np.mean([r.nan_fraction for r in reports])),
        imputation=reports[0].imputation,
    )
    return combined, merged_report


def build_flow_tensor_low_disk(
    sd_meta: pd.DataFrame, delete_h5_after_year: bool = True
) -> tuple[pd.DataFrame, MissingValueReport]:
    frames, reports = [], []
    for year in YEARS:
        resampled, report = _process_year_h5(sd_meta, year)
        frames.append(resampled)
        reports.append(report)
        if delete_h5_after_year:
            raw_h5_path(year).unlink(missing_ok=True)
    combined = pd.concat(frames, axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    merged_report = MissingValueReport(
        sensors_with_any_nan=max(r.sensors_with_any_nan for r in reports),
        total_nan_cells=sum(r.total_nan_cells for r in reports),
        nan_fraction=float(np.mean([r.nan_fraction for r in reports])),
        imputation=reports[0].imputation,
    )
    return combined, merged_report


def save_processed(sd_meta: pd.DataFrame, flow: pd.DataFrame, report: MissingValueReport) -> dict:
    ensure_dirs()
    sd_meta.to_csv(DATA_PROCESSED / "sd_meta.csv", index=False)
    flow_values = flow.values.astype(np.float32)
    np.save(DATA_PROCESSED / "sd_flow_2020_2021_15min.npy", flow_values)
    np.save(DATA_PROCESSED / "flow_15min.npy", flow_values)
    np.save(DATA_PROCESSED / "timestamps.npy", flow.index.values)
    flow.index.to_series().to_csv(DATA_PROCESSED / "sd_flow_timestamps.csv", header=["timestamp"])

    _, rationale = resample_policy()
    manifest = {
        "start": flow.index[0].strftime("%Y-%m-%d %H:%M:%S"),
        "freq": RESAMPLE_RULE,
        "num_nodes": SD_NUM_SENSORS,
        "years": list(YEARS),
    }
    with open(DATA_PROCESSED / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    meta = {
        "num_sensors": SD_NUM_SENSORS,
        "years": list(YEARS),
        "resample_minutes": 15,
        "aggregation": "mean",
        "aggregation_rationale": rationale,
        "flow_shape": list(flow_values.shape),
        "missing_values": {
            "sensors_with_any_nan_pre_fill": report.sensors_with_any_nan,
            "total_nan_cells_pre_fill": report.total_nan_cells,
            "mean_nan_fraction_pre_fill": report.nan_fraction,
            "imputation": report.imputation,
        },
        "sensor_id_order": sd_meta["ID"].astype(str).tolist(),
    }
    with open(DATA_PROCESSED / "sd_flow_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def run_preprocessing(low_disk: bool = False) -> dict:
    sd_meta = load_sd_metadata()
    if low_disk:
        flow, report = build_flow_tensor_low_disk(sd_meta)
    else:
        flow, report = build_flow_tensor(sd_meta)
    return save_processed(sd_meta, flow, report)