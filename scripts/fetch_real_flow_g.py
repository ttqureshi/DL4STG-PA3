"""Download 2020/2021 HDF5 to G: data/raw one at a time and build real flow tensor."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

import os

REPO = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("DL4STG_RAW_DIR", REPO / "data" / "raw"))
WORK = Path(os.environ.get("DL4STG_WORK_DIR", REPO / "data" / "processed"))
PARTIAL = WORK / "_flow_2020_15min.pkl"
G_PROCESSED = REPO / "data" / "processed"


def kaggle_file(name: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "liuxu77/largest",
        "-f",
        name,
        "-p",
        str(RAW),
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)
    h5 = RAW / name
    zpath = RAW / f"{name}.zip"
    if not h5.is_file() and zpath.is_file():
        print(f"Extracting {zpath.name} ...")
        with zipfile.ZipFile(zpath) as zf:
            with zf.open(name) as src, open(h5, "wb") as dst:
                while chunk := src.read(8 * 1024 * 1024):
                    dst.write(chunk)
        zpath.unlink(missing_ok=True)
    if not h5.is_file():
        raise FileNotFoundError(h5)


def main() -> int:
    sys.path.insert(0, str(REPO))
    os.environ.setdefault("DL4STG_RAW_DIR", str(RAW))
    os.environ.setdefault("DL4STG_PROCESSED_DIR", str(WORK))
    from src.data.paths import DATA_PROCESSED, raw_h5_path
    from src.data.preprocess_sd import (
        MissingValueReport,
        _process_year_h5,
        load_sd_metadata,
        save_processed,
    )

    sd_meta = load_sd_metadata()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    h2020 = raw_h5_path(2020)
    if not PARTIAL.is_file():
        if not h2020.is_file():
            print("Downloading ca_his_raw_2020.h5 ...")
            kaggle_file("ca_his_raw_2020.h5")
        print("Resampling 2020 ...")
        flow2020, rep2020 = _process_year_h5(sd_meta, 2020)
        flow2020.to_pickle(PARTIAL)
        h2020.unlink(missing_ok=True)
        print(f"  Cached {PARTIAL.name}; deleted 2020 HDF5.")
    else:
        flow2020 = pd.read_pickle(PARTIAL)
        rep2020 = None

    h2021 = raw_h5_path(2021)
    if not h2021.is_file():
        print("Downloading ca_his_raw_2021.h5 ...")
        kaggle_file("ca_his_raw_2021.h5")
    print("Resampling 2021 ...")
    flow2021, rep2021 = _process_year_h5(sd_meta, 2021)
    h2021.unlink(missing_ok=True)

    PARTIAL.unlink(missing_ok=True)
    combined = pd.concat([flow2020, flow2021], axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    if rep2020 is None:
        from src.data.preprocess_sd import missing_report

        rep2020 = MissingValueReport(0, 0, 0.0, "loaded from partial cache")
    merged = MissingValueReport(
        sensors_with_any_nan=max(rep2020.sensors_with_any_nan, rep2021.sensors_with_any_nan),
        total_nan_cells=rep2020.total_nan_cells + rep2021.total_nan_cells,
        nan_fraction=float((rep2020.nan_fraction + rep2021.nan_fraction) / 2),
        imputation=rep2021.imputation,
    )
    (REPO / "data/processed/synthetic_note.json").unlink(missing_ok=True)
    meta = save_processed(sd_meta, combined, merged)
    print("Real flow saved:", meta["flow_shape"], "->", DATA_PROCESSED)
    G_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name in (
        "flow_15min.npy",
        "sd_flow_2020_2021_15min.npy",
        "manifest.json",
        "sd_flow_metadata.json",
        "sd_meta.csv",
        "timestamps.npy",
        "sd_flow_timestamps.csv",
    ):
        src = DATA_PROCESSED / name
        if src.is_file():
            shutil.copy2(src, G_PROCESSED / name)
    (G_PROCESSED / "synthetic_note.json").unlink(missing_ok=True)
    print("Copied artifacts to", G_PROCESSED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())