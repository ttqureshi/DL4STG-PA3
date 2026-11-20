"""Canonical paths for LargeST San Diego PA3 pipeline."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = Path(os.environ.get("DL4STG_RAW_DIR", REPO_ROOT / "data" / "raw"))
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
FIGURES_DIR = REPO_ROOT / "report" / "figures" / "q3"
REPORT_SECTION = REPO_ROOT / "report" / "sections" / "q3.md"

# Expected filenames after manual or API download (Kaggle archive layout)
CA_META = DATA_RAW / "ca_meta.csv"
CA_RN_ADJ = DATA_RAW / "ca_rn_adj.npy"
RAW_H5_PATTERN = "ca_his_raw_{year}.h5"

SD_NUM_SENSORS = 716
SD_DISTRICT = 11
YEARS = (2020, 2021)
RESAMPLE_RULE = "15min"


def raw_h5_path(year: int) -> Path:
    return DATA_RAW / RAW_H5_PATTERN.format(year=year)


def ensure_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)