"""
Download LargeST CA files from Kaggle into data/raw/.

Manual fallback: https://www.kaggle.com/datasets/liuxu77/largest
Place ca_meta.csv, ca_rn_adj.npy, and ca_his_raw_YYYY.h5 in data/raw/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from src.data.paths import (
    CA_META,
    CA_RN_ADJ,
    DATA_RAW,
    RAW_H5_PATTERN,
    YEARS,
    ensure_dirs,
    raw_h5_path,
)


REQUIRED_FILES = [CA_META, CA_RN_ADJ] + [raw_h5_path(y) for y in YEARS]


def missing_files() -> list[Path]:
    return [p for p in REQUIRED_FILES if not p.is_file()]


def print_manual_instructions() -> None:
    print(
        """
LargeST raw data is not present locally.

1. Open: https://www.kaggle.com/datasets/liuxu77/largest
2. Download the dataset (Sign in / accept license if prompted).
3. Either:
   a) Unzip into: data/raw/
   b) Or run: python -m src.data.download --kaggle  (requires Kaggle API)

Required files in data/raw/:
  - ca_meta.csv
  - ca_rn_adj.npy
  - ca_his_raw_2020.h5
  - ca_his_raw_2021.h5

See README.md for Kaggle API setup (kaggle.json in ~/.kaggle/).
""".strip()
    )


def flatten_from_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            name = Path(member).name
            if not name:
                continue
            target = dest / name
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def download_via_kaggle(dest: Path) -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Install kaggle: pip install kaggle", file=sys.stderr)
        return False

    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files("liuxu77/largest", path=str(dest), unzip=True)
    # Kaggle may create a nested zip named largest.zip
    for z in dest.glob("*.zip"):
        flatten_from_zip(z, dest)
    return True


def verify_raw_data() -> bool:
    missing = missing_files()
    if missing:
        print("Missing:", *[str(p.relative_to(DATA_RAW.parent.parent)) for p in missing], sep="\n  ")
        return False
    print("All required raw files are present under data/raw/.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download LargeST data for PA3 Q3")
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Download via Kaggle API (needs credentials)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only check that required files exist",
    )
    args = parser.parse_args()
    ensure_dirs()

    if args.verify_only:
        return 0 if verify_raw_data() else 1

    if missing_files() and args.kaggle:
        print("Downloading liuxu77/largest ...")
        if not download_via_kaggle(DATA_RAW):
            return 1

    if missing_files():
        print_manual_instructions()
        return 1

    return 0 if verify_raw_data() else 1


if __name__ == "__main__":
    raise SystemExit(main())