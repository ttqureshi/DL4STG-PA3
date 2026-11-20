"""Run full PA3 Question 3 pipeline."""
from __future__ import annotations

import argparse
import sys

from src.data.download import missing_files, print_manual_instructions, verify_raw_data
from src.data.graph_sd import run_graph_construction
from src.data.paths import ensure_dirs
from src.data.preprocess_sd import run_preprocessing
from src.data.synthetic_flow import run_synthetic_preprocessing
from src.data.visualize_q3 import run_all_plots


def main() -> int:
    parser = argparse.ArgumentParser(description="PA3 Q3: LargeST San Diego preparation")
    parser.add_argument(
        "--skip-download-check",
        action="store_true",
        help="Run even if verify fails (not recommended)",
    )
    parser.add_argument(
        "--synthetic-flow",
        action="store_true",
        help="Use synthetic flow (real SD meta/adjacency) when HDF5 is unavailable",
    )
    args = parser.parse_args()
    ensure_dirs()

    if not args.synthetic_flow:
        if missing_files():
            print_manual_instructions()
            if not args.skip_download_check:
                return 1

        if not verify_raw_data() and not args.skip_download_check:
            return 1

    print("Preprocessing San Diego flow (2020–2021, 15-min)...")
    if args.synthetic_flow:
        meta = run_synthetic_preprocessing()
    else:
        meta = run_preprocessing()
    print(f"  Saved flow tensor shape: {meta['flow_shape']}")

    print("Building adjacency submatrix...")
    adj, stats = run_graph_construction()
    print(
        f"  Nodes: {stats.num_nodes}, nonzero (off-diag): {stats.num_nonzero_edges}, "
        f"symmetric: {stats.is_symmetric}, mean degree: {stats.mean_degree:.2f}"
    )

    print("Generating figures...")
    figs = run_all_plots()
    for p in figs:
        print(f"  {p}")

    print("Done. See report/sections/q3.md for write-up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())