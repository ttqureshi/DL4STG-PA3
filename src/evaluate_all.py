"""Run all four forecasting configs and aggregate test metrics (Q6)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    "configs/temporal_only.yaml",
    "configs/stgnn_fixed.yaml",
    "configs/stgnn_adaptive.yaml",
    "configs/stgnn_hybrid.yaml",
)


def _load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _output_dir(cfg: Dict[str, Any]) -> Path:
    base = REPO_ROOT / cfg.get("output_dir", "outputs")
    return base / cfg.get("run_name", cfg.get("model_type", "run"))


def _ensure_trained(config_path: Path, force: bool) -> Path:
    cfg = _load_config(config_path)
    out_dir = _output_dir(cfg)
    results_path = out_dir / "results.json"
    if results_path.exists() and not force:
        print(f"skip train (exists): {results_path}")
        return results_path
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if not results_path.exists():
        raise FileNotFoundError(f"Training finished but {results_path} missing")
    return results_path


def _collect_rows(config_paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in config_paths:
        cfg = _load_config(path)
        run_name = cfg.get("run_name", path.stem)
        out_dir = _output_dir(cfg)
        results_path = out_dir / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing {results_path}; run without --skip-missing or train first.")
        metrics = json.loads(results_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_name": run_name,
                "config": str(path.relative_to(REPO_ROOT)),
                "model_type": cfg.get("model_type", ""),
                "adjacency_mode": cfg.get("adjacency_mode", "—"),
                "val_mae": metrics.get("val_mae"),
                "val_rmse": metrics.get("val_rmse"),
                "test_mae": metrics.get("test_mae"),
                "test_rmse": metrics.get("test_rmse"),
                "train_seconds": metrics.get("train_seconds"),
                "epochs": metrics.get("epochs", cfg.get("epochs")),
                "seed": metrics.get("seed", cfg.get("seed")),
            }
        )
    return rows


def _markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        "| Model | Adjacency | Val MAE | Val RMSE | Test MAE | Test RMSE | Train time (s) | Seed |\n"
        "|-------|-----------|---------|----------|----------|-----------|----------------|------|\n"
    )
    lines = [header]
    for r in sorted(rows, key=lambda x: (x["test_mae"] is None, x["test_mae"] or 0)):
        vm, vr = r.get("val_mae"), r.get("val_rmse")
        vmae = f"{vm:.4f}" if vm is not None else "—"
        vrmse = f"{vr:.4f}" if vr is not None else "—"
        lines.append(
            f"| `{r['run_name']}` | {r['adjacency_mode']} | "
            f"{vmae} | {vrmse} | "
            f"{r['test_mae']:.4f} | {r['test_rmse']:.4f} | "
            f"{r['train_seconds']:.1f} | {r['seed']} |\n"
        )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train missing runs and build metrics comparison table.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[str(REPO_ROOT / p) for p in DEFAULT_CONFIGS],
        help="YAML config paths (default: all four PA models).",
    )
    parser.add_argument("--force", action="store_true", help="Re-train even if results.json exists.")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only aggregate existing results.json files (do not launch training).",
    )
    args = parser.parse_args()
    config_paths = [Path(c).resolve() for c in args.configs]

    if not args.skip_train:
        for path in config_paths:
            _ensure_trained(path, force=args.force)

    rows = _collect_rows(config_paths)
    table_md = _markdown_table(rows)
    out_base = REPO_ROOT / "outputs"
    out_base.mkdir(parents=True, exist_ok=True)
    (out_base / "comparison_table.md").write_text(table_md, encoding="utf-8")
    (out_base / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(table_md)
    print(f"Wrote {out_base / 'comparison_table.md'} and comparison.json")


if __name__ == "__main__":
    main()