"""Plot predicted vs actual traffic for three sensors on the test split (Q6)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.forecast_windows import ForecastDataModule, NormStats
from src.train import build_model, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "report" / "figures" / "q6"
DEFAULT_SENSORS = (0, 100, 200)


def _repo_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


@torch.no_grad()
def collect_horizon_one(
    model: torch.nn.Module,
    dm: ForecastDataModule,
    device: torch.device,
    sample_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return denormalized (pred_h1, actual_h1) each shape (len(sample_ends), N)."""
    model.eval()
    flow = dm.flow_raw
    stats = dm.stats
    preds: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    for t in sample_ends:
        t = int(t)
        x = flow[t - dm.input_len + 1 : t + 1]
        x_norm = stats.normalize(x)
        x_t = torch.from_numpy(x_norm).float().unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(x_t).cpu().numpy()[0, 0, :, 0]
        pred_u = stats.denormalize(torch.from_numpy(pred)).numpy()
        actual = flow[t + 1, :]
        preds.append(pred_u)
        acts.append(actual)
    return np.stack(preds), np.stack(acts)


def plot_three_sensors(
    timestamps,
    sample_ends: np.ndarray,
    pred: np.ndarray,
    actual: np.ndarray,
    sensor_indices: tuple[int, ...],
    out_path: Path,
    title: str,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t_axis = timestamps[sample_ends + 1]
    n = len(sensor_indices)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, idx in zip(axes, sensor_indices):
        ax.plot(t_axis, actual[:, idx], label="Actual", linewidth=1, alpha=0.9)
        ax.plot(t_axis, pred[:, idx], label="Predicted (h=1)", linewidth=1, alpha=0.9)
        ax.set_ylabel(f"Flow (idx {idx})")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (test split)")
    fig.suptitle(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _pick_test_slice(dm: ForecastDataModule, max_points: int = 672) -> np.ndarray:
    ends = dm.indices["test"]
    if len(ends) <= max_points:
        return ends
    start = len(ends) // 4
    return ends[start : start + max_points]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stgnn_hybrid.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override best_model.pt path.")
    parser.add_argument("--sensors", type=int, nargs=3, default=list(DEFAULT_SENSORS))
    parser.add_argument("--max-points", type=int, default=672, help="Test windows to plot (~1 week).")
    args = parser.parse_args()

    cfg_path = _repo_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))
    processed = _repo_path(cfg.get("processed_dir", "data/processed"))
    dm = ForecastDataModule(
        processed,
        batch_size=int(cfg.get("batch_size", 32)),
        input_len=int(cfg.get("input_len", 8)),
        horizon=int(cfg.get("horizon", 4)),
    )
    device = torch.device(cfg.get("device", "cpu"))
    model = build_model(
        cfg.get("model_type", "stgnn"),
        dm.num_nodes,
        dm.adj_raw,
        cfg,
    ).to(device)
    run_name = cfg.get("run_name", "model")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else REPO_ROOT / "outputs" / run_name / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    sample_ends = _pick_test_slice(dm, args.max_points)
    pred, actual = collect_horizon_one(model, dm, device, sample_ends)
    out = FIG_DIR / f"predictions_{run_name}_three_sensors.png"
    plot_three_sensors(
        dm.timestamps,
        sample_ends,
        pred,
        actual,
        tuple(args.sensors),
        out,
        f"Test split — 1-step forecast ({run_name})",
    )
    meta = {
        "run_name": run_name,
        "config": str(cfg_path.relative_to(REPO_ROOT)),
        "sensor_indices": list(args.sensors),
        "num_points": int(len(sample_ends)),
        "figure": str(out.relative_to(REPO_ROOT)),
    }
    (FIG_DIR / f"predictions_{run_name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()