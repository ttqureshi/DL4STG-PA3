"""Visualize learned adjacency and compare to fixed road-network graph (Q6b / Q7)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.forecast_windows import load_processed_arrays
from src.models.adjacency import symmetric_normalized_adjacency

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "report" / "figures" / "q7"


def _repo_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def _fixed_support(adj: np.ndarray) -> np.ndarray:
    a = adj.copy()
    np.fill_diagonal(a, 0.0)
    return (a > 0).astype(np.float32)


def topk_overlap(learned: np.ndarray, fixed_mask: np.ndarray, k: int) -> Dict[str, float]:
    """Per-row top-k overlap between learned weights and fixed edges."""
    n = learned.shape[0]
    overlaps: list[float] = []
    for i in range(n):
        l_idx = set(np.argsort(-learned[i])[:k].tolist())
        f_idx = set(np.where(fixed_mask[i] > 0)[0].tolist())
        if not l_idx:
            continue
        overlaps.append(len(l_idx & f_idx) / k)
    return {
        "top_k": k,
        "mean_topk_jaccard_with_fixed_support": float(np.mean(overlaps)) if overlaps else float("nan"),
    }


def correlation_on_fixed_edges(learned: np.ndarray, fixed_adj: np.ndarray) -> float:
    mask = _fixed_support(fixed_adj).astype(bool)
    if not mask.any():
        return float("nan")
    return float(np.corrcoef(learned[mask], fixed_adj[mask])[0, 1])


def load_learned_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    e1 = state["graph.adaptive.E1"].cpu().numpy()
    e2 = state["graph.adaptive.E2"].cpu().numpy()
    logits = np.maximum(0, e1 @ e2.T)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def plot_heatmaps(fixed: np.ndarray, learned: np.ndarray, out_prefix: Path, max_n: int = 128) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n = min(max_n, fixed.shape[0])
    saved: list[str] = []
    for name, arr in (("learned_adjacency", learned), ("fixed_normalized", fixed)):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(arr[:n, :n], cmap="magma", aspect="auto", interpolation="nearest")
        ax.set_title(f"{name} ({n}×{n} crop)")
        ax.set_xlabel("Sensor j")
        ax.set_ylabel("Sensor i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        path = out_prefix.parent / f"{out_prefix.name}_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path.relative_to(REPO_ROOT)))
    diff = learned[:n, :n] - fixed[:n, :n]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(diff, cmap="coolwarm", aspect="auto", interpolation="nearest")
    ax.set_title("Learned − fixed (normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = out_prefix.parent / f"{out_prefix.name}_difference.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(str(path.relative_to(REPO_ROOT)))
    return saved


def analyze_run(run_name: str, processed_dir: Path, k: int) -> Dict[str, Any]:
    _, adj_raw, _ = load_processed_arrays(processed_dir)
    out_dir = REPO_ROOT / "outputs" / run_name
    learned_path = out_dir / "learned_adjacency.npy"
    if learned_path.exists():
        learned = np.load(learned_path).astype(np.float64)
    else:
        learned = load_learned_matrix(out_dir / "best_model.pt")

    fixed_hat = symmetric_normalized_adjacency(adj_raw).cpu().numpy()
    fixed_mask = _fixed_support(adj_raw)
    metrics: Dict[str, Any] = {
        "run_name": run_name,
        "learned_density": float(np.mean(learned > 1e-6)),
        "learned_row_entropy_mean": float(
            np.mean(-np.sum(learned * np.log(learned + 1e-12), axis=1))
        ),
        "correlation_on_fixed_edges": correlation_on_fixed_edges(learned, adj_raw),
        "fixed_mean_degree": float(fixed_mask.sum(axis=1).mean()),
    }
    metrics.update(topk_overlap(learned, fixed_mask, k))

    fig_prefix = FIG_DIR / run_name
    metrics["figures"] = plot_heatmaps(fixed_hat, learned, fig_prefix)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stgnn_adaptive.yaml")
    parser.add_argument("--also-hybrid", action="store_true", help="Also analyze stgnn_hybrid.")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    cfg_path = _repo_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    processed = _repo_path(cfg.get("processed_dir", "data/processed"))
    runs = [cfg.get("run_name", "stgnn_adaptive")]
    if args.also_hybrid:
        runs.append("stgnn_hybrid")

    payload = {"runs": [analyze_run(name, processed, args.top_k) for name in runs]}
    out_json = REPO_ROOT / "outputs" / "graph_analysis.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()