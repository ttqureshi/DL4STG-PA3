"""Train STGNN / temporal-only models for Q4-Q5."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data.forecast_windows import ForecastDataModule, NormStats
from src.models.adjacency import AdjacencyConfig
from src.models.forecasting import ForecastingModel, TemporalOnlyModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(model_type: str, num_nodes: int, fixed_adj: np.ndarray, cfg: Dict[str, Any]) -> nn.Module:
    hidden_dim = int(cfg.get("hidden_dim", 32))
    input_len = int(cfg.get("input_len", 8))
    horizon = int(cfg.get("horizon", 4))
    if model_type == "temporal_only":
        return TemporalOnlyModel(hidden_dim=hidden_dim, input_len=input_len, horizon=horizon)
    adj_cfg = AdjacencyConfig(
        mode=cfg.get("adjacency_mode", "fixed"),
        hybrid_alpha=float(cfg.get("hybrid_alpha", 0.5)),
        embed_dim=int(cfg.get("embed_dim", 10)),
        top_k=cfg.get("top_k"),
    )
    return ForecastingModel(
        num_nodes=num_nodes,
        fixed_adj=fixed_adj,
        adjacency_config=adj_cfg,
        hidden_dim=hidden_dim,
        input_len=input_len,
        horizon=horizon,
    )


def _loss_fn(name: str) -> nn.Module:
    return nn.L1Loss() if name.lower() == "mae" else nn.MSELoss()


@torch.no_grad()
def evaluate(model: nn.Module, loader, stats: NormStats, loss_fn: nn.Module, device: torch.device):
    model.eval()
    total_loss = total_mae = total_mse = n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += loss_fn(pred, y).item()
        pred_u, y_u = stats.denormalize(pred), stats.denormalize(y)
        total_mae += torch.mean(torch.abs(pred_u - y_u)).item()
        total_mse += torch.mean((pred_u - y_u) ** 2).item()
        n_batches += 1
    if n_batches == 0:
        return {"loss": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    mse = total_mse / n_batches
    return {"loss": total_loss / n_batches, "mae": total_mae / n_batches, "rmse": float(np.sqrt(mse))}


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--processed-dir", type=str, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))
    repo_root = Path(__file__).resolve().parents[1]
    processed_dir = Path(args.processed_dir or cfg.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = repo_root / processed_dir
    dm = ForecastDataModule(
        processed_dir,
        batch_size=int(cfg.get("batch_size", 32)),
        input_len=int(cfg.get("input_len", 8)),
        horizon=int(cfg.get("horizon", 4)),
    )
    loaders = dm.dataloaders()
    model_type = cfg.get("model_type", "stgnn")
    device = torch.device(cfg.get("device", "cpu"))
    model = build_model(model_type, dm.num_nodes, dm.adj_raw, cfg).to(device)
    loss_fn = _loss_fn(cfg.get("loss", "mae"))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    out_dir = repo_root / cfg.get("output_dir", "outputs") / cfg.get("run_name", model_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    best_val, best_path = float("inf"), out_dir / "best_model.pt"
    for epoch in range(1, int(cfg.get("epochs", 20)) + 1):
        tr = train_one_epoch(model, loaders["train"], optimizer, loss_fn, device)
        val = evaluate(model, loaders["val"], dm.stats, loss_fn, device)
        print(f"epoch {epoch:03d} | train={tr:.4f} val MAE={val['mae']:.4f} RMSE={val['rmse']:.4f}")
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "norm_stats": {"mean": dm.stats.mean, "std": dm.stats.std},
                },
                best_path,
            )
    train_time = time.perf_counter() - t0
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test = evaluate(model, loaders["test"], dm.stats, loss_fn, device)
    results = {
        "test_mae": test["mae"],
        "test_rmse": test["rmse"],
        "train_seconds": train_time,
        "seed": int(cfg.get("seed", 42)),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if isinstance(model, ForecastingModel) and model.graph.adaptive is not None:
        np.save(out_dir / "learned_adjacency.npy", model.graph.adaptive().detach().cpu().numpy())
    print(f"test MAE={test['mae']:.4f} RMSE={test['rmse']:.4f} ({train_time:.1f}s) -> {out_dir}")


if __name__ == "__main__":
    main()