"""Add val_mae/val_rmse to existing results.json from saved best_model.pt (no retrain)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS = (
    "configs/temporal_only.yaml",
    "configs/stgnn_fixed.yaml",
    "configs/stgnn_adaptive.yaml",
    "configs/stgnn_hybrid.yaml",
)


def main() -> None:
    import sys

    sys.path.insert(0, str(REPO))
    from src.data.forecast_windows import ForecastDataModule
    from src.models.adjacency import AdjacencyConfig
    from src.models.forecasting import ForecastingModel, TemporalOnlyModel
    from src.train import build_model, evaluate, _loss_fn

    for rel in CONFIGS:
        cfg = yaml.safe_load((REPO / rel).read_text(encoding="utf-8"))
        run = cfg.get("run_name", Path(rel).stem)
        out = REPO / "outputs" / run
        ckpt_path = out / "best_model.pt"
        res_path = out / "results.json"
        if not ckpt_path.is_file() or not res_path.is_file():
            print("skip", run)
            continue
        dm = ForecastDataModule(
            REPO / cfg.get("processed_dir", "data/processed"),
            batch_size=int(cfg.get("batch_size", 32)),
            input_len=int(cfg.get("input_len", 8)),
            horizon=int(cfg.get("horizon", 4)),
        )
        device = torch.device(cfg.get("device", "cpu"))
        model_type = cfg.get("model_type", "stgnn")
        model = build_model(model_type, dm.num_nodes, dm.adj_raw, cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        loss_fn = _loss_fn(cfg.get("loss", "mae"))
        val = evaluate(model, dm.dataloaders()["val"], dm.stats, loss_fn, device)
        data = json.loads(res_path.read_text(encoding="utf-8"))
        data["val_mae"] = val["mae"]
        data["val_rmse"] = val["rmse"]
        data.setdefault("epochs", int(cfg.get("epochs", 20)))
        res_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(run, "val MAE", val["mae"], "test MAE", data.get("test_mae"))


if __name__ == "__main__":
    main()