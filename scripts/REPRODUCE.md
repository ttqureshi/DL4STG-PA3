# Reproduction guide (Q1–Q7)

## Hardware

Windows 11, Intel64 Family 6 Model 154 Stepping 4, GenuineIntel

Runs in this repository default to `device: cpu` in all YAML configs. GPU training is supported by setting `device: cuda` in the config files.

## Environment

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Random seeds

| Component | Seed |
|-----------|------|
| Training (`src/train.py`) | 42 (config field `seed`) |
| Q1–Q2 demos | See `src/q1_q2/run_all.py` |

Seeds touch Python `random`, NumPy, and PyTorch CPU.

## Hyperparameters (Q4–Q7 models)

| Parameter | Value |
|-----------|-------|
| `input_len` | 8 |
| `horizon` | 4 |
| `hidden_dim` | 10 (embeddings) / 32 (model) |
| `embed_dim` | 10 (adaptive & hybrid) |
| `top_k` | 20 (adaptive & hybrid) |
| `hybrid_alpha` | 0.5 |
| `batch_size` | 32 |
| `epochs` | 20 |
| `lr` | 0.001 |
| `loss` | MAE (`mae`) |
| `optimizer` | Adam |

Configs: `configs/temporal_only.yaml`, `configs/stgnn_fixed.yaml`, `configs/stgnn_adaptive.yaml`, `configs/stgnn_hybrid.yaml`.

## Data

1. Place LargeST CA files in `data/raw/` (see root `README.md`).
2. Q3 pipeline:

```bash
python -m src.data.download --verify-only
python -m src.data.run_q3
```

## Training and evaluation

```bash
# Single model
python -m src.train --config configs/stgnn_fixed.yaml

# All four models + comparison table
python -m src.evaluate_all

# Aggregate only (no training)
python -m src.evaluate_all --skip-train
```

## Figures and analysis (Q6–Q7)

```bash
python -m src.plot_predictions --config configs/stgnn_hybrid.yaml
python -m src.analyze_graph --config configs/stgnn_adaptive.yaml --also-hybrid
```

## Report

```bash
python -m src.q1_q2.run_all    # Q1–Q2 artifacts if needed
python report/build_report.py
```

Edit the GitHub URL placeholder at the top of `report/report.md` before export.

## Q1–Q2 only

```bash
python -m src.q1_q2.run_all
```

Uses `requirements-q1q2.txt` if you prefer a minimal env.