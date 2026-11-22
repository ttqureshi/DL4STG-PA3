# DL4STG — Programming Assignment 3

**Repository:** https://github.com/ttqureshi/DL4STG-PA3

Spatio-temporal graph learning on **LargeST San Diego** (716 sensors).

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. HDF5 input needs `tables` or `h5py`.

## LargeST raw data (required for Q3+)

The Kaggle dataset is **not** redistributed in this repo.

1. [LargeST on Kaggle](https://www.kaggle.com/datasets/liuxu77/largest) — download and unzip into `data/raw/`, **or**
2. Kaggle API: place `kaggle.json` in `~/.kaggle/`, then  
   `python -m src.data.download --kaggle`

**Required files in `data/raw/`:**

- `ca_meta.csv`
- `ca_rn_adj.npy`
- `ca_his_raw_2020.h5`
- `ca_his_raw_2021.h5`

Verify only:

```bash
python -m src.data.download --verify-only
```

## Question 3 — Run pipeline

```bash
python -m src.data.run_q3
```

Writes to `data/processed/` (flow, adjacency, stats, manifest) and figures under `report/figures/q3/`. Report: `report/sections/q3.md`.

## Reproduction (full assignment)

See [scripts/REPRODUCE.md](scripts/REPRODUCE.md).

- **Q1–Q2:** `python -m src.q1_q2.run_all`
- **Q4–Q5 training:**  
  `python -m src.train --config configs/temporal_only.yaml` (and `stgnn_fixed`, `stgnn_adaptive`, `stgnn_hybrid`)
- **Q6 batch eval / tables:** `python -m src.evaluate_all`
- **Prediction plots:** `python -m src.plot_predictions`
- **Learned-graph analysis:** `python -m src.analyze_graph --also-hybrid`
- **Merge report:** `python report/build_report.py`

## Layout

```
src/q1_q2/         # Q1–Q2 numpy experiments
src/data/          # Q3 download, preprocess, graph, windows
src/models/        # STGNN, adjacency modes
configs/           # training YAML
data/raw/          # user-provided LargeST files
data/processed/    # generated tensors
report/sections/   # q1.md … q7.md → report/report.md
scripts/           # REPRODUCE.md
```

**LMS report (PDF):** [`report/25280024_DL4STG_PA3.pdf`](report/25280024_DL4STG_PA3.pdf) — rebuild with `report/latex/compile.ps1`.