"""Question 2: ring-road slowdown dynamics and graph-layer fits."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

N_RING = 40


def ring_index(i: int) -> int:
    return i % N_RING


def step_backward(x: np.ndarray) -> np.ndarray:
    """x_i(t+1) = x_{i+1}(t)  <=> roll by -1 on the vector."""
    return np.roll(x, -1)


def step_forward(x: np.ndarray) -> np.ndarray:
    """x_i(t+1) = x_{i-1}(t)  <=> roll by +1."""
    return np.roll(x, 1)


def layer_symmetric(x: np.ndarray, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    left = np.roll(x, 1)
    right = np.roll(x, -1)
    return a * x + b * (left + right)


def layer_directional(
    x: np.ndarray, a: float = 0.0, w_minus: float = 0.0, w_plus: float = 1.0
) -> np.ndarray:
    left = np.roll(x, 1)
    right = np.roll(x, -1)
    return a * x + w_minus * left + w_plus * right


def layer_temporal(
    x_curr: np.ndarray, x_prev: np.ndarray, a: float = 0.0, b: float = 1.0, c: float = -1.0
) -> np.ndarray:
    left = np.roll(x_curr, 1)
    right = np.roll(x_curr, -1)
    return a * x_curr + b * (left + right) + c * x_prev


def isolated_spike() -> Tuple[np.ndarray, np.ndarray]:
    x0 = np.zeros(N_RING)
    x0[20] = 1.0
    return x0, step_backward(x0)


def fit_isolated_spike_ab() -> Tuple[float, float, Dict[str, np.ndarray]]:
    """
    Minimise SSE for symmetric layer on isolated spike at sensor 20.
    Analytic minimum: a=0, b=1.
    """
    x0, x1 = isolated_spike()
    a, b = 0.0, 1.0
    x_hat = layer_symmetric(x0, a, b)
    return a, b, {"x0": x0, "y_true": x1, "y_hat": x_hat}


def verify_directional_exact(n_trials: int = 20) -> float:
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(n_trials):
        x = rng.random(N_RING)
        pred = layer_directional(x, a=0.0, w_minus=0.0, w_plus=1.0)
        true = step_backward(x)
        max_err = max(max_err, float(np.max(np.abs(pred - true))))
    return max_err


def rollout_symmetric(x0: np.ndarray, steps: int, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    seq = np.zeros((steps + 1, N_RING))
    seq[0] = x0
    for t in range(steps):
        seq[t + 1] = layer_symmetric(seq[t], a, b)
    return seq


def rollout_true(x0: np.ndarray, steps: int, forward: bool = False) -> np.ndarray:
    seq = np.zeros((steps + 1, N_RING))
    seq[0] = x0
    step_fn = step_forward if forward else step_backward
    for t in range(steps):
        seq[t + 1] = step_fn(seq[t])
    return seq


def verify_temporal_backward() -> float:
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(30):
        x_prev = rng.random(N_RING)
        x_curr = step_backward(x_prev)
        pred = layer_temporal(x_curr, x_prev, a=0.0, b=1.0, c=-1.0)
        true = step_backward(x_curr)
        max_err = max(max_err, float(np.max(np.abs(pred - true))))
    return max_err


def verify_temporal_forward_same_coeffs() -> float:
    rng = np.random.default_rng(1)
    max_err = 0.0
    for _ in range(30):
        x_prev = rng.random(N_RING)
        x_curr = step_forward(x_prev)
        pred = layer_temporal(x_curr, x_prev, a=0.0, b=1.0, c=-1.0)
        true = step_forward(x_curr)
        max_err = max(max_err, float(np.max(np.abs(pred - true))))
    return max_err


def save_heatmaps(out_dir: Path, steps: int = 20) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    i = np.arange(N_RING, dtype=float)
    x0 = np.exp(-((i - 20.0) ** 2) / 8.0)
    true_seq = rollout_true(x0, steps, forward=False)
    model_seq = rollout_symmetric(x0, steps, a=0.0, b=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    vmax = float(max(true_seq.max(), model_seq.max()))
    vmin = 0.0
    ims = []
    for ax, data, title in zip(
        axes,
        [true_seq, model_seq],
        ["True backward motion", "Symmetric layer (a=0, b=1)"],
    ):
        im = ax.imshow(data, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Sensor index")
        ims.append(im)
    axes[0].set_ylabel("Time step")
    fig.colorbar(ims[0], ax=axes.ravel().tolist(), shrink=0.85, label="Slowdown")
    fig.tight_layout()
    path = out_dir / "q2a_bump_heatmaps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path