"""Question 1: GNN messages and aggregation (affine, encoding, sum/mean/max)."""

from __future__ import annotations

import itertools
from typing import Callable, Dict, Tuple

import numpy as np

Trend = int  # -1, 0, or +1


def affine_aggregate(neighbors: np.ndarray, w: float, c: float) -> float:
    """Sum of affine messages before rho."""
    n_v = len(neighbors)
    s_v = int(neighbors.sum())
    return w * s_v + n_v * c


def example_same_nv_sv_different_steady() -> Tuple[np.ndarray, np.ndarray]:
    """Two neighbourhoods: n_v=5, S_v=0, different counts of steady (+0) neighbours."""
    hood_a = np.array([-1, -1, 0, 1, 1], dtype=int)  # n_0 = 1
    hood_b = np.array([-1, 0, 0, 0, 1], dtype=int)  # n_0 = 3
    assert len(hood_a) == len(hood_b)
    assert hood_a.sum() == hood_b.sum() == 0
    return hood_a, hood_b


# --- Part (b): place-value encoding -----------------------------------------

def place_value_phis(k_max: int) -> Dict[int, int]:
    """Base B = K+1 so counts up to K occupy separate 'digits'."""
    base = k_max + 1
    return {-1: 1, 0: base, 1: base**2}


def encode_counts(
    n_minus: int, n_zero: int, n_plus: int, k_max: int
) -> int:
    b = k_max + 1
    return n_minus + b * n_zero + (b**2) * n_plus


def decode_message_sum(message_sum: int, k_max: int) -> Tuple[int, int, int]:
    base = k_max + 1
    n_minus = message_sum % base
    message_sum //= base
    n_zero = message_sum % base
    message_sum //= base
    n_plus = message_sum % base
    return n_minus, n_zero, n_plus


def non_steady_phis() -> Dict[int, int]:
    """Scalar messages: sum equals count of non-steady neighbours."""
    return {-1: 1, 0: 0, 1: 1}


def verify_place_value_encoding(k_max: int = 8) -> bool:
    """Check every (n_-, n_0, n_+) with 1 <= n_v <= K and trends in {-1,0,1}."""
    phis = place_value_phis(k_max)
    seen: Dict[int, Tuple[int, int, int]] = {}
    for n_v in range(1, k_max + 1):
        for n_minus in range(n_v + 1):
            for n_zero in range(n_v - n_minus + 1):
                n_plus = n_v - n_minus - n_zero
                total = (
                    n_minus * phis[-1]
                    + n_zero * phis[0]
                    + n_plus * phis[1]
                )
                triple = (n_minus, n_zero, n_plus)
                if total in seen and seen[total] != triple:
                    return False
                seen[total] = triple
                if decode_message_sum(total, k_max) != triple:
                    return False
    return True


# --- Part (c): one-hot messages and aggregators ------------------------------

ONE_HOT = {
    -1: np.array([1.0, 0.0, 0.0]),
    0: np.array([0.0, 1.0, 0.0]),
    1: np.array([0.0, 0.0, 1.0]),
}


def vec_messages(neighbors: np.ndarray) -> np.ndarray:
    return np.stack([ONE_HOT[int(x)] for x in neighbors])


def agg_sum(neighbors: np.ndarray) -> np.ndarray:
    return vec_messages(neighbors).sum(axis=0)


def agg_mean(neighbors: np.ndarray) -> np.ndarray:
    return vec_messages(neighbors).mean(axis=0)


def agg_max(neighbors: np.ndarray) -> np.ndarray:
    return vec_messages(neighbors).max(axis=0)


def counts_from_sum(vec: np.ndarray) -> Tuple[int, int, int]:
    return int(vec[0]), int(vec[1]), int(vec[2])


def task_steady_count(neighbors: np.ndarray) -> int:
    return int((neighbors == 0).sum())


def task_majority_slowing(neighbors: np.ndarray) -> bool:
    n_v = len(neighbors)
    n_minus = int((neighbors == -1).sum())
    return n_minus > n_v / 2


def task_any_slowing(neighbors: np.ndarray) -> bool:
    return bool((neighbors == -1).any())


if __name__ == "__main__":
    ok = verify_place_value_encoding(8)
    print("K=8 place-value encoding unique:", ok)
    a, b = example_same_nv_sv_different_steady()
    for w, c in [(1.0, 0.0), (2.0, -1.0)]:
        print(f"w={w}, c={c}: agg(a)={affine_aggregate(a, w, c)}, agg(b)={affine_aggregate(b, w, c)}")