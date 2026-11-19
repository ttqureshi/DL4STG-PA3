"""Run Q1 verification and Q2 experiments (prints + optional figures)."""

from pathlib import Path

from src.q1_q2.q1_aggregation import (
    example_same_nv_sv_different_steady,
    verify_place_value_encoding,
)
from src.q1_q2.q2_ring import (
    fit_isolated_spike_ab,
    save_heatmaps,
    verify_directional_exact,
    verify_temporal_backward,
    verify_temporal_forward_same_coeffs,
)


def main() -> None:
    print("=== Q1(b): place-value encoding K=8 ===")
    print("Unique decodable:", verify_place_value_encoding(8))

    a_hood, b_hood = example_same_nv_sv_different_steady()
    print("=== Q1(a): same n_v, S_v, different steady counts ===")
    print("Hood A trends:", a_hood.tolist(), "n_0=", (a_hood == 0).sum())
    print("Hood B trends:", b_hood.tolist(), "n_0=", (b_hood == 0).sum())

    print("=== Q2(a): isolated spike fit ===")
    a, b, det = fit_isolated_spike_ab()
    sse = float(((det["y_true"] - det["y_hat"]) ** 2).sum())
    print(f"a={a}, b={b}, SSE={sse}")

    print("=== Q2(b): directional layer ===")
    print("Max |error| over random inputs:", verify_directional_exact())

    print("=== Q2(c): temporal coefficients ===")
    print("Backward motion max error:", verify_temporal_backward())
    print("Forward motion max error (same a,b,c):", verify_temporal_forward_same_coeffs())

    repo = Path(__file__).resolve().parents[2]
    fig_dir = repo / "report" / "figures"
    save_heatmaps(fig_dir)
    print("Wrote heatmaps to", fig_dir / "q2a_bump_heatmaps.png")


if __name__ == "__main__":
    main()