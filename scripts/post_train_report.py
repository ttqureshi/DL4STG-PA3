"""After evaluate_all: prediction plots, graph analysis, patch Q6 table, rebuild report."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_full_pipeline", REPO / "scripts" / "run_full_pipeline.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
patch_q6_table = _mod.patch_q6_table
patch_reproduce_hardware = _mod.patch_reproduce_hardware


def patch_q6_heading_and_notes(rows: list[dict]) -> None:
    path = REPO / "report" / "sections" / "q6.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "### Expected comparisons (fill from your run)",
        "### Results (LargeST SD flow, 20 epochs, seed 42)",
    )
    by_mae = sorted(rows, key=lambda r: r["test_mae"] or 1e9)
    best = by_mae[0]
    temporal = next((r for r in rows if r["run_name"] == "temporal_only"), None)
    lines = []
    lines.append(
        f"\n**Summary:** Best test MAE is `{best['run_name']}` ({best['test_mae']:.4f}). "
    )
    if temporal:
        graph_better = [
            r["run_name"]
            for r in rows
            if r["run_name"] != "temporal_only" and (r["test_mae"] or 1e9) < (temporal["test_mae"] or 1e9)
        ]
        if graph_better:
            lines.append(
                f"Graph models beating the temporal baseline: {', '.join(f'`{n}`' for n in graph_better)}. "
            )
        else:
            lines.append("No graph model beats `temporal_only` on this split. ")
    import re

    text = re.sub(r"\n\*\*Summary:\*\*[^\n]*\n", "\n", text)
    text = text.replace(
        "**Interpretation prompts**",
        "".join(lines) + "\n\n**Interpretation prompts**",
        1,
    )
    path.write_text(text, encoding="utf-8")


def patch_q7_metrics() -> None:
    ga_path = REPO / "outputs" / "graph_analysis.json"
    if not ga_path.is_file():
        return
    data = json.loads(ga_path.read_text(encoding="utf-8"))
    lines = ["\n### Measured values (this run)\n\n"]
    lines.append("| Run | Row entropy (mean) | Corr. on fixed edges | Top-k Jaccard vs \\(A\\) |\n")
    lines.append("|-----|-------------------|----------------------|-------------------------|\n")
    for run in data.get("runs", []):
        lines.append(
            f"| `{run['run_name']}` | {run['learned_row_entropy_mean']:.3f} | "
            f"{run['correlation_on_fixed_edges']:.4f} | "
            f"{run['mean_topk_jaccard_with_fixed_support']:.4f} |\n"
        )
    block = "".join(lines)
    q7 = REPO / "report" / "sections" / "q7.md"
    text = q7.read_text(encoding="utf-8")
    import re

    text = re.sub(
        r"\n### Measured values \(this run\)\n\n\| Run \|.*?\n\|-----.*?\n(?:\| `[^`]+` \|[^\n]+\n)+",
        "",
        text,
        flags=re.DOTALL,
    )
    marker = "## Q7(b) — Quantitative alignment with \\(A\\)"
    if marker in text:
        text = text.replace(marker, block + marker, 1)
        q7.write_text(text, encoding="utf-8")


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


def main() -> None:
    comparison = REPO / "outputs" / "comparison_table.md"
    if not comparison.is_file():
        raise FileNotFoundError(f"Missing {comparison}; run python -m src.evaluate_all first")

    run([sys.executable, "-m", "src.plot_predictions", "--config", "configs/stgnn_hybrid.yaml"])
    run([sys.executable, "-m", "src.plot_predictions", "--config", "configs/stgnn_fixed.yaml"])
    run(
        [
            sys.executable,
            "-m",
            "src.analyze_graph",
            "--config",
            "configs/stgnn_adaptive.yaml",
            "--also-hybrid",
        ]
    )

    patch_q6_table(comparison.read_text(encoding="utf-8"))
    rows = json.loads((REPO / "outputs" / "comparison.json").read_text(encoding="utf-8"))
    patch_q6_heading_and_notes(rows)
    patch_q7_metrics()
    patch_reproduce_hardware()
    run([sys.executable, str(REPO / "report" / "build_report.py")])

    summary = {
        "comparison": json.loads((REPO / "outputs" / "comparison.json").read_text(encoding="utf-8")),
        "graph_analysis": json.loads((REPO / "outputs" / "graph_analysis.json").read_text(encoding="utf-8"))
        if (REPO / "outputs" / "graph_analysis.json").exists()
        else None,
    }
    (REPO / "outputs" / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Post-train report pipeline complete.")


if __name__ == "__main__":
    main()