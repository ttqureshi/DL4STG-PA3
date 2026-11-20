"""End-to-end: Q3 preprocess, train all models, plots, patch report sections."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


def patch_q6_table(table_md: str) -> None:
    path = REPO / "report" / "sections" / "q6.md"
    text = path.read_text(encoding="utf-8")
    pattern = r"(\| Model \| Adjacency.*?\n\|[^\n]*\n)(?:\|[^\n]*\n)*"
    replacement = table_md.rstrip() + "\n\n"
    if not re.search(pattern, text, flags=re.S):
        raise RuntimeError("Could not locate Q6 table placeholder in q6.md")
    text = re.sub(pattern, replacement, text, count=1, flags=re.S)
    path.write_text(text, encoding="utf-8")


def patch_reproduce_hardware() -> None:
    path = REPO / "scripts" / "REPRODUCE.md"
    text = path.read_text(encoding="utf-8")
    hw = f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}"
    text = text.replace(
        "<!-- HARDWARE: e.g. CPU model, RAM, GPU if used; fill before submission -->",
        hw,
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    run([sys.executable, "-m", "src.data.run_q3"])
    run([sys.executable, "-m", "src.evaluate_all"])
    run([sys.executable, "-m", "src.plot_predictions", "--config", "configs/stgnn_hybrid.yaml"])
    run([sys.executable, "-m", "src.plot_predictions", "--config", "configs/stgnn_fixed.yaml"])
    run([sys.executable, "-m", "src.analyze_graph", "--config", "configs/stgnn_adaptive.yaml", "--also-hybrid"])

    table = (REPO / "outputs" / "comparison_table.md").read_text(encoding="utf-8")
    patch_q6_table(table)
    patch_reproduce_hardware()
    run([sys.executable, str(REPO / "report" / "build_report.py")])

    summary = {
        "comparison": json.loads((REPO / "outputs" / "comparison.json").read_text(encoding="utf-8")),
        "graph_analysis": json.loads((REPO / "outputs" / "graph_analysis.json").read_text(encoding="utf-8"))
        if (REPO / "outputs" / "graph_analysis.json").exists()
        else None,
    }
    (REPO / "outputs" / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()