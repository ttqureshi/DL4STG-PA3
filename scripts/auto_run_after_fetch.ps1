# Waits for fetch_real_flow_g.py to finish, then Q3 figures + full 20-epoch train + report.
$repo = "g:\My Drive\MS AI\Semester_3\DLforTimeSpaceGraph\PAs\DL4STG-PA3"
$log = Join-Path $repo "outputs\auto_run.log"
New-Item -ItemType Directory -Force -Path (Join-Path $repo "outputs") | Out-Null
function Log($m) { "$(Get-Date -Format o) $m" | Tee-Object -FilePath $log -Append }

Log "Waiting for fetch_real_flow_g (PID $args[0])..."
Wait-Process -Id $args[0] -ErrorAction SilentlyContinue
Start-Sleep -Seconds 5

Set-Location $repo
if (-not (Test-Path "data\processed\flow_15min.npy")) {
    Log "ERROR: flow_15min.npy missing after fetch"
    exit 1
}
if (Test-Path "data\processed\synthetic_note.json") {
    Log "WARNING: synthetic_note still present"
}

Log "Q3 graph + figures (flow already in data/processed)..."
python -c "from src.data.graph_sd import run_graph_construction; from src.data.visualize_q3 import run_all_plots; run_graph_construction(); run_all_plots()" 2>&1 | Tee-Object -FilePath $log -Append

Log "Training all models (20 epochs, CPU)..."
python -m src.evaluate_all --force 2>&1 | Tee-Object -FilePath $log -Append

Log "Plots and graph analysis..."
python -m src.plot_predictions --config configs/stgnn_hybrid.yaml 2>&1 | Tee-Object -FilePath $log -Append
python -m src.analyze_graph --config configs/stgnn_adaptive.yaml --also-hybrid 2>&1 | Tee-Object -FilePath $log -Append

Log "Patch report..."
python -c "from scripts.run_full_pipeline import patch_q6_table, patch_reproduce_hardware; from pathlib import Path; p=Path('.'); patch_q6_table((p/'outputs/comparison_table.md').read_text(encoding='utf-8')); patch_reproduce_hardware()"
python report/build_report.py 2>&1 | Tee-Object -FilePath $log -Append
Log "AUTO-RUN COMPLETE"