# DL4STG PA3 — submission review sheet

**Repo:** https://github.com/ttqureshi/DL4STG-PA3  
**Last updated:** 2026-11-22 (aligned with `DL4STG_PA3_final_submission_review.md` on Desktop)

Use this page to review the whole assignment in one pass before LMS upload.

---

## 1. What was completed

### Data & pipeline
| Item | Status | Where |
|------|--------|--------|
| Real LargeST SD flow (2020–2021, 15 min) | Done | `data/processed/flow_15min.npy` shape `[70176, 716]` |
| Synthetic fallback | **Not used** for final runs | `synthetic_note.json` removed |
| Fetch script (HDF5 one-at-a-time) | Done | `scripts/fetch_real_flow_g.py` |
| Q3 graph + figures | Done | `data/processed/sd_adjacency_stats.json`, `report/figures/q3/` |

### Training (real flow, **20 epochs**, seed 42, CPU)
| Model | Val MAE | Test MAE | Checkpoint |
|-------|---------|----------|------------|
| `temporal_only` | 18.96 | **21.15** (best) | `outputs/temporal_only/` |
| `stgnn_adaptive` | 22.65 | 26.82 | `outputs/stgnn_adaptive/` |
| `stgnn_hybrid` | 24.49 | 28.16 | `outputs/stgnn_hybrid/` |
| `stgnn_fixed` | 53.89 | 60.37 | `outputs/stgnn_fixed/` |

Full table: `outputs/comparison_table.md` · Val metrics backfilled: `scripts/backfill_val_metrics.py`

### Theory & code fixes (your action plan)
| Item | Change |
|------|--------|
| **Q1(c)** | Mean **can** test majority slowing ($m_1 > 0.5$); tables updated in `report/sections/q1.md` |
| **Q2(a)** | Spike SSE → **$a=0$, $b=1/2$** (not $b=1$); `src/q1_q2/q2_ring.py` + heatmaps regenerated |
| **Q3** | Missing counts, edge count **17,319**, sensor **0 / 1114091** neighbourhood example |
| **Q6 metric** | Renamed **top-$k$ overlap / $k$** (precision-at-$k$), not Jaccard; `src/analyze_graph.py` |
| **Q6(d)** | Critical discussion (no ST-GNN beats temporal; hybrid **not** best vs adaptive) |
| **Validation metrics** | `val_mae` / `val_rmse` in `results.json` and comparison table |
| **Epochs** | All `configs/*.yaml` → **epochs: 20** (matches reported runs) |

### Reports
| Artifact | Path |
|----------|------|
| Markdown sections Q1–Q7 | `report/sections/*.md` |
| Merged markdown | `report/report.md` |
| **NeurIPS LaTeX (main submit PDF)** | **`report/DL4STG_PA3.pdf`** |
| LaTeX sources | `report/latex/main.tex`, `body.tex` |
| Rebuild PDF | `report/latex/compile.ps1` |

### Git
| Item | Status |
|------|--------|
| First big commit | **`223fcae`** (real flow pipeline + NeurIPS sources) |
| Checklist / final fixes | **Pending commit** after you review this sheet (see §2) |
| Pushed to GitHub | Run `git push origin master` after final commit |

### Verification log (2026-11-22, automated)
| Check | Result |
|-------|--------|
| `data/processed/flow_15min.npy` exists | Yes |
| `synthetic_note.json` | **Absent** |
| Flow shape | `[70176, 716]` |
| `outputs/*/results.json` | Nov 20, 2025 PM — real-data MAE (temporal test 21.15) |
| `val_mae` in results | Present after `backfill_val_metrics.py` |
| Config `epochs` | 20 in all four YAMLs |

---

## 2. Where **you** should review / decide

### Must review before upload
- [ ] Open **`report/DL4STG_PA3.pdf`** end-to-end (7 pages): figures, table, Q2 heatmap caption ($b=0.5$), no stale synthetic wording.
- [ ] Confirm **name/title** on PDF is correct (`report/latex/main.tex` → `\author{...}`).
- [ ] **LMS:** upload `report/DL4STG_PA3.pdf` (PDF is gitignored by design).
- [ ] **GitHub:** `git push origin master` after you are happy (and optionally a second commit for checklist fixes).

### Judgment calls (only you can sign off)
| Topic | Current choice | Your call? |
|-------|----------------|------------|
| **15-min aggregation** | **Mean** of 3×5-min bins (LargeST `process_ca_his` style; justified in Q3) | **You:** confirm against official PA3 PDF/handout. PeMS 5-min values are interval counts; assignment may allow mean *or* sum—we document mean. Rebuilding with **sum** needs re-fetch/preprocess + full retrain (~hours). |
| **Generative AI disclosure** | Short statement in Q7 + PDF back matter | Edit wording if your course wants more detail (tools, prompts). |
| **Hybrid discussion** | Honest: hybrid **worse** than adaptive here | OK unless you want different narrative. |
| **Mean vs assignment wording on Q1(c)** | Matches corrected math in plan | Skim `report/sections/q1.md` table once. |

### Optional cleanup (not required for grade)
- [ ] Delete duplicate `data/processed/sd_flow_2020_2021_15min.npy` if identical to `flow_15min.npy` (~192 MB on Google Drive).
- [ ] Empty **Google Drive Trash** if you need quota back after LargeST downloads.

### Not in GitHub (by `.gitignore` — intentional)
- `data/raw/**`, `data/processed/**`, `outputs/**`, `*.pdf`, `*.pt`

Reviewers/cloners need to run fetch + train locally; see `scripts/REPRODUCE.md`.

---

## 3. Quick reproducibility commands

```powershell
cd "G:\My Drive\MS AI\Semester_3\DLforTimeSpaceGraph\PAs\DL4STG-PA3"

# Q1–Q2 only
python -m src.q1_q2.run_all

# If processed flow already exists — refresh val columns in table
python scripts/backfill_val_metrics.py
python -m src.evaluate_all --skip-train

# Reports
python report/build_report.py
cd report\latex; .\compile.ps1
```

---

## 4. Known limitations (stated in report)

- **Temporal-only beats all ST-GNNs** on this real SD split — shallow single-layer GCN + static $A$ may be too weak.
- Learned adjacency **low overlap** with road graph (~5% top-$k$) — predictive edges ≠ map edges.
- **Static graph** — no time-of-day or incident-specific topology.

---

## 5. Checklist mapping (your desktop plan)

| Plan section | Addressed? |
|--------------|------------|
| §0 Real vs synthetic | Real flow documented; synthetic path only in code for fallback |
| §1 Q1(c), Q2(a) | Fixed in prose + code + Q2 figure |
| §2 Q3(a–c) | Stats from JSON files in report |
| §2 Q4 val+test | Table includes both |
| §2 Q5 adjacency | `report/sections/q5.md` |
| §2 Q6 table, plots, discussion | Updated; hybrid not claimed best |
| §2 Q6(b) Jaccard name | Renamed to top-$k$ overlap |
| §3 epochs / hardware / AI | 20 epochs; i5-1235U / 8 GB / Win11; disclosure added |
| §3 Push consistency | **Your input:** commit + push after review |

---

## 6. Final sequence (from Desktop review)

1. Read this file + open **`report/DL4STG_PA3.pdf`** (all 7 pages).
2. Confirm real-data claim (§1 verification table).
3. Mean vs sum: your sign-off (§2).
4. `git commit` + `git push` — verify GitHub shows latest `report/sections/*`.
5. Upload PDF to LMS.

**Do not** retrain for better MAE before deadline—consistency beats tuning.

---

## 7. If something looks wrong

| Symptom | Action |
|---------|--------|
| PDF metrics ≠ `outputs/comparison_table.md` | Run backfill + `evaluate_all --skip-train`, recompile PDF |
| GitHub README still old | Push; check `report/sections/q6.md` on remote |
| Instructor asks for validation-only table | Copy val columns from Q6 table in PDF |

**Contact in report:** GitHub URL on page 1 of PDF.