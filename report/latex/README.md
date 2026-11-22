# NeurIPS-style PDF report

```powershell
cd report/latex
.\compile.ps1
```

Outputs:

- `report/latex/DL4STG_PA3.pdf` (canonical build)
- `report/DL4STG_PA3.pdf` (copy for LMS upload)

Edit `body.tex` for prose; `main.tex` for title/abstract. Requires MiKTeX (`pdflatex`) and local `neurips_2024.sty`.