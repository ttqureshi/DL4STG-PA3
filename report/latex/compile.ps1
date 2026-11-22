$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pdf = Join-Path $here "DL4STG_PA3.pdf"
$pdflatex = "C:\Users\PMLS\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
Push-Location $here
try {
    & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
    & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
    Copy-Item -Force (Join-Path $here "main.pdf") $pdf
    Write-Host "Wrote $pdf"
} finally {
    Pop-Location
}