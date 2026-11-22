$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoReport = Split-Path -Parent $here
$submitName = "25280024_DL4STG_PA3.pdf"
$pdflatex = "C:\Users\PMLS\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
Push-Location $here
try {
    & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
    & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
    $built = Join-Path $here "main.pdf"
    Copy-Item -Force $built (Join-Path $here $submitName)
    Copy-Item -Force $built (Join-Path $repoReport $submitName)
    Write-Host "Wrote $here\$submitName and $repoReport\$submitName"
} finally {
    Pop-Location
}