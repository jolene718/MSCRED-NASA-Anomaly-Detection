$ErrorActionPreference = "Stop"
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot "..\\.mplconfig"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

& "D:\PyAnaconda\Ana\envs\DL\python.exe" .\utils\evaluate.py `
  --checkpoint-path .\checkpoints\release\nasa_mscred_best.pth `
  --score-topk-ratio 0.02 `
  --threshold-quantile 0.98 `
  --threshold-std-factor 1.5 `
  --smooth-window 1 `
  --metrics-path .\outputs\release\eval_metrics.json `
  --scores-path .\outputs\release\eval_scores.csv `
  --plots-dir .\outputs\release\eval_plots `
  --max-plots 81
