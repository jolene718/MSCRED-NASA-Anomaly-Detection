$ErrorActionPreference = "Stop"
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot "..\\.mplconfig"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

& "D:\PyAnaconda\Ana\envs\DL\python.exe" .\main.py `
  --processed-dir .\data\nasa_processed `
  --epochs 15 `
  --patience 4 `
  --batch-size 32 `
  --checkpoint-path .\checkpoints\release\nasa_mscred_best.pth `
  --metrics-path .\outputs\release\train_metrics.json `
  --scores-path .\outputs\release\train_scores.csv `
  --history-path .\outputs\release\train_history.json `
  --history-plot-path .\outputs\release\train_history.png `
  --plots-dir .\outputs\release\channel_plots `
  --max-plots 81
