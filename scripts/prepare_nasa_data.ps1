$ErrorActionPreference = "Stop"
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot "..\\.mplconfig"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

& "D:\PyAnaconda\Ana\envs\DL\python.exe" .\utils\matrix_generator.py `
  --raw-data-dir .\archive\data\data `
  --labels-path .\archive\labeled_anomalies.csv `
  --output-dir .\data\nasa_processed `
  --overwrite
