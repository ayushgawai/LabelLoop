# Detached via launch.ps1: LabelLoop as the Policy Space server on port 8001.
# Phase 1 passthrough: create C:\x2\passthrough.txt with one instruction; delete it for the full harness.
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONPATH='C:\x2\labelloop'
if (Test-Path C:\x2\passthrough.txt) { $env:LABELLOOP_PASSTHROUGH = (Get-Content C:\x2\passthrough.txt -Raw).Trim() }
Set-Location C:\x2\labelloop
& .venv\Scripts\policy-space.exe serve policies\labelloop\deploy.yaml --port 8001
"EXIT $LASTEXITCODE"
