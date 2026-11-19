# Detached via launch.ps1: policy-space serve (hold-position template, port 8001)
$env:PYTHONIOENCODING='utf-8'
Set-Location C:\x2\x2real\3rd\policy_space
& C:\x2\venv\Scripts\policy-space.exe serve policies/templates/bimanual_ee/deploy.yaml
"EXIT $LASTEXITCODE"
