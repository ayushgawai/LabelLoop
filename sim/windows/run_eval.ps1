# Detached via launch.ps1: zero-action smoke eval against 127.0.0.1:8001
$env:PYTHONIOENCODING='utf-8'; $env:OMNI_KIT_ACCEPT_EULA='YES'
$env:MANAENV_ASSET_ROOT='C:\x2\x2real-assets'
Set-Location C:\x2\x2real
& C:\x2\venv\Scripts\python.exe manaenv/scripts/x2real_eval.py --board artixon_arm6a_tabletop --env artixon_arm6a_G_pick_fruits_into_basket_id_env --num-episodes 1 --num-parallel-envs 1 --policy-config policy/x2real_ps@v1.0.0.yaml --server-address 127.0.0.1 --server-port 8001 --enable_cameras
"EXIT $LASTEXITCODE"
