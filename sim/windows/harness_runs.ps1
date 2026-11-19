# Full LabelLoop harness runs. Each line of C:\x2\harness_runs.txt = "<case> <episodes> [tag]".
# Uses policy/labelloop_ps@v1.0.0.yaml (1280x720 to the harness). Outputs under C:\x2\hruns\<tag or caseN>\.
$ErrorActionPreference = 'Continue'
Remove-Item C:\x2\passthrough.txt -ErrorAction SilentlyContinue
foreach ($line in Get-Content C:\x2\harness_runs.txt) {
    if (-not $line.Trim()) { continue }
    $case, $n, $tag = $line.Trim() -split '\s+', 3
    if (-not $tag) { $tag = "case$case" }
    $dir = "C:\x2\hruns\$tag"; New-Item -ItemType Directory -Force $dir | Out-Null
    "[$(Get-Date -Format HH:mm:ss)] $tag case$case x$n"
    $p = (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue).OwningProcess
    if ($p) { Stop-Process -Id $p -Force; Start-Sleep 3 }
    Remove-Item C:\x2\labelloop\runs\sim.jsonl -ErrorAction SilentlyContinue
    C:\x2\scripts\launch.ps1 C:\x2\scripts\run_labelloop_server.ps1 "$dir\server.log" | Out-Null
    while (-not (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue)) { Start-Sleep 2 }
    $envname = "artixon_arm6a_G_labelloop_qr_sort_case${case}_id_env"
    Set-Content C:\x2\eval_args.txt "labelloop_tabletop $envname $n policy/labelloop_ps@v1.0.0.yaml"
    $out = "C:\x2\x2real\outputs\x2real_eval\labelloop\benchmark\$envname"
    Remove-Item "$out\evaluation_report.jsonl" -ErrorAction SilentlyContinue
    $t0 = Get-Date
    & powershell -NoProfile -ExecutionPolicy Bypass -File C:\x2\scripts\run_eval_env.ps1 *> "$dir\eval.log"
    Copy-Item "$out\evaluation_report.jsonl" "$dir\report.jsonl" -ErrorAction SilentlyContinue
    Copy-Item C:\x2\labelloop\runs\sim.jsonl "$dir\runlog.jsonl" -ErrorAction SilentlyContinue
    Get-ChildItem "$out\videos\combined\episode_*.mp4" | Where-Object { $_.LastWriteTime -gt $t0 } | Copy-Item -Destination $dir
}
$p = (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue).OwningProcess
if ($p) { Stop-Process -Id $p -Force }
"RUNS DONE"
