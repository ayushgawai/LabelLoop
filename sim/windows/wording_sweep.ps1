# Level-3 wording test: each line of C:\x2\wording.txt = "<case> <episodes> <instruction...>".
# LabelLoop runs in passthrough with that instruction; logs/reports saved per line under C:\x2\wording\.
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force C:\x2\wording | Out-Null
$i = 0
foreach ($line in Get-Content C:\x2\wording.txt) {
    if (-not $line.Trim()) { continue }
    $i++; $case, $n, $instr = $line.Trim() -split '\s+', 3
    "[$(Get-Date -Format HH:mm:ss)] $i case$case x$n : $instr"
    $p = (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue).OwningProcess
    if ($p) { Stop-Process -Id $p -Force; Start-Sleep 3 }
    Set-Content C:\x2\passthrough.txt $instr
    Remove-Item C:\x2\labelloop\runs\sim.jsonl -ErrorAction SilentlyContinue
    C:\x2\scripts\launch.ps1 C:\x2\scripts\run_labelloop_server.ps1 "C:\x2\wording\server_$i.log" | Out-Null
    while (-not (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue)) { Start-Sleep 2 }
    $envname = "artixon_arm6a_G_labelloop_qr_sort_case${case}_id_env"
    Set-Content C:\x2\eval_args.txt "labelloop_tabletop $envname $n"
    $out = "C:\x2\x2real\outputs\x2real_eval\labelloop\benchmark\$envname"
    Remove-Item "$out\evaluation_report.jsonl" -ErrorAction SilentlyContinue
    & powershell -NoProfile -ExecutionPolicy Bypass -File C:\x2\scripts\run_eval_env.ps1 *> "C:\x2\wording\eval_$i.log"
    Copy-Item "$out\evaluation_report.jsonl" "C:\x2\wording\report_$i.jsonl" -ErrorAction SilentlyContinue
    Copy-Item C:\x2\labelloop\runs\sim.jsonl "C:\x2\wording\runlog_$i.jsonl" -ErrorAction SilentlyContinue
    Set-Content "C:\x2\wording\instr_$i.txt" $line
}
$p = (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue).OwningProcess
if ($p) { Stop-Process -Id $p -Force }
Remove-Item C:\x2\passthrough.txt -ErrorAction SilentlyContinue
"SWEEP DONE"
