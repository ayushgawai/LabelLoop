#!/bin/zsh
# One command to start the overnight sim training on the GPU box (runs unattended there; safe to close this terminal).
#   ./scripts/go_overnight.sh            # deploys the committed HEAD, rebuilds the sim (full-length cases), launches the batch
# Results land on the box in C:\x2\hb\n<round>_<case>\ (report.jsonl, runlog.jsonl, videos.txt); progress: C:\x2\logs\hb_night.log
# Cases 40-46 = each of the 7 parcels dropped ANYWHERE on the table with pre-filled bins (labels visible); 47 = 7-parcel stream.
set -e
BOX=ADS-Admin@100.125.217.67
cd "$(dirname "$0")/.."
T=$(mktemp -d)/ll.tar
git archive --format=tar HEAD -- labelloop configs policies scripts sim tests > "$T"
scp -q "$T" "$BOX:C:/x2/llh.tar"
ssh $BOX "cd C:\x2\labelloop; tar -xf C:\x2\llh.tar; (Get-Content configs\wallx.yaml) -replace '^backend: local','backend: cloud' | Set-Content configs\wallx.yaml; (Get-Content configs\task_qr_sort.yaml) -replace '^calib_test: true','calib_test: false' | Set-Content configs\task_qr_sort.yaml"
ssh $BOX "powershell -NoProfile -ExecutionPolicy Bypass -File C:\x2\scripts\build_props.ps1" | grep -E "usda|cases|DONE"
# 4 rounds x (7 single-parcel position cases x 2 episodes) + 1 stream per round; each line is restartable, results per round
L=""
for r in 1 2 3 4; do for c in 40 41 42 43 44 45 46; do L="$L'$c 2 n${r}_$c',"; done; L="$L'47 1 n${r}_47',"; done
ssh $BOX "Set-Content C:\x2\hbatch.txt ${L%,}; (Get-Content C:\x2\hbatch.txt).Count; powershell -NoProfile -File C:\x2\scripts\launch.ps1 C:\x2\scripts\harness_batch.ps1 C:\x2\logs\hb_night.log"
echo "overnight batch launched; watch: ssh $BOX \"Get-Content C:\x2\logs\hb_night.log\""
