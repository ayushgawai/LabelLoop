#!/bin/zsh
# Summarize the overnight sim training: pulls every finished C:\x2\hb\n*_* run from the box and prints success per parcel/case.
#   ./scripts/night_report.sh            # writes results/night/<tag>/... and results/night/summary.md
BOX=ADS-Admin@100.125.217.67
cd "$(dirname "$0")/.."; mkdir -p results/night
ssh $BOX "Get-Content C:\x2\logs\hb_night.log -Tail 3"
scp -rq "$BOX:C:/x2/hb/n*_*" results/night/ 2>/dev/null
python3 - <<'PY'
import json, glob, os, collections
names = {40: "bhindi (blue)", 41: "iqbar_a (red)", 42: "miro (red)", 43: "mailer (blue)", 44: "iqbar_b (reject)", 45: "perfume (blue)", 46: "wallet (reject)", 47: "7-parcel stream"}
tot = collections.defaultdict(lambda: [0, 0])
rows = []
for f in sorted(glob.glob("results/night/n*_*/report.jsonl")):
    tag = f.split("/")[-2]; case = int(tag.split("_")[1])
    r = json.loads(open(f).read().splitlines()[-1])
    for e in r["episodes"]:
        ok = e["terms"][0]["hit"]; tot[case][0] += ok; tot[case][1] += 1
        rows.append(f"| {tag} | {names.get(case, case)} | {'sorted' if ok else 'failed: ' + str(e.get('failed_reason'))} | {e['wall_duration_s']:.0f} s |")
out = ["# Overnight sim training", "", "| case | parcel | sorted / episodes |", "|---|---|---|"]
out += [f"| {c} | {names.get(c, c)} | {s} / {n} |" for c, (s, n) in sorted(tot.items())]
out += ["", "| run | parcel | result | wall time |", "|---|---|---|---|"] + rows
open("results/night/summary.md", "w").write("\n".join(out) + "\n"); print("\n".join(out[:12 + len(tot)]))
PY
