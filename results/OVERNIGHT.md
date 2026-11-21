# Overnight sim training: how to run and read it

Start (deploys the committed code to the GPU box, rebuilds the sim, launches; runs unattended there):

    ./scripts/go_overnight.sh

What runs: 4 rounds x (7 parcels x 2 episodes, each parcel dropped ANYWHERE on the table, bins pre-filled with labelled
parcels whose labels stay in view) + one 7-parcel stream per round. Cloud WALLX (configs/wallx.yaml backend: cloud).
About 10-15 min per episode, so a round takes ~3-4 h; results accumulate round by round.

Check progress:   ssh ADS-Admin@100.125.217.67 "Get-Content C:\x2\logs\hb_night.log"
Summary table:    ./scripts/night_report.sh   (writes results/night/summary.md)
Stop it:          on the box, kill the harness_batch / policy-space / manaenv python processes (only those).
