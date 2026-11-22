# LabelLoop demo plan (B-board, Nov 22) — built only on what worked on the real robot

## The one fact that decides everything
WALLX (wall-x 2610-dualhead-v2, the only model on the LAN server) reliably does ONE skill: **"put the box into the <colour> bin"**,
with a real bin on the table and a parcel it can grasp. Every real success today used that skill; pushes, tape squares, turning,
showing a face to the camera, directional arm commands all failed. Without bins on the table WALLX barely commits to a grasp.
Our harness wins by making that one skill reliable and wrapping it in perception, routing, hard safety rules and verification.

## Table setup (props are ours)
- Three bins along the far edge: blue (back left), red (back right), black (right edge). Use the shoe-box **lids** (low trays:
  WALLX grabbed the tall shoe boxes when no arm was locked) or the shoe boxes themselves (5/6 worked with the idle-arm lock).
- Parcels **standing on their narrow side, long axis front-to-back**, ~15 cm in front of their bin, a hand's width from the gripper
  (left-to-right boxes get knocked over; the gripper opens wider than the old 97.5 mm note: fully open = 1.526 in WALLX units).
- Labels: big QR (6-8 cm) taped flat on top so the head camera reads them (the 35-45 px QR codes today were physically unreadable).
  Until then the harness uses the demo code (`--assume-code`, says so in the log).

## What we show (cases), with the recipe that already worked
| # | Case | Command | Real result today |
|---|------|---------|-------------------|
| 1 | Read label -> red lane -> right arm sorts it, verifies, homes | `scripts/run_trial.py --scenario scenarios/s2_sort_red.json` | full success in 46 s (s2 run 3) |
| 2 | Same into blue with the left arm | `scenarios/s2_sort_blue.json` | atomic version a2: 2/3 |
| 3 | Atomic WALLX skill, idle arm locked | `scripts/atomic.py --recipe scenarios/a3_right_place_red.json` | 3/3 |
| 4 | Hard rule: a bin moves -> robot holds | (watchdog in every harness run) | caught every bin grab today |
| 5 | Verification: drop not confirmed -> retry; confirmed -> home at once | (harness) | fixed and confirmed in s2 run 3 |
| 6 | Teach & repeat (organisers allow scripted steps): replay Ayush's controller demo | `scenarios/r1/r2/r3_replay_*.json` | replays run; box must start where it was |

## What to say (the harness story)
1. Perception: head camera reads the label (QR, then printed ZIP OCR), finds the parcel even when it was there at the start.
2. Planning: label -> lane -> acting arm -> one short WALLX sub-goal ("put the box into the red bin").
3. Hard rules WALLX cannot break: the idle arm is locked (it cannot grab a bin), bins can be hidden from the image WALLX sees,
   a bin that moves latches HOLD.
4. Verification: the parcel left its spot + the target bin changed (3 calls) -> success -> home; else retry.
5. Tooling: live/replay dashboard (joint chains from IK, story mode, what WALLX saw), every trial recorded with video, scenario
   recipes reproducible with one command, teach-and-repeat from the robot's own bags, atomic WALLX tester, reviews (Fable/Astra).

## Morning checklist (30 min)
1. Bins (lids) on their marks; re-measure `profiles.real.bin_rois` from a snapshot if they moved (`run_trial.py X --snapshot-only`).
2. Run case 3 twice (atomic, proves WALLX + lock), then case 1 three times. Keep the best video (runs/scenarios/*/run_N/demo.mp4).
3. If homing errors appear: `scripts/rehome_master_arms.sh` (controller arms parked bent).
4. Dashboard: `.venv/bin/python -m labelloop.dashboard.server --port 8095`.
