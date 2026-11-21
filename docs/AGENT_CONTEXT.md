# Shared context for helper agents (Codex CLI, OpenAI, Anthropic, Meta Muse)

Read this first. It is the running thread for every helper agent; append a dated line under "Log" when you finish a task.

## What this is
LabelLoop: a harness around the organizers' WALLX VLA (X Square Robot hackathon, BRIGHT 2026, SF, Nov 21-22; competition ends Nov 22 16:00).
Dual-arm 6-DOF desktop robot (teletwin_desktop, G gripper 97.5 mm). Parcels (flat retail boxes) lie on a green table; one QR label
(payload `420/<ZIP5>/<seq>`) on top or bottom. Route by ZIP3: 940/941/950/951 -> blue bin, 900-902 -> red bin, else/no label -> black bin.
Official rules: WALLX must generate the actions; our harness plans sub-goals, reads labels, verifies, retries. Everything stays on the
workbench (black bin on the table, no floor drops). 8 min per case. Score = sum of 6 B-board case scores x difficulty K.

## How it runs at the event
- Robot host: xr@192.168.1.21 (only Ayush/Claude touch it). WALLX (organizer, LAN): ws://192.168.1.10:33677, 0.07 s/call, no auth.
- Our relay on Ayush's Mac: `.venv/bin/python -m labelloop.robots.event_server --host 0.0.0.0 --port 8001 --profile real` (fast, WALLX-only
  mode on by default; `--hold` = camera capture without motion; `--save-images DIR`).
- Real calibration + measured bin regions: `configs/task_qr_sort.yaml` -> `profiles.real`. Real table 1.17 m x 0.94 m (46 x 37 in).
- Real head frames: runs/event_images/<timestamp>/<n>_front.jpg (+ wrist images, state JSON).

## Code map
- labelloop/harness.py (perception -> state machine -> scripted/WALLX replies; apply_profile), labelloop/brain/state_machine.py,
  labelloop/perception/{scene,qr}.py, labelloop/look.py (camera model, IK), labelloop/robots/event_server.py (relay), labelloop/wallx_client.py.
- Tests: `.venv/bin/python -m pytest -q` must stay green (~170 tests).

## Rules for helper agents
- Code + tests only. Never touch the robot, never call WALLX, never read/print `.env` or keys. Don't commit unless told; Claude (Ayush's
  session) reviews and commits.

## Log
- 2026-11-21 11:50 Trial 1 on the real robot: harness never left IDLE: the parcel was already on the table at start (became the empty
  reference) and sat half under the left gripper (QR hidden). Fixes requested from Codex (see task below in git history).
