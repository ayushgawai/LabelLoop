# Task (Codex, thread "LabelLoop work"): hard rules for WALLX + stand-up grasp for wide flat parcels

Context: read docs/AGENT_CONTEXT.md, docs/research/CAPABILITIES.md, results/agent/REPORT_*.md (real-robot trial reports).
WALLX (vision-language-action model) only sees the camera images + ONE short instruction per call. On the real robot it keeps
grabbing the shoe-box BINS (blue/red/black, all open boxes) instead of the parcel, and single-arm pickups slide the wide flat
parcels (Miro 216x165x38 mm, Bhindi 165x102x38 mm; gripper opening 97.5 mm; their 38 mm thickness is easy for one gripper).
WALLX must generate all motion (event rule): our harness may only choose instructions, edit the images WALLX sees, and hold.

## Part 1: staged visibility (hard rule: WALLX cannot grab a bin it cannot see)
Split the event_fast delivery into WALLX sub-goals with per-phase image masking of the FRONT image forwarded to WALLX
(labelloop/perception/overlay.py already paints bins out: mode "mask"; the relay applies it in event_server._overlay_front):
- PICK: instruction "pick up the box" (template key PICK, `{arm}` / `{where}` allowed). ALL bins painted out (table green).
- CARRY/PLACE: once the parcel is held, instruction = the PLACE template; ONLY the target bin visible (others painted out).
- Phase switch PICK -> PLACE: the acting gripper opened then closed on something (reuse the gp phase logic of
  StateMachine._bad_grasp; real gripper reads ~0 closed-empty, ~0.3-0.85 when open; make thresholds config keys) for 2 calls,
  OR the parcel left its table spot (presence vs the reference near the parcel box drops). Timeout / retries as for PLACE.
- Make it config-driven: `event_fast_stage: true` (default false until tested on the robot), mask per phase chosen by the
  harness (expose the current phase so the relay picks the mask; keep `--overlay` working as before).

## Part 2: STAND sub-goal for wide flat parcels (before PICK)
- Wide = the parcel's smaller side is wider than the gripper: use labelloop/perception/table_grid.py (TableGrid, merged from
  Pratham) to measure the object box in table cm when it works on the frame, else a pixel fallback (smaller side > 130 px on
  the 1280x720 head image; real Miro box 313 x 219 px, Bhindi 281 x 148 px).
- STAND (WALLX, all bins painted out): template key STAND, default "use the {fix} arm to hold the {fix_side} side of the box
  still and use the {lift} arm to push the {lift_side} side up until the box stands on its long thin edge". {fix} = the arm
  on the parcel's far side from the lifting arm; default lift = the lane arm. Success: the object box footprint in the head
  image shrinks by >= 40 % vs its STAND start (it stands up) for 2 calls; tries 2, max_calls config; then PICK with
  "pick up the standing box from the top with the {arm} arm".
- Then PLACE as in Part 1.

## Rules
- Code + tests only (no robot, no WALLX, no keys). Edit labelloop/brain/state_machine.py, labelloop/harness.py,
  labelloop/robots/event_server.py, labelloop/perception/overlay.py, configs/task_qr_sort.yaml, tests/. Keep every existing
  behaviour unchanged when the new keys are off.
- `.venv/bin/python -m pytest -q` must pass (~200 tests). Add tests for the phase switch, the per-phase masks and STAND.
- SAVE PROGRESS as you go: after each part, append what you did + what is left to docs/tasks/HARD_RULES_STATUS.md, so a cut-off
  loses nothing. Do not commit; Claude reviews and commits.
