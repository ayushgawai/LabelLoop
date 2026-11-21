# Task for Pratham's coding agent: dashboard v2 — animated arm line art with joint angles + story mode

You are a coding agent helping team LabelLoop at the hackathon. Work ONLY on this task. Read docs/AGENT_CONTEXT.md first.

## Rules (must follow)
- Branch `pratham/dashboard-v2` from the latest `main`; push the branch; do NOT push to `main`, merge or force-push. Ayush's agent reviews and merges.
- Code + tests only. Do NOT touch the robot (no SSH to 192.168.1.21, no WebUI), do NOT call WALLX, do NOT read or print `.env` or keys.
- Only create/edit: `labelloop/dashboard/*` (server.py, index.html, new files there), `labelloop/dashboard/kinematics.py` (new),
  `tests/test_dashboard*.py`. Do NOT edit harness.py, state_machine.py, event_server.py, trial_recorder.py, configs (live code).
- No new dependencies (stdlib + numpy; the page is vanilla JS + inline SVG, no CDN). Full suite stays green:
  `.venv/bin/python -m pytest -q` (Windows: `.venv\Scripts\python -m pytest -q`).
- SAVE PROGRESS: commit + push after each numbered step below (small commits), and append a line per step under "Status" at the
  end of this file, so nothing is lost if your session or credits end.

## What exists (read it)
- `labelloop/dashboard/server.py` (stdlib HTTP, 127.0.0.1:8095) + `index.html`: live/replay of trials from `runs/trials/<YYYYmmdd_HHMMSS>/`
  (`meta.json`, `trial.jsonl` one row per robot request: t, index, goal, subgoal, source wallx|scripted|hold, harness {state, codes,
  lane, transitions, flags, presence, bin_change...}, state {left_ee, right_ee: position xyz m (HOME-RELATIVE: ~0 at rest),
  rotation_rpy rad, gripper}, latency_ms, elapsed_ms, actions (WALLX's predicted chunk, <= 8 steps per arm), `frames/<index>_<camera>.jpg`
  incl. `<index>_front_overlay.jpg` (the masked image WALLX saw), `summary.json`). Today it draws each arm as a single line base -> EE.
- `labelloop/look.py`: kinematics of the 6-DOF arm: `fk(q)`, `ik(pos, rpy)`, joint limits `LO`/`HI`, `TCP` (fingertip offset),
  `SIGN` (left/right base), `frame(arm, cfg)`. `scripts/home_pose_from_urdf.py`: the home pose. `tests/test_look.py` shows usage.
- Trial-agent runs: `runs/agent/<ts>/` (progress.jsonl, journal.json, t<N>.jsonl, montages) and reports `results/agent/REPORT_*.md`.
- Sample data for tests/dev: make a small synthetic trial in a temp dir (do not commit runs/).

## Steps
1. `labelloop/dashboard/kinematics.py`: `joints(arm, state_arm) -> {q: [6 rad], points: [[x,y,z] per joint origin + fingertip], ok}`.
   Absolute EE = home pose (from the URDF script / look.py) composed with the home-relative state; IK with look.ik (warm-start
   from the previous q for smooth playback); `ok=False` when IK fails or a joint is at its limit. Server route
   `/api/trials/<id>/joints` returning per-row joints for both arms (computed once and cached to `joints.json` in the trial dir).
   Test: fk(ik(p)) round trip on a few reachable poses; the route on a synthetic trial.
2. Arm line art in `index.html`: two linked SVG views on the notebook-grid background: TOP view (existing table/bins) and a SIDE
   view (y-z plane) of both arms drawn as joint chains (circles at joints, thick strokes for links, gripper fingers opening with
   `gripper`). Joint angles labelled (deg) next to each joint; a joint near its limit turns orange, IK failure grey dashed. Smooth
   tweening between rows; the moving arm glows; WALLX's predicted `actions` drawn as a dotted fingertip path.
3. Story mode: a page section listing every trial (runs/trials) and agent run (runs/agent + results/agent/REPORT_*.md) in time
   order with: wording used, overlay/stage, sub-goal timeline (coloured blocks per harness state), verdict, time, retries, and a
   "replay" button that loads it into the animated views; flags like `bin_moved`, `grasp:missed` shown as red markers.
   Server routes `/api/story` (merged list) and `/api/agent/<ts>` (journal + report text, markdown rendered as plain pre).
4. Camera strip shows the raw front frame and, when present, the masked `front_overlay` frame side by side ("what WALLX saw").
5. Performance panel: per trial calls, WALLX calls, mean/p95 latency, time per sub-goal, success rate across trials (small bar
   chart, inline SVG). Keep the page fast with 500+ row trials (render with requestAnimationFrame, no per-row DOM rebuilds).

## Status
(append one line per step: date time, step, commit hash)
- 2026-11-21 14:08, step 1, 4333892: `kinematics.py` (`joints`, `row_joints`, `trial_joints` -> `joints.json`, incremental for live trials, ~1 ms/row warm-started) + `GET /api/trials/<id>/joints` + `demo.py` (synthetic trials: `python -m labelloop.dashboard.demo --root runs/demo/trials`) + `tests/test_dashboard_v2.py`. Note: at rest (q = 0) joints 2 and 3 sit ON their URDF limits (LO[1] = 0, HI[2] = 0), so `ok` = IK converged and a joint at / near a limit is flagged in `near` (drawn orange), not an IK failure.
- 2026-11-21 14:14, step 2, 5f87bfa: `index.html` arm line art: linked TOP (x-y) + SIDE (y-z) SVG views, joint chains with angle labels, orange near-limit joints, grey dashed IK failure, gripper fingers, tweening + glow + trail, WALLX predicted fingertip path dotted. Root -> table placement uses `ROOT_NEAR_X` = 0.184 m (sim value; CALIBRATE on the real bench).
- 2026-11-21 14:20, step 3, b0cea21: story mode: `story.py` + `GET /api/story` (trials + agent runs + reports in time order: wording, overlay/stage, state blocks in seconds, verdict, time, retries, flags) + `GET /api/agent/<ts>` (journal, report text, progress); server `--agent` / `--reports`; Story card with replay (agent-run trials matched to their recorded trial by time) and report toggle. Works with report-only agent runs (results/agent without runs/agent).
- 2026-11-21 14:23, step 4, 4c4db14: camera strip: raw front + `front_overlay` ("what WALLX saw", nearest earlier overlay with its row, outlined when current; a note when the trial has none) + wrist cams; trial detail has a frame index, `?rows=0` skips the rows for the page's periodic refresh.
- 2026-11-21 14:29, step 5, 0c6dbf1: performance panel (calls, WALLX calls, mean/p95 latency, time per sub-goal bars, success rate across trials with a per-trial bar strip, inline SVG); incremental stats / log, timeline rebuilt <= 1/s with ticks as one SVG path, list polled every 2 s, server list skips rows of finished trials; "render x ms/frame" readout: ~4-5 ms/frame on a 650-row trial (played and followed live).
