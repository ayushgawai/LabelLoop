# Task: trial recorder + live/replay dashboard (local Mac, no new dependencies)

Goal: every real-robot trial is stored on the Mac with everything needed to replay it, and a local web dashboard
shows it live and in replay: animated, easy to read, good for a demo.

## 1. `labelloop/robots/trial_recorder.py` (new)
`TrialRecorder(root="runs/trials")`:
- `start(goal) -> trial_id`: makes `root/<YYYYmmdd_HHMMSS>/` and writes `meta.json` (goal, start time, profile flags passed in by the caller).
- `record(**kw)`: appends one JSON line to `trial.jsonl` per robot request. The caller passes:
  - `t`, `index`, `goal` (the robot's instruction), `subgoal` (the harness instruction sent to WALLX, or None)
  - `source`: "wallx" | "scripted" | "hold"
  - `harness`: a dict (state, codes, lane, transitions, flags, parcels, presence; from the harness, may be None)
  - `state`: the `_plain_state` dict (left_ee/right_ee position xyz m, rotation_rpy rad, gripper)
  - `latency_ms` (WALLX call time or None), `elapsed_ms` (whole request), `error`
  - `actions`: the reply's action chunk, reduced to at most 8 evenly spaced steps per arm as plain lists. Read
    `hold_response` / `wallx_client.py` for the reply format and extract the same per-arm fields. Never fail the relay if
    the format is unexpected: store None.
- `frame(index, images_b64)`: writes `frames/<index>_<camera>.jpg` (raw JPEG bytes) for every request.
- `end(summary)`: writes `summary.json` (calls, wallx_calls, mean/p95 latency, parcels, duration, last state).
- Thread-safe enough for the relay: plain appends, flush each line.

## 2. Relay wiring (`labelloop/robots/event_server.py`)
- New flag `--record DIR` (default `runs/trials`; `--no-record` disables it). Hold mode records too (source "hold").
- In `handle_wallx`: a new goal starts a new trial (end the previous one). Each request records frames and one line
  (`harness` = `harness.last_log` if you add one; see below). The connection ending calls `end()`.
- In `harness.py`, do NOT change logic. You may only add `self.last_log = {...}` holding the same fields it already
  passes to `self.runlog.write(...)` at the end of `step()` (keep the runlog call as is).
- Keep every existing test green. Never record keys/headers.

## 3. `labelloop/dashboard/server.py` (new, stdlib `http.server` only)
`python -m labelloop.dashboard.server [--root runs/trials] [--port 8095]`, bound to 127.0.0.1. Routes:
- `/`: `index.html`
- `/api/trials`: list (id, goal, start, calls, duration, parcels) newest first
- `/api/trials/<id>`: `{meta, rows: [...trial.jsonl], summary}`
- `/api/trials/<id>/rows?after=N`: rows with index > N (live tail)
- `/api/trials/<id>/frame/<index>/<camera>`: the JPEG (nearest earlier recorded index if missing)
- Reject path traversal (ids must match `^\d{8}_\d{6}$`, cameras `^[a-z_]+$`).

## 4. `labelloop/dashboard/index.html` (new, one file, vanilla JS + inline SVG + CSS, no CDN)
Look: notebook paper (off-white with faint blue 5 mm grid lines and a red margin line), hand-drawn-style dark line art,
smooth CSS/JS transitions. Light theme only is fine. Layout:
- **Header**: "LabelLoop × WALLX", trial picker (Live = newest, auto-follow), play/pause, scrub slider, speed 0.5/1/2/4x.
- **Tech chips** that light up and pulse when active on the current row: `Harness` (state name), `WALLX`
  (source == "wallx", latency ms), `Scripted`, `HOLD`, `QR decode` (codes non-empty), `Verify`.
- **Top-down table SVG** on a cm grid: table 111 × 68.5 cm, origin near-left, x right, y away from the robot.
  Blue bin and red bin side by side on the far edge (blue left of centre, red right of centre, each ~20 × 30 cm),
  black bin at the right edge mid-depth (~22 × 30 cm). Constants at the top of the script (`TABLE`, `BINS`,
  `ARM_HOME` = left home (30, 12) cm and right home (81, 12) cm, `AXIS` = how robot ee xyz deltas map to table
  x/y; default table_x = -ee_y, table_y = ee_x) so we can tune them. Each arm is a line from its base (on the near
  edge) to its end effector, with a gripper "V" whose opening follows `gripper`; the moving arm (EE speed above a
  threshold) glows and leaves a fading trail; WALLX's predicted `actions` are drawn as a dotted path ahead of it.
  The arm's z is shown as the shadow size / a small side gauge.
- **Joint/pose panel**: per arm x/y/z (mm), roll/pitch/yaw (deg), gripper; values tween; mini sparklines.
- **Camera strip**: head + two wrist frames for the current row.
- **State timeline**: coloured blocks of harness states over the trial with the playhead; transitions as ticks.
- **Performance**: elapsed vs the 480 s case clock (ring), calls, WALLX calls, mean/p95 latency, parcels, current lane.
- **Log**: scrolling list of transitions/flags/errors with timestamps, the current one highlighted.
Live mode polls `/rows?after=N` every 300 ms. Replay interpolates between rows using `t`.

## 5. Tests
`tests/test_trial_recorder.py`: record two rows with fake state + one fake image, check files, summary numbers,
and `actions` = None on an odd reply. `tests/test_dashboard.py`: start the server on a temp root in a thread, hit
`/api/trials`, `/api/trials/<id>`, `rows?after=0`, a frame, and a traversal attempt (404).
Run `.venv/bin/python -m pytest -q`; everything must pass.
