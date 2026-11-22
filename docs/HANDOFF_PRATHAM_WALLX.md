# Handoff: make scenario 4 work with WALLX (for Pratham's Claude)

You are continuing LabelLoop's real-robot work at the X Square hackathon (BRIGHT 2026). Read this whole file, then
`docs/TRIALS_HANDOFF.md` (robot commands, relay flags, safety) and `docs/AGENT_CONTEXT.md`.

## Goal
Make **scenario 4** happen with the organisers' VLA **WALLX actually generating the motion** (a mix with our harness /
recorded motion is fine, but WALLX must do real work, not only our replay):

> Bhindi box (label 420/95126 -> blue -> zone 1). Right arm picks it up, hands it to the left arm in the air,
> left arm puts it inside taped zone 1 (back left of the table). Arms return home.

Done = one recipe that succeeds **3 times in a row** with the box reset to its start marks before each run. Then freeze it
(see "When it works").

## DO NOT TOUCH (locked, read-only, 3/3 each on the real robot)
`scenarios/keepers/sc1`, `sc2`, `sc3`, `sc4` (+ `scenarios/keepers/finaldemo1*`). These are the final submission scenarios
(smoothed replays of Ayush's controller demos). Never edit, move or delete them; never change `labelloop/sequence.py` or
`scripts/record_teleop.py` behaviour they rely on (replay, move, release steps). Add new files instead.
| # | What | Prop |
|---|------|------|
| 1 | pick and place, zone 1 | padded mailer bag |
| 2 | right arm picks, zone 2 | Miro box |
| 3 | turn label to camera, set down, right arm to zone 2 | mailer bag |
| 4 | right arm hands to left arm in the air, zone 1 | Bhindi box |
You may READ `scenarios/keepers/sc4/scenario4_smooth.json` (the full both-arm trajectory + claw timing of scenario 4,
row = 0.0337 s, columns x, y, z, roll, pitch, yaw, gripper per arm, home-relative; gripper 0 closed .. 1.53 open).

## Setup on your laptop
- Same LAN as the robot (xr@192.168.1.21) and WALLX (ws://192.168.1.10:33677). Repo `main`, `.venv` with the repo deps.
- Robot SSH password: ask Ayush; put it in a private askpass script OUTSIDE the repo and `export ROBOT_ASKPASS=/path/to/it`.
- The robot client must reach YOUR relay: `export LL_RELAY_ADDR=<your laptop IP>:8001` (default is Ayush's Mac).
- One run (homes arms first and at the end, saves video + logs): `.venv/bin/python scripts/run_trial.py --scenario scenarios/<recipe>.json`
  -> `runs/scenarios/<name>/run_N/` (`demo.mp4`, `trial.jsonl`, `relay.out`, `summary.json`). Look at `demo.mp4` / a montage after every run.
- Only one session on the robot at a time. If homing errors appear ("motion planner status error", START_STATE_IN_COLLISION on
  master_*_arm links): `scripts/rehome_master_arms.sh` (controller arms parked bent; keep hands off them while it runs), then
  Stop/Reset the stale task in the WebUI.

## How to build a WALLX recipe (sequence mode, `labelloop/sequence.py`)
A recipe = `scenarios/<name>.json` (relay flags) + `scenarios/<name>_steps.json` (steps). Step types:
- `{"read": true}`: label read (QR -> printed ZIP OCR -> `code_hint` demo fallback) -> chooses the bin / zone.
- `{"say": "<WALLX instruction>", "arm": "left|right|lane", "calls": N, "until": "lifted|in_zone"}`: WALLX acts; the other arm is
  LOCKED (its WALLX motion replaced by holding still). `in_zone` = the parcel (tracked by colour `track_hsv`; Bhindi purple
  [125,165,70,40]) is inside `zones_px[bin]` for `zone_calls` calls -> WALLX is cut off immediately.
- `{"release": "left|right|lane"}`: opens that claw fully and lifts it 8 cm (scripted).
- `{"replay": "runs/demos/<file>.json"}`: replays recorded motion (e.g. a slice of the scenario 4 trajectory).
- `{"move": {"left": {"position": [...], "rotation_rpy": [...], "gripper": g}}, "max_step": 0.04}`: slow scripted move (home = zeros).
Relay flags that helped: `--slow 0.3..0.5` (WALLX chunks time-stretched: slower, safer), `--no-reset-on-subgoal`,
`--z-offset -1` (WALLX's one trained grasp-height knob, about 1-2 cm lower). AVOID `--force-roll` (fast, dangerous) and
`--open-wide` (made grasps worse).
Example to start from: `scenarios/w4a_replay_handover_wallx_place.json` + `_steps.json` (replays pick + handover, then WALLX places).

## What we know about WALLX (wall-x 2610-dualhead-v2, the only model on the LAN server)
- Its trained skill is pick-and-place INTO a container. The organisers' sample **"Put the spoon into the bowl."** scored
  100% on this robot. Our best WALLX results used the same pattern with real bins:
  "put the IQBAR box into the red bin" / "...blue bin", idle arm locked, parcel standing on its narrow side: 5/6.
- Short, single-action prompts in that style work best ("Put the X into the Y."). Long multi-step prompts make it hesitate.
  Naming an arm does not stop the other arm (that is why the harness locks it).
- It does NOT understand: taped squares as targets (stops at the edge or carries the parcel away), pushes, "turn it to the
  camera", directional commands, press-and-lift tip-ups. With no container on the table it barely commits to a grasp.
- No demo / file input exists: `prev_action_chunk` must be its own recent plan (server returns cb_prefix_mismatch otherwise).
- Rotation offset is not a trained parameter; flat boxes left-to-right get knocked over; front-to-back works better.

## Tried for scenario 4 so far (2026-11-22)
- V1 `w4a`: replayed right-arm pick + handover (works every time: left claw holds the box at the front of zone 1), then WALLX
  "Put the box into the square marked 1." -> WALLX carried the box AWAY and set it down mid-right. FAIL.
- Prompt sweep `w4p1..w4p5` (box wherever it lies, left arm, right locked, cut-off over zone 1): only #1 "Put the box into
  the left square." ran -> did not reach zone 1. #2-#5 untested:
  "Put the box on the number 1." / "Put the purple box into the square on the left side of the table." /
  "Put the box down inside the taped square with the number 1." / "Put the box into square 1."

## Ideas to try next (in this order)
1. Finish the prompt sweep (#2-#5) from the V1 handover pose.
2. Give WALLX a container: a low tray / shoe-box lid / bowl INSIDE zone 1, prompt "Put the box into the bowl." /
   "...into the tray." (the spoon example's exact grammar). Keep the tape; the container just sits in the square. (Ask Ayush.)
3. Split the handover into WALLX steps: "Pick up the box." (right, until lifted) -> "Give the box to the left hand." (both arms)
   -> "Put the box into the bowl." (left). Use the replayed handover if WALLX's fails.
4. If WALLX only fails the last centimetres, let it carry (cut off with `in_zone`) and release with the scripted step.

## Trial protocol
- Ayush / Pratham put the box back on the scenario 4 start marks (or another fixed spot) after EVERY try.
- One change per trial; write one line per trial in `scenarios/RESULTS.md` (recipe, prompt, result, video path).
- Safety: stop at once if a bin/prop moves unexpectedly, the arm moves fast, or anything is pressed into the table.

## When it works
Run the same recipe 3 times with resets. If 3/3: copy recipe + steps + any demo slices into `scenarios/keepers/sc4_wallx/`,
make them read-only (`chmod -R a-w`), add a line to `scenarios/RESULTS.md`, commit and push on a branch
(`pratham/sc4-wallx`), and tell Ayush the recipe path and the 3 video paths.
