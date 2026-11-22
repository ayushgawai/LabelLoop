# Scenario 4 with WALLX (Pratham, 2026-11-22)

Scenario 4 = Bhindi box (label ZIP 950xx -> blue -> zone 1): right arm picks it, hands it to the left arm in the air,
left arm puts it in taped zone 1 (back left), arms go home. Goal from `docs/HANDOFF_PRATHAM_WALLX.md`: WALLX must do real work.
Nothing in `scenarios/keepers/`, `labelloop/sequence.py` or `scripts/record_teleop.py` was changed.

| Folder | What |
|---|---|
| `demo_replay_pick_wallx_place/` | **The demo we use** (recipe name `sc4_wallx_v1`). Recorded pick + handover, WALLX places. |
| `wip_wallx_pick_anywhere/` | Work in progress: WALLX does the pick too (recipe name `sc4_wallx_pick_anywhere`, was `w4x_pickany4`). Not working yet. |
| `trials/` | Every recipe tried on the way (`w4x_try1`..`try5`, `w4x_demo`, `w4x_pickany`..`pickany3`), paths fixed so each still runs. One line per run in `scenarios/RESULTS.md`. |

Videos and logs are in `runs/scenarios/<recipe name>/run_N/` on Pratham's laptop (`runs/` is not in git).

## Run the demo

1. Box: **standing on its long narrow edge, big face to the camera, just in front of zone 2's front-left corner**
   (`demo_replay_pick_wallx_place/start_pose.jpg`). The pick is a recording, so the spot matters. Arms at rest, no other session on the robot.
2. Windows: `scripts\run_sc4_wallx_demo.cmd` (finds the laptop's 192.168.1.x address; `--check` only prints it).
   Anywhere: `LL_RELAY_ADDR=<laptop ip>:8001 .venv/bin/python scripts/run_trial.py --scenario scenarios/sc4_wallx/demo_replay_pick_wallx_place/recipe.json`
   (about 5 min with homing; video in `runs/scenarios/sc4_wallx_v1/run_N/demo.mp4`).

Relay flags: `--sequence .../steps.json --slow 0.4 --no-reset-on-subgoal`, 180 s.

| # | Step (`labelloop/sequence.py`) | Who moves |
|---|---|---|
| 1 | Read the label (QR -> printed ZIP -> fallback `420/95126/001`) -> blue -> zone 1 | no motion |
| 2 | Replay `sc4_pick_handover.json` = keeper `sc4/scenario4_smooth.json` rows 0-1756 (0-59.2 s): right picks, hands to left in the air | recording |
| 3 | Right arm up/out to (0.35, 0.203, 0.26), then home (<= 4 cm per call) | script |
| 4 | **WALLX, left arm, 40 calls** (right arm locked): `Put the purple box into the square on the left side of the table.` | **WALLX** |
| 5 | Release: left claw opens fully, lifts 8 cm | script |
| 6 | Both arms home (<= 5 cm per call) | script |

WALLX (wall-x 2610-dualhead-v2, ws://192.168.1.10:33677) gets per call the 3 camera images, both arms' state, the sentence
above and its own previous chunk. `--slow 0.4` executes 40 % of each chunk, stretched (slower); `--no-reset-on-subgoal`
resets its history only at the start.

**Results (5 runs of this recipe):** pick + handover 5/5. Box placement by WALLX: half inside zone 1 once (try 3 run 1);
about 20 cm in front of zone 1 twice (Pratham: "good enough" for the demo); about 15 cm in front of zone 1's right corner
once (WALLX never let go, the scripted release dropped it); in front of zone 2 once. Not 3/3 inside zone 1, so it is
not in `scenarios/keepers/`.

**Known issue:** after the handover the held box covers the LEFT wrist camera (until about call 85-90), so WALLX places
mostly blind; the runs where it cleared sooner placed better.

**Next idea (not built):** replay further (keeper 59.2-77 s carries the box to just above zone 1) and give WALLX only the
set-down (10-15 calls), then the scripted release.

## Pick from anywhere (work in progress)

`wip_wallx_pick_anywhere/`: WALLX right arm `Put the purple box into the square on the left side of the table.` until the
box is lifted (45 calls) -> scripted lift 10 cm -> scripted move to the recording's 34 s pose (wrist turned in 3 steps) ->
replay 34-59.2 s (`sc4_from_lift.json`, the handover) -> same as demo steps 3-6, home with claws open.
Relay adds `--z-offset -1 --open-wide 1.5 --grasp-dz 0.015 --grasp-floor -0.015`, 300 s.

- **Bug found and fixed in the recipe:** with the default scene config the "lifted" check took the parked left claw as
  "the box" (and the default `bin_rois` hid the box), so it could never fire. `steps.json` now carries a `scene` override
  (parked claws, taped zones and the floor beside the table masked, `bin_rois: {}`, `lift_calls: 1`). Checked offline on
  6 box positions and on the recorded frames (fires one call after the real lift, never on a failed pick).
  It has not fired live yet, so the WALLX-pick -> recorded handover transition has never run on the robot.
- `Pick up the purple box.` makes WALLX grip but never lift (`w4x_pickany`); a destination sentence makes it lift
  (`w4x_pickany2`, call 21), but then it carries the box toward zone 1, which is why the cut-off matters.
- The box must stand **short end to the camera**: side-on it gets pushed over (`w4x_try5`); WALLX's wrist angle barely changes.
- Half-open claw grasps slip (`w4x_pickany3`); with the deeper-grasp flags and the box diagonal mid-table WALLX never went for it (`w4x_pickany4`).
- Best bet to continue: box standing short end to camera about 10 cm in front of the right claw (where `w4x_pickany2` lifted it).

## Windows notes

`scripts/run_trial.py`: uses `.venv/Scripts/python.exe` when `.venv/bin/python` is missing, and on Windows asks you to stop a
stale relay on port 8001 (no `pkill`). Its verdict prompt also fires on Windows `NUL`; the launcher feeds an empty file instead.
