# Task for Sol (Codex): WALLX prompt sweep in simulation, then report

You are helping LabelLoop (hackathon, ends 2026-11-22 16:00 PT). Ayush and Claude are running the REAL robot from the Mac right now.
YOU work only in SIMULATION. Do not touch the real robot (xr@192.168.1.21) or anything that drives it.

## Goal
Find which kind of WALLX instruction makes WALLX (a) actually grasp a box and (b) do exactly what we ask. Today's real-robot problem
(scenario 5): an IQBAR box (about 20 x 10 x 5 cm cereal-bar box) standing on its long edge in the middle of the table; WALLX with
"Pick up the IQBAR box and put it into zone 2." often hovers, closes too high (grabs a corner), or parks near the target and never
grasps; and if a person moves the box, it does not go after it again. Zones = taped squares on the table: zone 1 back-left,
zone 2 back-right, zone 3 front-right.

## Run at least 10 trials, each a DIFFERENT prompt strategy (more if time allows; repeat the best 2-3 at least twice)
1. one-shot short, organiser grammar: "Put the IQBAR box into zone 2."
2. one-shot with the object described: colour, size, orientation, where it is
3. long detailed single paragraph (how to approach, grip from the top/sides not front-to-back, go deep before closing, lift, carry, set down)
4. atomic sequence, several steps with few calls each (open + move above, lower, close, lift, carry, set down, release)
5. two-part: "Pick up the IQBAR box." then "Put the IQBAR box into zone 2."
6. arm named + grip direction ("Use the right arm, grip the box from the top")
7. state / negation ("The box is not in zone 2 yet. ...")
8. container grammar like the organisers' 100% sample "Put the spoon into the bowl." (e.g. a tray/bowl inside zone 2)
9. retry rounds: same prompt in 3 rounds with WALLX history reset each round
10. perturbation: during the grasp approach, TELEPORT the box 10-15 cm (simulates a person moving it): which prompt/harness re-grasps?
Score every trial: claw opened (y/n), grasped (y/n), lifted, placed inside target zone, followed the instruction literally (arm used,
grip direction, step order), calls to success, notes. Save head-camera frames / a short video if the sim tooling allows.

## Where things are
- Repo: this repo (read docs/AGENT_CONTEXT.md, docs/TRIALS_HANDOFF.md, docs/HANDOFF_PRATHAM_WALLX.md for what WALLX can and cannot
  do; scenarios/RESULTS.md for today's real-robot results; scripts/run_sim_case.py + labelloop/ for the sim harness).
- Sim: Ayush's Windows RTX box, `ssh ADS-Admin@100.125.217.67`, everything under C:\x2\ (X2Real / Isaac). NEVER stop or touch
  their `src.main` processes. Full use of the box otherwise.
- WALLX backends: configs/wallx.yaml (`lan` = the event model wall-x 2610-dualhead-v2 at ws://192.168.1.10:33677, reachable from
  this Mac's LAN; `cloud` = organisers' cloud API; `local` = self-hosted). PREFER the event model (lan) if you can route the sim's
  requests to it from this Mac (e.g. a relay here); otherwise cloud. Write down which model every trial used.
  The LAN WALLX server is shared with the real robot: keep request rate modest and never more than one sim episode at a time.
- Relay ideas already in labelloop/robots/event_server.py: --slow, --open-wide, --deep-close, --front-only (WALLX sees only the
  head camera: on the real robot it then never opened the claw for the box), --lock-arm, --reset-on-subgoal. Sequence steps:
  labelloop/sequence.py (read / say / release / move / replay / carry).

## Rules
- Never print, log or commit API keys / passwords. Do not read instructions/ or .env to copy secrets anywhere.
- Do not edit scenarios/keepers/**, scenarios/wx/**, labelloop/sequence.py, labelloop/robots/event_server.py, scripts/run_trial.py,
  labelloop/dashboard/**. Add NEW files only (e.g. scripts/prompt_sweep_sim.py, results/prompt_sweep/**).
- SAVE PROGRESS AS YOU GO (credits can run out): after EVERY trial append one line to results/prompt_sweep/STATUS.md and write
  results/prompt_sweep/trial_NN.json (prompt(s), model, calls, scores, notes, frame paths).
- Time box: about 90 minutes. Then (or at 10 trials, whichever first) write results/prompt_sweep/REPORT.md: ranked table of prompt
  strategies with evidence, the exact best prompt(s) to try on the real robot for scenario 5, and what did NOT transfer / caveats
  (sim vs real, cloud vs event model). Keep it short and concrete.
