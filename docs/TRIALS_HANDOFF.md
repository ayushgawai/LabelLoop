# Real-robot trials: full context for the Codex session

You are continuing team LabelLoop's real-robot trials (X Square hackathon, BRIGHT 2026, SF; competition ends Nov 22 16:00).
Ayush (the human) drives you from his Codex app. He will show you videos he recorded of the motion he wants; your job is to
make our harness + WALLX reproduce it on the robot, trial by trial. Read this whole file, then docs/AGENT_CONTEXT.md.

## 1. The task and the rules
- Sort parcels by QR label (`420/<ZIP5>/<seq>`): ZIP3 940/941/950/951 -> blue bin, 900-902 -> red bin, else / no label -> black bin.
- Event rule: **WALLX** (the organisers' vision-language-action model) must generate every arm motion. Our harness only reads
  labels, chooses sub-goals, writes the ONE short instruction WALLX gets per call, may edit the images WALLX sees, verifies, retries,
  or holds still. No scripted arm motion at the event (event_fast mode = WALLX-only).
- 8 min per case, fully autonomous. Everything stays on the table (black bin is on the table).
- Props and bin types are the team's choice.

## 2. Hardware and network
- Robot: dual 6-DOF arms (teletwin_desktop, G gripper; opens much wider than the old 97.5 mm note, units 0..1.526 = fully open), head camera 1280x720 looking down from behind the robot,
  two wrist cameras. Host `xr@192.168.1.21`; the client runs in Docker container `desktop_client-robot_client-1`.
- WALLX on the LAN: `ws://192.168.1.10:33677` (0.07 s/call). Our relay on this Mac: `0.0.0.0:8001` (Mac IP 192.168.1.157).
- SSH password: NEVER write it anywhere. Use the askpass helper `~/.config/labelloop/robot_askpass.sh` (outside the repo).
- Dashboard (live + replay of every trial): `.venv/bin/python -m labelloop.dashboard.server --port 8095` -> http://127.0.0.1:8095.

## 3. Safety (always)
- Before every robot session: the table must be clear of hands; Ayush must have reset the scene. If unsure, take a snapshot (5.3)
  and look at it first.
- Stop and tell Ayush whenever a bin moved, something fell, an arm looks stuck, or a parcel is held in the air at the end.
- The harness latches HOLD by itself when a bin moves (flag `bin_moved`). Never resume motion after that without a human reset.
- Physical e-stop is on the robot; Ayush is beside it.

## 4. How a harness trial runs
1. Relay (terminal 1, keep running; restart it to change flags/wording):
   ```
   .venv/bin/python -m labelloop.robots.event_server --host 0.0.0.0 --port 8001 --profile real --stage \
       --wording runs/manual/wording.json --log runs/manual/t1.jsonl --save-images runs/manual/images --record runs/trials
   ```
   Flags: `--stage` = staged delivery with bin hiding (below); `--overlay none|parcel|mask|grid` = extra marks on WALLX's head image;
   `--wording FILE` = JSON overriding instruction templates; `--hold` = camera capture only (no motion); `--passthrough` = no harness,
   WALLX gets the client's instruction verbatim (use for a single direct command, e.g. a table reset).
2. Robot session (terminal 2; ends after `timeout` seconds and homes the arms):
   ```
   SSH_ASKPASS=~/.config/labelloop/robot_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -o ConnectTimeout=8 xr@192.168.1.21 \
     'docker exec desktop_client-robot_client-1 bash -c "source /opt/xr/bot/x2robot_client/share/x2robot_client/scripts/session_env.sh >/dev/null 2>&1; timeout -s INT 240 python3 /opt/xr/bot/x2robot_client/share/x2robot_client/scripts/start_desktop.py --model-address 192.168.1.157:8001 --instruction \"Read each parcel label and sort it: ZIP 940/941/950/951 to the blue bin, 900-902 to the red bin, anything else to the black bin.\" --robot-mode end_pose --infer-type sync --embodiment teletwin_desktop/gripper-g --image-wire-format base64 --attach --home-on-stop"'
   ```
   With `--passthrough` the `--instruction` text IS the WALLX command. With `--hold` use `--instruction "hold still"` and ~12 s.
3. Analyse: `runs/manual/t1.jsonl` (one row per call: state, instruction, transitions, flags, codes/confirmed, bin_change, presence,
   grippers), head frames every 10th call in `runs/manual/images/<ts>/<n>_front.jpg`, and every frame in `runs/trials/<ts>/frames/`
   (incl. `<n>_front_overlay.jpg` = the masked image WALLX saw). `scripts/trial_agent.py` has `summarize(log)` and a montage helper.
- Automated loop (optional): `ROBOT_ASKPASS=~/.config/labelloop/robot_askpass.sh .venv/bin/python -I scripts/trial_agent.py --sessions 14 --minutes 80`
  (an OpenAI gpt-6.2-sol agent with fixed tools; reports in results/agent/REPORT_*.md). Its SYSTEM prompt holds the full run history.

## 5. Harness behaviour (event_fast + `--stage`)
- IDLE -> SCAN: reads the label from the head camera (QR with an x3 upscale retry; small dense QR -> OCR of the printed ZIP, confirmed
  over 2 frames). A parcel already on the table at the start is handled (inpainted out of the empty-table reference).
- Staged delivery (hard rule: WALLX cannot grab a bin it cannot see):
  - wide parcel (narrow side > gripper): PICK_WIDE "use both arms to squeeze the short ends of the box and lift it up" (ALL bins painted
    out of the head image) -> PLACE_WIDE "put the box into the {mat} bin with both arms" (ONLY the target bin visible).
  - narrow parcel: PICK "pick up the box" -> PLACE.
  - STAND (tip onto the thin edge) exists but is off (`tries.STAND: 0`): 0/3 on the robot.
  - PICK done = gripper opened then closed on something for 2 calls, or the parcel left its table spot. Up to 35 calls per try, 2 tries.
- Watchdog: a bin region changing >= 0.8 for 3 calls (target bin only while the drop has not succeeded; off once the parcel left its
  spot) -> HOLD. Success = zone empty + target bin region changed.
- Wording templates (configs/task_qr_sort.yaml `instructions`, override with `--wording`): keys PICK, PICK_WIDE, PICK_STANDING, PLACE,
  PLACE_WIDE, STAND, REJECT, INSPECT, CENTER, HOME, LIFT2. Placeholders: {mat} bin colour, {arm} acting arm, {where} parcel place
  ("left", "front right", ... frozen once routed), {bin_where} target bin place ("at the back right" red, "at the back left" blue,
  "at the right edge" black), {obj}, STAND roles {fix} {fix_side} {lift} {lift_side}. Example `runs/manual/wording.json`:
  `{"PICK_WIDE": "use both arms to squeeze the short ends of the box and lift it up", "PLACE_WIDE": "keep holding the box and put it into the {mat} bin {bin_where}"}`

## 6. Scene and props (real table)
- Table ~1.11 x 0.685 m usable (measured mat 1.17 x 0.94 m). Bins (open shoe boxes): blue = navy box far edge left of centre, red =
  red Nike box far edge right of centre, black = black/white box at the right edge mid-depth. Head-image regions in
  configs/task_qr_sort.yaml `profiles.real.bin_rois`; RE-MEASURE them if the bins are moved (take a snapshot, fractions x0,y0,x1,y1).
- Parcels (mm): Miro black box 216x165x38 (420/90012 -> red), Bhindi carton 165x102x38 (420/95126 -> blue), IQBAR 267x102x64,
  perfume 152x102x51, wallet 114x89x19 (the gripper opens wider than the old 97.5 mm note, so the 102 mm faces may fit when fully open). docs/props/README.md has all.

## 7. What happened so far (real robot, 2026-11-21)
- Trials 1-2: label not read (small taped QR) -> fixed. "flip the box over" -> WALLX grabbed the RED BIN.
- Agent runs 1-3: every PLACE wording (incl. the organisers' verbatim "put the box into the {mat} bin") ended with WALLX grabbing a
  bin (blue tipped, black lifted into red, red tipped). -> staged bin hiding + bin watchdog.
- Run 4: STAND failed 3/3; "use both arms to squeeze the short ends ... and lift" LIFTED the Miro box clean off the table (first real
  pick of a wide parcel); a false bin alarm (lifted box hid a bin) froze it -> fixed.
- Run 5: squeeze + retry: the RIGHT arm lifted the box, then carried it to the BLACK bin at the right edge (not the red target) and
  bumped the blue bin. WALLX's right arm seems to drop at the right edge by habit; its wrist camera still sees the black bin.
- Open ideas: wording with {bin_where}; a bin layout matched to where each arm naturally drops (props/layout are our choice); a
  narrower parcel (wallet) for single-arm picks; Ayush's recorded demonstration videos (extract frames with
  `trial_agent.video_grids(path, out_dir)` or ffmpeg, study grasp / path / release, translate into sub-goal wording).

## 8. Working rules
- Code changes: keep `.venv/bin/python -m pytest -q` green (~225 tests). Small commits on main are fine when Ayush asks; never commit
  `.env`, keys, the askpass file, or runs/ data.
- Save progress as you go in `runs/manual/NOTES.md` (one line per trial: wording, flags, start pose, result, what to change), so a
  cut-off session loses nothing.
- Never print or read `.env` / keys. Ask Ayush before every robot session unless he has said to keep going.
