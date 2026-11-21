# Event runbook (hackathon day)

Facts: robot `xr@192.168.1.21` (password from the team, not stored here), WebUI http://192.168.1.21/robot/home,
organizer WALLX `ws://192.168.1.10:33677` (0.07 s/call, `backend: lan` in `configs/wallx.yaml`),
our Mac relay `ws://192.168.1.157:8001` (re-check the IP, it changes with the network).
Robot task file `~/diversity/local_labelloop_pipeline_test.json` already points at the relay, instruction "put the box into the blue bin".

## Go / no-go (all five must be true before any harness run on the real arm)
1. Physical stop (e-stop) tested by the person standing next to it.
2. Scripted motions, wrist reads and retries are OFF.
3. Stationary-image checks pass: empty-table reference, smallest parcel, correct lanes, zero bin-code reads.
4. Embodiment (`teletwin_desktop/gripper-g`), units and frames verified against `results/event/`.
5. `max_parcels = 1`, supervised, one person with a hand on the stop.

## 0. Safety
- Know where the e-stop is. Test it before the first run.
- Clear the workspace: only the parcel(s) under test, bins, nothing loose, no cables in reach.
- ONE person drives the WebUI. Everyone else keeps hands, bodies and keyboards off.
- After arming: hands off. No typing on any laptop during a run. Stop = e-stop first, talk later.

## 1. Start the relay (passthrough) and verify the robot reaches it
```bash
cd ~/Documents/GitHub/LabelLoop
ipconfig getifaddr en0                       # must be 192.168.1.157, else fix the task file (step 2)
nc -z 192.168.1.10 33677 && echo WALLX up    # organizer model reachable from the Mac
.venv/bin/python -m labelloop.robots.event_server --host 0.0.0.0 --port 8001 --passthrough
# if present (real profile): add  --profile real --real-safe --save-images
```
Verify from the robot:
```bash
ssh xr@192.168.1.21
nc -zv 192.168.1.157 8001                    # must say succeeded
cat ~/diversity/local_labelloop_pipeline_test.json | grep model_address   # must be ws://<Mac IP>:8001
```
If the Mac IP differs, edit `model_address` in that file (keep the rest unchanged).

## 2. Run the pipeline test from the WebUI
1. Open http://192.168.1.21/robot/home. Select task `local_labelloop_pipeline_test`.
2. Place one box in view. Instruction is "put the box into the blue bin".
3. Confirm workspace clear, one person arms and starts. Hands off.
4. Watch the relay terminal: requests should scroll in. Stop immediately if motion looks wrong.

## 3. Check results
```bash
tail -n 20 runs/event.jsonl                  # one line per call: instruction, latency, action summary
.venv/bin/python - <<'E'
import json;L=[json.loads(l) for l in open("runs/event.jsonl")]
print(len(L),"calls", L[-1].keys())
E
ls -t runs | head                            # saved camera images (only with --save-images)
```
Expect ~0.07 s WALLX latency plus relay overhead. Missing lines = robot never reached the relay (section 7).

## 4. Harness in real-safe mode
Only after steps 1-3 pass and the go/no-go list is green.
- Setup: ONE parcel, top label up, in the top-label lane. `max_parcels = 1`.
- Head-camera reads only (no wrist reads, no retries, no scripted swipes).
- On reject / unread / oversize the harness HOLDS (does not move the parcel). Operator removes it by hand with the arm stopped.
```bash
.venv/bin/python -m labelloop.robots.event_server --host 0.0.0.0 --port 8001 --profile real --real-safe --save-images   # if present
```
Which label goes on which item (`docs/labels/labels.csv`):
| Bin | Label | Item |
|---|---|---|
| blue | LL-08 | Bhindi Masala box |
| blue | LL-09 | padded mailer |
| blue | LL-10 | perfume box (lies flat) |
| red | LL-11 | IQBAR pack |
| red | LL-12 | Miro box |
| reject | none | unlabelled IQBAR, wallet |

Stick the label up, flat, QR clear of tape/creases. Bin labels must never be read (zero bin-code reads in the stationary check).

## 5. Calibration before any scripted arm motion
Do not enable scripted motion until: head intrinsics (`results/event/intrinsics.yaml`) and `results/event/merged_calibration.yaml` are loaded,
a tabletop point projected from the head image lands where the arm tip actually goes (jog to it by hand/WebUI, supervised),
and units (m vs mm), frame (robot base vs camera) and gripper range match the embodiment. Any mismatch: stay passthrough/real-safe.

## 6. Switch back to cloud / home
```bash
sed -i '' 's/^backend: lan.*/backend: cloud   # at the event; '"'"'cloud'"'"' at home/' configs/wallx.yaml
# or per run, no edit:  WALLX_BACKEND=cloud <command>
grep '^backend' configs/wallx.yaml
```
Cloud needs `WALLX_ENDPOINT` and `WALLX_API_KEY` in `.env`. Do not commit the event IP changes by accident; `git diff configs/` first.

## 7. Troubleshooting
- Relay not reachable from robot: macOS firewall (System Settings > Network > Firewall: allow incoming for python, or turn off for the event);
  Mac and robot on the same 192.168.1.x network; relay bound to `0.0.0.0`; Mac IP changed (`ipconfig getifaddr en0`, update task file).
- WALLX unreachable: `nc -z 192.168.1.10 33677`. Fails = wrong network/VPN, ask organizers; fall back to `WALLX_BACKEND=cloud` only if internet exists.
- Wrong embodiment / garbage actions: must be `teletwin_desktop/gripper-g`, `robot_type DESKTOP`, `EndPose`, `normalization provided` (compare with
  `results/event/local_desktop_g_spoon_into_bowl.json`). Wrong units/frames = stop, do not tune on the real arm.
- Robot idle, relay silent: wrong `model_address` in the task file, or task not selected in the WebUI.
