# LabelLoop

Harness that makes X Square's dual-arm desktop robot sort parcels by QR label, reading codes with the wrist cameras only.
WALLX (cloud VLA, zero-shot) makes all arm motion; LabelLoop picks one short sub-goal per model call, from QR evidence,
the left gripper value and a vision agent (Gemini Robotics-ER via a Chat Completions endpoint).
Plan: `instructions/LabelLoop Simulation-First Plan.pdf` (gitignored: the organizer PDFs contain the API key).

```
labelloop/harness.py            shared core: QR -> agent (every 4 calls) -> state machine -> instruction | hold
labelloop/brain/                state_machine (SURVEY LOOK GRASP PRESENT PLACE REJECT DONE), agent, memory, instructions
labelloop/perception/qr.py      zxing-cpp, OpenCV fallback, 2-frame confirm
labelloop/wallx_client.py       public_v2 msgpack client, Bearer auth, reset_history, hold replies
labelloop/pose.py               X2Real home-relative <-> WALLX absolute, gripper 0..1.89 <-> 0..1.526
labelloop/robots/x2real_server.py   sim link (33x14 chunks, row 0 = state echo)
labelloop/robots/event_server.py    event-day relay for the X Square client (organizer harness.py pattern)
policies/labelloop/             Policy Space deployment X2Real connects to
configs/                        task_qr_sort.yaml (tunables, wording), ex001_6r.yaml (gripper ranges, home pose)
sim/                            QR box textures, X2Real task draft (see sim/README.md), scripts/run_sim_case.py
scripts/probe_wallx.py          live WALLX link test
```

## Setup (Mac or Linux)
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env            # fill WALLX_API_KEY, AGENT_API_KEY (Gemini key)
.venv/bin/python -m pytest -q   # unit + offline 3-box episode + both links end to end against a stub WALLX
PYTHONPATH=. .venv/bin/python scripts/probe_wallx.py --n 50 --images <dir with front/left_wrist/right_wrist.jpg>
```

## Sim (Linux + RTX, X2Real installed per the plan)
```bash
# terminal 1: LabelLoop as the policy server (policy_space installed in this env)
PYTHONPATH=$PWD policy-space serve $PWD/policies/labelloop/deploy.yaml --port 8001
# terminal 2: first run on the built-in fruit task (Phase 1 gate), then our task after sim/README.md setup
python manaenv/scripts/x2real_eval.py --board artixon_arm6a_tabletop --env artixon_arm6a_G_pick_fruits_into_basket_id_env \
  --num-episodes 1 --num-parallel-envs 1 --policy-config policy/x2real_ps@v1.0.0.yaml \
  --server-address 127.0.0.1 --server-port 8001 --enable_cameras
X2REAL_ROOT=~/x2real python scripts/run_sim_case.py --case 1 --runs 10
```
Logs: `runs/sim.jsonl`, one line per model call. **First thing on the sim box:** round-trip test (hold reply → arm must
stay still), then check the arm moves the way the instruction says. Both confirm the `home_pose` frame in `configs/ex001_6r.yaml`.

## Event day
```bash
.venv/bin/python -m labelloop.robots.event_server --host <laptop LAN IP> --port 8001 [--passthrough]
```
WebUI model address `ws://<laptop LAN IP>:8001`, EndPose, sync. Stop the robot from the WebUI, never by killing the relay.
