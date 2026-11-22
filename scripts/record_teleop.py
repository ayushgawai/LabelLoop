#!/usr/bin/env python3
"""Record a controller (teleop) demo ourselves (the robot's debug recorder stopped writing bags): a ROS listener inside the robot
client container logs /left_arm/end_pose, /right_arm/end_pose and the gripper commands; on stop it is fetched and converted to a
replayable demo (same format as scripts/demo_from_bag.py: home-relative poses + gripper, 0.0337 s rows).

  .venv/bin/python scripts/record_teleop.py start NAME        # starts recording on the robot (max 15 min)
  .venv/bin/python scripts/record_teleop.py stop NAME [SPEED] [SMOOTH_S]   # stops, fetches, writes runs/demos/NAME.json
     SPEED (default 2): playback speed-up of a slow hand demo; SMOOTH_S (default 0.5): moving-average window on poses
  .venv/bin/python scripts/record_teleop.py convert NAME [SPEED] [SMOOTH_S]  # re-convert an already fetched NAME.raw.jsonl
"""
import json, os, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from demo_from_bag import DT, GRIP_SCALE, quat_rpy  # noqa: E402

ASKPASS = os.environ.get("ROBOT_ASKPASS", str(Path.home() / ".config/labelloop/robot_askpass.sh"))
ENV = {**os.environ, "SSH_ASKPASS": ASKPASS, "SSH_ASKPASS_REQUIRE": "force", "DISPLAY": ":0"}
SSH = ["ssh", "-o", "ConnectTimeout=8", "xr@192.168.1.21"]
LISTENER = r'''
import rclpy, json, time, sys
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
rclpy.init(); n = rclpy.create_node("ll_teleop_rec"); out = open(sys.argv[1], "w")
def pose(arm):
    return lambda m: out.write(json.dumps({"t": time.time(), "k": arm, "p": [m.pose.position.x, m.pose.position.y, m.pose.position.z],
                                           "q": [m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w]}) + "\n")
def grip(arm):
    return lambda m: out.write(json.dumps({"t": time.time(), "k": arm + "_g", "g": float(m.data[0])}) + "\n")
for a in ("left", "right"):
    n.create_subscription(PoseStamped, "/%s_arm/end_pose" % a, pose(a), 50)
    n.create_subscription(Float64MultiArray, "/%s_gripper_controller/commands" % a, grip(a), 50)
t0 = time.time()
while time.time() - t0 < 900:
    rclpy.spin_once(n, timeout_sec=0.05); out.flush()
'''


def sh(cmd, **kw):
    return subprocess.run(SSH + [cmd], env=ENV, capture_output=True, text=True, **kw)


def start(name):
    (ROOT / "runs/demos").mkdir(parents=True, exist_ok=True)
    sh("cat > /tmp/ll_rec.py", input=LISTENER)
    sh("docker cp /tmp/ll_rec.py desktop_client-robot_client-1:/tmp/ll_rec.py")
    sh("docker exec -d desktop_client-robot_client-1 bash -c 'source /opt/xr/bot/x2robot_client/share/x2robot_client/scripts/session_env.sh "
       ">/dev/null 2>&1; python3 /tmp/ll_rec.py /tmp/ll_%s.jsonl'" % name)
    print("recording", name)


def stop(name, speed="2", smooth="0.5"):
    sh("docker exec desktop_client-robot_client-1 pkill -f ll_rec.py")
    raw = sh("docker exec desktop_client-robot_client-1 cat /tmp/ll_%s.jsonl" % name).stdout
    (ROOT / "runs/demos" / (name + ".raw.jsonl")).write_text(raw)
    convert(name, speed, smooth)


def smooth_cols(X, w):  # centred moving average per column, edges padded with the end values
    if w < 2:
        return X
    pad = np.pad(X, ((w // 2, w - 1 - w // 2), (0, 0)), mode="edge")
    k = np.ones(w) / w
    return np.column_stack([np.convolve(pad[:, j], k, mode="valid") for j in range(X.shape[1])])


def convert(name, speed="2", smooth="0.5"):
    speed, smooth = float(speed), float(smooth)
    raw = (ROOT / "runs/demos" / (name + ".raw.jsonl")).read_text()
    rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    pose = {a: np.array([[r["t"], *r["p"], *quat_rpy(*r["q"])] for r in rows if r["k"] == a]) for a in ("left", "right")}
    grip = {a: np.array([[r["t"], r["g"]] for r in rows if r["k"] == a + "_g"]) for a in ("left", "right")}
    t0, t1 = max(pose[a][0, 0] for a in pose), min(pose[a][-1, 0] for a in pose)
    ts = np.arange(t0, t1, DT * speed)  # sample the slow demo every speed x DT, play at DT: speed x faster
    out = {}
    for a in ("left", "right"):
        P = np.column_stack([np.interp(ts, pose[a][:, 0], np.unwrap(pose[a][:, k]) if k >= 4 else pose[a][:, k]) for k in range(1, 7)])
        P = smooth_cols(P, max(1, int(round(smooth / DT))))
        g = GRIP_SCALE * np.interp(ts, grip[a][:, 0], grip[a][:, 1]) if len(grip[a]) else np.zeros(len(ts))
        out[a] = np.column_stack([P, g]).round(5).tolist()
    moving = np.zeros(len(ts), bool)  # trim idle start / end (both arms within 3 mm of their first / last pose, grippers unchanged)
    for a in ("left", "right"):
        X = np.asarray(out[a])
        moving |= (np.linalg.norm(X[:, :3] - X[0, :3], axis=1) > 0.003) | (np.linalg.norm(X[:, :3] - X[-1, :3], axis=1) > 0.003)
        moving |= (np.abs(X[:, 6] - X[0, 6]) > 0.05) & (np.abs(X[:, 6] - X[-1, 6]) > 0.05)
    idx = np.flatnonzero(moving)
    lo, hi = (max(idx[0] - 10, 0), min(idx[-1] + 10, len(ts))) if len(idx) else (0, len(ts))
    out = {a: out[a][lo:hi] for a in out}
    ts = ts[lo:hi]
    p = ROOT / "runs/demos" / (name + ".json")
    p.write_text(json.dumps({"dt": DT, "source": "record_teleop", "speed": speed, "smooth_s": smooth, "left": out["left"], "right": out["right"]}))
    print(f"{p}: {len(ts)} rows = {len(ts) * DT:.1f} s")


if __name__ == "__main__":
    {"start": start, "stop": stop, "convert": convert}[sys.argv[1]](*sys.argv[2:])
