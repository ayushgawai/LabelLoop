#!/usr/bin/env python3
"""Record a controller (teleop) demo ourselves (the robot's debug recorder stopped writing bags): a ROS listener inside the robot
client container logs /left_arm/end_pose, /right_arm/end_pose and the gripper commands; on stop it is fetched and converted to a
replayable demo (same format as scripts/demo_from_bag.py: home-relative poses + gripper, 0.0337 s rows).

  .venv/bin/python scripts/record_teleop.py start NAME        # starts recording on the robot (max 15 min)
  .venv/bin/python scripts/record_teleop.py stop NAME         # stops, fetches, writes runs/demos/NAME.json
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


def stop(name):
    sh("docker exec desktop_client-robot_client-1 pkill -f ll_rec.py")
    raw = sh("docker exec desktop_client-robot_client-1 cat /tmp/ll_%s.jsonl" % name).stdout
    rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    (ROOT / "runs/demos" / (name + ".raw.jsonl")).write_text(raw)
    pose = {a: np.array([[r["t"], *r["p"], *quat_rpy(*r["q"])] for r in rows if r["k"] == a]) for a in ("left", "right")}
    grip = {a: np.array([[r["t"], r["g"]] for r in rows if r["k"] == a + "_g"]) for a in ("left", "right")}
    t0, t1 = max(pose[a][0, 0] for a in pose), min(pose[a][-1, 0] for a in pose)
    ts = np.arange(t0, t1, DT)
    out = {}
    for a in ("left", "right"):
        P = np.column_stack([np.interp(ts, pose[a][:, 0], pose[a][:, k]) for k in range(1, 7)])
        g = GRIP_SCALE * np.interp(ts, grip[a][:, 0], grip[a][:, 1]) if len(grip[a]) else np.zeros(len(ts))
        out[a] = np.column_stack([P, g]).round(5).tolist()
    p = ROOT / "runs/demos" / (name + ".json")
    p.write_text(json.dumps({"dt": DT, "source": "record_teleop", "left": out["left"], "right": out["right"]}))
    print(f"{p}: {len(ts)} rows = {len(ts) * DT:.1f} s")


if __name__ == "__main__":
    {"start": start, "stop": stop}[sys.argv[1]](sys.argv[2])
