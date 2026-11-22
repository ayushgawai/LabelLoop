#!/usr/bin/env python3
"""Teach and repeat: fetch the newest robot control bag (the debug recorder saves every session, teleop included) and turn it into a
replayable demo (home-relative end poses + grippers, resampled to WALLX's row time 0.0337 s).

  .venv/bin/python scripts/demo_from_bag.py NAME [--bag LOCAL.mcap] [--trim 0.002]
-> runs/demos/NAME.json {"dt": 0.0337, "left": [[x, y, z, roll, pitch, yaw, gripper], ...], "right": [...]}
Idle time at the start / end (both arms within --trim m of their first / last pose) is cut. Replay with a sequence step
{"replay": "runs/demos/NAME.json", "calls": N} (labelloop/sequence.py).
"""
import argparse, json, os, subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASKPASS = os.environ.get("ROBOT_ASKPASS", str(Path.home() / ".config/labelloop/robot_askpass.sh"))
DT = 0.0337
GRIP_SCALE = 0.55  # robot gripper command units (0 closed .. 1.53 open) -> relay / WALLX units (0 .. ~0.84 open)


def quat_rpy(x, y, z, w):  # extrinsic xyz (roll, pitch, yaw), as the robot client reports rotation_rpy
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def fetch_latest(out):
    env = {**os.environ, "SSH_ASKPASS": ASKPASS, "SSH_ASKPASS_REQUIRE": "force", "DISPLAY": ":0"}
    ssh = ["ssh", "-o", "ConnectTimeout=8", "xr@192.168.1.21"]
    d = subprocess.run(ssh + ["ls -td /home/xr/bagfiles/debug/control/sealed/*/ | head -1"], env=env, capture_output=True, text=True).stdout.strip()
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", "-q", "-o", "ConnectTimeout=8", "xr@192.168.1.21:%s*.mcap" % d, str(out)], env=env, check=True)
    return sorted(out.glob(Path(d.rstrip("/")).name + "*.mcap"))[0]


def read(bag):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    pose, grip = {"left": [], "right": []}, {"left": [], "right": []}
    with open(bag, "rb") as fh:
        for _, ch, raw, m in make_reader(fh, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=["/left_arm/end_pose", "/right_arm/end_pose", "/left_gripper_controller/commands", "/right_gripper_controller/commands"]):
            t = raw.log_time * 1e-9
            if ch.topic.endswith("gripper_controller/commands"):  # commanded, not measured: a held box stops the fingers at 0.2-0.6
                grip[ch.topic.split("_")[0].strip("/")].append((t, m.data[0]))  # while the command is 0 (squeeze); replaying that would barely grip
            else:
                p, q = m.pose.position, m.pose.orientation
                rpy = quat_rpy(q.x, q.y, q.z, q.w)
                pose[ch.topic.split("_")[0].strip("/")].append((t, p.x, p.y, p.z, *rpy))
    return {a: np.array(v) for a, v in pose.items()}, {a: np.array(v) for a, v in grip.items()}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("name")
    a.add_argument("--bag", default="")
    a.add_argument("--trim", type=float, default=0.002)
    o = a.parse_args()
    bag = Path(o.bag) if o.bag else fetch_latest(ROOT / "runs/bags")
    pose, grip = read(bag)
    t0 = max(pose["left"][0, 0], pose["right"][0, 0])
    t1 = min(pose["left"][-1, 0], pose["right"][-1, 0])
    ts = np.arange(t0, t1, DT)
    out = {}
    for arm in ("left", "right"):
        P = np.column_stack([np.interp(ts, pose[arm][:, 0], pose[arm][:, k]) for k in range(1, 7)])
        g = GRIP_SCALE * np.interp(ts, grip[arm][:, 0], grip[arm][:, 1]) if len(grip[arm]) else np.zeros(len(ts))
        out[arm] = np.column_stack([P, g])
    moving = np.zeros(len(ts), bool)
    for arm in ("left", "right"):
        X = out[arm][:, :3]
        moving |= (np.linalg.norm(X - X[0], axis=1) > o.trim) | (np.linalg.norm(X - X[-1], axis=1) > o.trim)
    idx = np.flatnonzero(moving)
    lo, hi = (max(idx[0] - 15, 0), min(idx[-1] + 15, len(ts))) if len(idx) else (0, len(ts))
    demo = {"dt": DT, "bag": bag.name, "left": out["left"][lo:hi].round(5).tolist(), "right": out["right"][lo:hi].round(5).tolist()}
    p = ROOT / "runs/demos" / (o.name + ".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(demo))
    n = hi - lo
    print(f"{p}: {n} rows = {n * DT:.1f} s from {bag.name}; right arm max reach {np.abs(out['right'][lo:hi, :3]).max(0).round(3)}, "
          f"left {np.abs(out['left'][lo:hi, :3]).max(0).round(3)}; needs ~{int(np.ceil(n / 32))} calls")


if __name__ == "__main__":
    main()
