#!/usr/bin/env python3
"""Camera -> arm calibration by measurement: move one claw to a grid of home-relative EE positions at a fixed height (scripted, slow),
find the claw tip in the head image (difference against the arms-home frame; the arm blob enters from the bottom corner; the tip is
its point farthest from that corner), fit a homography EE (x, y) -> pixel, save it.

  .venv/bin/python scripts/calibrate_claw.py --arm right --z 0.12
-> runs/calib/claw_<arm>_z<z>.json {"H": 3x3 (EE xy -> pixel), "Hinv": pixel -> EE xy, "points": [[x, y, u, v], ...], "rms_px"}
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_trial import fronts, home, session  # noqa: E402


def tip(ref, img, arm):
    d = cv2.absdiff(cv2.GaussianBlur(img, (5, 5), 0), cv2.GaussianBlur(ref, (5, 5), 0)).max(axis=2)
    m = (d > 40).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    h, w = m.shape
    corner = np.array([w - 1, h - 1]) if arm == "right" else np.array([0, h - 1])
    best = None
    for i in range(1, n):  # the arm blob touches the bottom edge on the arm's side
        x, y, bw, bh, a = st[i]
        if a < 6000 or not (y + bh >= h - 5 or y <= 5 or (arm == "right" and x + bw >= w - 5) or (arm == "left" and x <= 5)):
            continue  # the arm enters from the bottom / its side edge
        if best is None or a > st[best][4]:
            best = i
    if best is None:
        return None
    ys, xs = np.nonzero(lab == best)
    edge = (xs <= 2) | (xs >= w - 3) | (ys <= 2) | (ys >= h - 3)  # where the arm enters the image
    if edge.any():
        corner = np.array([xs[edge].mean(), ys[edge].mean()])
    dist = np.hypot(xs - corner[0], ys - corner[1])
    k = np.argmax(dist)
    return (float(xs[k]), float(ys[k])) if dist[k] > 150 else None  # a claw near its parked corner is not usable


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--arm", default="right")
    a.add_argument("--z", type=float, default=0.12)
    a.add_argument("--refit", default="", help="refit from an existing runs/calib/<ts> dir (no robot)")
    o = a.parse_args()
    xs, ys = (0.12, 0.24, 0.36), ((0.0, 0.10, 0.20) if o.arm == "right" else (-0.20, -0.10, 0.0))  # toward the table centre (in view)
    grid = [(x, y) for x in xs for y in ys]
    out = Path(o.refit) if o.refit else ROOT / "runs/calib" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    steps = [{"move": {o.arm: {"position": [0.0, 0.0, o.z], "rotation_rpy": [0, 0, 0], "gripper": 0.0}}, "calls": 3, "max_step": 0.05}]
    for x, y in grid:
        steps.append({"move": {o.arm: {"position": [x, y, o.z], "rotation_rpy": [0, 0, 0], "gripper": 0.0}}, "calls": 4, "max_step": 0.05})
    steps.append({"move": {o.arm: {"position": [0.0, 0.0, 0.0], "rotation_rpy": [0, 0, 0], "gripper": 0.0}}, "calls": 6, "max_step": 0.05})
    (out / "steps.json").write_text(json.dumps({"watchdog": False, "steps": steps}))
    if not o.refit:
      home(out)
      session(out, ["--sequence", str(out / "steps.json"), "--save-images", str(out / "images"), "--save-every", "1", "--no-record"],
            "calibrate", 6 * len(steps) + 10, home=True)
    frames = sorted((out / "images").glob("*/*_front.jpg"), key=lambda f: (f.parent.name, int(f.name.split("_")[0])))
    states = [json.loads(f.with_name(f.name.split("_")[0] + ".json").read_text())["state"] for f in frames]
    ref = cv2.imread(str(frames[0]))
    pts = []
    for f, s in zip(frames, states):
        p = np.asarray(s[o.arm + "_ee"]["position"])
        if abs(p[2] - o.z) > 0.01 or np.linalg.norm(p[:2]) < 0.05:
            continue
        t = tip(ref, cv2.imread(str(f)), o.arm)
        if t:
            pts.append([p[0], p[1], t[0], t[1]])
    pts = np.array(pts)
    if len(pts) < 4:
        raise SystemExit(f"only {len(pts)} usable points; images in {out}")
    H, _ = cv2.findHomography(pts[:, :2], pts[:, 2:], cv2.RANSAC, 15.0)
    proj = cv2.perspectiveTransform(pts[None, :, :2].astype(np.float64), H)[0]
    rms = float(np.sqrt(((proj - pts[:, 2:]) ** 2).sum(1).mean()))
    res = {"arm": o.arm, "z": o.z, "H": H.tolist(), "Hinv": np.linalg.inv(H).tolist(), "points": pts.round(4).tolist(), "rms_px": rms, "dir": str(out)}
    p = ROOT / "runs/calib" / f"claw_{o.arm}_z{o.z:.2f}.json"
    p.write_text(json.dumps(res, indent=1))
    print(f"{p}: {len(pts)} points, rms {rms:.1f} px")


if __name__ == "__main__":
    main()
