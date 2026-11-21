"""Arm joint chains for the dashboard: a recorded EE state (HOME-RELATIVE, ~0 at rest) -> absolute EE in the arm frame -> look.ik
(warm-started row to row) -> joint angles + joint origins in the robot root frame (+x forward, +y left, +z up). numpy only.

Absolute EE = the home pose composed with the relative state, as pose.sim_to_wallx does: p = p_home + p_rel, R = R(rpy_rel) @ R_home;
the home pose is look.fk(0) = (0.1185, 0, 0.181), rpy 0 (configs/ex001_6r.yaml home_pose). At q = 0 joints 2 and 3 sit ON their URDF
limits (LO[1] = 0, HI[2] = 0), so a joint at a limit is flagged (`near`), not an IK failure: `ok` = the IK converged."""
import json
import os
import threading
from pathlib import Path

import numpy as np
import yaml

from labelloop import look
from labelloop.pose import matrix_to_rpy, rpy_to_matrix

ARMS = ("left_ee", "right_ee")
HOME_P, HOME_R = look.fk(np.zeros(6))  # EE (gripper_base_link) at rest, arm frame
GRIP_MAX = 1.526  # WALLX gripper units at fully open (97.5 mm between the fingers)
NEAR = np.radians(5)  # a joint this close to a limit is flagged (drawn orange)
IK_POS, IK_ROT = 0.01, 0.05  # accepted IK residuals (m, ||R - Rt||_F), as look.reach
VERSION = 1  # joints.json layout; bump to invalidate every cache
_CFG = Path(__file__).resolve().parents[2] / "configs" / "task_qr_sort.yaml"


def _frame(profile="real"):
    """arm_y / arm_z / table_z of a task-yaml profile (the arm bases are root + (0, +-arm_y, arm_z))."""
    try:
        p = yaml.safe_load(_CFG.read_text())["profiles"][profile]
        return {k: float(p[k]) for k in ("arm_y", "arm_z", "table_z")}
    except (OSError, KeyError, TypeError, ValueError):
        return {"arm_y": 0.263, "arm_z": 0.0, "table_z": -0.015}


FRAME = _frame()


def base_of(arm, frame=FRAME):
    """Arm base in the root frame."""
    return np.array([0.0, look.SIGN[arm] * frame["arm_y"], frame["arm_z"]])


def absolute(state_arm):
    """Home-relative EE state -> (position, rotation matrix) of the EE in the arm frame."""
    return HOME_P + np.asarray(state_arm["position"], float), rpy_to_matrix(state_arm["rotation_rpy"]) @ HOME_R


def chain(q):
    """(joint origins (6, 3), EE rotation) in the arm frame: look.fk keeping every origin (the last is the EE)."""
    p, R, pts = np.zeros(3), np.eye(3), []
    for o, ax, a in zip(look._ORG, look._AX, q):
        p, R = p + R @ o, R @ look._rot(ax, a)
        pts.append(p)
    return np.array(pts), R


def fingertip(arm, state_arm, frame=FRAME):
    """Root-frame fingertip of a (home-relative) EE state, no IK (for WALLX's predicted chunk)."""
    p, R = absolute(state_arm)
    return p + R @ look.TCP + base_of(arm, frame)


def joints(arm, state_arm, q0=None, frame=FRAME):
    """{q: [6 rad], points: [[x, y, z]] the 6 joint origins + the fingertip (root frame), base, tool / jaw (unit tool x = toward the
    fingertips, tool y = the jaw axis), gripper (0..1 open), ok (IK converged), err (m), near: [6 bool] (within NEAR of a limit)}.
    q0 = the previous row's joints (warm start: smooth playback, few iterations); None = cold start."""
    p, R = absolute(state_arm)
    q, e, er = look.ik(p, matrix_to_rpy(R), tries=8 if q0 is None else 2, q0=q0)
    pts, Rq = chain(q)
    b = base_of(arm, frame)
    near = (q - look.LO < NEAR) | (look.HI - q < NEAR)
    return {"q": np.round(q, 4).tolist(), "points": np.round(np.vstack([pts, pts[-1] + Rq @ look.TCP]) + b, 4).tolist(),
            "base": b.tolist(), "tool": np.round(Rq[:, 0], 4).tolist(), "jaw": np.round(Rq[:, 1], 4).tolist(),
            "gripper": round(float(np.clip(float(state_arm.get("gripper", 0.0)) / GRIP_MAX, 0, 1)), 3),
            "ok": bool(e < IK_POS and er < IK_ROT), "err": round(float(e), 4), "near": near.tolist()}


def _key(s):
    return tuple(np.round([*s["position"], *s["rotation_rpy"], s.get("gripper", 0.0)], 5))


def row_joints(rows, prev=None, frame=FRAME):
    """Per-row joints of both arms + WALLX's predicted fingertip path. prev = the last cached entry (warm start across calls).
    A row whose state is missing / malformed gets None for that arm; a state identical to the previous row reuses its result."""
    out, last = [], dict(prev or {})
    for i, r in enumerate(rows):
        st, ent = r.get("state") if isinstance(r, dict) else None, {"index": r.get("index", i) if isinstance(r, dict) else i}
        for arm in ARMS:
            try:
                s = st[arm]
                k = _key(s)
                p = last.get(arm)
                if p and p.get("_key") == list(k):
                    ent[arm] = p
                    continue
                j = joints(arm, s, np.array(p["q"]) if p else None, frame)  # a failed row's q is still the best seed nearby
                j["_key"] = list(k)
                ent[arm] = last[arm] = j
            except (KeyError, TypeError, ValueError, IndexError):
                ent[arm] = None
        pred = {}
        for arm, a in ((r.get("actions") or {}) if isinstance(r, dict) else {}).items():
            try:
                pred[arm] = [np.round(fingertip(arm, {"position": a["position"][k], "rotation_rpy": a["rotation_rpy"][k]}, frame), 4).tolist()
                             for k in range(len(a["position"]))]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        ent["pred"] = pred
        out.append(ent)
    return out


_LOCK = threading.Lock()


def trial_joints(trial_dir, rows):
    """Joints of every row of a trial, cached in <trial_dir>/joints.json; a live trial's cache is extended (warm-started from its last
    row), a stale one (other VERSION, fewer rows than cached) is rebuilt. Returns the API payload."""
    path = Path(trial_dir) / "joints.json"
    with _LOCK:
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
            if cache.get("version") != VERSION or len(cache.get("joints", [])) > len(rows):
                cache = None
        except (OSError, ValueError):
            cache = None
        done = cache["joints"] if cache else []
        if len(done) < len(rows):
            prev = {a: done[-1].get(a) for a in ARMS} if done else None
            done = done + row_joints(rows[len(done):], prev)
            payload = {"version": VERSION, "rows": len(done), "frame": FRAME, "lo": look.LO.tolist(), "hi": look.HI.tolist(),
                       "near_deg": float(np.degrees(NEAR)), "joints": done}
            tmp = path.with_suffix(".json.tmp")
            try:
                tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
                os.replace(tmp, path)
            except OSError:  # a read-only trial dir still gets its joints, just not cached
                pass
            return payload
        return cache
