"""Synthetic trials for the dashboard (dev + tests, no robot, no WALLX): the right / left arm picks a parcel and places it in its bin,
recorded in the trial_recorder layout with HOME-RELATIVE EE states that the arm model can reach (keyframes planned with look.reach,
joint-space interpolation between them).

  python -m labelloop.dashboard.demo [--root runs/demo/trials]     then   python -m labelloop.dashboard.server --root runs/demo/trials
"""
import argparse
import functools
import json
import time
from pathlib import Path

import cv2
import numpy as np

from labelloop import look
from labelloop.dashboard.kinematics import FRAME, GRIP_MAX, HOME_R, chain
from labelloop.pose import matrix_to_rpy

REAL_FRONT = Path(__file__).resolve().parents[2] / "tests" / "data" / "real_head_qr.jpg"
LANES = {"A": ("left_ee", "blue", (0.58, 0.17)), "B": ("right_ee", "red", (0.58, -0.17))}  # arm, bin, root xy over the bin
CODES = {"A": "420/95126/001", "B": "420/90012/001"}


def state_of(q, grip):
    """Home-relative WALLX-style EE state of joint angles q (the inverse of kinematics.absolute)."""
    pts, R = chain(q)
    return {"position": np.round(pts[-1] - look.fk(np.zeros(6))[0], 5).tolist(),
            "rotation_rpy": np.round(matrix_to_rpy(R @ HOME_R.T), 5).tolist(), "gripper": round(float(grip), 4)}


@functools.lru_cache(maxsize=None)  # deterministic (look.reach seeds its IK): plan once per arm + spot
def _keyframes(arm, parcel):
    """Joint keyframes (q, gripper) of one pick-and-place: home, above the parcel, down, grip, lift, over the bin, release, home."""
    cfg = {"arm_y": FRAME["arm_y"], "arm_z": FRAME["arm_z"]}
    over = dict(LANES)["A" if arm == "left_ee" else "B"][2]
    tips = [(*parcel, 0.16, GRIP_MAX), (*parcel, 0.05, GRIP_MAX), (*parcel, 0.05, 0.3), (*parcel, 0.2, 0.3),
            (*over, 0.22, 0.3), (*over, 0.22, GRIP_MAX)]
    keys, q = [(np.zeros(6), 0.0)], None
    for x, y, z, g in tips:
        r = next(filter(None, (look.reach(arm, np.array([x, y, look.TABLE_Z + z]), np.radians(p), cfg, q0=q) for p in (90, 80, 70, 60, 45, 30, 15, 0))), None)
        if r is None:
            raise ValueError("demo keyframe unreachable: %s" % ((x, y, z),))
        q = r[2]
        keys.append((q, g))
    return keys + [(np.zeros(6), 0.0)]


def _path(keys, n):
    """n (q, gripper) samples through the keyframes: smoothstep per leg, equal calls per leg."""
    out, legs = [], len(keys) - 1
    for k in range(n):
        u = k / max(n - 1, 1) * legs
        i = min(int(u), legs - 1)
        f = u - i
        f = f * f * (3 - 2 * f)
        (q0, g0), (q1, g1) = keys[i], keys[i + 1]
        out.append((q0 + f * (q1 - q0), g0 + f * (g1 - g0)))
    return out


def _jpeg(img_rgb, q=70):
    return cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def make_trial(root, tid, lane="B", n=90, start_epoch=None, outcome="ok", retry=False, flags=(), frames=True, frame_every=3,
               overlay=True, goal="Read each parcel label and sort it", parcel=(0.40, -0.06), wording=None):
    """One synthetic trial folder root/<tid>/ (meta.json, trial.jsonl, frames/, summary.json). outcome "ok" | "failed";
    retry = a second PLACE try after a failed first; flags = [(row, flag)] harness flags (e.g. "bin_moved", "grasp:missed")."""
    d = Path(root) / tid
    (d / "frames").mkdir(parents=True, exist_ok=True)
    t0 = start_epoch if start_epoch is not None else time.mktime(time.strptime(tid, "%Y%m%d_%H%M%S"))
    arm, mat, _ = LANES[lane]
    other = "left_ee" if arm == "right_ee" else "right_ee"
    place = wording or "use the {arm} arm to put the box into the {mat} bin"
    instr = place.format(arm=arm[:-3], mat=mat)
    idle, scan, home = 4, 5, 6
    k = n - idle - scan - home - 1  # PLACE calls: the trial has exactly n rows (the last one DONE)
    moves = _path(_keyframes(arm, tuple(parcel)), k)
    if retry:  # first try: the same motion at double speed whose gripper never closes, then the real one
        half = moves[::2]
        moves = ([(q, GRIP_MAX) for q, _ in half] + half)[:k]
    plan = [("IDLE", None)] * idle + [("SCAN", None)] * scan + [("PLACE", m) for m in moves] + [("HOME", None)] * home + [("DONE", None)]
    flag_at = {int(i): f for i, f in flags}
    rest = state_of(np.zeros(6), 0.0)
    base = cv2.cvtColor(cv2.imread(str(REAL_FRONT)), cv2.COLOR_BGR2RGB) if frames and REAL_FRONT.exists() else None
    rows, t, last_q, prev_state = [], t0, (np.zeros(6), 0.0), None
    rng = np.random.default_rng(int(tid[-6:]))
    with open(d / "trial.jsonl", "w", encoding="utf-8") as f:
        for i, (st, m) in enumerate(plan):
            if m is not None:
                last_q = m
            elif st == "HOME":
                f_ = (i - idle - scan - len(moves) + 1) / home
                last_q = (last_q[0] * (1 - f_), last_q[1] * (1 - f_))
            tr = []
            if st != prev_state and prev_state is not None:
                tr = [[prev_state, st, {"SCAN": "object in zone", "PLACE": "routed", "HOME": "outcome ok" if outcome == "ok" else "budget",
                                        "DONE": outcome}[st]]]
            if retry and st == "PLACE" and i == idle + scan + len(moves) // 2:
                tr = [["PLACE", "PLACE", "retry"]]
            prev_state = st
            wall = st in ("PLACE", "HOME")
            state = {arm: state_of(*last_q), other: rest}
            nxt = [plan[k][1] for k in range(i + 1, min(len(plan), i + 25)) if plan[k][1] is not None][::3][:8]
            actions = {arm: {"position": [state_of(*x)["position"] for x in nxt], "rotation_rpy": [state_of(*x)["rotation_rpy"] for x in nxt],
                             "gripper": [state_of(*x)["gripper"] for x in nxt]}} if wall and nxt else None
            codes = [[CODES[lane], [640, 430]]] if st in ("SCAN", "PLACE") else []
            h = {"call": i + 1, "state": st, "instruction": instr if st == "PLACE" else None, "codes": codes, "confirmed": [c[0] for c in codes][:1] if i > idle + 1 else [],
                 "lane": lane if st != "IDLE" and st != "SCAN" else None, "presence": 0.0 if st == "IDLE" and i < 2 else 0.12,
                 "bin_change": {"blue": 0.0, "red": 0.02 if outcome == "ok" and st in ("HOME", "DONE") else 0.0},
                 "transitions": tr, "flags": [flag_at[i]] if i in flag_at else [], "parcels": int(st == "DONE"), "spent_s": round(t - t0, 1)}
            row = {"t": round(t, 3), "index": i, "goal": goal, "subgoal": instr if wall and st == "PLACE" else ("put both arms back to the home position" if wall else None),
                   "source": "wallx" if wall else "hold", "harness": h, "state": state,
                   "latency_ms": round(float(rng.normal(85, 18)), 1) if wall else None, "elapsed_ms": round(float(rng.normal(140, 25)), 1),
                   "error": None, "actions": actions}
            f.write(json.dumps(row) + "\n")
            rows.append(row)
            if base is not None and i % frame_every == 0:
                img = base.copy()
                cv2.putText(img, "%s #%d %s" % (tid, i, st), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                (d / "frames" / ("%d_front.jpg" % i)).write_bytes(_jpeg(img))
                for cam in ("left_wrist", "right_wrist"):
                    w = np.full((240, 320, 3), 60, np.uint8)
                    cv2.putText(w, cam, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
                    (d / "frames" / ("%d_%s.jpg" % (i, cam))).write_bytes(_jpeg(w))
                if overlay and wall:
                    ov = img.copy()
                    h_, w_ = ov.shape[:2]
                    for bx0, by0, bx1, by1 in ((0.29, 0.06, 0.49, 0.47), (0.52, 0.06, 0.75, 0.47), (0.79, 0.45, 1.0, 1.0)):
                        if not (lane == "B" and bx0 == 0.52 or lane == "A" and bx0 == 0.29):
                            ov[int(by0 * h_):int(by1 * h_), int(bx0 * w_):int(bx1 * w_)] = 0
                    (d / "frames" / ("%d_front_overlay.jpg" % i)).write_bytes(_jpeg(ov))
            t += 0.6 + (0.25 if wall else 0.0)
    meta = {"trial_id": tid, "goal": goal, "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), "start_epoch": round(t0, 3),
            "profile": "real", "real_safe": True, "fast": True, "hold": False, "passthrough": False, "demo": True}
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
    (d / "summary.json").write_text(json.dumps({"trial_id": tid, "goal": goal, "calls": len(rows), "wallx_calls": len(lat),
                                                 "mean_latency_ms": round(sum(lat) / len(lat), 1), "p95_latency_ms": lat[int(0.95 * (len(lat) - 1))],
                                                 "parcels": 1, "duration_s": round(t - t0, 1), "last_state": "DONE"}, indent=1), encoding="utf-8")
    return d


def make_agent_run(agent_root, reports_root, ts, trial_dirs, notes=None, model="demo-model"):
    """A trial-agent run like scripts/trial_agent.py writes: runs/agent/<ts>/{journal.json, t<N>.jsonl (runlog rows: t, call, state,
    instruction, transitions, flags, confirmed), progress.jsonl, wording.json} + results/agent/REPORT_<ts>.md; trial N = trial_dirs[N-1]."""
    d = Path(agent_root) / ts
    d.mkdir(parents=True, exist_ok=True)
    journal, wording = [], {"PLACE": ["use the {arm} arm to put the box into the {mat} bin"]}
    for n, td in enumerate(trial_dirs, 1):
        rows = [json.loads(l) for l in (Path(td) / "trial.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        with open(d / ("t%d.jsonl" % n), "w", encoding="utf-8") as f:
            for r in rows:
                h = r["harness"]
                f.write(json.dumps({"t": r["t"], "call": h["call"], "state": h["state"], "instruction": h["instruction"],
                                    "transitions": h["transitions"], "flags": h["flags"], "confirmed": h["confirmed"]}) + "\n")
                if r["latency_ms"] is not None:
                    f.write(json.dumps({"t": r["t"], "call": h["call"], "latency_s": r["latency_ms"] / 1000, "error": None}) + "\n")
        outs = [t[2] for r in rows for t in r["harness"]["transitions"] if t[1] == "DONE"]
        journal.append({"trial": n, "note": (notes or {}).get(n, "demo trial %d" % n), "overlay": "parcel" if n % 2 else "none",
                        "wording": wording, "verdict": outs[0] if outs else "no verdict",
                        "seconds_to_verdict": round(rows[-1]["t"] - rows[0]["t"], 1), "retries": sum(t[2] == "retry" for r in rows for t in r["harness"]["transitions"])})
    (d / "journal.json").write_text(json.dumps(journal, indent=1), encoding="utf-8")
    (d / "wording.json").write_text(json.dumps(wording, indent=1), encoding="utf-8")
    with open(d / "progress.jsonl", "w", encoding="utf-8") as f:
        for j in journal:
            f.write(json.dumps({"t": "13:%02d:00" % (20 + j["trial"]), "tool": "run_trial", "args": {"note": j["note"]}, "out": json.dumps(j)}) + "\n")
    rep = Path(reports_root) / ("REPORT_%s.md" % ts)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("# Trial agent report %s (%s, %d robot sessions)\n\n# Demo run: %d trials, %d ok\n\nSynthetic report for the dashboard demo.\n\n"
                   "## Journal\n%s\n\nArtifacts: %s\n" % (ts, model, len(journal), len(journal), sum(j["verdict"] == "ok" for j in journal),
                                                         "\n".join("- " + json.dumps(j) for j in journal), d), encoding="utf-8")
    return d, rep


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m labelloop.dashboard.demo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs/demo/trials")
    p.add_argument("--agent", default="runs/demo/agent")
    p.add_argument("--reports", default="runs/demo/reports")
    a = p.parse_args(argv)
    dirs = []
    for tid, kw in (("20261121_130500", dict(lane="B")), ("20261121_132000", dict(lane="A", retry=True, flags=[(40, "grasp:missed")])),
                    ("20261121_134500", dict(lane="B", outcome="failed", n=150, flags=[(60, "bin_moved")]))):
        dirs.append(make_trial(a.root, tid, **kw))
        print("wrote", dirs[-1])
    print("wrote", *make_agent_run(a.agent, a.reports, "20261121_131900", dirs[1:], {1: "retry after a missed grasp", 2: "bin moved: HOLD"}))
    print("serve: python -m labelloop.dashboard.server --root %s --agent %s --reports %s" % (a.root, a.agent, a.reports))


if __name__ == "__main__":
    main()
