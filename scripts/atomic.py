#!/usr/bin/env python3
"""Atomic WALLX tests: one raw instruction per short session (relay --passthrough, no harness), arms homed before and after.
Per instruction: did the parcel move (WALLX engaged it) and did a bin move (mat uncovered where it stood)? Stops when a bin moves.

  .venv/bin/python scripts/atomic.py --seconds 25 "pick up the box" "pick up the brown box in the middle of the table" ...
Writes runs/atomic/<ts>/<k>/ (images, relay.out), results.json and montage.jpg (4 frames per instruction).
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_trial import fronts, home, session  # noqa: E402

sys.path.insert(0, str(ROOT))
from labelloop.config import load_yaml  # noqa: E402
from labelloop.harness import apply_profile  # noqa: E402
from labelloop.perception import scene  # noqa: E402

CFG = apply_profile({**load_yaml(ROOT / "configs/task_qr_sort.yaml"), "profile": "real", "event_fast": True})["scene"]


def rgb(f):
    return cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)


def iou(a, b):
    if a is None or b is None:
        return 0.0 if (a is None) != (b is None) else 1.0
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def judge(before, after):
    b, a = rgb(before), rgb(after)
    ob, oa = scene.object_box(b, None, CFG), scene.object_box(a, None, CFG)
    vac = {c: round(scene.mat_fraction(a, CFG, r) - scene.mat_fraction(b, CFG, r), 2) for c, r in CFG["bin_rois"].items()}
    return {"parcel_moved": iou(ob, oa) < 0.7, "parcel_gone": oa is None and ob is not None, "bins_vacated": vac,
            "bin_moved": [c for c, v in vac.items() if v >= 0.15]}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("instructions", nargs="*")
    a.add_argument("--recipe", default="", help="scenarios/<atomic recipe>.json (instruction, relay, seconds); runs append to scenarios/RESULTS.md")
    a.add_argument("--seconds", type=int, default=25)
    a.add_argument("--relay", default="", help="extra relay flags for every session, e.g. '--lock-arm right'")
    o = a.parse_args()
    rec = json.loads(Path(o.recipe).read_text()) if o.recipe else None
    if rec:
        o.instructions, o.relay, o.seconds = [rec["instruction"]], rec.get("relay", ""), rec.get("seconds", o.seconds)
    out = ROOT / "runs/atomic" / (((rec["name"] + "_") if rec else "") + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)
    results, tiles = [], []
    for k, instr in enumerate(o.instructions):
        d = out / str(k)
        d.mkdir()
        home(d)
        n0 = len(fronts(d))
        session(d, ["--passthrough", "--no-record", *o.relay.split()], instr, o.seconds, home=True)
        frames = fronts(d)[n0:]
        home(d)
        after = fronts(d)[-1]
        r = {"k": k, "relay": o.relay, "instruction": instr, **judge(frames[0], after)} if frames else {"k": k, "instruction": instr, "error": "no frames"}
        results.append(r)
        print(json.dumps(r), flush=True)
        row = [frames[i] for i in np.linspace(0, len(frames) - 1, 3).astype(int)] + [after] if frames else []
        ims = [cv2.resize(cv2.imread(str(f)), (400, 225)) for f in row]
        if ims:
            cv2.putText(ims[0], "%d: %s" % (k, instr[:48]), (6, 20), 0, 0.5, (0, 0, 255), 2)
            tiles.append(np.hstack(ims))
        (out / "results.json").write_text(json.dumps(results, indent=1))
        if rec:
            with open(ROOT / "scenarios/RESULTS.md", "a") as f:
                f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} | {rec['name']} | parcel_moved {r.get('parcel_moved')} gone {r.get('parcel_gone')} | "
                        f"bin_moved {r.get('bin_moved')} | {out.relative_to(ROOT)} | observed: (fill in)\n")
        if r.get("bin_moved"):
            print("STOP: bin moved:", r["bin_moved"], flush=True)
            break
    if tiles:
        w = max(t.shape[1] for t in tiles)
        cv2.imwrite(str(out / "montage.jpg"), np.vstack([np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in tiles]), [cv2.IMWRITE_JPEG_QUALITY, 70])
    print("results:", out)


if __name__ == "__main__":
    main()
