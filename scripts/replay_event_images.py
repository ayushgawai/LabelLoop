"""Offline checker for real robot images saved by the event relay: DIR/<episode>/<call>_<camera>.jpg (+ <call>_state.json).
usage: python scripts/replay_event_images.py runs/event_images [--episode NAME] [--profile real|sim] [--grid]
Per front image: QR codes (1x + the harness's 2x changed-crop retry), bin-region split, change vs the episode's first frame; annotated PNG in <episode>/annotated/.
--profile (default real): the scene section of configs/task_qr_sort.yaml with that profile's bin_rois / induction_roi / whole_table overrides applied
(as harness.apply_profile does), so the bin regions are the ones the relay uses. --grid: the table grid (labelloop.perception.table_grid) under the
annotations; skipped with a note while that module is not merged."""
import argparse
import importlib
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labelloop.config import load_yaml
from labelloop.perception import scene
from labelloop.perception.qr import locate

CFG = Path(__file__).resolve().parent.parent / "configs" / "task_qr_sort.yaml"


def read(p):
    img = cv2.imread(str(p))
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def scene_cfg(cfg_path=CFG, profile="real"):
    """The task yaml's scene section with the profile's overrides (bin_rois, induction_roi, whole_table), as harness.apply_profile applies them."""
    y = load_yaml(cfg_path)
    p = y["profiles"][profile]
    return {**y["scene"], **{k: p[k] for k in ("bin_rois", "induction_roi", "whole_table") if p.get(k) is not None}}


def table_grid():
    """The labelloop.perception.table_grid module, or None while it is not merged."""
    try:
        return importlib.import_module("labelloop.perception.table_grid")
    except ImportError:
        return None


def front_codes(img, ref, cfg):
    """Harness logic: locate on the frame; if nothing, retry at 2x on the padded changed-pixel crop."""
    codes = locate(img)
    cb = scene.changed_box(img, ref, cfg) if ref is not None else None
    if not codes and cb:
        cb = (max(cb[0] - 20, 0), max(cb[1] - 20, 0), cb[2] + 20, cb[3] + 20)
        crop = cv2.resize(img[cb[1]:cb[3], cb[0]:cb[2]], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        codes = [(t, (cb[0] + x / 2, cb[1] + y / 2)) for t, (x, y) in locate(crop)]
    return codes


def analyse(img, ref, cfg):
    """-> dict(accepted, ignored: [(text, (x, y))], frac, box)."""
    codes = front_codes(img, ref, cfg)
    ign = [c for c in codes if scene.in_bin(c[1], img, cfg)]
    acc = [c for c in codes if c not in ign]
    try:
        frac, box = scene.presence(img, ref, cfg), scene.changed_box(img, ref, cfg)
    except Exception as e:  # odd image / config mismatch: keep the code report
        print(f"  change detection failed: {e}", file=sys.stderr)
        frac, box = float("nan"), None
    return dict(accepted=acc, ignored=ign, frac=frac, box=box)


def annotate(img, res, cfg):
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    m = cfg.get("bin_margin", 0.03)
    for name, b in cfg["bin_rois"].items():
        x0, y0, x1, y1 = scene._frac_box(img, (b[0] - m, b[1] - m, b[2] + m, b[3] + m))
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 128, 0), 2)
        cv2.putText(out, name, (x0 + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
    if res["box"]:
        cv2.rectangle(out, tuple(int(v) for v in res["box"][:2]), tuple(int(v) for v in res["box"][2:]), (0, 255, 255), 2)
    for lst, col in ((res["accepted"], (0, 200, 0)), (res["ignored"], (0, 0, 255))):
        for t, (x, y) in lst:
            cv2.circle(out, (int(x), int(y)), 8, col, -1)
            cv2.putText(out, t[:24], (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return out


def run(root, episode=None, cfg_path=CFG, profile="real", grid=False):
    cfg = scene_cfg(cfg_path, profile)
    tg = table_grid() if grid else None
    if grid and tg is None:
        print("--grid: labelloop.perception.table_grid is not available (Task 1 not merged yet): no grid", file=sys.stderr)
    print(f"profile {profile}: bin_rois {cfg['bin_rois']}, induction_roi {cfg['induction_roi']}, whole_table {cfg.get('whole_table')}")
    root = Path(root)
    eps = [root / episode] if episode else sorted(d for d in root.iterdir() if d.is_dir())
    rows = []
    for ep in eps:
        ref = None
        print(f"== {ep.name}")
        print(f"{'call':>6} {'accepted':<28} {'ignored(bin)':<28} {'change':>7}  box / wrist codes")
        for f in sorted(ep.glob("*_front.jpg"), key=lambda p: p.name.split("_")[0].zfill(8)):
            call = f.name.split("_")[0]
            img = read(f)
            if img is None:
                print(f"{call:>6} unreadable {f.name}")
                continue
            if ref is None:
                ref = scene.zone_ref(img, cfg)
            res = analyse(img, ref, cfg)
            wr = {}
            for cam in ("left_wrist", "right_wrist"):
                w = read(ep / f"{call}_{cam}.jpg")
                if w is not None:
                    wr[cam] = sorted({t for t, _ in locate(w)})
            res["quad"] = tg.find_table_quad(img) if tg else None
            if tg:
                print(f"{call:>6} grid corners {res['quad'].round(1).tolist()}" if res["quad"] is not None else f"{call:>6} grid: no table quad found")
            (ep / "annotated").mkdir(exist_ok=True)
            base = tg.TableGrid(res["quad"]).draw(img) if res["quad"] is not None else img
            cv2.imwrite(str(ep / "annotated" / f"{call}_front.png"), annotate(base, res, cfg))
            box = tuple(int(v) for v in res["box"]) if res["box"] else None
            fmt = lambda l: ",".join(t for t, _ in l) or "-"
            print(f"{call:>6} {fmt(res['accepted']):<28} {fmt(res['ignored']):<28} {res['frac']:7.3f}  {box} {wr or ''}")
            for t, (x, y) in res["accepted"] + res["ignored"]:
                print(f"         {t!r} at ({x:.0f},{y:.0f}) {'IGNORED(bin)' if (t, (x, y)) in res['ignored'] else 'accepted'}")
            rows.append((ep.name, call, res))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--episode")
    ap.add_argument("--profile", choices=("real", "sim"), default="real", help="scene overrides of this task-yaml profile (bin / induction regions)")
    ap.add_argument("--grid", action="store_true", help="draw the table grid (needs labelloop.perception.table_grid)")
    a = ap.parse_args()
    run(a.dir, a.episode, profile=a.profile, grid=a.grid)
