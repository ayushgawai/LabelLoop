"""Parcel tracking for the dashboard's head-camera frames: find the parcel in the first frame (largest object on the table that is not
mat, not tape, not a parked claw), then follow it frame by frame with a hue/saturation histogram (CamShift). Cached per trial in
track.json; draw() puts the box on a frame. ponytail: colour tracking, so a white parcel can jump to the white arm when they overlap."""
import json
import threading
from pathlib import Path

import cv2
import numpy as np


def _table_mask(hsv):
    H, W = hsv.shape[:2]
    mat = ((hsv[..., 0] > 70) & (hsv[..., 0] < 100) & (hsv[..., 1] > 60)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mat)
    inside = np.zeros((H, W), np.uint8)
    if n > 1:
        k = 1 + int(np.argmax(st[1:, 4]))
        cv2.fillConvexPoly(inside, cv2.convexHull(np.column_stack(np.nonzero(lab == k)[::-1]).astype(np.int32)), 1)
    inside[int(0.55 * H):, :int(0.20 * W)] = 0  # parked claws
    inside[int(0.55 * H):, int(0.78 * W):] = 0
    return inside, mat


def find_parcels(img_bgr, min_area=1500):
    """[(x, y, w, h)] of every loose parcel on the table (not a tray / bin in a zone, not the table edge), largest first."""
    from labelloop.sequence import ZONE_BOXES
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    inside, mat = _table_mask(hsv)  # the mat's hull (no wall), minus the zones (trays / bins) and parked claws
    H, W = inside.shape
    for x0, y0, x1, y1 in ZONE_BOXES:
        inside[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = 0
    m = ((mat == 0) & (inside > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))  # thin tape lines and digits drop out
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h, a in sorted(st[1:], key=lambda r: -r[4]) if a > min_area]


def find_parcel(img_bgr):
    """(x, y, w, h) of the loose parcel (the largest one), or None."""
    boxes = find_parcels(img_bgr)
    return boxes[0] if boxes else None


_STATE, _LOCK = {}, threading.Lock()  # per trial: tracker state kept in memory, so a live trial only processes its NEW frames


def track(trial_dir):
    """{row index: [x, y, w, h]} of the parcel in every recorded head frame. Incremental (new frames only) and thread-safe;
    finished trials are also cached in <trial>/track.json."""
    d = Path(trial_dir)
    cache = d / "track.json"
    frames = sorted((d / "frames").glob("*_front.jpg"), key=lambda p: int(p.name.split("_")[0]))
    with _LOCK:
        st = _STATE.get(str(d))
        if st is None and cache.exists():
            try:
                c = json.loads(cache.read_text())
                if c.get("n") == len(frames):
                    return {int(k): v for k, v in c["boxes"].items()}
            except (OSError, ValueError):
                pass
        if st is None:
            st = _STATE[str(d)] = {"boxes": {}, "hist": None, "win": None, "seen": set()}
        todo = [p for p in frames if p.name not in st["seen"]]
        if not todo:
            return dict(st["boxes"])
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        boxes = st["boxes"]
        for p in todo:
            img = cv2.imread(str(p))
            if img is None:  # still being written: try again next time
                break
            st["seen"].add(p.name)
            n = int(p.name.split("_")[0])
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if st["hist"] is None:
                b = find_parcel(img)
                if not b:
                    continue
                x, y, w, h = b
                roi, mroi = hsv[y:y + h, x:x + w], cv2.inRange(hsv[y:y + h, x:x + w], (0, 0, 0), (180, 255, 255))
                mroi[(roi[..., 0] > 70) & (roi[..., 0] < 100) & (roi[..., 1] > 60)] = 0  # the parcel's colours, not the mat
                hist = cv2.calcHist([roi], [0, 1], mroi, [30, 32], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                st["hist"], st["win"] = hist, b
                boxes[n] = list(b)
                continue
            inside, _ = _table_mask(hsv)
            bp = cv2.calcBackProject([hsv], [0, 1], st["hist"], [0, 180, 0, 256], 1)
            bp[inside == 0] = 0
            _, w2 = cv2.CamShift(bp, tuple(int(v) for v in st["win"]), crit)
            x, y, w, h = [int(v) for v in w2]
            conf = float(bp[y:y + h, x:x + w].mean()) if w > 8 and h > 8 else 0.0
            if conf < 40:  # not sure (lifted out of view / hidden by the arm): re-find it if a table object has the parcel's colours
                f = find_parcel(img)
                if f and float(bp[f[1]:f[1] + f[3], f[0]:f[0] + f[2]].mean()) >= 15:
                    st["win"] = f
                    boxes[n] = list(f)
                    continue
                last = boxes[max(boxes)] if boxes else st["win"]  # else no box this frame, search near the last one
                st["win"] = (max(0, last[0] - 40), max(0, last[1] - 40), last[2] + 80, last[3] + 80)
                continue
            st["win"] = (x, y, w, h)
            boxes[n] = [x, y, w, h]
        if len(st["seen"]) == len(frames):
            try:
                cache.write_text(json.dumps({"n": len(frames), "boxes": boxes}))
            except OSError:
                pass
        return dict(boxes)


GRID_X, GRID_Y = (0.38, 0.62), (0.42, 0.62)  # = labelloop.sequence.where_words thresholds (image fractions)


def draw_grid(img, cell_xy=None):
    """The harness's 3x3 table grid (back/middle/front x left/middle/right) on a head frame; the cell holding cell_xy filled."""
    H, W = img.shape[:2]
    xs, ys = [0.05, *GRID_X, 0.95], [0.17, *GRID_Y, 0.99]
    rows, cols = ("back", "middle", "front"), ("left", "middle", "right")
    if cell_xy:
        c = sum(cell_xy[0] >= v for v in GRID_X)
        r = sum(cell_xy[1] >= v for v in GRID_Y)
        over = img.copy()
        cv2.rectangle(over, (int(xs[c] * W), int(ys[r] * H)), (int(xs[c + 1] * W), int(ys[r + 1] * H)), (40, 140, 255), -1)
        img[:] = cv2.addWeighted(over, 0.28, img, 0.72, 0)
    for v in GRID_X:
        cv2.line(img, (int(v * W), int(ys[0] * H)), (int(v * W), int(ys[-1] * H)), (255, 255, 255), 2)
    for v in GRID_Y:
        cv2.line(img, (int(xs[0] * W), int(v * H)), (int(xs[-1] * W), int(v * H)), (255, 255, 255), 2)
    for r in range(3):
        for c in range(3):
            name = "centre" if (r, c) == (1, 1) else "%s %s" % (rows[r], cols[c])
            cv2.putText(img, name, (int(xs[c] * W) + 8, int(ys[r] * H) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img


def draw(jpeg_path, box, grid=False, cell_xy=None):
    """JPEG bytes of the frame with the tracked parcel boxed (and the harness grid when asked)."""
    img = cv2.imread(str(jpeg_path))
    if grid:
        draw_grid(img, cell_xy)
    if box:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 255), 3)
        cv2.putText(img, "parcel", (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
