"""Parcel tracking for the dashboard's head-camera frames: find the parcel in the first frame (largest object on the table that is not
mat, not tape, not a parked claw), then follow it frame by frame with a hue/saturation histogram (CamShift). Cached per trial in
track.json; draw() puts the box on a frame. ponytail: colour tracking, so a white parcel can jump to the white arm when they overlap."""
import json
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


def find_parcel(img_bgr):
    """(x, y, w, h) of the parcel at rest, or None."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    inside, mat = _table_mask(hsv)
    m = ((mat == 0) & (inside > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))  # thin tape lines and digits drop out
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    x, y, w, h, a = st[i]
    return (int(x), int(y), int(w), int(h)) if a > 1500 else None


def track(trial_dir):
    """{row index: [x, y, w, h]} of the parcel in every recorded head frame; cached in <trial>/track.json."""
    d = Path(trial_dir)
    cache = d / "track.json"
    frames = sorted((d / "frames").glob("*_front.jpg"), key=lambda p: int(p.name.split("_")[0]))
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("n") == len(frames):
                return {int(k): v for k, v in c["boxes"].items()}
        except (OSError, ValueError):
            pass
    boxes, hist, win = {}, None, None
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    for p in frames:
        img = cv2.imread(str(p))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if hist is None:
            b = find_parcel(img)
            if not b:
                continue
            x, y, w, h = b
            roi, mroi = hsv[y:y + h, x:x + w], cv2.inRange(hsv[y:y + h, x:x + w], (0, 0, 0), (180, 255, 255))
            mat = ((roi[..., 0] > 70) & (roi[..., 0] < 100) & (roi[..., 1] > 60))
            mroi[mat] = 0  # the parcel's own colours, not the mat around it
            hist = cv2.calcHist([roi], [0, 1], mroi, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            win = b
        else:
            inside, _ = _table_mask(hsv)
            bp = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            bp[inside == 0] = 0
            _, w2 = cv2.CamShift(bp, tuple(int(v) for v in win), crit)
            x, y, w, h = [int(v) for v in w2]
            conf = float(bp[y:y + h, x:x + w].mean()) if w > 8 and h > 8 else 0.0
            if conf < 40:  # not sure (lifted out of view / hidden by the arm): re-find it if a table object has the parcel's colours
                f = find_parcel(img)
                if f and float(bp[f[1]:f[1] + f[3], f[0]:f[0] + f[2]].mean()) >= 15:
                    win = f
                    boxes[int(p.name.split("_")[0])] = list(f)
                    continue
                last = boxes[max(boxes)] if boxes else win  # else no box this frame, search near the last one
                win = (max(0, last[0] - 40), max(0, last[1] - 40), last[2] + 80, last[3] + 80)
                continue
            win = (x, y, w, h)
        boxes[int(p.name.split("_")[0])] = [int(v) for v in win]
    try:
        cache.write_text(json.dumps({"n": len(frames), "boxes": boxes}))
    except OSError:
        pass
    return boxes


def draw(jpeg_path, box):
    """JPEG bytes of the frame with the tracked parcel boxed."""
    img = cv2.imread(str(jpeg_path))
    if box:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 255), 3)
        cv2.putText(img, "parcel", (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
