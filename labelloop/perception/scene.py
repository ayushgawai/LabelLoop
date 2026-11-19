import cv2
import numpy as np


def _mask(img, lo, hi, k):
    m = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), np.array(lo), np.array(hi))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))


def _blobs(mask, min_area):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [tuple(int(v) for v in stats[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def bins(img, cfg):
    """{"blue": (x0, y0, x1, y1) | None, "orange": ...}: bbox of the largest blob of each bin colour."""
    h, w = img.shape[:2]
    out = {}
    for color in ("blue", "orange"):
        found = _blobs(_mask(img, *cfg[color], max(2, w // 200)), cfg["bin_min_area"] * h * w)
        x, y, bw, bh, _ = max(found, key=lambda b: b[4]) if found else (0, 0, 0, 0, 0)
        out[color] = (x, y, x + bw, y + bh) if found else None
    return out


def inside(p, box):
    return box is not None and box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def blocks(img, bin_boxes, cfg):
    """Centres of kraft blobs outside the bins. Closing merges a cube with its white QR label."""
    h, w = img.shape[:2]
    m = _mask(img, *cfg["kraft"], max(2, w // 200))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((w // 30, w // 30), np.uint8))
    boxes = [(x + bw / 2, y + bh / 2) for x, y, bw, bh, _ in _blobs(m, cfg["block_min_area"] * h * w)]
    return [c for c in boxes if not any(inside(c, b) for b in bin_boxes.values())]


def kraft_in_bins(img, bin_boxes, cfg):
    """Fraction of the image taken by kraft pixels inside each bin bbox: a cube that landed there raises it."""
    h, w = img.shape[:2]
    m = _mask(img, *cfg["kraft"], max(2, w // 200))
    return {c: 0.0 if b is None else float((m[b[1]:b[3], b[0]:b[2]] > 0).sum()) / (h * w) for c, b in bin_boxes.items()}


def where(cx, w):
    return "on the left" if cx < w / 3 else "in the middle" if cx < 2 * w / 3 else "on the right"
