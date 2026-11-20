import cv2
import numpy as np


def _mask(img, ranges, k):
    """Union of HSV [lo, hi] ranges (more than one for red, which wraps at H 0), opened with a k x k kernel."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))


def _blobs(mask, min_area):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [tuple(int(v) for v in stats[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def mats(img, cfg):
    """{colour: (x0, y0, x1, y1) | None} for each lane bin colour in cfg["bins"]: bbox of the largest blob of that colour."""
    h, w = img.shape[:2]
    out = {}
    for color, ranges in cfg["bins"].items():
        found = _blobs(_mask(img, ranges, max(2, w // 200)), cfg["mat_min_area"] * h * w)
        x, y, bw, bh, _ = max(found, key=lambda b: b[4]) if found else (0, 0, 0, 0, 0)
        out[color] = (x, y, x + bw, y + bh) if found else None
    return out


def inside(p, box):
    return box is not None and box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def kraft_in_mats(img, mat_boxes, cfg):
    """Fraction of the image taken by kraft pixels inside each mat bbox: a parcel that landed there raises it."""
    h, w = img.shape[:2]
    m = _mask(img, [cfg["kraft"]], max(2, w // 200))
    return {c: 0.0 if b is None else float((m[b[1]:b[3], b[0]:b[2]] > 0).sum()) / (h * w) for c, b in mat_boxes.items()}


def roi_box(img, cfg):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = cfg["induction_roi"]
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def zone_ref(img, cfg):
    """Blurred gray crop of the induction zone: the empty-zone reference, or the current frame to compare to it."""
    x0, y0, x1, y1 = roi_box(img, cfg)
    return cv2.GaussianBlur(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY), (img.shape[1] // 100 | 1,) * 2, 0)


def presence(img, ref, cfg):
    """Fraction of the induction zone that differs from the empty-zone reference."""
    cur = zone_ref(img, cfg)
    return float((cv2.absdiff(cur, cv2.resize(ref, cur.shape[::-1])) > cfg["presence_thr"]).mean())


def kraft_in_roi(img, cfg):
    """Fraction of the induction zone that is kraft-coloured (parcel vs garbage)."""
    x0, y0, x1, y1 = roi_box(img, cfg)
    return float((_mask(img, [cfg["kraft"]], max(2, img.shape[1] // 200))[y0:y1, x0:x1] > 0).mean())
