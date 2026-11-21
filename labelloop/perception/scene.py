import cv2
import numpy as np


def _mask(img, ranges, k):
    """Union of HSV [lo, hi] ranges (more than one for red, which wraps at H 0), opened with a k x k kernel."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))


def inside(p, box):
    return box is not None and box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def roi_box(img, cfg, key="induction_roi", sub=None):
    """Pixel box of cfg[key] (or cfg[key][sub], e.g. a bin colour in bin_rois)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = cfg[key] if sub is None else cfg[key][sub]
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def zone_ref(img, cfg, key="induction_roi", sub=None):
    """Blurred gray crop of the induction zone (or another ROI): the reference, or the current frame to compare to it."""
    x0, y0, x1, y1 = roi_box(img, cfg, key, sub)
    return cv2.GaussianBlur(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY), (img.shape[1] // 100 | 1,) * 2, 0)


def presence(img, ref, cfg, key="induction_roi", sub=None):
    """Fraction of the ROI (default the induction zone) that differs from its reference."""
    return float(_changed(img, ref, cfg, key, sub).mean())


def _changed(img, ref, cfg, key="induction_roi", sub=None):
    cur = zone_ref(img, cfg, key, sub)
    return cv2.absdiff(cur, cv2.resize(ref, cur.shape[::-1])) > cfg["presence_thr"]


def centroid(img, ref, cfg):
    """(x, y) of the changed pixels of the induction zone in full-image coords (zone centre if nothing changed)."""
    x0, y0, x1, y1 = roi_box(img, cfg)
    ys, xs = np.nonzero(_changed(img, ref, cfg))
    return (x0 + xs.mean(), y0 + ys.mean()) if len(xs) else ((x0 + x1) / 2, (y0 + y1) / 2)


def kraft_in_roi(img, cfg):
    """Fraction of the induction zone that is kraft-coloured (parcel vs garbage)."""
    x0, y0, x1, y1 = roi_box(img, cfg)
    return float((_mask(img, [cfg["kraft"]], max(2, img.shape[1] // 200))[y0:y1, x0:x1] > 0).mean())
